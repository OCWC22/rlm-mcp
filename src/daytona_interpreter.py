"""
Daytona-based code interpreter for Fleet RLM.

Replaces Modal sandboxes with Daytona sandboxes for secure,
remote Python code execution.  Uses the Daytona SDK v0.10+:

    process.code_run()   → stateless single-script execution
    code_interpreter     → stateful REPL (variables persist)
    process.exec()       → raw shell commands
"""

import atexit
import json
import logging
import os
import time
from typing import Any, Callable, Optional

from daytona import (
    CreateSandboxFromSnapshotParams,
    Daytona,
    DaytonaConfig,
)

logger = logging.getLogger(__name__)

# Registry so atexit can clean up every interpreter that was started
_LIVE_INTERPRETERS: list["DaytonaInterpreter"] = []


def _cleanup_all():
    for interp in list(_LIVE_INTERPRETERS):
        try:
            interp.shutdown()
        except Exception:
            pass


atexit.register(_cleanup_all)


class DaytonaInterpreter:
    """Code execution engine backed by a Daytona sandbox.

    Drop-in replacement for fleet-rlm's ModalInterpreter.
    Interface: start(), execute(), shutdown(), context-manager.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        target: Optional[str] = None,
        timeout: int = 600,
        auto_stop_interval: int = 30,
        env_vars: Optional[dict[str, str]] = None,
        summarize_stdout: bool = True,
    ):
        self.api_key = api_key or os.getenv("DAYTONA_API_KEY", "")
        self.api_url = api_url or os.getenv("DAYTONA_API_URL", "https://app.daytona.io/api")
        self.target = target or os.getenv("DAYTONA_TARGET", "us")
        self.timeout = timeout
        self.auto_stop_interval = auto_stop_interval
        self.env_vars = env_vars or {}
        self.summarize_stdout = summarize_stdout

        self._daytona: Optional[Daytona] = None
        self._sandbox: Optional[Any] = None
        self._started = False

        # Tools callable from sandbox code (future extension point)
        self.tools: dict[str, Callable] = {}

        # Execution history (trajectory)
        self._history: list[dict] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Create and start a Daytona sandbox."""
        if self._started:
            return

        if not self.api_key:
            raise RuntimeError(
                "DAYTONA_API_KEY is required.  Set it in .env.local or as an env var."
            )

        logger.info("Creating Daytona sandbox …")
        config = DaytonaConfig(
            api_key=self.api_key,
            api_url=self.api_url,
            target=self.target,
        )
        self._daytona = Daytona(config)

        # Forward LLM API keys into sandbox so code can call litellm/openai
        sandbox_env = {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("OPENAI_", "ANTHROPIC_", "DSPY_", "LITELLM_"))
        }
        sandbox_env.update(self.env_vars)

        params = CreateSandboxFromSnapshotParams(
            language="python",
            env_vars=sandbox_env if sandbox_env else None,
            auto_stop_interval=self.auto_stop_interval,
        )

        self._sandbox = self._daytona.create(params, timeout=self.timeout)
        logger.info("Daytona sandbox ready: %s", self._sandbox.id)

        # Pre-install common data-science packages
        self._exec_shell(
            "pip install -q numpy pandas requests 2>/dev/null || true",
            timeout=120,
        )
        self._started = True
        _LIVE_INTERPRETERS.append(self)

    def shutdown(self):
        """Delete the Daytona sandbox and release resources."""
        if self in _LIVE_INTERPRETERS:
            _LIVE_INTERPRETERS.remove(self)
        if self._sandbox and self._daytona:
            try:
                logger.info("Deleting Daytona sandbox %s", self._sandbox.id)
                self._daytona.delete(self._sandbox)
            except Exception as exc:
                logger.warning("Error deleting sandbox: %s", exc)
        self._sandbox = None
        self._daytona = None
        self._started = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False

    # ------------------------------------------------------------------
    # Shell helper
    # ------------------------------------------------------------------

    def _exec_shell(self, command: str, timeout: int = 60) -> str:
        """Run a shell command in the sandbox.  Returns stdout."""
        resp = self._sandbox.process.exec(command, timeout=timeout)
        if resp.exit_code != 0:
            logger.warning(
                "Shell command failed (exit %d): %s", resp.exit_code, command[:120]
            )
        return resp.result

    # ------------------------------------------------------------------
    # Code execution  (stateless — each call is a fresh script)
    # ------------------------------------------------------------------

    def execute(
        self,
        code: str,
        variables: Optional[dict[str, Any]] = None,
        timeout: int = 60,
    ) -> str:
        """Execute Python code via ``process.code_run()``.

        Returns stdout, or a formatted ``__SUBMIT__`` result if the
        code calls ``SUBMIT(answer=…)``.
        """
        if not self._started:
            self.start()

        full_code = self._build_execution_script(code, variables)

        start = time.time()
        try:
            # process.code_run → ExecuteResponse  (.exit_code, .result)
            resp = self._sandbox.process.code_run(full_code, timeout=timeout)
            elapsed = time.time() - start

            output = resp.result or ""
            exit_code = resp.exit_code

            self._history.append(
                {"code": code, "output": output, "exit_code": exit_code, "elapsed": f"{elapsed:.2f}s"}
            )

            if exit_code != 0:
                logger.warning("Code execution failed (exit %d)", exit_code)
                return f"[ERROR exit_code={exit_code}]\n{output}"

            # Detect SUBMIT marker written by the helper
            for line in reversed(output.strip().splitlines()):
                if line.startswith("__SUBMIT__:"):
                    payload = line[len("__SUBMIT__:"):]
                    try:
                        return json.dumps(json.loads(payload), indent=2)
                    except json.JSONDecodeError:
                        return payload

            # Truncate huge stdout to protect LLM context
            if self.summarize_stdout and len(output) > 10_000:
                output = output[:5000] + "\n…[truncated]…\n" + output[-2000:]

            return output

        except Exception as exc:
            elapsed = time.time() - start
            msg = f"[EXECUTION ERROR after {elapsed:.2f}s] {type(exc).__name__}: {exc}"
            self._history.append(
                {"code": code, "output": msg, "exit_code": -1, "elapsed": f"{elapsed:.2f}s"}
            )
            return msg

    # ------------------------------------------------------------------
    # Code execution  (stateful — variables persist across calls)
    # ------------------------------------------------------------------

    def execute_stateful(self, code: str, timeout: int = 60) -> str:
        """Execute code via ``code_interpreter.run_code()`` (stateful REPL)."""
        if not self._started:
            self.start()

        # code_interpreter.run_code → ExecutionResult  (.stdout, .stderr, .error)
        resp = self._sandbox.code_interpreter.run_code(code, timeout=timeout)

        output = resp.stdout or ""
        error_text = ""
        if resp.error:
            error_text = f"\n[ERROR] {resp.error.name}: {resp.error.value}"
            if resp.error.traceback:
                error_text += f"\n{resp.error.traceback}"
        if resp.stderr:
            error_text += f"\n[STDERR] {resp.stderr}"

        combined = (output + error_text).strip()

        self._history.append(
            {"code": code, "output": combined, "exit_code": 1 if resp.error else 0}
        )
        return combined

    # ------------------------------------------------------------------
    # Script builder
    # ------------------------------------------------------------------

    def _build_execution_script(
        self,
        code: str,
        variables: Optional[dict[str, Any]] = None,
    ) -> str:
        """Wrap user code with variable injection and the SUBMIT helper."""
        parts: list[str] = [
            "import sys, os, json, re, math, collections, itertools, functools",
        ]

        if variables:
            for name, value in variables.items():
                parts.append(f"{name} = {json.dumps(value)}")

        parts.append(
            'def SUBMIT(**kwargs):\n'
            '    """Signal the final answer from the RLM loop."""\n'
            '    import json as _json\n'
            '    print("__SUBMIT__:" + _json.dumps(kwargs))\n'
            '    sys.exit(0)\n'
        )

        parts.append(code)
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # File operations
    # ------------------------------------------------------------------

    def upload_file(self, content: bytes, remote_path: str):
        """Upload bytes to *remote_path* inside the sandbox."""
        if not self._started:
            self.start()
        self._sandbox.fs.upload_file(content, remote_path)

    def download_file(self, remote_path: str) -> bytes:
        """Download a file from the sandbox and return its bytes."""
        if not self._started:
            self.start()
        return self._sandbox.fs.download_file(remote_path)

    def list_files(self, path: str = "/home/daytona") -> list[dict]:
        """List files/dirs at *path*.  Returns list of dicts."""
        if not self._started:
            self.start()
        entries = self._sandbox.fs.list_files(path)
        return [
            {"name": f.name, "is_dir": f.is_dir, "size": getattr(f, "size", 0)}
            for f in entries
        ]

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def get_history(self) -> list[dict]:
        """Return the full execution trajectory."""
        return list(self._history)
