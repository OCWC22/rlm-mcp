"""
Daytona-based code interpreter for Fleet RLM.

Replaces Modal sandboxes with Daytona sandboxes for secure,
remote Python code execution. Communicates with the sandbox
via the Daytona SDK (process.exec / code_interpreter).
"""

import os
import json
import logging
import time
from typing import Any, Optional, Callable

from daytona import Daytona, DaytonaConfig, CreateSandboxFromSnapshotParams

logger = logging.getLogger(__name__)


class DaytonaInterpreter:
    """Code execution engine using Daytona sandboxes.

    Drop-in replacement for fleet-rlm's ModalInterpreter.
    Provides the same interface: start(), execute(), shutdown(),
    context manager, and tools dictionary.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: str = "https://app.daytona.io/api",
        target: str = "us",
        timeout: int = 600,
        auto_stop_interval: int = 30,
        env_vars: Optional[dict[str, str]] = None,
        sub_lm: Optional[Any] = None,
        max_llm_calls: int = 50,
        llm_call_timeout: int = 60,
        summarize_stdout: bool = True,
    ):
        self.api_key = api_key or os.getenv("DAYTONA_API_KEY", "")
        self.api_url = api_url or os.getenv("DAYTONA_API_URL", "https://app.daytona.io/api")
        self.target = target or os.getenv("DAYTONA_TARGET", "us")
        self.timeout = timeout
        self.auto_stop_interval = auto_stop_interval
        self.env_vars = env_vars or {}
        self.sub_lm = sub_lm
        self.max_llm_calls = max_llm_calls
        self.llm_call_timeout = llm_call_timeout
        self.summarize_stdout = summarize_stdout

        self._daytona: Optional[Daytona] = None
        self._sandbox: Optional[Any] = None
        self._started = False
        self._llm_calls_made = 0

        # Tools that can be called from sandbox code
        self.tools: dict[str, Callable] = {}

        # Execution history
        self._history: list[dict[str, str]] = []

    def start(self):
        """Create and start a Daytona sandbox."""
        if self._started:
            return

        logger.info("Creating Daytona sandbox...")
        config = DaytonaConfig(
            api_key=self.api_key,
            api_url=self.api_url,
            target=self.target,
        )
        self._daytona = Daytona(config)

        # Merge env vars: pass through LLM API keys so sandbox code
        # can call litellm / openai if needed.
        sandbox_env = {
            k: v for k, v in os.environ.items()
            if k.startswith(("OPENAI_", "ANTHROPIC_", "DSPY_", "LITELLM_"))
        }
        sandbox_env.update(self.env_vars)

        params = CreateSandboxFromSnapshotParams(
            language="python",
            env_vars=sandbox_env,
            auto_stop_interval=self.auto_stop_interval,
        )

        self._sandbox = self._daytona.create(params, timeout=120)
        logger.info(f"Daytona sandbox created: {self._sandbox.id}")

        # Install commonly needed packages
        self._exec_shell("pip install -q numpy pandas requests 2>/dev/null || true", timeout=120)
        self._started = True

    def _exec_shell(self, command: str, timeout: int = 60) -> str:
        """Run a shell command in the sandbox."""
        resp = self._sandbox.process.exec(command, timeout=timeout)
        if resp.exit_code != 0:
            logger.warning(f"Shell command failed (exit {resp.exit_code}): {command[:100]}")
        return resp.result

    def execute(
        self,
        code: str,
        variables: Optional[dict[str, Any]] = None,
        timeout: int = 60,
    ) -> str:
        """Execute Python code in the Daytona sandbox.

        Args:
            code: Python code to execute
            variables: Optional variables to inject into the execution namespace
            timeout: Execution timeout in seconds

        Returns:
            Stdout output from the code execution, or structured
            output if SUBMIT() was called.
        """
        if not self._started:
            self.start()

        # Build the full script: inject variables, helpers, then user code
        full_code = self._build_execution_script(code, variables)

        start_time = time.time()
        try:
            resp = self._sandbox.process.code_run(full_code, timeout=timeout)
            elapsed = time.time() - start_time

            output = resp.result or ""
            exit_code = resp.exit_code

            # Record in history
            self._history.append({
                "code": code,
                "output": output,
                "exit_code": exit_code,
                "elapsed": f"{elapsed:.2f}s",
            })

            if exit_code != 0:
                logger.warning(f"Code execution failed (exit {exit_code})")
                return f"[ERROR exit_code={exit_code}]\n{output}"

            # Check for SUBMIT output (JSON on last line prefixed with __SUBMIT__)
            lines = output.strip().split("\n")
            for line in reversed(lines):
                if line.startswith("__SUBMIT__:"):
                    submit_json = line[len("__SUBMIT__:"):]
                    try:
                        return json.dumps(json.loads(submit_json), indent=2)
                    except json.JSONDecodeError:
                        return submit_json

            # Truncate very long output to prevent context blowup
            if self.summarize_stdout and len(output) > 10000:
                output = output[:5000] + "\n...[truncated]...\n" + output[-2000:]

            return output

        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = f"[EXECUTION ERROR after {elapsed:.2f}s] {type(e).__name__}: {e}"
            self._history.append({
                "code": code,
                "output": error_msg,
                "exit_code": -1,
                "elapsed": f"{elapsed:.2f}s",
            })
            return error_msg

    def execute_stateful(self, code: str) -> str:
        """Execute code using the stateful code interpreter (variables persist)."""
        if not self._started:
            self.start()

        resp = self._sandbox.code_interpreter.run_code(code)
        output = resp.result or ""

        self._history.append({
            "code": code,
            "output": output,
            "exit_code": getattr(resp, "exit_code", 0),
        })

        return output

    def _build_execution_script(
        self,
        code: str,
        variables: Optional[dict[str, Any]] = None,
    ) -> str:
        """Build a complete Python script with variable injection and helpers."""
        parts = []

        # Standard library imports
        parts.append("import sys, os, json, re, math, collections, itertools, functools")

        # Inject variables
        if variables:
            for name, value in variables.items():
                parts.append(f"{name} = {json.dumps(value)}")

        # SUBMIT helper: prints a marker that execute() parses
        parts.append('''
def SUBMIT(**kwargs):
    """Signal structured output from the RLM execution."""
    import json
    print("__SUBMIT__:" + json.dumps(kwargs))
    sys.exit(0)
''')

        # User code
        parts.append(code)

        return "\n".join(parts)

    def upload_file(self, local_content: bytes, remote_path: str):
        """Upload a file to the sandbox."""
        if not self._started:
            self.start()
        self._sandbox.fs.upload_file(local_content, remote_path)

    def download_file(self, remote_path: str) -> bytes:
        """Download a file from the sandbox."""
        if not self._started:
            self.start()
        return self._sandbox.fs.download_file(remote_path)

    def list_files(self, path: str = "/home/daytona") -> list[dict]:
        """List files in the sandbox."""
        if not self._started:
            self.start()
        files = self._sandbox.fs.list_files(path)
        return [{"name": f.name, "is_dir": f.is_dir, "size": f.size} for f in files]

    def get_history(self) -> list[dict]:
        """Get execution history (trajectory)."""
        return list(self._history)

    def shutdown(self):
        """Stop and clean up the Daytona sandbox."""
        if self._sandbox and self._daytona:
            try:
                logger.info(f"Stopping Daytona sandbox {self._sandbox.id}")
                self._daytona.delete(self._sandbox)
            except Exception as e:
                logger.warning(f"Error deleting sandbox: {e}")
        self._sandbox = None
        self._daytona = None
        self._started = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False
