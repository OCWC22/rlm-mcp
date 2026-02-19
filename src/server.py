#!/usr/bin/env python3
"""
Fleet RLM MCP Server — Daytona Edition

Provides RLM (Recursive Language Model) capabilities backed by
Daytona sandboxes for secure, remote Python code execution.

    LLM writes code  →  Daytona executes it  →  output fed back  →  iterate until SUBMIT()

Usage:
    export DAYTONA_API_KEY=…
    export OPENAI_API_KEY=…        # or ANTHROPIC_API_KEY
    python -m src
"""

import atexit
import json
import logging
import os
import re
import signal
import sys
import time
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Load .env.local  (first one found wins; values never overwrite existing env)
# ---------------------------------------------------------------------------
for _env_path in (
    Path(__file__).parent.parent / ".env.local",
    Path(__file__).parent.parent.parent / ".env.local",
):
    if _env_path.exists():
        with open(_env_path) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    os.environ.setdefault(_k.strip(), _v.strip().strip("\"'"))
        break

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
)
logger = logging.getLogger("fleet-rlm")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RLM_MODEL = os.getenv("RLM_MODEL", "openai/gpt-4o")
RLM_SUBTASK_MODEL = os.getenv("RLM_SUBTASK_MODEL", "openai/gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DAYTONA_API_KEY = os.getenv("DAYTONA_API_KEY", "")
DAYTONA_API_URL = os.getenv("DAYTONA_API_URL", "https://app.daytona.io/api")
DAYTONA_TARGET = os.getenv("DAYTONA_TARGET", "us")
RLM_MAX_ITERATIONS = int(os.getenv("RLM_MAX_ITERATIONS", "15"))
MCP_TRANSPORT = os.getenv("MCP_TRANSPORT", "stdio")  # "stdio" | "streamable-http" | "sse"
MCP_PORT = int(os.getenv("MCP_PORT", "8000"))

# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------
mcp = FastMCP("fleet-rlm-daytona", host="0.0.0.0", port=MCP_PORT)

# ---------------------------------------------------------------------------
# Interpreter lifecycle
# ---------------------------------------------------------------------------
_interpreter = None


def get_interpreter():
    """Lazy-create a DaytonaInterpreter that lives for the process lifetime."""
    global _interpreter
    if _interpreter is None:
        from .daytona_interpreter import DaytonaInterpreter

        _interpreter = DaytonaInterpreter(
            api_key=DAYTONA_API_KEY,
            api_url=DAYTONA_API_URL,
            target=DAYTONA_TARGET,
            timeout=600,
            auto_stop_interval=30,
        )
        _interpreter.start()
        logger.info("Daytona sandbox ready — id=%s", _interpreter._sandbox.id)
    return _interpreter


def _shutdown_interpreter():
    global _interpreter
    if _interpreter is not None:
        _interpreter.shutdown()
        _interpreter = None


atexit.register(_shutdown_interpreter)


def _resolve_api_key(model: str) -> str:
    """Pick the right API key for a litellm model string."""
    if model.startswith("anthropic/"):
        return ANTHROPIC_API_KEY or ""
    return OPENAI_API_KEY or ""


# ---------------------------------------------------------------------------
# RLM execution loop
# ---------------------------------------------------------------------------

def _llm_completion(messages: list[dict], model: str = None) -> str:
    """Call an LLM via litellm (chat-style)."""
    import litellm

    model = model or RLM_MODEL
    api_key = _resolve_api_key(model)

    resp = litellm.completion(
        model=model,
        messages=messages,
        api_key=api_key,
        max_tokens=4096,
    )
    return resp.choices[0].message.content


_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
_CODE_STARTERS = (
    "import ", "from ", "def ", "class ", "for ", "while ", "if ",
    "print(", "x ", "y ", "result", "#", "SUBMIT",
)


def _extract_code_block(text: str) -> Optional[str]:
    """Extract the first fenced code block, or the whole text if it looks like code."""
    m = _CODE_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    first = text.strip().split("\n", 1)[0].strip()
    if any(first.startswith(s) for s in _CODE_STARTERS):
        return text.strip()
    return None


def _is_submit_result(output: str) -> Optional[str]:
    """If *output* contains a SUBMIT payload, return the answer string."""
    stripped = output.strip()
    # The SUBMIT helper prints  __SUBMIT__:{json}
    # The interpreter already decodes it into pretty JSON.
    # Case 1: raw JSON object with "answer" key
    if stripped.startswith("{") and '"answer"' in stripped:
        try:
            return json.loads(stripped).get("answer", stripped)
        except (json.JSONDecodeError, AttributeError):
            pass
    # Case 2: still has the marker (shouldn't normally reach here)
    for line in reversed(stripped.splitlines()):
        if line.startswith("__SUBMIT__:"):
            payload = line[len("__SUBMIT__:"):]
            try:
                return json.loads(payload).get("answer", payload)
            except (json.JSONDecodeError, AttributeError):
                return payload
    return None


def rlm_loop(
    task: str,
    context: str = "",
    max_iterations: Optional[int] = None,
    model: Optional[str] = None,
) -> dict:
    """Core RLM iterative code-execution loop.

    Returns ``{"answer": str, "trajectory": list, "execution_time": float}``.
    """
    max_iter = max_iterations or RLM_MAX_ITERATIONS
    model = model or RLM_MODEL
    interp = get_interpreter()
    trajectory: list[dict] = []
    start = time.time()

    system_msg = (
        "You are an RLM (Recursive Language Model) agent.\n"
        "You solve tasks by writing and executing Python code in a Daytona sandbox.\n\n"
        "RULES:\n"
        "- Write Python code to explore data, compute results, test hypotheses.\n"
        "- After each execution you see stdout/stderr.  Use it to decide your next step.\n"
        "- When you have the final answer, call  SUBMIT(answer=<your answer>)  in your code.\n"
        "- Available packages: numpy, pandas, requests, plus the stdlib.\n"
        "- Keep code concise.  Print intermediate results so you can inspect them.\n"
        "- Do NOT just describe what you would do — write executable code.\n"
    )

    messages: list[dict] = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"TASK: {task}"},
    ]
    if context:
        messages.append(
            {"role": "user", "content": f"CONTEXT (available as variable `context`):\n{context}"}
        )

    for iteration in range(max_iter):
        if iteration == 0:
            messages.append({"role": "user", "content": "Write your first Python code to start solving this task:"})
        else:
            messages.append({"role": "user", "content": "Based on the output above, write your next Python code (or call SUBMIT if done):"})

        llm_response = _llm_completion(messages, model=model)
        messages.append({"role": "assistant", "content": llm_response})

        code = _extract_code_block(llm_response)
        if code is None:
            trajectory.append({"iteration": iteration + 1, "llm_response": llm_response, "code": None, "output": None})
            messages.append({"role": "user", "content": "Please write executable Python code.  Use SUBMIT(answer=…) when done."})
            continue

        # Execute in Daytona
        variables = {}
        if context and iteration == 0:
            variables["context"] = context

        output = interp.execute(code, variables=variables, timeout=60)

        trajectory.append({"iteration": iteration + 1, "code": code, "output": output})
        messages.append({"role": "user", "content": f"[Execution output]:\n{output}"})

        # Check for SUBMIT
        answer = _is_submit_result(output)
        if answer is not None:
            return {"answer": answer, "trajectory": trajectory, "execution_time": time.time() - start}

    # Fallback — ask LLM to summarize
    messages.append({"role": "user", "content": "You've run out of iterations.  Based on everything above, provide your best final answer in plain text.  Do NOT write code."})
    answer = _llm_completion(messages, model=model)

    return {"answer": answer, "trajectory": trajectory, "execution_time": time.time() - start}


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def rlm_execute(task: str, context: str = "") -> str:
    """Execute a task using the RLM code-execution loop.

    The LLM writes Python code, executes it in a secure Daytona sandbox,
    reads the output, iterates, and returns a verified answer.

    Args:
        task: The task to accomplish (e.g. "Calculate the 50th Fibonacci number")
        context: Optional data or context to make available to the code
    """
    result = rlm_loop(task=task, context=context)

    trace = ""
    for step in result["trajectory"]:
        if step.get("code"):
            trace += f"\n**Iteration {step['iteration']}:**\n```python\n{step['code'][:500]}\n```\n"
            if step.get("output"):
                trace += f"Output: `{step['output'][:300]}`\n"

    return (
        f"## RLM Result\n\n"
        f"**Task:** {task}\n"
        f"**Model:** {RLM_MODEL}\n"
        f"**Time:** {result['execution_time']:.1f}s\n\n"
        f"### Answer\n\n{result['answer']}\n\n"
        f"### Trace\n{trace}"
    )


@mcp.tool()
def rlm_analyze(data: str, question: str) -> str:
    """Analyze data by writing and executing Python code.

    Args:
        data: The data to analyze (text, JSON, CSV, numbers)
        question: What to find out about the data
    """
    result = rlm_loop(task=f"Analyze the following data and answer: {question}", context=data)
    return (
        f"## Analysis\n\n**Question:** {question}\n\n"
        f"### Findings\n\n{result['answer']}\n\n"
        f"---\n*{result['execution_time']:.1f}s · {len(result['trajectory'])} iterations*"
    )


@mcp.tool()
def rlm_code(description: str, language: str = "python") -> str:
    """Generate, test, and return working code via the RLM loop.

    Args:
        description: What the code should do
        language: Target language (default: python)
    """
    task = f"Write {language} code that: {description}. Test it, fix any bugs, and SUBMIT the final working version."
    result = rlm_loop(task=task)
    return (
        f"## Generated Code\n\n**Description:** {description}\n\n"
        f"{result['answer']}\n\n"
        f"---\n*{result['execution_time']:.1f}s*"
    )


@mcp.tool()
def rlm_decompose(complex_task: str, num_subtasks: int = 5) -> str:
    """Break a complex task into subtasks, solve each with code, synthesize.

    Args:
        complex_task: A complex task to break down and solve
        num_subtasks: Approximate number of subtasks (default: 5)
    """
    task = (
        f"Break this task into ~{num_subtasks} subtasks and solve each one:\n\n"
        f"{complex_task}\n\n"
        "For each subtask: describe it, write and execute code, then call "
        "SUBMIT(answer=<synthesized result>) when all subtasks are done."
    )
    result = rlm_loop(task=task, max_iterations=RLM_MAX_ITERATIONS + 5)
    return (
        f"## Decomposed Solution\n\n**Task:** {complex_task}\n\n"
        f"### Solution\n\n{result['answer']}\n\n"
        f"---\n*{result['execution_time']:.1f}s · {len(result['trajectory'])} iterations*"
    )


@mcp.tool()
def sandbox_exec(code: str) -> str:
    """Execute Python code directly in the Daytona sandbox (no RLM loop).

    Args:
        code: Python code to execute
    """
    return get_interpreter().execute(code, timeout=60)


@mcp.tool()
def sandbox_exec_stateful(code: str) -> str:
    """Execute Python in a stateful REPL — variables persist between calls.

    Args:
        code: Python code to execute
    """
    return get_interpreter().execute_stateful(code, timeout=60)


@mcp.tool()
def sandbox_upload(content: str, path: str) -> str:
    """Upload a text file into the Daytona sandbox.

    Args:
        content: File content (text)
        path: Destination path in the sandbox (e.g. /home/daytona/data.csv)
    """
    get_interpreter().upload_file(content.encode("utf-8"), path)
    return f"Uploaded {len(content)} bytes → {path}"


@mcp.tool()
def sandbox_download(path: str) -> str:
    """Download a text file from the Daytona sandbox.

    Args:
        path: File path in the sandbox to download
    """
    data = get_interpreter().download_file(path)
    return data.decode("utf-8", errors="replace")


@mcp.tool()
def sandbox_files(path: str = "/home/daytona") -> str:
    """List files in the Daytona sandbox.

    Args:
        path: Directory to list (default: /home/daytona)
    """
    files = get_interpreter().list_files(path)
    if not files:
        return f"No files in {path}"
    lines = [
        f"  {'dir' if f['is_dir'] else 'file'}  {f['size']:>8}  {f['name']}"
        for f in files
    ]
    return f"Files in {path}:\n" + "\n".join(lines)


@mcp.tool()
def sandbox_shell(command: str) -> str:
    """Run a shell command in the Daytona sandbox.

    Args:
        command: Shell command to execute (e.g. "ls -la", "pip install scipy")
    """
    return get_interpreter()._exec_shell(command, timeout=60)


@mcp.tool()
def rlm_status() -> str:
    """Get Fleet RLM + Daytona system status and configuration."""
    interp = _interpreter
    running = interp is not None and interp._started
    sandbox_id = interp._sandbox.id if running else "N/A"

    keys = {
        "OpenAI": bool(OPENAI_API_KEY),
        "Anthropic": bool(ANTHROPIC_API_KEY),
        "Daytona": bool(DAYTONA_API_KEY),
    }

    return (
        f"## Fleet RLM + Daytona Status\n\n"
        f"**Model:** {RLM_MODEL}  |  **Subtask:** {RLM_SUBTASK_MODEL}\n"
        f"**Max iterations:** {RLM_MAX_ITERATIONS}\n\n"
        f"**Sandbox:** {'Running' if running else 'Not started'}  "
        f"(id: {sandbox_id}, target: {DAYTONA_TARGET})\n\n"
        f"**API keys:** " + ", ".join(f"{k}: {'set' if v else 'NOT SET'}" for k, v in keys.items())
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Start the Fleet RLM MCP server."""
    logger.info("Fleet RLM MCP Server (Daytona Edition)")
    logger.info("  model=%s  subtask=%s  target=%s", RLM_MODEL, RLM_SUBTASK_MODEL, DAYTONA_TARGET)
    logger.info("  transport=%s  port=%d", MCP_TRANSPORT, MCP_PORT)

    # Graceful shutdown on SIGTERM — interpreter cleanup via atexit
    def _handle_signal(sig, _frame):
        logger.info("Received signal %s — shutting down", sig)
        # Restore default handler to avoid double-signal issues
        signal.signal(sig, signal.SIG_DFL)
        _shutdown_interpreter()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _handle_signal)

    mcp.run(transport=MCP_TRANSPORT)


if __name__ == "__main__":
    main()
