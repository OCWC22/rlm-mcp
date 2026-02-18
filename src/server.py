#!/usr/bin/env python3
"""
Fleet RLM MCP Server — Daytona Edition

Provides RLM (Recursive Language Model) capabilities backed by
Daytona sandboxes for secure, remote Python code execution.

The LLM writes code → Daytona executes it in an isolated sandbox →
output returns to the LLM → iterate until SUBMIT().

Usage:
    export DAYTONA_API_KEY=your_daytona_key
    export OPENAI_API_KEY=your_key        # or ANTHROPIC_API_KEY
    python -m src.server
"""

import os
import json
import time
import logging
from typing import Optional
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Load .env.local
# ---------------------------------------------------------------------------
for env_path in [
    Path(__file__).parent.parent / ".env.local",
    Path(__file__).parent.parent.parent / ".env.local",
]:
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    value = value.strip("\"'")
                    os.environ.setdefault(key, value)
        break

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
RLM_MAX_OUTPUT_CHARS = int(os.getenv("RLM_MAX_OUTPUT_CHARS", "10000"))
MCP_TRANSPORT = os.getenv("MCP_TRANSPORT", "stdio")  # "stdio" | "streamable-http" | "sse"
MCP_PORT = int(os.getenv("MCP_PORT", "8000"))

# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------
mcp = FastMCP("fleet-rlm-daytona", host="0.0.0.0", port=MCP_PORT)

# ---------------------------------------------------------------------------
# Interpreter lifecycle — shared per server process
# ---------------------------------------------------------------------------
_interpreter = None


def get_interpreter():
    """Lazy-create a DaytonaInterpreter that lives for the server's lifetime."""
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
        logger.info("Daytona sandbox ready")
    return _interpreter


def _resolve_api_key(model: str) -> str:
    """Pick the right API key for a litellm model string."""
    if model.startswith("anthropic/"):
        return ANTHROPIC_API_KEY or ""
    return OPENAI_API_KEY or ""


# ---------------------------------------------------------------------------
# RLM execution loop  (the core "write code → execute → iterate" loop)
# ---------------------------------------------------------------------------

def _llm_completion(prompt: str, model: str = None) -> str:
    """Call an LLM via litellm."""
    import litellm

    model = model or RLM_MODEL
    api_key = _resolve_api_key(model)

    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        api_key=api_key,
        max_tokens=4096,
    )
    return response.choices[0].message.content


def _extract_code_block(text: str) -> Optional[str]:
    """Extract the first ```python ... ``` block, or the whole text if no block."""
    import re

    # Try fenced code block first
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # If the entire response looks like code (starts with import/def/for/etc.)
    lines = text.strip().split("\n")
    code_starters = ("import ", "from ", "def ", "class ", "for ", "while ", "if ",
                     "print(", "x ", "y ", "result", "#")
    if lines and any(lines[0].strip().startswith(s) for s in code_starters):
        return text.strip()

    return None


def rlm_loop(
    task: str,
    context: str = "",
    max_iterations: int = None,
    model: str = None,
) -> dict:
    """Run the RLM iterative code-execution loop.

    1. LLM reads the task + history and writes Python code
    2. Code executes in Daytona sandbox
    3. Output returns to LLM
    4. Repeat until SUBMIT() or max_iterations

    Returns dict with keys: answer, trajectory, execution_time
    """
    max_iter = max_iterations or RLM_MAX_ITERATIONS
    model = model or RLM_MODEL
    interp = get_interpreter()
    trajectory = []
    start_time = time.time()

    system_prompt = f"""You are an RLM (Recursive Language Model) agent.
You solve tasks by writing and executing Python code in a sandbox.

RULES:
- Write Python code to explore data, compute results, test hypotheses.
- After each execution you see stdout. Use it to decide your next step.
- When you have the final answer, call SUBMIT(answer=<your answer>) in your code.
- You have access to: numpy, pandas, requests, re, json, math, collections.
- Keep code concise. Print intermediate results so you can inspect them.
- Do NOT just describe what you would do — actually write executable code.

TASK: {task}
"""
    if context:
        system_prompt += f"\nCONTEXT (available as the variable `context`):\n{context}\n"

    conversation = [system_prompt]

    for iteration in range(max_iter):
        # Build the prompt with history
        history_text = "\n".join(conversation)
        if iteration == 0:
            history_text += "\n\nWrite your first piece of Python code to start solving this task:"
        else:
            history_text += "\n\nBased on the output above, write your next piece of Python code (or call SUBMIT if done):"

        # Ask LLM to write code
        llm_response = _llm_completion(history_text, model=model)
        code = _extract_code_block(llm_response)

        if code is None:
            # LLM didn't produce code — maybe it has the answer in prose
            trajectory.append({
                "iteration": iteration + 1,
                "llm_response": llm_response,
                "code": None,
                "output": None,
            })
            # Try one more time asking explicitly for code
            conversation.append(f"\n[LLM Response (no code)]:\n{llm_response}")
            conversation.append("\nPlease write executable Python code. Use SUBMIT(answer=...) when done.")
            continue

        # Execute in Daytona
        variables = {}
        if context and iteration == 0:
            variables["context"] = context

        output = interp.execute(code, variables=variables, timeout=60)

        trajectory.append({
            "iteration": iteration + 1,
            "code": code,
            "output": output,
        })

        conversation.append(f"\n[Code (iteration {iteration + 1})]:\n```python\n{code}\n```")
        conversation.append(f"\n[Output]:\n{output}")

        # Check if SUBMIT was called (output starts with JSON from __SUBMIT__)
        if output.strip().startswith("{") and '"answer"' in output:
            try:
                result = json.loads(output.strip())
                return {
                    "answer": result.get("answer", output),
                    "trajectory": trajectory,
                    "execution_time": time.time() - start_time,
                }
            except json.JSONDecodeError:
                pass

        # Also check for the __SUBMIT__ marker that got JSON-decoded already
        if "__SUBMIT__:" not in output and "SUBMIT" in code and output.strip():
            # SUBMIT was in code and we got clean output — probably the answer
            return {
                "answer": output.strip(),
                "trajectory": trajectory,
                "execution_time": time.time() - start_time,
            }

    # Fallback: ask LLM to extract answer from history
    fallback_prompt = "\n".join(conversation)
    fallback_prompt += "\n\nYou've run out of iterations. Based on everything above, provide your best final answer in plain text. Do NOT write code."
    answer = _llm_completion(fallback_prompt, model=model)

    return {
        "answer": answer,
        "trajectory": trajectory,
        "execution_time": time.time() - start_time,
    }


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def rlm_execute(task: str, context: str = "") -> str:
    """Execute a task using the RLM code-execution loop.

    The LLM writes Python code, executes it in a secure Daytona sandbox,
    inspects output, iterates, and returns a verified answer.

    Args:
        task: The task to accomplish (e.g. "Calculate compound interest for 5 years")
        context: Optional context/data to work with

    Returns:
        RLM's final answer after iterative code execution
    """
    result = rlm_loop(task=task, context=context)

    trajectory_summary = ""
    for step in result["trajectory"]:
        if step.get("code"):
            trajectory_summary += f"\n**Iteration {step['iteration']}:**\n```python\n{step['code'][:500]}\n```\n"
            if step.get("output"):
                out = step["output"][:300]
                trajectory_summary += f"Output: `{out}`\n"

    return f"""## RLM Execution Result

**Task:** {task}
**Model:** {RLM_MODEL}
**Execution Time:** {result['execution_time']:.2f}s
**Sandbox:** Daytona

### Answer

{result['answer']}

### Execution Trace
{trajectory_summary}
"""


@mcp.tool()
def rlm_analyze(data: str, question: str) -> str:
    """Analyze data using the RLM code-execution loop.

    Provide data (text, JSON, CSV, numbers) and a question.
    RLM writes and executes Python code to analyze it.

    Args:
        data: The data to analyze
        question: What to find out about the data

    Returns:
        Analysis results from code execution
    """
    task = f"Analyze the following data and answer: {question}"
    result = rlm_loop(task=task, context=data)

    return f"""## RLM Data Analysis

**Question:** {question}

### Findings

{result['answer']}

---
**Execution Time:** {result['execution_time']:.2f}s
**Iterations:** {len(result['trajectory'])}
"""


@mcp.tool()
def rlm_code(description: str, language: str = "python") -> str:
    """Generate and test code using the RLM loop.

    RLM writes code, executes it to verify it works,
    fixes bugs, and returns working code.

    Args:
        description: What the code should do
        language: Programming language (default: python)

    Returns:
        Working, tested code
    """
    task = f"Write {language} code that: {description}. Test it with example inputs, fix bugs, and return the final working code."
    result = rlm_loop(task=task)

    return f"""## RLM Code Generation

**Description:** {description}
**Language:** {language}

### Generated Code

{result['answer']}

---
**Execution Time:** {result['execution_time']:.2f}s
"""


@mcp.tool()
def rlm_decompose(complex_task: str, num_subtasks: int = 5) -> str:
    """Decompose a complex task into subtasks and solve each.

    RLM breaks down the task, solves each part with code execution,
    then synthesizes results.

    Args:
        complex_task: A complex task to break down
        num_subtasks: Approximate number of subtasks (default: 5)

    Returns:
        Solution with subtask breakdown
    """
    task = f"""Break this task into ~{num_subtasks} subtasks and solve each one:

{complex_task}

For each subtask:
1. Describe it
2. Write and execute code to solve it
3. After all subtasks are done, call SUBMIT(answer=<synthesized result>)
"""
    result = rlm_loop(task=task, max_iterations=RLM_MAX_ITERATIONS + 5)

    return f"""## RLM Task Decomposition

**Task:** {complex_task}

### Solution

{result['answer']}

---
**Execution Time:** {result['execution_time']:.2f}s
**Iterations:** {len(result['trajectory'])}
"""


@mcp.tool()
def sandbox_exec(code: str) -> str:
    """Execute Python code directly in the Daytona sandbox.

    Low-level tool — runs code without the RLM loop.
    Useful for quick computations or file operations.

    Args:
        code: Python code to execute

    Returns:
        Stdout from the execution
    """
    interp = get_interpreter()
    output = interp.execute(code, timeout=60)
    return output


@mcp.tool()
def sandbox_upload(content: str, path: str) -> str:
    """Upload a file to the Daytona sandbox.

    Args:
        content: File content (text)
        path: Destination path in sandbox (e.g. /home/daytona/data.csv)

    Returns:
        Confirmation message
    """
    interp = get_interpreter()
    interp.upload_file(content.encode("utf-8"), path)
    return f"Uploaded {len(content)} bytes to {path}"


@mcp.tool()
def sandbox_files(path: str = "/home/daytona") -> str:
    """List files in the Daytona sandbox.

    Args:
        path: Directory to list (default: /home/daytona)

    Returns:
        File listing
    """
    interp = get_interpreter()
    files = interp.list_files(path)
    if not files:
        return f"No files in {path}"
    lines = []
    for f in files:
        kind = "dir" if f["is_dir"] else "file"
        lines.append(f"  {kind}  {f['size']:>8}  {f['name']}")
    return f"Files in {path}:\n" + "\n".join(lines)


@mcp.tool()
def rlm_status() -> str:
    """Get Fleet RLM + Daytona system status and configuration."""
    interp = _interpreter
    sandbox_status = "Running" if (interp and interp._started) else "Not started"
    sandbox_id = interp._sandbox.id if (interp and interp._sandbox) else "N/A"

    return f"""## Fleet RLM + Daytona Status

**Root Model:** {RLM_MODEL}
**Subtask Model:** {RLM_SUBTASK_MODEL}
**Max Iterations:** {RLM_MAX_ITERATIONS}

**Sandbox:**
- Provider: Daytona
- Status: {sandbox_status}
- ID: {sandbox_id}
- Target: {DAYTONA_TARGET}

**Capabilities:**
- Iterative code execution (RLM loop)
- Secure sandbox (Daytona)
- File upload/download
- Data analysis
- Code generation & testing
- Task decomposition

**API Keys:**
- OpenAI: {"Set" if OPENAI_API_KEY else "NOT SET"}
- Anthropic: {"Set" if ANTHROPIC_API_KEY else "NOT SET"}
- Daytona: {"Set" if DAYTONA_API_KEY else "NOT SET"}
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run the Fleet RLM MCP server."""
    logger.info("Starting Fleet RLM MCP Server (Daytona Edition)...")
    logger.info(f"Root Model: {RLM_MODEL}")
    logger.info(f"Subtask Model: {RLM_SUBTASK_MODEL}")
    logger.info(f"Daytona Target: {DAYTONA_TARGET}")
    logger.info(f"Transport: {MCP_TRANSPORT} (port {MCP_PORT})")
    mcp.run(transport=MCP_TRANSPORT)


if __name__ == "__main__":
    main()
