#!/usr/bin/env python3
"""
RLM MCP Server - Pure Code Execution with LLM

Provides RLM (Recursive Language Model) capabilities:
- Execute Python code with LLM reasoning
- Decompose complex tasks into subtasks
- Iterative problem solving with code execution

No external dependencies (no Qdrant, no PDF-RAG).

Usage:
    # API keys are automatically loaded from .env.local
    # Or set manually:
    export OPENAI_API_KEY=your_key
    export ANTHROPIC_API_KEY=your_key
    python -m src.server
"""

import os
import logging
from typing import Optional
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# Load environment variables from .env.local if it exists
env_local_path = Path(__file__).parent.parent.parent / ".env.local"
if env_local_path.exists():
    with open(env_local_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                # Remove quotes if present
                value = value.strip('"\'')
                os.environ[key] = value

# Lazy import RLM
_rlm_instance = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
RLM_MODEL = os.getenv("RLM_MODEL", "openai/gpt-5.2-2025-12-11")
RLM_SUBTASK_MODEL = os.getenv("RLM_SUBTASK_MODEL", "anthropic/claude-opus-4-6")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
RLM_MAX_DEPTH = int(os.getenv("RLM_MAX_DEPTH", "2"))
RLM_MAX_ITERATIONS = int(os.getenv("RLM_MAX_ITERATIONS", "20"))

# Initialize MCP Server
mcp = FastMCP("rlm")


def get_rlm():
    """Get or create RLM instance (lazy load).

    Configuration:
    - max_depth: Levels of recursion (root + subtasks)
    - max_iterations: Room for complex multi-step analysis
    - other_backends: Uses cheaper model for subtasks
    - logger: Saves trajectories for visualization

    Cost optimization:
    - Root task: openai/gpt-5.2-2025-12-11 (powerful, handles decomposition)
    - Subtasks: anthropic/claude-opus-4-6 (high-quality, handles individual work)
    """
    global _rlm_instance
    if _rlm_instance is None:
        from rlm import RLM
        from rlm.logger import RLMLogger

        # Log to visualizer-compatible format
        log_dir = os.getenv("RLM_LOG_DIR", os.path.expanduser("~/.rlm/logs"))
        os.makedirs(log_dir, exist_ok=True)
        rlm_logger = RLMLogger(log_dir=log_dir)

        logger.info(f"Initializing RLM - Root: {RLM_MODEL}, Subtasks: {RLM_SUBTASK_MODEL}")
        logger.info(f"Max depth: {RLM_MAX_DEPTH}, Max iterations: {RLM_MAX_ITERATIONS}")
        logger.info(f"Logging trajectories to: {log_dir}")

        # Configure API keys based on model provider
        if RLM_MODEL.startswith("openai/"):
            api_key = OPENAI_API_KEY
        elif RLM_MODEL.startswith("anthropic/"):
            api_key = ANTHROPIC_API_KEY
        else:
            api_key = OPENAI_API_KEY  # default to OpenAI
        
        if RLM_SUBTASK_MODEL.startswith("openai/"):
            subtask_api_key = OPENAI_API_KEY
        elif RLM_SUBTASK_MODEL.startswith("anthropic/"):
            subtask_api_key = ANTHROPIC_API_KEY
        else:
            subtask_api_key = OPENAI_API_KEY  # default to OpenAI

        _rlm_instance = RLM(
            backend="litellm",
            backend_kwargs={
                "model_name": RLM_MODEL,
                "api_key": api_key,
            },
            environment="local",
            max_depth=RLM_MAX_DEPTH,
            max_iterations=RLM_MAX_ITERATIONS,
            persistent=False,  # Fresh context each call (prevents context bleed)
            verbose=True,
            logger=rlm_logger,
            # Use cheaper model for subtask decomposition
            other_backends=["litellm"],
            other_backend_kwargs=[{
                "model_name": RLM_SUBTASK_MODEL,
                "api_key": subtask_api_key,
            }],
        )
    return _rlm_instance


# =============================================================================
# MCP Tools
# =============================================================================

@mcp.tool()
def rlm_execute(
    task: str,
    context: str = ""
) -> str:
    """
    Execute a task with RLM code execution.

    RLM will write and execute Python code to solve the task,
    iterating until it produces a final answer.

    Args:
        task: The task to accomplish (e.g., "Calculate compound interest for 5 years")
        context: Optional context/data to work with

    Returns:
        RLM's final answer after code execution
    """
    prompt = f"""You have Python execution capabilities.

TASK: {task}
"""
    if context:
        prompt += f"""
CONTEXT:
{context}
"""
    prompt += """
INSTRUCTIONS:
1. Analyze the task
2. Write Python code to solve it
3. Execute and verify results
4. Provide a clear final answer

Begin:
"""

    rlm = get_rlm()
    result = rlm.completion(prompt)

    return f"""## RLM Execution Result

**Task:** {task}

### Answer

{result.response}

---
**Execution Time:** {result.execution_time:.2f}s
**Model:** {RLM_MODEL}
"""


@mcp.tool()
def rlm_analyze(
    data: str,
    question: str
) -> str:
    """
    Analyze data with RLM code execution.

    Provide data (text, numbers, JSON, etc.) and ask a question.
    RLM will write code to analyze it.

    Args:
        data: The data to analyze (text, JSON, CSV, etc.)
        question: What to find out about the data

    Returns:
        Analysis results with code execution
    """
    prompt = f"""You are a data analyst with Python execution capabilities.

DATA:
{data}

QUESTION: {question}

INSTRUCTIONS:
1. Parse and understand the data
2. Write Python code to analyze it
3. Answer the question with evidence
4. Show key findings

Begin analysis:
"""

    rlm = get_rlm()
    result = rlm.completion(prompt)

    return f"""## RLM Data Analysis

**Question:** {question}

### Findings

{result.response}

---
**Execution Time:** {result.execution_time:.2f}s
"""


@mcp.tool()
def rlm_code(
    description: str,
    language: str = "python"
) -> str:
    """
    Generate and test code with RLM.

    Describe what you want, RLM will write code,
    execute it to verify it works, then return it.

    Args:
        description: What the code should do
        language: Programming language (default: python)

    Returns:
        Working, tested code
    """
    prompt = f"""You are a code generator with execution capabilities.

TASK: Write {language} code that: {description}

INSTRUCTIONS:
1. Write the code
2. Test it with example inputs
3. Fix any bugs found during testing
4. Return the final working code with usage examples

Begin:
"""

    rlm = get_rlm()
    result = rlm.completion(prompt)

    return f"""## RLM Code Generation

**Description:** {description}
**Language:** {language}

### Generated Code

{result.response}

---
**Execution Time:** {result.execution_time:.2f}s
"""


@mcp.tool()
def rlm_decompose(
    complex_task: str,
    num_subtasks: int = 5
) -> str:
    """
    Decompose a complex task into subtasks and solve each.

    RLM will break down the task, solve each subtask with
    code execution, then synthesize the results.

    Args:
        complex_task: A complex task to break down
        num_subtasks: Approximate number of subtasks (default: 5)

    Returns:
        Solution with subtask breakdown
    """
    prompt = f"""You are a problem solver with Python execution and sub-LLM capabilities.

COMPLEX TASK: {complex_task}

INSTRUCTIONS:
1. Break this into ~{num_subtasks} logical subtasks
2. Use llm_query() or llm_query_batched() to delegate subtasks
3. Execute code as needed for calculations/processing
4. Synthesize subtask results into final answer

Example subtask delegation:
```python
subtasks = ["subtask 1", "subtask 2", "subtask 3"]
results = llm_query_batched(subtasks)
for i, result in enumerate(results):
    print(f"Subtask {{i+1}}: {{result}}")
```

Begin decomposition:
"""

    rlm = get_rlm()
    result = rlm.completion(prompt)

    return f"""## RLM Task Decomposition

**Task:** {complex_task}

### Solution

{result.response}

---
**Execution Time:** {result.execution_time:.2f}s
"""


@mcp.tool()
def rlm_status() -> str:
    """
    Get RLM system status and configuration.

    Returns current model settings and capabilities.
    """
    return f"""## RLM Status

**Root Model:** {RLM_MODEL}
**Subtask Model:** {RLM_SUBTASK_MODEL}
**Max Depth:** {RLM_MAX_DEPTH}
**Max Iterations:** {RLM_MAX_ITERATIONS}

**Capabilities:**
- Python code execution
- Sub-LLM queries (llm_query, llm_query_batched)
- Iterative problem solving
- Task decomposition

**Log Directory:** {os.getenv("RLM_LOG_DIR", "~/.rlm/logs")}

**API Keys:** 
- OpenAI: {"Set" if OPENAI_API_KEY else "NOT SET"}
- Anthropic: {"Set" if ANTHROPIC_API_KEY else "NOT SET"}
"""


# =============================================================================
# Main
# =============================================================================

def main():
    """Run MCP server."""
    logger.info("Starting RLM MCP Server...")
    logger.info(f"Root Model: {RLM_MODEL}")
    logger.info(f"Subtask Model: {RLM_SUBTASK_MODEL}")

    # Run server with stdio transport
    mcp.run()


if __name__ == "__main__":
    main()
