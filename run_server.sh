#!/bin/bash
# RLM MCP Server Launcher
# Pure code execution with LLM - no Qdrant dependency

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Check for venv, auto-setup if missing
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python"
if [ ! -f "$VENV_PYTHON" ]; then
    echo "Virtual environment not found. Running setup..." >&2
    "$SCRIPT_DIR/setup.sh" >&2
fi

# RLM library path (optional - only needed if RLM not installed via pip)
if [ -n "$RLM_LIB_PATH" ]; then
    export PYTHONPATH="$RLM_LIB_PATH:$PYTHONPATH"
elif [ -d "$HOME/rlm" ]; then
    export PYTHONPATH="$HOME/rlm:$PYTHONPATH"
fi

# Default config (can be overridden)
export RLM_MODEL="${RLM_MODEL:-openrouter/x-ai/grok-code-fast-1}"
export RLM_SUBTASK_MODEL="${RLM_SUBTASK_MODEL:-openrouter/openai/gpt-4o-mini}"
export RLM_MAX_DEPTH="${RLM_MAX_DEPTH:-2}"
export RLM_MAX_ITERATIONS="${RLM_MAX_ITERATIONS:-20}"

exec "$VENV_PYTHON" -m src.server
