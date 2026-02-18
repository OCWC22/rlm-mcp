#!/bin/bash
# Fleet RLM MCP Server Launcher (Daytona Edition)
# Code execution via Daytona sandboxes — no Modal, no local sandbox

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Check for venv, auto-setup if missing
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python"
if [ ! -f "$VENV_PYTHON" ]; then
    echo "Virtual environment not found. Running setup..." >&2
    "$SCRIPT_DIR/setup.sh" >&2
fi

# Default config (can be overridden via env or .env.local)
export RLM_MODEL="${RLM_MODEL:-openai/gpt-4o}"
export RLM_SUBTASK_MODEL="${RLM_SUBTASK_MODEL:-openai/gpt-4o-mini}"
export RLM_MAX_ITERATIONS="${RLM_MAX_ITERATIONS:-15}"
export DAYTONA_TARGET="${DAYTONA_TARGET:-us}"

# Validate Daytona key
if [ -z "$DAYTONA_API_KEY" ]; then
    # Try loading from .env.local
    if [ -f "$SCRIPT_DIR/.env.local" ]; then
        export $(grep -v '^#' "$SCRIPT_DIR/.env.local" | xargs)
    fi
    if [ -z "$DAYTONA_API_KEY" ]; then
        echo "WARNING: DAYTONA_API_KEY not set. Sandbox creation will fail." >&2
    fi
fi

exec "$VENV_PYTHON" -m src.server
