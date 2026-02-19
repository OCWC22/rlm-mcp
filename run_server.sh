#!/bin/bash
# Fleet RLM MCP Server Launcher (Daytona Edition)
#
# Usage:
#   ./run_server.sh              # local stdio (for Claude Desktop / CLI)
#   ./run_server.sh --http       # HTTP on port 8000 (for remote access)
#   ./run_server.sh --docker     # build & run in Docker (always-on)
#   ./run_server.sh --deploy     # deploy to Daytona workspace

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

# Load .env.local if keys aren't set
if [ -z "$DAYTONA_API_KEY" ] || [ -z "$OPENAI_API_KEY" ]; then
    if [ -f "$SCRIPT_DIR/.env.local" ]; then
        set -a
        . "$SCRIPT_DIR/.env.local"
        set +a
    fi
fi

if [ -z "$DAYTONA_API_KEY" ]; then
    echo "WARNING: DAYTONA_API_KEY not set. Sandbox creation will fail." >&2
fi

# Parse mode
case "${1:-}" in
    --http)
        export MCP_TRANSPORT="streamable-http"
        export MCP_PORT="${MCP_PORT:-8000}"
        echo "Starting Fleet RLM MCP on http://0.0.0.0:${MCP_PORT}/mcp" >&2
        exec "$VENV_PYTHON" -m src
        ;;
    --docker)
        echo "Building & starting Docker container..." >&2
        docker compose -f "$SCRIPT_DIR/docker-compose.yml" up --build -d
        echo "Fleet RLM MCP running at http://localhost:${MCP_PORT:-8000}/mcp" >&2
        echo "Logs: docker compose logs -f fleet-rlm" >&2
        exit 0
        ;;
    --deploy)
        echo "Deploying to Daytona..." >&2
        exec "$VENV_PYTHON" "$SCRIPT_DIR/deploy.py"
        ;;
    *)
        # Default: stdio transport (for local MCP hosts)
        export MCP_TRANSPORT="stdio"
        exec "$VENV_PYTHON" -m src
        ;;
esac
