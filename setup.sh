#!/bin/bash
# Fleet RLM MCP Server Setup (Daytona Edition)
# Installs dependencies into a virtual environment

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

echo "Setting up Fleet RLM MCP Server (Daytona Edition)..."

# Create venv if needed
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Install dependencies
echo "Installing dependencies..."
"$VENV_DIR/bin/pip" install --quiet --upgrade pip
"$VENV_DIR/bin/pip" install --quiet -r "$SCRIPT_DIR/requirements.txt"

# Verify Daytona SDK installed
if "$VENV_DIR/bin/python" -c "import daytona" 2>/dev/null; then
    echo "Daytona SDK: OK"
else
    echo "WARNING: daytona-sdk not installed properly. Try:"
    echo "  $VENV_DIR/bin/pip install daytona-sdk"
fi

# Verify litellm installed
if "$VENV_DIR/bin/python" -c "import litellm" 2>/dev/null; then
    echo "LiteLLM: OK"
else
    echo "WARNING: litellm not installed properly."
fi

echo ""
echo "Setup complete!"
echo "Venv Python: $VENV_DIR/bin/python"
echo ""
echo "Required environment variables:"
echo "  DAYTONA_API_KEY  - Your Daytona API key"
echo "  OPENAI_API_KEY   - OpenAI API key (or ANTHROPIC_API_KEY)"
echo ""
echo "Optional:"
echo "  RLM_MODEL          - Root model (default: openai/gpt-4o)"
echo "  RLM_SUBTASK_MODEL  - Subtask model (default: openai/gpt-4o-mini)"
echo "  DAYTONA_TARGET     - Region: us, eu, asia (default: us)"
