#!/bin/bash
# RLM MCP Server Setup
# Installs dependencies into a virtual environment

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

echo "Setting up RLM MCP Server..."

# Create venv if needed
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Install dependencies
echo "Installing dependencies..."
"$VENV_DIR/bin/pip" install --quiet --upgrade pip
"$VENV_DIR/bin/pip" install --quiet -r "$SCRIPT_DIR/requirements.txt"

# Install RLM library (from local source or PyPI)
if [ -n "$RLM_LIB_PATH" ] && [ -d "$RLM_LIB_PATH" ]; then
    echo "Installing RLM library from $RLM_LIB_PATH..."
    "$VENV_DIR/bin/pip" install -e "$RLM_LIB_PATH"
elif [ -d "$HOME/rlm" ]; then
    echo "Installing RLM library from $HOME/rlm..."
    "$VENV_DIR/bin/pip" install -e "$HOME/rlm"
else
    echo "Note: RLM library not found locally. Install from source:"
    echo "  git clone https://github.com/alexzhang13/rlm.git \$HOME/rlm"
    echo "  Then re-run this setup script."
fi

echo "Setup complete!"
echo "Venv Python: $VENV_DIR/bin/python"
