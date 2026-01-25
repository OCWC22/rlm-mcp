# RLM MCP Server

MCP (Model Context Protocol) server wrapper for [RLM (Recursive Language Models)](https://github.com/alexzhang13/rlm).

> **Note:** This is an MCP interface for the RLM library. The core RLM implementation is by **Alex Zhang, Tim Kraska, and Omar Khattab** at MIT CSAIL. See [Acknowledgments](#acknowledgments) for full credits.

RLM enables verified code execution with LLM reasoning - it writes and executes Python code iteratively until producing a verified answer.

## Features

- **rlm_execute** - Execute tasks with Python code and LLM reasoning
- **rlm_analyze** - Analyze data with code execution
- **rlm_code** - Generate and test code
- **rlm_decompose** - Break complex tasks into subtasks
- **rlm_status** - Check system status

## Prerequisites

1. **RLM Library**
   ```bash
   git clone https://github.com/alexzhang13/rlm.git $HOME/rlm
   cd $HOME/rlm
   pip install -e .
   ```

2. **OpenRouter API Key**
   ```bash
   export OPENROUTER_API_KEY="your-key-here"
   ```

## Installation

```bash
# Create MCP server directory
mkdir -p $HOME/.claude/mcp-servers/rlm

# Download files
curl -o $HOME/.claude/mcp-servers/rlm/src/server.py \
  https://raw.githubusercontent.com/eesb99/rlm-mcp/main/src/server.py
curl -o $HOME/.claude/mcp-servers/rlm/run_server.sh \
  https://raw.githubusercontent.com/eesb99/rlm-mcp/main/run_server.sh
curl -o $HOME/.claude/mcp-servers/rlm/setup.sh \
  https://raw.githubusercontent.com/eesb99/rlm-mcp/main/setup.sh
curl -o $HOME/.claude/mcp-servers/rlm/requirements.txt \
  https://raw.githubusercontent.com/eesb99/rlm-mcp/main/requirements.txt

# Setup
chmod +x $HOME/.claude/mcp-servers/rlm/*.sh
$HOME/.claude/mcp-servers/rlm/setup.sh
```

## Configuration

Add to `$HOME/.mcp.json`:

```json
{
  "mcpServers": {
    "rlm": {
      "command": "bash",
      "args": ["/YOUR/HOME/PATH/.claude/mcp-servers/rlm/run_server.sh"]
    }
  }
}
```

Replace `/YOUR/HOME/PATH` with your actual home directory (run `echo $HOME` to find it).

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | (required) | OpenRouter API key |
| `RLM_MODEL` | `openrouter/x-ai/grok-code-fast-1` | Root execution model |
| `RLM_SUBTASK_MODEL` | `openrouter/openai/gpt-4o-mini` | Subtask model |
| `RLM_MAX_DEPTH` | `2` | Max recursion depth |
| `RLM_MAX_ITERATIONS` | `20` | Max iterations per task |
| `RLM_LOG_DIR` | `~/.rlm/logs` | Directory for execution logs |
| `RLM_LIB_PATH` | `$HOME/rlm` | Path to RLM library (if not pip installed) |

## Usage with mcporter

```bash
# Install mcporter
npm install -g mcporter

# Check server is available
mcporter list | grep rlm

# Execute a calculation
mcporter call 'rlm.rlm_execute(task: "calculate the first 20 prime numbers")'

# Analyze data
mcporter call 'rlm.rlm_analyze(data: "[1,2,3,4,5]", question: "what is the mean?")'

# Check status
mcporter call 'rlm.rlm_status()'
```

## Security Notice

RLM executes arbitrary Python code by design. Only use with trusted inputs. The code runs in a local Python environment without additional sandboxing.

## Acknowledgments

This MCP server is a wrapper for the **Recursive Language Models (RLM)** library developed by:

- **Alex L. Zhang** (MIT CSAIL)
- **Tim Kraska** (MIT CSAIL)
- **Omar Khattab** (MIT CSAIL)

The RLM concept and implementation are their original work. This repository only provides an MCP interface to make RLM accessible via the Model Context Protocol.

**Citation:**
```bibtex
@article{zhang2025rlm,
  title={Recursive Language Models},
  author={Zhang, Alex L. and Kraska, Tim and Khattab, Omar},
  journal={arXiv preprint arXiv:2512.24601},
  year={2025}
}
```

## References

- **Paper:** [Recursive Language Models](https://arxiv.org/abs/2512.24601) (Zhang, Kraska, Khattab 2025)
- **RLM Library:** [github.com/alexzhang13/rlm](https://github.com/alexzhang13/rlm)
- **MCP SDK:** [modelcontextprotocol.io](https://modelcontextprotocol.io)
- **mcporter:** [mcporter.dev](http://mcporter.dev)

## License

MIT License - see [LICENSE](LICENSE)
