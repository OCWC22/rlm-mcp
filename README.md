# Fleet RLM MCP Server — Daytona Edition

MCP server that provides [RLM (Recursive Language Model)](https://arxiv.org/abs/2512.24601) capabilities with **Daytona** sandboxes for secure, remote code execution.

> **Architecture:** LLM writes Python code → executes in Daytona sandbox → inspects output → iterates → returns verified answer. No local code execution, no Modal dependency.

Based on the [Fleet RLM](https://github.com/Qredence/fleet-rlm) patterns by Qredence, with Daytona replacing Modal for sandboxed execution.

## Features

| Tool | Description |
|------|-------------|
| `rlm_execute` | Execute tasks with iterative code execution (full RLM loop) |
| `rlm_analyze` | Analyze data with code execution |
| `rlm_code` | Generate, test, and fix code |
| `rlm_decompose` | Break complex tasks into subtasks and solve each |
| `sandbox_exec` | Execute Python code directly in sandbox (no RLM loop) |
| `sandbox_upload` | Upload files to the sandbox |
| `sandbox_files` | List files in the sandbox |
| `rlm_status` | Check system status |

## How It Works

```
┌─────────────┐     ┌──────────┐     ┌──────────────────┐
│  MCP Client │────▶│ RLM Loop │────▶│ Daytona Sandbox  │
│ (Claude, etc)│     │          │     │ (secure Python)  │
└─────────────┘     │ 1. LLM   │     │                  │
                    │    writes │     │ numpy, pandas,   │
                    │    code   │     │ requests, etc.   │
                    │ 2. Execute│────▶│                  │
                    │ 3. Read   │◀────│ stdout/stderr    │
                    │    output │     │                  │
                    │ 4. Repeat │     │ SUBMIT() → done  │
                    └──────────┘     └──────────────────┘
```

## Prerequisites

1. **Daytona API Key** — get one from [app.daytona.io](https://app.daytona.io)
2. **LLM API Key** — OpenAI or Anthropic
3. **Python 3.10+**

## Quick Start

```bash
# Clone
git clone https://github.com/OCWC22/rlm-mcp.git
cd rlm-mcp

# Setup
chmod +x setup.sh run_server.sh
./setup.sh

# Configure (create .env.local or export)
export DAYTONA_API_KEY="your-daytona-key"
export OPENAI_API_KEY="your-openai-key"

# Run
./run_server.sh
```

## Configuration

Add to your MCP client config (`~/.mcp.json` or Claude Desktop settings):

```json
{
  "mcpServers": {
    "fleet-rlm": {
      "command": "bash",
      "args": ["/path/to/rlm-mcp/run_server.sh"],
      "env": {
        "DAYTONA_API_KEY": "your-daytona-key",
        "OPENAI_API_KEY": "your-openai-key"
      }
    }
  }
}
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DAYTONA_API_KEY` | (required) | Daytona API key |
| `OPENAI_API_KEY` | (required*) | OpenAI API key |
| `ANTHROPIC_API_KEY` | (optional) | Anthropic API key |
| `RLM_MODEL` | `openai/gpt-4o` | Root LLM for the RLM loop |
| `RLM_SUBTASK_MODEL` | `openai/gpt-4o-mini` | Model for subtasks |
| `RLM_MAX_ITERATIONS` | `15` | Max iterations per RLM loop |
| `DAYTONA_API_URL` | `https://app.daytona.io/api` | Daytona API endpoint |
| `DAYTONA_TARGET` | `us` | Daytona region (us, eu, asia) |

*One of OPENAI_API_KEY or ANTHROPIC_API_KEY is required.

## Architecture

### DaytonaInterpreter (`src/daytona_interpreter.py`)

Drop-in replacement for Fleet RLM's `ModalInterpreter`. Manages the Daytona sandbox lifecycle:

- `start()` — creates a Daytona sandbox with Python environment
- `execute(code, variables, timeout)` — runs Python code in sandbox
- `execute_stateful(code)` — stateful execution (variables persist between calls)
- `upload_file()` / `download_file()` / `list_files()` — file operations
- `get_history()` — returns full execution trajectory
- `shutdown()` — deletes the sandbox

### RLM Loop (`src/server.py::rlm_loop`)

The core iterative loop:

1. Build system prompt with task + context
2. Ask LLM to write Python code
3. Execute code in Daytona sandbox via `DaytonaInterpreter`
4. Append code + output to conversation history
5. If `SUBMIT()` called → return answer
6. Otherwise → go to step 2
7. After max iterations → fallback extraction

### SUBMIT Protocol

Inside the sandbox, code can call `SUBMIT(answer=...)` to signal completion:

```python
# This runs in the Daytona sandbox
result = some_computation()
SUBMIT(answer=result)
```

## Comparison: Modal vs Daytona

| Aspect | Modal (fleet-rlm) | Daytona (this) |
|--------|-------------------|----------------|
| Setup | `modal setup` + secrets | API key only |
| Sandbox creation | ~2-5s | ~90ms |
| Persistence | Modal Volumes | Sandbox filesystem |
| Protocol | JSON-line over stdin/stdout | SDK API calls |
| Self-host | No | Yes (open source) |
| Free tier | Limited | $200 free compute |

## Acknowledgments

- **RLM concept:** Alex L. Zhang, Tim Kraska, Omar Khattab (MIT CSAIL) — [paper](https://arxiv.org/abs/2512.24601)
- **Fleet RLM:** [Qredence](https://github.com/Qredence/fleet-rlm) — patterns and architecture
- **Daytona:** [daytona.io](https://www.daytona.io) — sandbox infrastructure
- **MCP:** [modelcontextprotocol.io](https://modelcontextprotocol.io)

## License

MIT License — see [LICENSE](LICENSE)
