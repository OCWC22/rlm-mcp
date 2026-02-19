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
| `sandbox_exec_stateful` | Stateful REPL — variables persist between calls |
| `sandbox_upload` | Upload files to the sandbox |
| `sandbox_download` | Download files from the sandbox |
| `sandbox_files` | List files in the sandbox |
| `sandbox_shell` | Run shell commands in the sandbox |
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
cp .env.example .env.local
# Edit .env.local — add your keys

chmod +x setup.sh run_server.sh
./setup.sh
```

### Option A: Run locally (stdio, for Claude Desktop / CLI)

```bash
./run_server.sh
```

MCP client config (`~/.mcp.json`):

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

### Option B: Run locally over HTTP (for remote clients)

```bash
./run_server.sh --http
# Server starts on http://0.0.0.0:8000/mcp
```

Connect from Claude CLI:

```bash
claude mcp add fleet-rlm --transport http http://your-ip:8000/mcp
```

### Option C: Docker (always-on, production)

```bash
# With .env.local
./run_server.sh --docker

# Or directly
docker compose up --build -d
```

Server starts on `http://localhost:8000/mcp` with a health check.

```bash
claude mcp add fleet-rlm --transport http http://localhost:8000/mcp
```

### Option D: Deploy to Daytona (always-on, connect from anywhere)

```bash
./run_server.sh --deploy
```

This creates a Daytona workspace, uploads the server, installs deps, and starts it on HTTP. The server runs remotely — you just connect to it.

```bash
claude mcp add fleet-rlm --transport http https://<sandbox-url>/mcp
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
| `MCP_TRANSPORT` | `stdio` | `stdio`, `streamable-http`, or `sse` |
| `MCP_PORT` | `8000` | HTTP port (when using streamable-http) |

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

## Three Ways to Run

| Mode | Command | Transport | Use case |
|------|---------|-----------|----------|
| Local stdio | `./run_server.sh` | stdio | Claude Desktop, local CLI |
| Local HTTP | `./run_server.sh --http` | streamable-http | Remote clients on same network |
| Docker | `./run_server.sh --docker` | streamable-http | Production, always-on |
| Daytona deploy | `./run_server.sh --deploy` | streamable-http | Always-on, connect from anywhere |

## Acknowledgments

- **RLM concept:** Alex L. Zhang, Tim Kraska, Omar Khattab (MIT CSAIL) — [paper](https://arxiv.org/abs/2512.24601)
- **Fleet RLM:** [Qredence](https://github.com/Qredence/fleet-rlm) — patterns and architecture
- **Daytona:** [daytona.io](https://www.daytona.io) — sandbox infrastructure
- **MCP:** [modelcontextprotocol.io](https://modelcontextprotocol.io)

## License

MIT License — see [LICENSE](LICENSE)
