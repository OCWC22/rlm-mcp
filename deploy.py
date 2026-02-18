#!/usr/bin/env python3
"""
Deploy Fleet RLM MCP Server to a Daytona workspace.

Creates a Daytona sandbox, uploads the server code, installs deps,
and starts the MCP server on HTTP — then prints the URL you connect to.

Usage:
    python deploy.py

Requires:
    DAYTONA_API_KEY  — Daytona API key
    OPENAI_API_KEY   — (or ANTHROPIC_API_KEY) for the RLM loop
"""

import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Load .env.local so deploy.py can find keys without exporting manually
# ---------------------------------------------------------------------------
for env_path in [Path(".env.local"), Path(__file__).parent / ".env.local"]:
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    value = value.strip("\"'")
                    os.environ.setdefault(key, value)
        break

from daytona import Daytona, DaytonaConfig, CreateSandboxFromSnapshotParams

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DAYTONA_API_KEY = os.environ.get("DAYTONA_API_KEY", "")
DAYTONA_API_URL = os.environ.get("DAYTONA_API_URL", "https://app.daytona.io/api")
DAYTONA_TARGET = os.environ.get("DAYTONA_TARGET", "us")
MCP_PORT = int(os.environ.get("MCP_PORT", "8000"))

if not DAYTONA_API_KEY:
    print("ERROR: DAYTONA_API_KEY not set. Put it in .env.local or export it.")
    sys.exit(1)

# Env vars to forward into the sandbox
FORWARD_KEYS = [
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
    "RLM_MODEL", "RLM_SUBTASK_MODEL", "RLM_MAX_ITERATIONS",
    "DAYTONA_API_KEY", "DAYTONA_API_URL", "DAYTONA_TARGET",
]

# ---------------------------------------------------------------------------
# Files to upload  (everything the server needs)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
SRC_DIR = PROJECT_ROOT / "src"

FILES = {
    "src/__init__.py": (SRC_DIR / "__init__.py").read_text(),
    "src/__main__.py": (SRC_DIR / "__main__.py").read_text(),
    "src/server.py": (SRC_DIR / "server.py").read_text(),
    "src/daytona_interpreter.py": (SRC_DIR / "daytona_interpreter.py").read_text(),
    "requirements.txt": (PROJECT_ROOT / "requirements.txt").read_text(),
}


def main():
    print("=== Fleet RLM MCP — Deploying to Daytona ===\n")

    # Build sandbox env vars
    sandbox_env = {
        "MCP_TRANSPORT": "streamable-http",
        "MCP_PORT": str(MCP_PORT),
    }
    for key in FORWARD_KEYS:
        val = os.environ.get(key)
        if val:
            sandbox_env[key] = val

    # Connect to Daytona
    config = DaytonaConfig(
        api_key=DAYTONA_API_KEY,
        api_url=DAYTONA_API_URL,
        target=DAYTONA_TARGET,
    )
    daytona = Daytona(config)

    # Create sandbox
    print(f"Creating sandbox (target={DAYTONA_TARGET})...")
    params = CreateSandboxFromSnapshotParams(
        language="python",
        env_vars=sandbox_env,
        auto_stop_interval=0,  # don't auto-stop — this is our server
    )
    sandbox = daytona.create(params, timeout=120)
    print(f"Sandbox created: {sandbox.id}\n")

    # Upload server files
    base = "/home/daytona/fleet-rlm"
    sandbox.process.exec(f"mkdir -p {base}/src")
    for rel_path, content in FILES.items():
        remote = f"{base}/{rel_path}"
        print(f"  Uploading {rel_path}")
        sandbox.fs.upload_file(content.encode("utf-8"), remote)

    # Install dependencies
    print("\nInstalling dependencies...")
    resp = sandbox.process.exec(
        f"cd {base} && pip install -q -r requirements.txt",
        timeout=180,
    )
    if resp.exit_code != 0:
        print(f"WARNING: pip install exited {resp.exit_code}")
        print(resp.result)

    # Start the MCP server in a background session
    print(f"\nStarting MCP server (port {MCP_PORT})...")
    session_id = "fleet-rlm-mcp"
    sandbox.process.create_session(session_id)
    from daytona import SessionExecuteRequest
    sandbox.process.execute_session_command(
        session_id=session_id,
        req=SessionExecuteRequest(
            command=f"cd {base} && python -m src",
            run_async=True,
        ),
    )

    # Wait a moment for startup, then check
    time.sleep(3)
    check = sandbox.process.exec(f"curl -sf http://localhost:{MCP_PORT}/mcp -X POST -d '{{}}' -H 'Content-Type: application/json' || echo 'waiting...'")
    print(f"Health check: {check.result[:200]}")

    # Print connection info
    print("\n" + "=" * 60)
    print("DEPLOYED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nSandbox ID:  {sandbox.id}")
    print(f"MCP URL:     (use sandbox proxy URL)/mcp")
    print(f"\nTo connect from Claude CLI:")
    print(f'  claude mcp add fleet-rlm --transport http "<sandbox-url>:{MCP_PORT}/mcp"')
    print(f"\nTo stop:")
    print(f"  python -c \"from daytona import Daytona, DaytonaConfig; import os; d=Daytona(DaytonaConfig(api_key=os.environ['DAYTONA_API_KEY'])); d.delete(d.get('{sandbox.id}'))\"")
    print(f"\nOr just let it auto-archive after it's idle.")
    print(f"\nSandbox env vars forwarded: {', '.join(k for k in FORWARD_KEYS if os.environ.get(k))}")


if __name__ == "__main__":
    main()
