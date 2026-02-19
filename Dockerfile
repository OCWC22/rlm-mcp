FROM python:3.11-slim

WORKDIR /app

# Install curl for the health check
RUN apt-get update -qq && apt-get install -y -qq curl && rm -rf /var/lib/apt/lists/*

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/

# Default: HTTP transport on port 8000
ENV MCP_TRANSPORT=streamable-http \
    MCP_PORT=8000 \
    RLM_MODEL=openai/gpt-4o \
    RLM_SUBTASK_MODEL=openai/gpt-4o-mini \
    RLM_MAX_ITERATIONS=15 \
    DAYTONA_TARGET=us

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -sf -X POST http://localhost:8000/mcp \
        -H "Content-Type: application/json" \
        -H "Accept: application/json, text/event-stream" \
        -d '{"jsonrpc":"2.0","id":0,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"healthcheck","version":"0.1"}}}' \
        || exit 1

CMD ["python", "-m", "src"]
