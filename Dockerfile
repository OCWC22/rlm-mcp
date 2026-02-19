FROM python:3.11-slim

WORKDIR /app

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

CMD ["python", "-m", "src"]
