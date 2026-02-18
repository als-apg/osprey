# ARIEL Web Interface
# Production-grade web interface for ARIEL search service
# This uses the native osprey.interfaces.ariel module

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy requirements first for layer caching
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Copy startup script
COPY start.sh ./

# Make start script executable
RUN chmod +x start.sh

ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8085

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8085/health || exit 1

# Run via startup script (handles dev mode)
CMD ["./start.sh"]
