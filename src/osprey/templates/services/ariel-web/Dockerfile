# ARIEL Web Interface
# Production-grade web interface for ARIEL search service

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./

# Set Python path to include mounted osprey
ENV PYTHONPATH="/app/src:${PYTHONPATH}"
ENV PYTHONUNBUFFERED=1

# Create non-root user for security
RUN useradd -m -u 1000 ariel && chown -R ariel:ariel /app
USER ariel

# Expose port
EXPOSE 8085

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8085/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8085"]
