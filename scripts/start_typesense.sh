#!/usr/bin/env bash
# Start Typesense server for AccelPapers local development.
#
# Usage:
#   ./scripts/start_typesense.sh
#
# Prerequisites:
#   brew install typesense/tap/typesense-server
#
# NOTE: Homebrew's default API key is "xyz". This script uses "accelpapers-dev"
#       instead. Make sure ACCELPAPERS_TYPESENSE_API_KEY matches when indexing.
#
# Compatibility:
#   Typesense 27.1 works with OpenAI-compatible auto-embedding (Ollama).
#   Typesense 28.0+ / 30.x has a regression — the server never connects to
#   the embedding endpoint. Pin to 27.1 until upstream fixes this:
#     brew install typesense/tap/typesense-server@27.1
#
# Ollama must be running for auto-embedding to work:
#   ollama serve &
#   ollama pull nomic-embed-text
#
# Environment:
#   ACCELPAPERS_TYPESENSE_API_KEY  API key (default: accelpapers-dev)

set -euo pipefail

DATA_DIR="${HOME}/.accelpapers/typesense-data"
mkdir -p "${DATA_DIR}"

# Warn if Ollama isn't reachable (embedding will fail at index time)
if ! curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "WARNING: Ollama is not running on localhost:11434."
    echo "         Auto-embedding will fail during indexing."
    echo "         Start Ollama with: ollama serve"
fi

echo "Starting Typesense server (data: ${DATA_DIR})"
exec typesense-server \
    --data-dir "${DATA_DIR}" \
    --api-key "${ACCELPAPERS_TYPESENSE_API_KEY:-accelpapers-dev}" \
    --enable-cors
