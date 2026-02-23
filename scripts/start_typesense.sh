#!/usr/bin/env bash
# Start Typesense server for AccelPapers local development.
#
# Usage:
#   ./scripts/start_typesense.sh
#
# Prerequisites:
#   brew install typesense/tap/typesense-server
#
# Environment:
#   ACCELPAPERS_TYPESENSE_API_KEY  API key (default: accelpapers-dev)

set -euo pipefail

DATA_DIR="${HOME}/.accelpapers/typesense-data"
mkdir -p "${DATA_DIR}"

echo "Starting Typesense server (data: ${DATA_DIR})"
exec typesense-server \
    --data-dir "${DATA_DIR}" \
    --api-key "${ACCELPAPERS_TYPESENSE_API_KEY:-accelpapers-dev}" \
    --enable-cors
