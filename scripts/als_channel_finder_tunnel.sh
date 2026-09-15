#!/usr/bin/env bash
# Open an SSH tunnel to the ChannelFinder REST API via the jump host.
#
# Usage:
#   ./scripts/als_channel_finder_tunnel.sh        # open tunnel
#   ./scripts/als_channel_finder_tunnel.sh stop    # close tunnel
#
# The ChannelFinder host and the SSH jump host that reaches it are this
# deployment's own:
#   CHANNEL_FINDER_HOST=channelfinder.example.org
#   CHANNEL_FINDER_JUMP_HOST=bastion.example.org
#
# Once open, the API is reachable at https://localhost:8443/ChannelFinder/
# Configure osprey with:
#   channel_finder:
#     direct:
#       backend: als_channel_finder
#       backend_url: https://localhost:8443/ChannelFinder

set -euo pipefail

LOCAL_PORT=8443
REMOTE_HOST="${CHANNEL_FINDER_HOST:?set CHANNEL_FINDER_HOST to your ChannelFinder host}"
REMOTE_PORT=443
JUMP_HOST="${CHANNEL_FINDER_JUMP_HOST:?set CHANNEL_FINDER_JUMP_HOST to the SSH host that reaches it}"

if [[ "${1:-}" == "stop" ]]; then
    # Kill existing tunnel
    pkill -f "ssh.*-L ${LOCAL_PORT}:${REMOTE_HOST}:${REMOTE_PORT}" 2>/dev/null \
        && echo "Tunnel closed." \
        || echo "No tunnel found."
    exit 0
fi

# Check if tunnel is already running
if lsof -i ":${LOCAL_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Tunnel already open on localhost:${LOCAL_PORT}"
    exit 0
fi

ssh -fnNT -L "${LOCAL_PORT}:${REMOTE_HOST}:${REMOTE_PORT}" "${JUMP_HOST}"
echo "Tunnel open: localhost:${LOCAL_PORT} → ${REMOTE_HOST}:${REMOTE_PORT} (via ${JUMP_HOST})"
