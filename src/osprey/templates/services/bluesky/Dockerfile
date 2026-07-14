# Image for the Bluesky bridge service (osprey.services.bluesky_bridge): a
# FastAPI process owning the Bluesky RunEngine + ophyd-async devices, reached
# by the `scan` MCP server over HTTP.
#
# osprey install strategy (matches the framework's --dev convention):
#   * dev builds  — `osprey deploy up --dev` drops a locally-built wheel into the
#     build context; if a *.whl is present we install that (so unreleased code
#     is included).
#   * prod builds — no wheel present, so install the pinned release from PyPI.
# Both paths install the `bluesky-bridge` extra on top of the framework's core
# FastAPI/uvicorn deps, pulling in bluesky/ophyd-async/tiled for the real
# (non-Fake) Scanner implementation.
# This image is built locally by `osprey deploy up`; override with
# OSPREY_BLUESKY_BRIDGE_IMAGE to use a prebuilt/published image.

FROM python:3.11-slim

WORKDIR /app

# Install osprey: prefer a locally-built wheel (dev), else the pinned PyPI release.
# A C toolchain is needed at install time to compile native deps; it's purged
# in the same layer so it doesn't bloat the final image.
ARG OSPREY_VERSION=""
COPY . /tmp/ctx/
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential python3-dev \
    && if ls /tmp/ctx/*.whl >/dev/null 2>&1; then \
        echo "Installing osprey from local wheel (dev build)" \
        && whl="$(ls /tmp/ctx/*.whl | head -n1)" \
        && pip install --no-cache-dir "${whl}[bluesky-bridge]" ; \
    elif [ -n "$OSPREY_VERSION" ]; then \
        echo "Installing osprey-framework[bluesky-bridge]==$OSPREY_VERSION from PyPI" \
        && pip install --no-cache-dir "osprey-framework[bluesky-bridge]==$OSPREY_VERSION" ; \
    else \
        echo "Installing osprey-framework[bluesky-bridge] (latest) from PyPI" \
        && pip install --no-cache-dir "osprey-framework[bluesky-bridge]" ; \
    fi \
    && apt-get purge -y build-essential python3-dev \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* /tmp/ctx
