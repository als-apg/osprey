# Image for the Bluesky Panels sidecar service (osprey.services.bluesky_panels): a
# FastAPI process serving the operator web panels, same-origin read-proxying
# the bluesky bridge, and exposing /health/full, reached by the web-terminal
# reverse proxy at /panel/{id}.
#
# osprey install strategy (matches the framework's --dev convention):
#   * dev builds  — `osprey deploy up --dev` drops a locally-built wheel into the
#     build context; if a *.whl is present we install that (so unreleased code
#     is included).
#   * prod builds — no wheel present, so install the pinned release from PyPI.
# The framework's core FastAPI/uvicorn/httpx deps cover everything this
# sidecar needs, so both paths are a plain install.
# This image is built locally by `osprey deploy up`; override with
# OSPREY_BLUESKY_PANELS_IMAGE to use a prebuilt/published image.

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
        && pip install --no-cache-dir "${whl}" ; \
    elif [ -n "$OSPREY_VERSION" ]; then \
        echo "Installing osprey-framework==$OSPREY_VERSION from PyPI" \
        && pip install --no-cache-dir "osprey-framework==$OSPREY_VERSION" ; \
    else \
        echo "Installing osprey-framework (latest) from PyPI" \
        && pip install --no-cache-dir "osprey-framework" ; \
    fi \
    && apt-get purge -y build-essential python3-dev \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* /tmp/ctx
