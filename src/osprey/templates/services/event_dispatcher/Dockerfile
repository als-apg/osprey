# Shared image for the event-dispatcher and dispatch-worker services.
#
# Both services are plain `python -m ...` entrypoints (the compose `command:`
# selects which one), so they run from a single image. The worker executes the
# agent via the Claude Code CLI, which is a Node package — hence Node + the CLI
# are installed here alongside the Python framework.
#
# osprey install strategy (matches the framework's --dev convention):
#   * dev builds  — `osprey deploy up --dev` drops a locally-built wheel into the
#     build context; if a *.whl is present we install that (so unreleased code,
#     e.g. the dispatch feature itself, is included).
#   * prod builds — no wheel present, so install the pinned release from PyPI.
# This image is built locally by `osprey deploy up`; override with
# OSPREY_DISPATCH_IMAGE / OSPREY_WORKER_IMAGE to use a prebuilt/published image.

FROM python:3.11-slim

# Node.js 20 + the Claude Code CLI (the dispatch-worker shells out to `claude`).
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    # Pin the Claude Code CLI so an upstream release cannot silently change the
    # headless agent's behavior between image builds (matches the cli_version
    # OSPREY documents in config.yml). Bump deliberately.
    && npm install -g @anthropic-ai/claude-code@2.1.146 \
    && apt-get purge -y curl gnupg \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install osprey: prefer a locally-built wheel (dev), else the pinned PyPI release.
# A C toolchain is needed at install time to compile native deps (e.g.
# accelerator-toolbox); it's purged in the same layer so it doesn't bloat the
# final image.
ARG OSPREY_VERSION=""
COPY . /tmp/ctx/
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential python3-dev \
    && if ls /tmp/ctx/*.whl >/dev/null 2>&1; then \
        echo "Installing osprey from local wheel (dev build)" \
        && pip install --no-cache-dir /tmp/ctx/*.whl ; \
    elif [ -n "$OSPREY_VERSION" ]; then \
        echo "Installing osprey-framework==$OSPREY_VERSION from PyPI" \
        && pip install --no-cache-dir "osprey-framework==$OSPREY_VERSION" ; \
    else \
        echo "Installing osprey-framework (latest) from PyPI" \
        && pip install --no-cache-dir osprey-framework ; \
    fi \
    && apt-get purge -y build-essential python3-dev \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* /tmp/ctx
