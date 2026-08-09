# Image for the Bluesky bridge service (osprey.services.bluesky_bridge): a
# FastAPI process owning the queue facade, reached by the `scan` MCP server
# over HTTP. The same image also runs the `queueserver` compose service
# (`start-re-manager`), which spawns the RunEngine worker that owns the
# ophyd-async devices — one image, two entrypoints, so the worker namespace
# and the bridge always agree on the osprey code they share.
#
# osprey install strategy (two cache-friendly layers):
#   * deps layer  — primes the pinned framework release (+ its dependencies)
#     from PyPI in one RUN. The framework's core deps already include
#     bluesky/ophyd-async/tiled (Bluesky is a core part of OSPREY), so a plain
#     prime carries everything the real (non-Fake) PlanRunner needs — plus any
#     local dependency delta staged as osprey-local-requirements.txt on dev
#     builds. Its build cache is shared across projects and only rebuilds when
#     the staged manifest changes.
#   * wheel layer — when `osprey deploy up --dev` stages a locally-built wheel
#     into the build context, this later layer overlays it (so unreleased code
#     is included). With no wheel staged it is a no-op.
# This image is built locally by `osprey deploy up`; override with
# OSPREY_BLUESKY_BRIDGE_IMAGE to use a prebuilt/published image.

FROM python:3.11-slim

WORKDIR /app

# Debian apt mirrors over HTTPS (plain-HTTP bulk fetches are throttled or
# broken by middleboxes on some networks; deb.debian.org supports HTTPS), and
# bounded apt retries with backoff so a transient network blip mid-build does
# not fail the whole image.
RUN find /etc/apt \( -name '*.sources' -o -name '*.list' \) \
    -exec sed -i 's|http://deb.debian.org|https://deb.debian.org|g' {} + \
 && printf 'Acquire::Retries "5";\n' > /etc/apt/apt.conf.d/80-osprey-retries

# ── deps layer ───────────────────────────────────────────────────────────────
# Prime the image with the pinned framework release and its dependencies. A C
# toolchain is needed at install time to compile any native deps; it is purged
# in this same RUN so it does not bloat the final image. Under `--dev` an
# unreleased pin may not exist on PyPI: OSPREY_DEV=1 relaxes the failure to an
# unpinned prime (the wheel layer below then overlays the real code); without it
# a pin miss stays fatal. Any local dependency delta staged as
# osprey-local-requirements.txt (dev builds only) is installed after the primer,
# while the toolchain is still available; the `.dockerignore` COPY sibling keeps
# the glob matching when no manifest is staged, so this cache only busts when
# the manifest content changes.
#
# `bluesky-queueserver` (the RE Manager itself, providing `start-re-manager`)
# arrives transitively via the framework's `bluesky-queueserver-api` client
# dependency, but transitively means unpinned: it is installed again here at an
# exact version so the manager the image runs and the client the bridge talks
# to are a version pair chosen deliberately, not whatever resolved that day.
#
# This literal is MANUALLY MAINTAINED and lock-checked — dependabot does not
# see it (there is no docker ecosystem configured, and it would not parse an
# ARG regardless). `test_bluesky_compose_render.py` asserts it equals the
# version uv.lock resolves, so a lockfile bump that leaves this behind fails
# CI rather than silently shipping a manager/client version skew.
#
# The closing `pip check` is SKIPPED on dev builds, and only here. On a `--dev`
# build this layer holds the last RELEASED framework (the unreleased pin misses
# PyPI and falls back to an unpinned prime, above) alongside the WORKING TREE's
# dependency pins from osprey-local-requirements.txt. Those two disagree by
# construction whenever a pin has moved since the last release — mid-cycle,
# that is the normal state — so a strict check here fails every dev build for a
# mismatch this image does not ship: the wheel layer below installs the local
# wheel, whose own metadata is what the local pins belong to, and runs its own
# unconditional `pip check` on the result. Non-dev builds install exactly the
# pinned release and keep the strict check.
ARG OSPREY_VERSION=""
ARG OSPREY_DEV=""
ARG BLUESKY_QUEUESERVER_VERSION="0.0.25"
COPY .dockerignore osprey-local-requirements.tx[t] /tmp/deps-ctx/
RUN [ -n "$OSPREY_VERSION" ] || { echo "ERROR: OSPREY_VERSION build-arg is required" >&2; exit 1; } \
    && apt-get update \
    && apt-get install -y --no-install-recommends build-essential python3-dev \
    && { pip install --no-cache-dir "osprey-framework==$OSPREY_VERSION" \
         || if [ "$OSPREY_DEV" = "1" ]; then \
                echo "WARNING: pin unreleased, priming with latest" \
                && pip install --no-cache-dir "osprey-framework" ; \
            else \
                exit 1 ; \
            fi ; } \
    && pip install --no-cache-dir "bluesky-queueserver==$BLUESKY_QUEUESERVER_VERSION" \
    && if [ -f /tmp/deps-ctx/osprey-local-requirements.txt ]; then \
           pip install --no-cache-dir -r /tmp/deps-ctx/osprey-local-requirements.txt ; \
       fi \
    && { [ "$OSPREY_DEV" = "1" ] || pip check ; } \
    && apt-get purge -y build-essential python3-dev \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* /tmp/deps-ctx

# ── wheel layer ──────────────────────────────────────────────────────────────
# Overlay a locally-built wheel when `osprey deploy up --dev` stages one into
# the build context. `.dockerignore` is a guaranteed sibling of the COPY, so
# the glob always matches at least one file; `*.wh[l]` optionally pulls in the
# wheel. The wheel is installed plain (no extra — the deps layer already primed
# every extra), then force-reinstalled --no-deps to guarantee its own modules
# win, and `pip check` guards against residual mismatches. No wheel staged →
# no-op, image already complete after the deps layer.
COPY .dockerignore *.wh[l] /tmp/ctx/
RUN if ls /tmp/ctx/*.whl >/dev/null 2>&1; then \
        echo "Overlaying locally-built osprey wheel (dev build)" \
        && pip install --no-cache-dir /tmp/ctx/*.whl \
        && pip install --no-cache-dir --no-deps --force-reinstall /tmp/ctx/*.whl \
        && pip check ; \
    fi \
    && rm -rf /tmp/ctx

# Project metadata, kept as the final metadata-only layer so the shared deps
# cache chain above stays identical across projects (a per-project value here
# never invalidates the framework install below it).
ARG OSPREY_PROJECT_NAME=""
LABEL com.osprey.project=$OSPREY_PROJECT_NAME
