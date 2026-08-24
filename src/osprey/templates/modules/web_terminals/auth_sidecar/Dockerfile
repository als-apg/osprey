# Image for the web-terminal auth sidecar (osprey.services.auth_sidecar): the
# FastAPI process that answers nginx `auth_request` subrequests for every
# `/u/<user>/` location and serves the login flow.
#
# NO ENTRYPOINT, deliberately: the rendered docker-compose.web.yml supplies the
# FULL uvicorn argv via `command:`, so anything prepended here would either
# swallow those arguments or duplicate them. The healthcheck in that same file
# is the in-image python-urllib probe every other OSPREY service uses, so both
# `python` and `uvicorn` must stay on PATH — the base image provides the first
# and the framework install below provides the second.
#
# osprey install strategy (two cache-friendly layers):
#   * deps layer  — primes the pinned framework release (+ its dependencies)
#     from PyPI in one RUN. The framework's core FastAPI/uvicorn/authlib/
#     itsdangerous deps cover everything this sidecar needs, so a plain prime
#     suffices — plus any local dependency delta staged as
#     osprey-local-requirements.txt on dev builds. Its build cache is shared
#     across projects and only rebuilds when the staged manifest changes.
#   * wheel layer — when `osprey up --dev` stages a locally-built wheel
#     into the build context, this later layer overlays it (so unreleased code
#     is included). With no wheel staged it is a no-op.
# This image is built locally by `osprey up` in `image_source: local`
# mode; a registry-mode deployment sets modules.web_terminals.auth.image to a
# published image instead, and publishing that image is the facility CI's job.
#
# WHY THIS LIVES UNDER templates/modules/ AND NOT templates/services/: that tree
# is catalog-bound — every directory in it must be a registered BuildArtifactCatalog
# entry copied into rendered projects, and `_copy_service_templates` only copies
# services a project names in `deployed_services` or declares under `services:`.
# The sidecar is a member of the web-terminals stack, not a service a project
# declares, so an entry there would never be acted on. Do not "tidy" this back.

FROM python:3.11-slim

WORKDIR /app

# Debian apt mirrors over HTTPS (plain-HTTP bulk fetches are throttled or
# broken by middleboxes on some networks; deb.debian.org supports HTTPS), and
# bounded apt retries with backoff so a transient network blip mid-build does
# not fail the whole image. Pipelining is disabled alongside those retries
# because retries alone were seen not to cover every blip: a CI build lost one
# 4.6 kB .deb to a peer connection reset while 70 other packages fetched fine
# from the same host, with Acquire::Retries already in effect. Apt's default of
# up to 10 requests per connection is the part of that failure we can act on —
# one request per connection is marginally slower and strictly easier to
# recover. Set for both schemes deliberately: these mirrors are rewritten to
# HTTPS just above, and apt keeps no https entry in its config tree unless one
# is written, so relying on a fallback from the http key is how this would
# quietly become a no-op.
RUN export http_proxy="${http_proxy:-${HTTP_PROXY:-}}" https_proxy="${https_proxy:-${HTTPS_PROXY:-}}" no_proxy="${no_proxy:-${NO_PROXY:-}}"; \
    find /etc/apt \( -name '*.sources' -o -name '*.list' \) \
    -exec sed -i 's|http://deb.debian.org|https://deb.debian.org|g' {} + \
 && printf 'Acquire::Retries "5";\nAcquire::http::Pipeline-Depth "0";\nAcquire::https::Pipeline-Depth "0";\n' > /etc/apt/apt.conf.d/80-osprey-retries

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
ARG OSPREY_VERSION=""
ARG OSPREY_DEV=""
COPY .dockerignore osprey-local-requirements.tx[t] /tmp/deps-ctx/
# setuptools 84.0.0 breaks setuptools_dso's compile-probe error handling and
# fails sdist compiles of the EPICS toolchain (pvxslibs/epicscorelibs/softioc)
# where no binary wheel exists (notably linux/arm64); PIP_CONSTRAINT reaches
# pip's isolated build environments. Drop once a fixed release is out.
RUN export http_proxy="${http_proxy:-${HTTP_PROXY:-}}" https_proxy="${https_proxy:-${HTTPS_PROXY:-}}" no_proxy="${no_proxy:-${NO_PROXY:-}}"; \
    [ -n "$OSPREY_VERSION" ] || { echo "ERROR: OSPREY_VERSION build-arg is required" >&2; exit 1; } \
    && printf 'setuptools<84\n' > /tmp/deps-ctx/pip-constraints.txt \
    && export PIP_CONSTRAINT=/tmp/deps-ctx/pip-constraints.txt \
    && apt-get update \
    && apt-get install -y --no-install-recommends build-essential python3-dev \
    && { pip install --no-cache-dir "osprey-framework==$OSPREY_VERSION" \
         || if [ "$OSPREY_DEV" = "1" ]; then \
                echo "WARNING: pin unreleased, priming with latest" \
                && pip install --no-cache-dir "osprey-framework" ; \
            else \
                exit 1 ; \
            fi ; } \
    && if [ -f /tmp/deps-ctx/osprey-local-requirements.txt ]; then \
           pip install --no-cache-dir -r /tmp/deps-ctx/osprey-local-requirements.txt ; \
       fi \
    && apt-get purge -y build-essential python3-dev \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* /tmp/deps-ctx

# ── wheel layer ──────────────────────────────────────────────────────────────
# Overlay a locally-built wheel when `osprey up --dev` stages one into
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
