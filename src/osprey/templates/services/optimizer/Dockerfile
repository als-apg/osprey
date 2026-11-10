# Image for the optimizer service (osprey_optimizer): a FastAPI process that
# runs Xopt-driven optimizations against the control system through the
# connector-host pool, keeping its jobs in the Redis sidecar beside it.
#
# It installs the optimizer package with its `service` extra, not the
# framework: the service needs the engine, its HTTP layer and the connectors,
# and nothing else the framework carries.
#
# osprey install strategy (two cache-friendly layers):
#   * deps layer  — primes the pinned `osprey-optimizer[service]` release (+ its
#     dependencies) in one RUN, plus any local dependency delta staged as
#     osprey-local-requirements.txt on dev builds. Torch is taken from the
#     PyTorch CPU wheel index (OPTIMIZER_TORCH_INDEX_URL): from PyPI a Linux
#     torch is the CUDA build, gigabytes the service never uses.
#   * wheel layer — when `osprey up --dev` stages locally-built wheels (the
#     optimizer and the connectors) into the build context, this later layer
#     overlays them with the `service` extra, so unreleased code and any
#     dependency the dev wheel adds to that extra are included. With no wheel
#     staged it is a no-op.
#   * source builds — `osprey up --dev` also stages osprey-lock-constraints.txt,
#     the checkout's uv.lock exported as constraints. With it present the deps
#     layer primes no release, and every install layer holds each third-party
#     package to the version the lock holds (see the deps layer).
# This image is built locally by `osprey up`; override with
# OSPREY_OPTIMIZER_IMAGE to use a prebuilt/published image.

# Registry standing in for docker.io; images.base_registry points it at a site mirror of Docker Hub.
ARG OSPREY_BASE_REGISTRY="docker.io"
FROM ${OSPREY_BASE_REGISTRY}/library/python:3.11-slim

WORKDIR /app

# Optional site CA, for building (and running) behind a TLS-intercepting proxy
# that re-signs traffic with a site CA — the same layer, in the same place, as
# the project image's. OSPREY_SITE_CA names a CA file (PEM) staged in the build
# context; the compose build stages the operator's `images.site_ca` there and
# passes the name, because a COPY cannot reach outside the context. The
# `.osprey-layer-anchor` sibling keeps the glob COPY matching when nothing is
# staged, and with the ARG unset the RUN is a no-op. This layer sits BEFORE the
# apt install below deliberately: that fetch verifies TLS against the system
# store this extends.
#
# The ENVs point each tool family at the merged Debian bundle the install lands
# in — pip trusts only its bundled certifi without PIP_CERT, and SSL_CERT_FILE
# / REQUESTS_CA_BUNDLE cover Python's ssl module and requests. They are set
# unconditionally and always name the merged bundle: with no CA staged they
# restate each tool's own default, whereas a variable naming a path that may
# not exist crashes an httpx client at construction. No NODE_EXTRA_CA_CERTS —
# this image has no Node.
ARG OSPREY_SITE_CA=""
COPY .osprey-layer-anchor *.cr[t] *.pe[m] /tmp/ca-ctx/
RUN if [ -n "$OSPREY_SITE_CA" ]; then \
        cp "/tmp/ca-ctx/${OSPREY_SITE_CA}" /usr/local/share/ca-certificates/osprey-site-ca.crt \
        && update-ca-certificates; \
    fi \
 && rm -rf /tmp/ca-ctx
ENV PIP_CERT=/etc/ssl/certs/ca-certificates.crt \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

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
#
# The base image's docker-clean hook deletes every downloaded .deb after each
# install, which keeps a plain layer small but would empty the apt cache mount
# the installing RUNs below share across builds, so it is removed here. Every
# RUN that installs packages mounts /var/cache/apt and /var/lib/apt/lists as
# caches, so neither the archives nor the lists land in a layer.
RUN export http_proxy="${http_proxy:-${HTTP_PROXY:-}}" https_proxy="${https_proxy:-${HTTPS_PROXY:-}}" no_proxy="${no_proxy:-${NO_PROXY:-}}"; \
    find /etc/apt \( -name '*.sources' -o -name '*.list' \) \
    -exec sed -i 's|http://deb.debian.org|https://deb.debian.org|g' {} + \
 && printf 'Acquire::Retries "5";\nAcquire::http::Pipeline-Depth "0";\nAcquire::https::Pipeline-Depth "0";\n' > /etc/apt/apt.conf.d/80-osprey-retries \
 && rm -f /etc/apt/apt.conf.d/docker-clean

# ── deps layer ───────────────────────────────────────────────────────────────
# Prime the image with the pinned framework release and its dependencies. A C
# toolchain is needed at install time to compile any native deps; it is purged
# in this same RUN so it does not bloat the final image. Under `--dev` an
# unreleased pin may not exist on PyPI: OSPREY_DEV=1 relaxes the failure to an
# unpinned prime (the wheel layer below then overlays the real code); without it
# a pin miss stays fatal. Any local dependency delta staged as
# osprey-local-requirements.txt (dev builds only) is installed after the primer,
# while the toolchain is still available; the `.osprey-layer-anchor` COPY
# sibling keeps the glob matching when no manifest is staged, so this cache only
# busts when the manifest content changes.
#
# Everything from FROM through this layer is the same, instruction for
# instruction, in every Python service recipe, and `.osprey-layer-anchor` is the
# same bytes in every context. That is what lets BuildKit run the framework
# install once and share the layer across every service image a deploy builds;
# whatever one image needs beyond it goes in a layer below this one.
#
# pip and apt fetch through BuildKit cache mounts, so a deps layer that does
# rebuild (a new pin, a changed manifest) downloads only what changed, and
# images building side by side draw on one cache. The apt mounts are locked:
# apt keeps its own lock files in both directories, so two builds cannot use
# them at once, and podman's builder accepts no private sharing mode. The lock
# lasts the whole RUN, so two different deps layers building together take
# turns; the service recipes share one deps layer, which builds only once.
#
# The installs here run under uv rather than pip: uv resolves without pip's
# one-candidate-at-a-time metadata crawl and downloads in parallel, which on a
# slow index is most of this layer's time. It is installed with pip from the
# same index (so a site mirror still applies), handed pip's settings (the
# constraint file for runtime and build dependencies, the site indexes with
# pip's consider-every-index strategy, bytecode compilation), and uninstalled
# before the RUN ends, so no image ships it. Later layers install with pip.
ARG OSPREY_VERSION=""
ARG OSPREY_DEV=""
# OSPREY_PIP_PRE=1 says the pin is a pre-release. The optimizer and its
# connectors ship as a pair from one tag, and an installer prefers a stable
# release for a requirement that names none (the optimizer's own connectors
# requirement), so the deps layer then resolves as a whole with
# --prerelease=allow.
ARG OSPREY_PIP_PRE=""
# The three pip ARGs are the rest of the site build settings the compose build
# hands this image, declared so that they arrive (see the bluesky-web image for
# the full account). PIP_NO_PROXY is the proxy bypass list, mapped onto
# NO_PROXY/no_proxy in the deps RUN below only when it is set.
ARG PIP_NO_PROXY=""
ARG PIP_INDEX_URL=""
ARG PIP_EXTRA_INDEX_URL=""
# Where torch's CPU build comes from. A site whose mirror serves CPU torch
# itself sets this empty, and the install resolves torch from the site indexes
# alone.
ARG OPTIMIZER_TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
# A build from an osprey source checkout (`osprey up --dev`) also stages
# osprey-lock-constraints.txt (the checkout's uv.lock exported as constraints)
# and osprey-extra-requirements.txt (the `service` extra's own requirements),
# and the lock file's presence, not OSPREY_DEV, puts this layer on the
# lock-driven install: nothing is primed from a released optimizer, and the
# staged wheels' base requirements and the extra's are installed here, with
# the toolchain, every third-party package held to the version the lock holds.
# The lock's lines are appended to the setuptools constraint file, so they
# reach uv's build environments too, and the file is kept at
# /opt/osprey/lock-constraints.txt for the wheel layer below. With no lock file
# staged (a release build) the layer is exactly what it was.
COPY .osprey-layer-anchor osprey-local-requirements.tx[t] osprey-lock-constraints.tx[t] osprey-extra-requirements.tx[t] /tmp/deps-ctx/
# The torch index rides in UV_INDEX beside any site extra index, and uv's
# consider-every-index strategy then takes the `+cpu` build, the highest
# version matching the extra's `torch==` pin. On the lock-driven path the lock
# itself pins that build (`torch==<version>+cpu` off macOS), which only the
# CPU index serves: a site that empties OPTIMIZER_TORCH_INDEX_URL must then
# mirror the `+cpu` build itself, as a mirror carrying only PyPI's plain torch
# has no candidate the lock admits and the layer fails.
#
# A dev build of an unreleased optimizer has no pinned release AND no earlier
# one to prime from, so that case skips the prime instead of failing: the wheel
# layer below then installs the whole service stack from the staged wheels.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    export http_proxy="${http_proxy:-${HTTP_PROXY:-}}" https_proxy="${https_proxy:-${HTTPS_PROXY:-}}" no_proxy="${no_proxy:-${NO_PROXY:-}}"; \
    [ -z "$PIP_NO_PROXY" ] || export NO_PROXY="$PIP_NO_PROXY" no_proxy="$PIP_NO_PROXY"; \
    [ -n "$OSPREY_VERSION" ] || { echo "ERROR: OSPREY_VERSION build-arg is required" >&2; exit 1; } \
    && printf 'setuptools<84\n' > /tmp/deps-ctx/pip-constraints.txt \
    && export PIP_CONSTRAINT=/tmp/deps-ctx/pip-constraints.txt \
              UV_CONSTRAINT=/tmp/deps-ctx/pip-constraints.txt \
              UV_BUILD_CONSTRAINT=/tmp/deps-ctx/pip-constraints.txt \
              UV_SYSTEM_PYTHON=1 UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 \
              UV_INDEX_STRATEGY=unsafe-best-match \
    && { [ -z "$PIP_INDEX_URL" ] || export UV_DEFAULT_INDEX="$PIP_INDEX_URL"; } \
    && extra_indexes="${PIP_EXTRA_INDEX_URL} ${OPTIMIZER_TORCH_INDEX_URL}" \
    && { [ -z "${extra_indexes# }" ] || export UV_INDEX="$extra_indexes"; } \
    && apt-get update \
    && apt-get install -y --no-install-recommends build-essential python3-dev \
    && pip install "uv==0.12.21" \
    && if [ -f /tmp/deps-ctx/osprey-lock-constraints.txt ]; then \
           echo "Source build: installing the staged wheels' dependencies at the checkout's uv.lock versions" \
           && mkdir -p /opt/osprey \
           && cp /tmp/deps-ctx/osprey-lock-constraints.txt /opt/osprey/lock-constraints.txt \
           && cat /opt/osprey/lock-constraints.txt >> /tmp/deps-ctx/pip-constraints.txt \
           && uv pip install -r /tmp/deps-ctx/osprey-local-requirements.txt \
                  -r /tmp/deps-ctx/osprey-extra-requirements.txt ; \
       else \
           { uv pip install ${OSPREY_PIP_PRE:+--prerelease=allow} "osprey-optimizer[service]==$OSPREY_VERSION" \
             || if [ "$OSPREY_DEV" = "1" ]; then \
                    echo "WARNING: pin unreleased, priming with latest" \
                    && { uv pip install "osprey-optimizer[service]" \
                         || echo "WARNING: no osprey-optimizer release to prime from; the wheel layer installs the service stack" ; } ; \
                else \
                    exit 1 ; \
                fi ; } \
           && if [ -f /tmp/deps-ctx/osprey-local-requirements.txt ]; then \
                  uv pip install -r /tmp/deps-ctx/osprey-local-requirements.txt ; \
              fi ; \
       fi \
    && pip uninstall -y uv \
    && apt-get purge -y build-essential python3-dev \
    && apt-get autoremove -y \
    && rm -rf /tmp/deps-ctx

# ── wheel layer ──────────────────────────────────────────────────────────────
# Overlay the locally-built wheels when `osprey up --dev` stages them into the
# build context. `.osprey-layer-anchor` is a guaranteed sibling of the COPY, so
# the glob always matches at least one file. The optimizer wheel is installed
# WITH its `service` extra (the deps layer primed only the released extra set,
# and `pip check` cannot see a missing extra), beside the connectors wheel it
# requires; then both are force-reinstalled --no-deps so their own modules
# win, and `pip check` guards against residual mismatches.
#
# On the lock-driven path (see the deps layer) the first install runs under
# /opt/osprey/lock-constraints.txt, so anything the wheels add beyond what the
# earlier layers installed is held to the checkout's uv.lock as well; the
# staged wheels satisfy one another, so nothing resolves an osprey
# distribution from the index.
COPY .osprey-layer-anchor *.wh[l] /tmp/ctx/
RUN --mount=type=cache,target=/root/.cache/pip \
    if ls /tmp/ctx/*.whl >/dev/null 2>&1; then \
        echo "Overlaying locally-built osprey wheels (dev build)" \
        && lock= && { [ ! -f /opt/osprey/lock-constraints.txt ] || lock="-c /opt/osprey/lock-constraints.txt" ; } \
        && whl="$(ls /tmp/ctx/osprey_optimizer-*.whl)" \
        && pip install $lock ${OPTIMIZER_TORCH_INDEX_URL:+--extra-index-url "$OPTIMIZER_TORCH_INDEX_URL"} \
               "${whl}[service]" /tmp/ctx/osprey_connectors-*.whl \
        && pip install --no-deps --force-reinstall /tmp/ctx/*.whl \
        && pip check ; \
    fi \
    && rm -rf /tmp/ctx

# Project metadata, kept as the final metadata-only layer so the shared deps
# cache chain above stays identical across projects.
ARG OSPREY_PROJECT_NAME=""
LABEL com.osprey.project=$OSPREY_PROJECT_NAME
