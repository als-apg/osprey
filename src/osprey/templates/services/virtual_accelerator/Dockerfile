# Image for the Virtual Accelerator (osprey.services.virtual_accelerator): a
# PyAT-backed EPICS server, reached over Channel Access by an unmodified
# EPICSConnector, with the physics model's own variables served on PVAccess
# from the same process.
#
# osprey install strategy (cache-friendly layers):
#   * deps layer  — primes the pinned framework release (+ its dependencies)
#     from PyPI in one RUN, plus any local dependency delta staged as
#     osprey-local-requirements.txt on dev builds. It is the same layer, built
#     once, as every other Python service image's, so a deploy that builds the
#     bridge too installs the framework once. Its build cache is shared across
#     projects and only rebuilds when the staged manifest changes.
#   * virtual-accelerator layer — adds the framework's `virtual-accelerator`
#     extra on top: the serving stack (`lume-pva-apg[ca,pva]`) and the archiver
#     recorder's CA client, resolved from the extra so this file cannot pin a
#     dependency that pyproject.toml disagrees with. PyAT and pymongo (the
#     recorder's store client) are core and arrive with the deps layer. The
#     recorder runs THIS image with a different command (compose template only,
#     no Dockerfile of its own), which is why its deps ride along here.
#   * wheel layer — when `osprey up --dev` stages a locally-built wheel
#     into the build context, this later layer overlays it (so unreleased code
#     is included). The first wheel install carries the `[virtual-accelerator]`
#     extra so a dev wheel that adds or bumps a dep inside the extra picks it up
#     (the primer only covers the *released* extra set, and `pip check` cannot
#     see missing extras); the second install is a fast, deps-free
#     force-reinstall without the extra. With no wheel staged it is a no-op.
#   * source builds — `osprey up --dev` also stages osprey-lock-constraints.txt,
#     the checkout's uv.lock exported as constraints. With it present the deps
#     layer primes no release, and every install layer holds each third-party
#     package to the version the lock holds (see the deps layer).
#
# No install layer pins a native-dependency version — the extra + core deps
# resolve them, so they can't drift from pyproject.toml's declared constraints.
# The one exception is the arm64 native-wheels stage below, which builds pcaspy
# from its sdist at a pinned version inside the range pyproject.toml admits.
#
# ARCHITECTURE: native, linux/amd64 or linux/arm64 — whichever the build
# targets. On amd64 every framework dependency resolves to a prebuilt manylinux
# wheel. On arm64 several have no linux/aarch64 wheel at the interpreter this
# image runs. p4p, the EPICS libraries under it and accelerator-toolbox (PyAT)
# build from source with the deps layer's own toolchain, exactly as they do in
# the other service images that share that layer. pcaspy, the Channel Access
# server behind the serving stack, is built by the `native-wheels-arm64` stage
# from its pinned sdist against EPICS Base and the PCAS module, which that
# stage compiles first, statically, so the wheel carries them — and handed to
# the final stage in a find-links directory, where the virtual-accelerator
# layer installs it like any other wheel. On amd64 that directory is empty and
# the resolve is exactly what it is without it.
# The stage depends on nothing but its pins, so BuildKit builds it once per
# host and serves it from cache after; the install layers, which rebuild with
# every new framework version, never recompile EPICS. An architecture with no
# native-wheels stage fails at the FROM that selects one.
#
# This image is built locally by `osprey up`; override with
# OSPREY_VA_IMAGE to use a prebuilt/published image.

# The build's target architecture, set by BuildKit and by buildah; it selects
# the native-wheels stage.
ARG TARGETARCH
# Registry standing in for docker.io; images.base_registry points it at a site mirror of Docker Hub.
ARG OSPREY_BASE_REGISTRY="docker.io"

# ── base: site trust and apt settings, shared by every stage ────────────────
FROM ${OSPREY_BASE_REGISTRY}/library/python:3.11-slim AS va-base

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

# ── native wheels ────────────────────────────────────────────────────────────
# One stage per architecture, each leaving its wheels in /wheels. amd64 needs
# none: PyPI has a wheel for every dependency there.
FROM va-base AS native-wheels-amd64
RUN mkdir -p /wheels

# arm64: pcaspy from its sdist. Every download is pinned by version and checked
# against its sha256, the sdist through pip's hash mode. PyAT is not built here:
# it is a core dependency, so the shared deps layer has already installed it by
# the time this stage's wheels are in reach.
#
# pcaspy links EPICS Base and the PCAS module, which Base 7 no longer bundles.
# Both build here as position-independent static libraries, and PCAS's headers
# and libraries are merged into Base's tree: pcaspy's setup.py links statically
# when every library it needs sits under EPICS_BASE, so its extension carries
# EPICS inside it and the image needs no EPICS tree at run time. The pcaspy pin
# keeps lume-pva-apg's `>=0.8.1` floor: 0.8.0's `SimplePV.writeNotify` aborts
# the server on any asynchronous write. swig generates pcaspy's bindings.
#
# The stage is a builder only — nothing but /wheels leaves it — so its
# toolchain is never shipped and needs no purge.
FROM va-base AS native-wheels-arm64
ARG EPICS_BASE_VERSION=7.0.10
ARG EPICS_BASE_SHA256=44193e962793de514ead442e9b5096a13109f1ee275f131995826468aa89b839
ARG PCAS_VERSION=4.13.3
ARG PCAS_SHA256=5004e39339c8e592fcb9b4275c84143635c6e688c0fbe01f17dafe19850398a0
ARG PCASPY_VERSION=0.8.1
ARG PCASPY_SHA256=80f6aafacdd796c471b62737ad242a322cff22d8b8f18d9c56feed4270d221bf
ARG PIP_NO_PROXY=""
ARG PIP_INDEX_URL=""
ARG PIP_EXTRA_INDEX_URL=""
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    export http_proxy="${http_proxy:-${HTTP_PROXY:-}}" https_proxy="${https_proxy:-${HTTPS_PROXY:-}}" no_proxy="${no_proxy:-${NO_PROXY:-}}"; \
    [ -z "$PIP_NO_PROXY" ] || export NO_PROXY="$PIP_NO_PROXY" no_proxy="$PIP_NO_PROXY"; \
    set -e; \
    apt-get update; \
    apt-get install -y --no-install-recommends build-essential perl swig curl; \
    base=/tmp/native/epics; pcas=/tmp/native/pcas; \
    mkdir -p "$base" "$pcas" /wheels; \
    curl -fsSL --retry 5 -o /tmp/native/base.tar.gz \
        "https://github.com/epics-base/epics-base/releases/download/R${EPICS_BASE_VERSION}/base-${EPICS_BASE_VERSION}.tar.gz"; \
    echo "${EPICS_BASE_SHA256}  /tmp/native/base.tar.gz" | sha256sum -c -; \
    curl -fsSL --retry 5 -o /tmp/native/pcas.tar.gz \
        "https://github.com/epics-modules/pcas/archive/refs/tags/v${PCAS_VERSION}.tar.gz"; \
    echo "${PCAS_SHA256}  /tmp/native/pcas.tar.gz" | sha256sum -c -; \
    tar -xzf /tmp/native/base.tar.gz --strip-components=1 -C "$base"; \
    tar -xzf /tmp/native/pcas.tar.gz --strip-components=1 -C "$pcas"; \
    printf 'STATIC_BUILD = YES\nSHARED_LIBRARIES = NO\nUSR_CFLAGS += -fPIC\nUSR_CXXFLAGS += -fPIC\n' \
        > "$base/configure/CONFIG_SITE.local"; \
    printf 'EPICS_BASE = %s\n' "$base" > "$pcas/configure/RELEASE.local"; \
    cp "$base/configure/CONFIG_SITE.local" "$pcas/configure/CONFIG_SITE.local"; \
    make -C "$base" -j"$(nproc)"; \
    make -C "$pcas" -j"$(nproc)"; \
    export EPICS_BASE="$base" EPICS_HOST_ARCH="$(perl "$base/src/tools/EpicsHostArch.pl")"; \
    cp "$pcas/lib/$EPICS_HOST_ARCH"/*.a "$base/lib/$EPICS_HOST_ARCH/"; \
    cp -R "$pcas/include/." "$base/include/"; \
    printf 'pcaspy==%s --hash=sha256:%s\n' \
        "$PCASPY_VERSION" "$PCASPY_SHA256" > /tmp/native/requirements.txt; \
    pip wheel --no-deps --no-binary pcaspy --require-hashes \
        -r /tmp/native/requirements.txt -w /wheels; \
    ls /wheels/pcaspy-*_aarch64.whl; \
    rm -rf /tmp/native

FROM native-wheels-${TARGETARCH} AS native-wheels

# ── final image ──────────────────────────────────────────────────────────────
# The final image starts again from the Python base and repeats va-base's steps
# word for word, rather than building `FROM va-base`. That is what lets the deps
# layer below be the one every other Python service image builds: BuildKit keys
# a RUN in a stage built on an earlier stage differently from the same RUN in a
# single-stage recipe once that earlier stage sets an ENV (va-base sets the CA
# variables), so a `FROM va-base` final stage would install the framework a
# second time however identical its instructions are. The steps below are the
# bluesky recipe's, and the shared-deps test holds them to it; va-base keeps them
# for the native-wheels builder stages, where nothing is shared.
FROM ${OSPREY_BASE_REGISTRY}/library/python:3.11-slim
WORKDIR /app
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
RUN export http_proxy="${http_proxy:-${HTTP_PROXY:-}}" https_proxy="${https_proxy:-${HTTPS_PROXY:-}}" no_proxy="${no_proxy:-${NO_PROXY:-}}"; \
    find /etc/apt \( -name '*.sources' -o -name '*.list' \) \
    -exec sed -i 's|http://deb.debian.org|https://deb.debian.org|g' {} + \
 && printf 'Acquire::Retries "5";\nAcquire::http::Pipeline-Depth "0";\nAcquire::https::Pipeline-Depth "0";\n' > /etc/apt/apt.conf.d/80-osprey-retries \
 && rm -f /etc/apt/apt.conf.d/docker-clean

# ── deps layer ───────────────────────────────────────────────────────────────
# The same framework install as every other Python service image, instruction
# for instruction (see the bluesky recipe's deps layer for what each line is
# for): the pinned release and its dependencies, the dev fallback for an
# unreleased pin, and the dev-staged osprey-local-requirements.txt delta, with
# the toolchain purged in the same RUN. Nothing VA-specific happens here, so
# BuildKit builds this layer once and a deploy that also builds the bridge or
# bluesky-web image shares it rather than installing the framework a second
# time; `test_python_services_share_one_deps_layer` holds the recipes to that.
# Everything the VA needs beyond the framework is in the layer after this one.
ARG OSPREY_VERSION=""
ARG OSPREY_DEV=""
# OSPREY_PIP_PRE=1 says the pin is a pre-release. osprey-connectors releases
# on its own tags, the framework declares only a floor on it, and an installer
# prefers a stable release for a requirement that names no pre-release (that
# floor), so the deps layer then resolves as a whole with --prerelease=allow,
# which lets it take the newest connectors pre-release.
ARG OSPREY_PIP_PRE=""
# The three pip ARGs are the rest of the site build settings the compose build
# hands this image (the same producer that passes OSPREY_SITE_CA above).
# Declaring them is what makes them arrive: Docker drops a --build-arg no
# recipe declares, with a warning nobody reads, so an internal mirror
# configured once in `config.yml` would reach the project image and quietly
# leave this one resolving from PyPI. PIP_INDEX_URL and PIP_EXTRA_INDEX_URL are
# pip's own environment names, so the declaration alone delivers them and an
# unset one arrives empty, which pip drops before it parses its configuration.
# PIP_NO_PROXY is not a pip name — it is the proxy bypass list, mapped onto
# NO_PROXY/no_proxy in the deps RUN below, and only when it is set: exporting
# it empty would undo the bridge on the line above it.
ARG PIP_NO_PROXY=""
ARG PIP_INDEX_URL=""
ARG PIP_EXTRA_INDEX_URL=""
# A build from an osprey source checkout (`osprey up --dev`) also stages
# osprey-lock-constraints.txt, the checkout's uv.lock exported as constraints,
# and the file's presence (not OSPREY_DEV) puts this layer on the lock-driven
# install: nothing is primed from a released framework, the staged wheels' own
# dependency list is installed with every third-party package held to the
# version the lock holds, and the file is kept at
# /opt/osprey/lock-constraints.txt so that every later install layer is held to
# it as well. The lock's lines are appended to the setuptools constraint file
# below rather than replacing it, so they reach uv's build environments too,
# and a lock that pins setuptools>=84 fails this layer loudly instead of
# quietly dropping the fix that constraint is for. With no file staged (a
# release build) the layer is exactly what it was. The `.osprey-layer-anchor`
# sibling keeps the glob COPY matching either way, and the file is the same
# bytes in every context, so the layer stays shared.
COPY .osprey-layer-anchor osprey-local-requirements.tx[t] osprey-lock-constraints.tx[t] /tmp/deps-ctx/
# setuptools 84.0.0 breaks setuptools_dso's compile-probe error handling and
# fails sdist compiles of the EPICS toolchain (pvxslibs/epicscorelibs/softioc)
# where no binary wheel exists (notably linux/arm64); PIP_CONSTRAINT reaches
# pip's isolated build environments. Drop once a fixed release is out.
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
    && { [ -z "$PIP_EXTRA_INDEX_URL" ] || export UV_INDEX="$PIP_EXTRA_INDEX_URL"; } \
    && apt-get update \
    && apt-get install -y --no-install-recommends build-essential python3-dev \
    && pip install "uv==0.12.21" \
    && if [ -f /tmp/deps-ctx/osprey-lock-constraints.txt ]; then \
           echo "Source build: installing the staged wheels' dependencies at the checkout's uv.lock versions" \
           && mkdir -p /opt/osprey \
           && cp /tmp/deps-ctx/osprey-lock-constraints.txt /opt/osprey/lock-constraints.txt \
           && cat /opt/osprey/lock-constraints.txt >> /tmp/deps-ctx/pip-constraints.txt \
           && uv pip install -r /tmp/deps-ctx/osprey-local-requirements.txt ; \
       else \
           { uv pip install ${OSPREY_PIP_PRE:+--prerelease=allow} "osprey-framework==$OSPREY_VERSION" \
             || if [ "$OSPREY_DEV" = "1" ]; then \
                    echo "WARNING: pin unreleased, priming with latest" \
                    && uv pip install "osprey-framework" ; \
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

# ── virtual-accelerator layer ────────────────────────────────────────────────
# The `virtual-accelerator` extra on top of the shared install: the serving
# stack (`lume-pva-apg[ca,pva]`, with pcaspy under it) and the archiver
# recorder's CA client. It is resolved from the framework's own extra rather
# than pinned here, so this file cannot pin a dependency pyproject.toml
# disagrees with; the framework is already installed, so the resolve adds the
# extra's packages and leaves the rest in place. The same dev fallback as the
# deps layer applies, because an unreleased pin is just as absent from PyPI
# here.
#
# The native wheels arrive here, after the shared layer, and are in place for
# this resolve and for the wheel layer's. On amd64 the directory is empty and
# the variables change nothing.
#
# `--only-binary pcaspy` is a guard, not an optimisation: pcaspy's sdist wants
# a full EPICS base + epics-modules/pcas build that only the native-wheels
# stage is set up to do, and a wheel always exists here — PyPI's on amd64, that
# stage's on arm64 — so if pip ever reaches for that sdist, something upstream
# is wrong and the build should say so immediately instead of failing minutes
# later inside a compile.
#
# The toolchain comes back for this RUN and is purged in it: on arm64 an extra
# dependency without a wheel may still compile. The setuptools constraint is
# the deps layer's, for the same reason.
#
# On the lock-driven path (see the deps layer) no framework is installed yet
# and none may come from the index, so the layer installs the extra's own
# requirements instead, staged beside the wheels as
# osprey-extra-requirements.txt, with the lock appended to the setuptools
# constraint exactly as the deps layer appends it. The arm64 pcaspy wheel is
# built at PCASPY_VERSION, which a test holds equal to the lock's pin, so the
# constrained resolve still takes it from the find-links directory.
COPY --from=native-wheels /wheels/ /opt/osprey-native-wheels/
COPY .osprey-layer-anchor osprey-extra-requirements.tx[t] /tmp/deps-ctx/
ENV PIP_FIND_LINKS=/opt/osprey-native-wheels \
    UV_FIND_LINKS=/opt/osprey-native-wheels
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    export http_proxy="${http_proxy:-${HTTP_PROXY:-}}" https_proxy="${https_proxy:-${HTTPS_PROXY:-}}" no_proxy="${no_proxy:-${NO_PROXY:-}}"; \
    [ -z "$PIP_NO_PROXY" ] || export NO_PROXY="$PIP_NO_PROXY" no_proxy="$PIP_NO_PROXY"; \
    mkdir -p /tmp/deps-ctx \
    && printf 'setuptools<84\n' > /tmp/deps-ctx/pip-constraints.txt \
    && export PIP_CONSTRAINT=/tmp/deps-ctx/pip-constraints.txt \
              UV_CONSTRAINT=/tmp/deps-ctx/pip-constraints.txt \
              UV_BUILD_CONSTRAINT=/tmp/deps-ctx/pip-constraints.txt \
              UV_SYSTEM_PYTHON=1 UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 \
              UV_INDEX_STRATEGY=unsafe-best-match \
    && { [ -z "$PIP_INDEX_URL" ] || export UV_DEFAULT_INDEX="$PIP_INDEX_URL"; } \
    && { [ -z "$PIP_EXTRA_INDEX_URL" ] || export UV_INDEX="$PIP_EXTRA_INDEX_URL"; } \
    && apt-get update \
    && apt-get install -y --no-install-recommends build-essential python3-dev \
    && pip install "uv==0.12.21" \
    && if [ -f /opt/osprey/lock-constraints.txt ]; then \
           cat /opt/osprey/lock-constraints.txt >> /tmp/deps-ctx/pip-constraints.txt \
           && uv pip install -r /tmp/deps-ctx/osprey-extra-requirements.txt --only-binary pcaspy ; \
       else \
           { uv pip install ${OSPREY_PIP_PRE:+--prerelease=allow} "osprey-framework[virtual-accelerator]==$OSPREY_VERSION" --only-binary pcaspy \
             || if [ "$OSPREY_DEV" = "1" ]; then \
                    echo "WARNING: pin unreleased, adding the extra to the latest release" \
                    && uv pip install "osprey-framework[virtual-accelerator]" --only-binary pcaspy ; \
                else \
                    exit 1 ; \
                fi ; } ; \
       fi \
    && pip uninstall -y uv \
    && apt-get purge -y build-essential python3-dev \
    && apt-get autoremove -y \
    && rm -rf /tmp/deps-ctx

# ── wheel layer ──────────────────────────────────────────────────────────────
# Overlay a locally-built wheel when `osprey up --dev` stages one into the build
# context. `.osprey-layer-anchor` is a guaranteed sibling of the COPY, so the
# glob always matches at least one file; `*.wh[l]` optionally pulls in the
# wheel. The first wheel install carries the `[virtual-accelerator]` extra so a
# dev wheel whose extra adds/bumps a dep resolves it (the primer only covers the
# released extra set and `pip check` cannot detect missing extras); every other
# staged wheel -- one per workspace member -- rides in the same call so the
# framework's requirement on each resolves locally rather than from PyPI. The
# wheels are then force-reinstalled --no-deps (no extra) to guarantee their own
# modules win, and `pip check` guards against residual mismatches. No wheel
# staged → no-op, image already complete after the deps layer.
#
# The overlay then asserts that the `pyaml` import package has exactly one
# owning distribution (osprey.deployment.pyaml_owner): every pyAML distribution
# writes the same `pyaml/` files, and neither the resolver nor `pip check` sees
# two of them installed together. The manifest above is staged only beside these
# wheels, so every build that installed it runs the assertion.
#
# On the lock-driven path (see the deps layer) the first install runs under
# /opt/osprey/lock-constraints.txt, so anything the wheels add beyond what the
# earlier layers installed is held to the checkout's uv.lock as well; the
# staged wheels satisfy one another, so nothing resolves an osprey
# distribution from the index.
COPY .osprey-layer-anchor *.wh[l] /tmp/ctx/
RUN --mount=type=cache,target=/root/.cache/pip \
    if ls /tmp/ctx/*.whl >/dev/null 2>&1; then \
        echo "Overlaying locally-built osprey wheel (dev build)" \
        && lock= && { [ ! -f /opt/osprey/lock-constraints.txt ] || lock="-c /opt/osprey/lock-constraints.txt" ; } \
        && whl="$(ls /tmp/ctx/osprey_framework-*.whl)" \
        && pip install $lock "${whl}[virtual-accelerator]" $(ls /tmp/ctx/*.whl | grep -v /osprey_framework-) --only-binary pcaspy \
        && pip install --no-deps --force-reinstall /tmp/ctx/*.whl \
        && python -m osprey.deployment.pyaml_owner \
        && pip check ; \
    fi \
    && rm -rf /tmp/ctx

# The serving stack must be importable before the image counts as built. An
# environment marker that does not match is not an error — pip simply installs
# nothing for it — so a framework release whose `virtual-accelerator` extra
# does not admit this platform would otherwise build cleanly and fail at
# `import pcaspy` inside serving/runner.py, long after the build reported
# success. Fail here instead, where the message can say why.
RUN python -c "import pcaspy, p4p, lume_pva_apg.runner" \
    || { echo "ERROR: the virtual accelerator image has no Channel Access/PVAccess serving stack on $(uname -m): the framework's virtual-accelerator extra did not install lume-pva-apg[ca,pva] here." >&2; exit 1; }

# The image runs the shipped entrypoint module by name. `exec` replaces the
# shell so SIGTERM reaches python directly and entrypoint.py's shutdown
# handlers still run on `docker stop`.
#
# EPICS_CAS_SERVER_PORT is derived here rather than set in the compose
# environment, because it has to track EPICS_CA_SERVER_PORT and cannot be
# allowed to drift from it: the CA *server* library reads the CAS variable and
# does not fall back to the client-side one, so a project that moves
# services.virtual_accelerator.port would otherwise publish the new port while
# the server kept binding 5064 — and a CA search reply carries the server's own
# port, so clients would be handed an address nothing listens on, with no
# useful error.
CMD ["/bin/sh", "-c", "export EPICS_CAS_SERVER_PORT=\"${EPICS_CAS_SERVER_PORT:-${EPICS_CA_SERVER_PORT:-5064}}\"; exec python -u -m osprey.services.virtual_accelerator.entrypoint"]

# Project metadata, kept as the final metadata-only layer so the shared deps
# cache chain above stays identical across projects (a per-project value here
# never invalidates the framework install below it).
ARG OSPREY_PROJECT_NAME=""
LABEL com.osprey.project=$OSPREY_PROJECT_NAME
