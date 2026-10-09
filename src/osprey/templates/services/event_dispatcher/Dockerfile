# Shared image for the event-dispatcher and dispatch-worker services.
#
# Both services are plain `python -m ...` entrypoints (the compose `command:`
# selects which one), so they run from a single image. The worker executes the
# agent via the Claude Code CLI, which is a Node package — hence Node + the CLI
# are installed here alongside the Python framework.
#
# osprey install strategy (cache-friendly layers):
#   * deps layer  — primes the pinned framework release (+ its dependencies)
#     from PyPI in one RUN, plus any local dependency delta staged as
#     osprey-local-requirements.txt on dev builds. Its build cache is shared
#     across projects and only rebuilds when the staged manifest changes.
#   * node layer  — Node and the pinned Claude Code CLI the worker runs.
#   * wheel layer — when `osprey up --dev` stages a locally-built wheel
#     into the build context, this later layer overlays it (so unreleased code
#     is included). With no wheel staged it is a no-op and the image is complete
#     after the deps layer.
# This image is built locally by `osprey up`; override with
# OSPREY_DISPATCH_IMAGE / OSPREY_WORKER_IMAGE to use a prebuilt/published image.

FROM python:3.11-slim

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
# not exist crashes an httpx client at construction. NODE_EXTRA_CA_CERTS, for
# the Node tooling, is set in the node layer further down rather than here:
# everything down to the deps layer is shared with the other service images.
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
# images building side by side draw on one cache. The apt mounts are private
# rather than locked: a lock would hold this whole RUN, pip install included,
# against every other image building its own deps layer at the same time.
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
# OSPREY_PIP_PRE=1 says the pin is a pre-release. The framework and its
# connectors ship as a pair from one tag, and an installer prefers a stable
# release for a requirement that names none (the framework's own connectors
# requirement), so the deps layer then resolves as a whole with
# --prerelease=allow.
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
COPY .osprey-layer-anchor osprey-local-requirements.tx[t] /tmp/deps-ctx/
# setuptools 84.0.0 breaks setuptools_dso's compile-probe error handling and
# fails sdist compiles of the EPICS toolchain (pvxslibs/epicscorelibs/softioc)
# where no binary wheel exists (notably linux/arm64); PIP_CONSTRAINT reaches
# pip's isolated build environments. Drop once a fixed release is out.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/var/cache/apt,sharing=private \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=private \
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
    && { uv pip install ${OSPREY_PIP_PRE:+--prerelease=allow} "osprey-framework==$OSPREY_VERSION" \
         || if [ "$OSPREY_DEV" = "1" ]; then \
                echo "WARNING: pin unreleased, priming with latest" \
                && uv pip install "osprey-framework" ; \
            else \
                exit 1 ; \
            fi ; } \
    && if [ -f /tmp/deps-ctx/osprey-local-requirements.txt ]; then \
           uv pip install -r /tmp/deps-ctx/osprey-local-requirements.txt ; \
       fi \
    && pip uninstall -y uv \
    && apt-get purge -y build-essential python3-dev \
    && apt-get autoremove -y \
    && rm -rf /tmp/deps-ctx

# ── node layer ───────────────────────────────────────────────────────────────
# Node + the Claude Code CLI (the dispatch-worker shells out to `claude`).
# Below the deps layer rather than above it, so the framework install stays
# the one every service image shares, and a CLI pin bump rebuilds only this
# layer and the wheel layer instead of the whole framework install.
#
# Node comes from the base image's own Debian release rather than a third-party
# apt repo, so the image has a single package provenance and needs no network
# beyond the Debian mirror to build. The CLI's engines floor is node >= 18,
# which every Debian release the slim bases resolve to satisfies.
#
# npm is a runtime dependency, not a build-time convenience: the agent is
# launched as `npx -y @anthropic-ai/claude-code@<version>` (agent_runner/launcher.py),
# so npm must remain in the final image.
#
# NODE_EXTRA_CA_CERTS points Node at the merged bundle the site-CA layer wrote,
# since Node reads neither the system store nor the Python trust variables.
ENV NODE_EXTRA_CA_CERTS=/etc/ssl/certs/ca-certificates.crt
RUN --mount=type=cache,target=/var/cache/apt,sharing=private \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=private \
    export http_proxy="${http_proxy:-${HTTP_PROXY:-}}" https_proxy="${https_proxy:-${HTTPS_PROXY:-}}" no_proxy="${no_proxy:-${NO_PROXY:-}}"; \
    apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates nodejs npm \
    # Pin the Claude Code CLI so an upstream release cannot silently change the
    # headless agent's behavior between image builds (matches the cli_version
    # OSPREY documents in config.yml). Bump deliberately; the npm channel the
    # value is taken from is written beside _DEFAULT_CLAUDE_CLI_VERSION in
    # src/osprey/cli/templates/scaffolding.py.
    && npm install -g @anthropic-ai/claude-code@2.1.267
# The pin is the point, so the CLI's background self-updater is off: a
# successful update would silently replace the pinned version. Same rationale
# as the project image (Dockerfile.j2); bump the pin and rebuild instead.
ENV DISABLE_AUTOUPDATER=1


# ── wheel layer ──────────────────────────────────────────────────────────────
# Overlay a locally-built wheel when `osprey up --dev` stages one into the build
# context. `.osprey-layer-anchor` is a guaranteed sibling of the COPY, so the
# glob always matches at least one file; `*.wh[l]` optionally pulls in the
# wheel. The wheel is installed plain (no extra — the deps layer already primed
# every extra), then force-reinstalled --no-deps to guarantee its own modules
# win, and `pip check` guards against residual mismatches. No wheel staged →
# no-op, image already complete after the deps layer.
COPY .osprey-layer-anchor *.wh[l] /tmp/ctx/
RUN --mount=type=cache,target=/root/.cache/pip \
    if ls /tmp/ctx/*.whl >/dev/null 2>&1; then \
        echo "Overlaying locally-built osprey wheel (dev build)" \
        && pip install /tmp/ctx/*.whl \
        && pip install --no-deps --force-reinstall /tmp/ctx/*.whl \
        && pip check ; \
    fi \
    && rm -rf /tmp/ctx

# Project metadata, kept as the final metadata-only layer so the shared deps
# cache chain above stays identical across projects (a per-project value here
# never invalidates the framework install below it).
ARG OSPREY_PROJECT_NAME=""
LABEL com.osprey.project=$OSPREY_PROJECT_NAME
