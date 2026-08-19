# qmd search sidecar image.
#
# Runs `@tobilu/qmd` as a long-lived indexer + MCP search server next to the
# OSPREY agent. The running container works offline: the qmd CLI and all three
# GGUF models it can load are in place before it starts, so a cold container
# never reaches out to huggingface.co on a user's first query.
#
# The models get there one of two ways. By default they are baked in at build
# time, which needs a build host that can reach huggingface.co. A build host
# that cannot is told so with OSPREY_QMD_MODELS_MOUNTED (see below): the
# fetches are skipped and the same three files arrive at runtime over the
# read-only bind mount that `services.qmd.models_dir` configures.
#
# Node 22 is the floor, not a preference: @tobilu/qmd declares
# `"engines": {"node": ">=22.0.0"}` and refuses to run on Node 20.
#
# This image is built locally by `osprey up`; override with OSPREY_QMD_IMAGE to
# use a prebuilt/published image.

FROM node:22-bookworm-slim

# Bounded apt retries with backoff so a transient network blip mid-build does
# not fail the whole image (with pipelining off — see any sibling service
# Dockerfile for why retries alone were not enough), then the CA bundle.
# `node:22-bookworm-slim` ships with no ca-certificates at all, so this first
# fetch has to go over plain HTTP — switching the mirror to HTTPS before the CA
# bundle exists fails apt with "certificate issuer is unknown".
RUN export http_proxy="${http_proxy:-${HTTP_PROXY:-}}" https_proxy="${https_proxy:-${HTTPS_PROXY:-}}" no_proxy="${no_proxy:-${NO_PROXY:-}}"; \
    printf 'Acquire::Retries "5";\nAcquire::http::Pipeline-Depth "0";\nAcquire::https::Pipeline-Depth "0";\n' > /etc/apt/apt.conf.d/80-osprey-retries \
 && apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# With the CA bundle in place, move apt to HTTPS for the bulk fetch below
# (plain-HTTP bulk fetches are throttled or broken by middleboxes on some
# networks; deb.debian.org supports HTTPS).
#
# curl  — build-time model fetch; the slim base has no curl either.
# socat — runtime port forwarder. qmd hardcodes `httpServer.listen(port,
#         "localhost")`, with no --host flag and no env override, so the daemon
#         only ever answers on a loopback address — unreachable from outside its
#         own network namespace — and which loopback family "localhost" resolves
#         to depends on the host. The entrypoint runs qmd on the internal port,
#         probes both families for the one that answers, and forwards the
#         routable port to it. This keeps the security posture: qmd itself never
#         listens on a routable address.
RUN export http_proxy="${http_proxy:-${HTTP_PROXY:-}}" https_proxy="${https_proxy:-${HTTPS_PROXY:-}}" no_proxy="${no_proxy:-${NO_PROXY:-}}"; \
    find /etc/apt \( -name '*.sources' -o -name '*.list' \) \
    -exec sed -i 's|http://deb.debian.org|https://deb.debian.org|g' {} + \
 && apt-get update \
 && apt-get install -y --no-install-recommends curl socat \
 && rm -rf /var/lib/apt/lists/*

# ── qmd CLI ──────────────────────────────────────────────────────────────────
# Pinned so an upstream release cannot silently change index format, query
# semantics, or the default model set between image builds. Bump deliberately —
# a default-embedder change forces a full reindex (~40 min at ALS scale).
#
# Several of qmd's transitive dependencies are native addons that publish no
# prebuilt binary for every platform (the tree-sitter grammars in particular),
# so node-gyp compiles them here and needs a C++ toolchain plus python3. The
# toolchain is purged in this same RUN — the compiled .node files survive it —
# so it does not bloat the image. This also means the image is architecture-
# specific: build it for the architecture it will run on.
ARG QMD_VERSION=2.5.3
RUN export http_proxy="${http_proxy:-${HTTP_PROXY:-}}" https_proxy="${https_proxy:-${HTTPS_PROXY:-}}" no_proxy="${no_proxy:-${NO_PROXY:-}}"; \
    apt-get update \
 && apt-get install -y --no-install-recommends build-essential python3 \
 && npm install -g "@tobilu/qmd@${QMD_VERSION}" \
 && npm cache clean --force \
 && qmd --version \
 && apt-get purge -y build-essential python3 \
 && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

# ── runtime paths ────────────────────────────────────────────────────────────
# qmd resolves its model cache to $XDG_CACHE_HOME/qmd/models, falling back to
# $HOME/.cache/qmd/models, and its config dir to $HOME/.config/qmd (unless
# QMD_CONFIG_DIR is set). Both are pinned here so the baked models are found
# regardless of which user or HOME the container is started with; the two
# settings deliberately agree on the same directory.
#
# OSPREY_QMD_MODEL_DIR is ours, not a qmd knob — qmd reads only XDG_CACHE_HOME
# and HOME for this. It exists so the fetch steps below and the entrypoint can
# name the directory once.
ENV HOME=/opt/qmd \
    XDG_CACHE_HOME=/opt/qmd/.cache \
    OSPREY_QMD_MODEL_DIR=/opt/qmd/.cache/qmd/models
RUN mkdir -p "$OSPREY_QMD_MODEL_DIR" /opt/qmd/.config/qmd

# Pin the model URIs explicitly rather than inheriting qmd's compiled-in
# defaults, so a qmd version bump cannot swap the embedder out from under an
# existing index. These are the 2.5.3 defaults, stated.
ENV QMD_EMBED_MODEL=hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf \
    QMD_RERANK_MODEL=hf:ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf \
    QMD_GENERATE_MODEL=hf:tobil/qmd-query-expansion-1.7B-gguf/qmd-query-expansion-1.7B-q4_k_m.gguf

# ── model pins ───────────────────────────────────────────────────────────────
# The SHA256 of each of the three model files, written once here and read by
# everything downstream: the fetch steps below, the identity file, and (via the
# ENV promotion) the entrypoint's check of the mounted files. The
# `com.osprey.qmd.*_sha256` labels are the one deliberate second copy — see the
# comment on the LABEL block for why they cannot reference these.
#
# Declared here rather than at the top of the file so that bumping a pin
# invalidates only the model layers below it, not the apt and npm layers above.
# A bumped value does change the text of the RUN that reads it, which is what
# makes the layer miss (and what CI's pin-keyed layer cache assumes).
ARG OSPREY_QMD_EMBED_SHA256=b5ce9d77a3fc4b3b39ccb5643c36777911cc4eb46a66962eadfa3f5f60490d63
ARG OSPREY_QMD_RERANK_SHA256=22c9979ce4fbcdc5acdc310c6641c32797eff1aa980b8f7a2db8a8ea23429a48
ARG OSPREY_QMD_GENERATE_SHA256=000dfb1c06efa6a049e9f64ba921c3740e2454f62abab6fa10e77bd30bb2bcc0

# Set to any non-empty value — the compose fragment passes "1" — to declare
# that the models will be supplied at runtime by the read-only bind mount
# `services.qmd.models_dir` configures. The three fetches below then do
# nothing, so the image builds on a host with no route to huggingface.co. The
# pins above still travel with the image: skipping the *download* does not skip
# verification, it moves it to container startup, where the entrypoint checks
# the mounted files against them before qmd is allowed to load one.
ARG OSPREY_QMD_MODELS_MOUNTED=""

# Promote the pins to ENV so the entrypoint reads exactly what the build used.
# The ENV shadows the ARG of the same name for every instruction below, at the
# same value — including a value passed with --build-arg.
ENV OSPREY_QMD_EMBED_SHA256=${OSPREY_QMD_EMBED_SHA256} \
    OSPREY_QMD_RERANK_SHA256=${OSPREY_QMD_RERANK_SHA256} \
    OSPREY_QMD_GENERATE_SHA256=${OSPREY_QMD_GENERATE_SHA256}

# ── models ───────────────────────────────────────────────────────────────────
# Fetched from pinned HuggingFace *commit* revisions (not `main`, which moves)
# and SHA256-verified; a mismatch fails the build rather than baking an
# unverified 300 MB - 1.2 GB blob into the image.
#
# The on-disk names are node-llama-cpp's cache naming for an `hf:` URI —
# `hf_<org>_<basename>`. They must match exactly or qmd treats the model as
# absent and re-downloads it on first use — which is also why the runtime mount
# has to carry these same three names.
#
# One RUN per model so a network failure on the 1.2 GB file does not discard the
# two already-verified layers.

# Embeddings (318 MB). Needed to BUILD an index — the fail-closed startup pass
# in the entrypoint depends on this one.
RUN set -eu; \
    if [ -n "${OSPREY_QMD_MODELS_MOUNTED:-}" ]; then \
      echo "models arrive by runtime mount; skipping embed model fetch"; exit 0; \
    fi; \
    dest="$OSPREY_QMD_MODEL_DIR/hf_ggml-org_embeddinggemma-300M-Q8_0.gguf"; \
    want="$OSPREY_QMD_EMBED_SHA256"; \
    curl -fSL --retry 5 --retry-delay 5 --retry-connrefused -o "$dest" \
      "https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF/resolve/0f741b5a6585bd53aeb15cd1372c56f2a0f65e12/embeddinggemma-300M-Q8_0.gguf"; \
    got="$(sha256sum "$dest" | cut -d' ' -f1)"; \
    [ "$got" = "$want" ] || { echo "ERROR: SHA256 mismatch for embedding model: want $want, got $got" >&2; exit 1; }; \
    echo "verified embed model $got"

# Reranking (610 MB). Loaded on first query, not on index build.
RUN set -eu; \
    if [ -n "${OSPREY_QMD_MODELS_MOUNTED:-}" ]; then \
      echo "models arrive by runtime mount; skipping rerank model fetch"; exit 0; \
    fi; \
    dest="$OSPREY_QMD_MODEL_DIR/hf_ggml-org_qwen3-reranker-0.6b-q8_0.gguf"; \
    want="$OSPREY_QMD_RERANK_SHA256"; \
    curl -fSL --retry 5 --retry-delay 5 --retry-connrefused -o "$dest" \
      "https://huggingface.co/ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/resolve/a02f48bb4f057028298c21fa033da2b30d7742d5/qwen3-reranker-0.6b-q8_0.gguf"; \
    got="$(sha256sum "$dest" | cut -d' ' -f1)"; \
    [ "$got" = "$want" ] || { echo "ERROR: SHA256 mismatch for rerank model: want $want, got $got" >&2; exit 1; }; \
    echo "verified rerank model $got"

# Query expansion / HyDE (1.2 GB). Loaded on first `hyde` search.
RUN set -eu; \
    if [ -n "${OSPREY_QMD_MODELS_MOUNTED:-}" ]; then \
      echo "models arrive by runtime mount; skipping generate model fetch"; exit 0; \
    fi; \
    dest="$OSPREY_QMD_MODEL_DIR/hf_tobil_qmd-query-expansion-1.7B-q4_k_m.gguf"; \
    want="$OSPREY_QMD_GENERATE_SHA256"; \
    curl -fSL --retry 5 --retry-delay 5 --retry-connrefused -o "$dest" \
      "https://huggingface.co/tobil/qmd-query-expansion-1.7B-gguf/resolve/7816de0b72572c6c860ca1eddf97ba9e7fb8cc65/qmd-query-expansion-1.7B-q4_k_m.gguf"; \
    got="$(sha256sum "$dest" | cut -d' ' -f1)"; \
    [ "$got" = "$want" ] || { echo "ERROR: SHA256 mismatch for generate model: want $want, got $got" >&2; exit 1; }; \
    echo "verified generate model $got"

# ── embedder identity ────────────────────────────────────────────────────────
# The startup pass compares this against the identity recorded in the index. A
# mismatch means the vectors in the index were produced by a different embedder
# and the index must be rebuilt from scratch. Emitted as a shell-sourceable file
# for the entrypoint and as image labels for `docker inspect`.
#
# Names are OSPREY_-prefixed on purpose: sourcing this file must not clobber the
# QMD_* variables qmd itself reads.
ENV OSPREY_QMD_MODEL_IDENTITY_FILE=/opt/qmd/model-identity.env
RUN printf '%s\n' \
    'OSPREY_QMD_EMBED_MODEL_URI=hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf' \
    'OSPREY_QMD_EMBED_MODEL_FILE=hf_ggml-org_embeddinggemma-300M-Q8_0.gguf' \
    "OSPREY_QMD_EMBED_MODEL_SHA256=${OSPREY_QMD_EMBED_SHA256}" \
    "OSPREY_QMD_EMBED_MODEL_ID=hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf@sha256:${OSPREY_QMD_EMBED_SHA256}" \
    "OSPREY_QMD_VERSION=${QMD_VERSION}" \
    > "$OSPREY_QMD_MODEL_IDENTITY_FILE"

# The pins name the models this image REQUIRES: the ones it baked in on an
# online build, and the ones it will insist on seeing — same names, same
# digests — on the runtime mount when the fetches were skipped.
#
# The three `*_sha256` values are spelled out here instead of referencing the
# ARGs above because CI reads them straight out of this file, before any build,
# to key the layer cache on the pins (`.github/workflows/ci.yml`, "Derive the
# model-pin cache key"); it greps for exactly three 64-hex literals and fails
# the job if it finds any other number. A unit test asserts these three match
# the ARG defaults, so the copy cannot drift unnoticed.
#
# `model_delivery` records which of the two paths this image took, as "baked"
# or "mounted", for `docker inspect` and for the entrypoint (which reads the
# ENV of the same name). `:+` turns any non-empty OSPREY_QMD_MODELS_MOUNTED
# into "mounted" and the `:-` supplies the other half. It carries no digest, so
# it stays clear of the cache-key grep.
ARG OSPREY_QMD_MODEL_DELIVERY="${OSPREY_QMD_MODELS_MOUNTED:+mounted}"
ENV OSPREY_QMD_MODEL_DELIVERY=${OSPREY_QMD_MODEL_DELIVERY:-baked}
LABEL com.osprey.qmd.version="2.5.3" \
      com.osprey.qmd.embed_model="hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf" \
      com.osprey.qmd.embed_sha256="b5ce9d77a3fc4b3b39ccb5643c36777911cc4eb46a66962eadfa3f5f60490d63" \
      com.osprey.qmd.rerank_sha256="22c9979ce4fbcdc5acdc310c6641c32797eff1aa980b8f7a2db8a8ea23429a48" \
      com.osprey.qmd.generate_sha256="000dfb1c06efa6a049e9f64ba921c3740e2454f62abab6fa10e77bd30bb2bcc0" \
      com.osprey.qmd.model_delivery="${OSPREY_QMD_MODEL_DELIVERY}"

# ── entrypoint ───────────────────────────────────────────────────────────────
# `entrypoint.sh` is a sibling of this Dockerfile in the rendered service
# directory (which is the build context). `.dockerignore` is a guaranteed
# sibling, so the COPY always matches at least one file and the image can still
# be built — models and all — before the entrypoint exists.
COPY .dockerignore entrypoint.s[h] /tmp/ctx/
RUN if [ -f /tmp/ctx/entrypoint.sh ]; then \
        install -m 0755 /tmp/ctx/entrypoint.sh /usr/local/bin/qmd-sidecar-entrypoint ; \
    else \
        echo "WARNING: no entrypoint.sh in the build context; this image cannot start as a sidecar" >&2 ; \
    fi \
 && rm -rf /tmp/ctx

WORKDIR /opt/qmd
ENTRYPOINT ["/usr/local/bin/qmd-sidecar-entrypoint"]

# Project metadata, kept as the final metadata-only layer so the model layers
# above stay identical across projects (a per-project value here would otherwise
# invalidate the 2.1 GB of cache below it).
ARG OSPREY_PROJECT_NAME=""
LABEL com.osprey.project=$OSPREY_PROJECT_NAME
