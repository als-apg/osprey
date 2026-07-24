# Image for the Virtual Accelerator soft-IOC (osprey.services.virtual_accelerator):
# a PyAT-backed EPICS Channel Access server, reached over CA by an unmodified
# EPICSConnector (see src/osprey/services/virtual_accelerator/probe/README.md
# for the Phase-1 reachability gate this shape was proven against).
#
# osprey install strategy (two cache-friendly layers):
#   * deps layer  — primes the pinned framework release with the
#     `virtual-accelerator` extra (+ all dependencies) from PyPI in one RUN,
#     plus any local dependency delta staged as osprey-local-requirements.txt
#     on dev builds. Its build cache is shared across projects and only
#     rebuilds when the staged manifest changes.
#   * wheel layer — when `osprey deploy up --dev` stages a locally-built wheel
#     into the build context, this later layer overlays it (so unreleased code
#     is included). The first wheel install carries the `[virtual-accelerator]`
#     extra so a dev wheel that adds or bumps a dep inside the extra picks it up
#     (the primer only covers the *released* extra set, and `pip check` cannot
#     see missing extras); the second install is a fast, deps-free
#     force-reinstall without the extra. With no wheel staged it is a no-op.
# Deliberately no pinned accelerator-toolbox/softioc versions here — the extra +
# core deps resolve them, so this Dockerfile can't drift from pyproject.toml's
# declared constraints.
#
# Architecture-agnostic: this image builds for the host's native architecture
# (no `--platform` pin). On arm64 the native deps (accelerator-toolbox, softioc,
# and softioc's epicscorelibs/pvxslibs deps) publish no manylinux_aarch64 wheels,
# so pip compiles them from sdist using the build toolchain installed in the
# deps layer below — and purged in that same RUN — for native execution with no
# x86 emulation. This arm64 sdist compile now happens once, in the shared deps
# layer, so per-project dev rebuilds (the wheel layer) never re-pay it. On
# x86_64 pip installs the prebuilt wheels as before. See the probe README's
# platform note for the historical amd64 rationale.
#
# This image is built locally by `osprey deploy up`; override with
# OSPREY_VA_IMAGE to use a prebuilt/published image.

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
# Prime the image with the pinned framework release + `virtual-accelerator`
# extra and their dependencies. A C toolchain is needed at install time to
# compile the arm64 native deps (accelerator-toolbox, softioc,
# epicscorelibs/pvxslibs); it is purged in this same RUN so it does not bloat
# the final image. Under `--dev` an unreleased pin may not exist on PyPI:
# OSPREY_DEV=1 relaxes the failure to an unpinned prime (the wheel layer below
# then overlays the real code); without it a pin miss stays fatal. Any local
# dependency delta staged as osprey-local-requirements.txt (dev builds only) is
# installed after the primer, while the toolchain is still available; the
# `.dockerignore` COPY sibling keeps the glob matching when no manifest is
# staged, so this cache only busts when the manifest content changes.
ARG OSPREY_VERSION=""
ARG OSPREY_DEV=""
COPY .dockerignore osprey-local-requirements.tx[t] /tmp/deps-ctx/
RUN [ -n "$OSPREY_VERSION" ] || { echo "ERROR: OSPREY_VERSION build-arg is required" >&2; exit 1; } \
    && apt-get update \
    && apt-get install -y --no-install-recommends build-essential python3-dev \
    && { pip install --no-cache-dir "osprey-framework[virtual-accelerator]==$OSPREY_VERSION" \
         || if [ "$OSPREY_DEV" = "1" ]; then \
                echo "WARNING: pin unreleased, priming with latest" \
                && pip install --no-cache-dir "osprey-framework[virtual-accelerator]" ; \
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
# Overlay a locally-built wheel when `osprey deploy up --dev` stages one into
# the build context. `.dockerignore` is a guaranteed sibling of the COPY, so
# the glob always matches at least one file; `*.wh[l]` optionally pulls in the
# wheel. The first wheel install carries the `[virtual-accelerator]` extra so a
# dev wheel whose extra adds/bumps a dep resolves it (the primer only covers
# the released extra set and `pip check` cannot detect missing extras); the
# wheel is then force-reinstalled --no-deps (no extra) to guarantee its own
# modules win, and `pip check` guards against residual mismatches. No wheel
# staged → no-op, image already complete after the deps layer.
COPY .dockerignore *.wh[l] /tmp/ctx/
RUN if ls /tmp/ctx/*.whl >/dev/null 2>&1; then \
        echo "Overlaying locally-built osprey wheel (dev build)" \
        && whl="$(ls /tmp/ctx/*.whl)" \
        && pip install --no-cache-dir "${whl}[virtual-accelerator]" \
        && pip install --no-cache-dir --no-deps --force-reinstall /tmp/ctx/*.whl \
        && pip check ; \
    fi \
    && rm -rf /tmp/ctx

# The entrypoint module is configuration, not a bake-time constant: a
# facility may supply its own entrypoint module (e.g. one serving a
# file-backed manifest with no lattice) without rebuilding the image.
# `exec` replaces the shell so SIGTERM reaches python directly and the
# IOC's wait_for_quit() shutdown path still runs on `docker stop`.
# `:-` treats the compose passthrough's empty string the same as unset.
CMD ["/bin/sh", "-c", "exec python -m ${VA_ENTRYPOINT_MODULE:-osprey.services.virtual_accelerator.entrypoint}"]

# Project metadata, kept as the final metadata-only layer so the shared deps
# cache chain above stays identical across projects (a per-project value here
# never invalidates the framework install below it).
ARG OSPREY_PROJECT_NAME=""
LABEL com.osprey.project=$OSPREY_PROJECT_NAME
