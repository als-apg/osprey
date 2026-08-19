# check=skip=FromPlatformFlagConstDisallowed
# (The constant --platform below is deliberate -- see the architecture note.
# Parser directives must lead the file, so this one cannot sit beside it.)

# Image for the Virtual Accelerator (osprey.services.virtual_accelerator): a
# PyAT-backed EPICS server, reached over Channel Access by an unmodified
# EPICSConnector, with the physics model's own variables served on PVAccess
# from the same process.
#
# osprey install strategy (two cache-friendly layers):
#   * deps layer  — primes the pinned framework release with the
#     `virtual-accelerator` extra (+ all dependencies) from PyPI in one RUN,
#     plus any local dependency delta staged as osprey-local-requirements.txt
#     on dev builds. Its build cache is shared across projects and only
#     rebuilds when the staged manifest changes. The extra is what carries the
#     serving stack (`lume-pva-apg[ca,pva]`, PyAT) and the archiver recorder's
#     CA client — deliberately, so this file cannot pin a dependency that
#     pyproject.toml disagrees with. The recorder's other client-side dep,
#     pymongo, is core and arrives with the base install. The recorder
#     runs THIS image with a different command (compose template only, no
#     Dockerfile of its own), which is why its deps ride along here.
#   * wheel layer — when `osprey up --dev` stages a locally-built wheel
#     into the build context, this later layer overlays it (so unreleased code
#     is included). The first wheel install carries the `[virtual-accelerator]`
#     extra so a dev wheel that adds or bumps a dep inside the extra picks it up
#     (the primer only covers the *released* extra set, and `pip check` cannot
#     see missing extras); the second install is a fast, deps-free
#     force-reinstall without the extra. With no wheel staged it is a no-op.
# Deliberately no pinned native-dependency versions here — the extra + core
# deps resolve them, so this Dockerfile can't drift from pyproject.toml's
# declared constraints.
#
# linux/amd64, pinned — deliberately single-arch, and arm64 is not built.
# pcaspy, the Channel Access server behind the serving stack, publishes no
# linux/aarch64 wheel at any interpreter, so an arm64 image would have to
# compile EPICS base and epics-modules/pcas from source before it could even
# start on pcaspy itself. amd64 is what CI runs, and it is where every
# framework dependency resolves to a prebuilt manylinux_x86_64 wheel.
#
# WHAT THIS COSTS YOU, since this file ships into generated projects: on an
# Apple Silicon machine your virtual accelerator runs under emulation, and is
# correspondingly slower. That is a deliberate trade, not an oversight. The
# alternative is a ~10-minute EPICS source build on every cold image build, and
# the real fix — a prebuilt EPICS base image to build aarch64 against — is
# planned separately. Please don't drop the pin to "make it native" without
# reading the guard below; on aarch64 the result is not a slower VA, it is a VA
# with no Channel Access server.
#
# This image is built locally by `osprey up`; override with
# OSPREY_VA_IMAGE to use a prebuilt/published image.

FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app

# Refuse to build anywhere but amd64. The pin above should make this
# unreachable, but a `--platform` override, or someone lifting this recipe into
# their own file, would otherwise produce an image that BUILDS CLEANLY AND
# CANNOT SERVE: osprey's `virtual-accelerator` extra marks pcaspy
# `sys_platform == 'linux' and platform_machine == 'x86_64'`, and an
# environment marker that does not match is not an error — pip simply installs
# nothing for it. The first sign would be a runtime ImportError on
# `import pcaspy` inside serving/runner.py, long after the build reported
# success. Fail here instead, where the message can say why.
RUN arch="$(dpkg --print-architecture)"; [ "$arch" = "amd64" ] \
    || { echo "ERROR: the virtual accelerator image is linux/amd64 only (this build is $arch). No pcaspy wheel exists for linux/aarch64 at any interpreter, so an aarch64 image would carry no Channel Access server." >&2; exit 1; }

# Debian apt mirrors over HTTPS (plain-HTTP bulk fetches are throttled or
# broken by middleboxes on some networks; deb.debian.org supports HTTPS), and
# bounded apt retries with backoff so a transient network blip mid-build does
# not fail the whole image.
RUN export http_proxy="${http_proxy:-${HTTP_PROXY:-}}" https_proxy="${https_proxy:-${HTTPS_PROXY:-}}" no_proxy="${no_proxy:-${NO_PROXY:-}}"; \
    find /etc/apt \( -name '*.sources' -o -name '*.list' \) \
    -exec sed -i 's|http://deb.debian.org|https://deb.debian.org|g' {} + \
 && printf 'Acquire::Retries "5";\n' > /etc/apt/apt.conf.d/80-osprey-retries

# ── deps layer ───────────────────────────────────────────────────────────────
# Prime the image with the pinned framework release + `virtual-accelerator`
# extra and their dependencies. Under `--dev` an unreleased pin may not exist
# on PyPI: OSPREY_DEV=1 relaxes the failure to an unpinned prime (the wheel
# layer below then overlays the real code); without it a pin miss stays fatal.
# Any local dependency delta staged as osprey-local-requirements.txt (dev
# builds only) is installed after the primer, while the toolchain is still
# available — that delta is arbitrary, so it is the one install here that may
# still need to compile something; the `.dockerignore` COPY sibling keeps the
# glob matching when no manifest is staged, so this cache only busts when the
# manifest content changes. The toolchain is purged in this same RUN so it
# never bloats the final image.
#
# `--only-binary pcaspy` is a guard, not an optimisation: pcaspy's sdist wants
# a full EPICS base + epics-modules/pcas build that this image is not set up
# to do, and on the pinned platform a wheel always exists — so if pip ever
# reaches for that sdist, something upstream is wrong and the build should say
# so immediately instead of failing minutes later inside a compile. It trails
# the requirement rather than leading it purely for readability of the pinned
# spec; pip applies it to the whole resolve either way.
# setuptools 84.0.0 breaks setuptools_dso's compile-probe error handling and
# fails those arm64 sdist compiles; the PIP_CONSTRAINT below (which reaches
# pip's isolated build environments, where a plain requirement pin would not)
# holds setuptools <84 until a fixed release is out.
ARG OSPREY_VERSION=""
ARG OSPREY_DEV=""
COPY .dockerignore osprey-local-requirements.tx[t] /tmp/deps-ctx/
RUN export http_proxy="${http_proxy:-${HTTP_PROXY:-}}" https_proxy="${https_proxy:-${HTTPS_PROXY:-}}" no_proxy="${no_proxy:-${NO_PROXY:-}}"; \
    [ -n "$OSPREY_VERSION" ] || { echo "ERROR: OSPREY_VERSION build-arg is required" >&2; exit 1; } \
    && printf 'setuptools<84\n' > /tmp/deps-ctx/pip-constraints.txt \
    && export PIP_CONSTRAINT=/tmp/deps-ctx/pip-constraints.txt \
    && apt-get update \
    && apt-get install -y --no-install-recommends build-essential python3-dev \
    && { pip install --no-cache-dir "osprey-framework[virtual-accelerator]==$OSPREY_VERSION" --only-binary pcaspy \
         || if [ "$OSPREY_DEV" = "1" ]; then \
                echo "WARNING: pin unreleased, priming with latest" \
                && pip install --no-cache-dir "osprey-framework[virtual-accelerator]" --only-binary pcaspy ; \
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
# wheel. The first wheel install carries the `[virtual-accelerator]` extra so a
# dev wheel whose extra adds/bumps a dep resolves it (the primer only covers
# the released extra set and `pip check` cannot detect missing extras); the
# staged osprey-connectors wheel rides in the same call so the framework's
# requirement on it resolves locally rather than from PyPI. The wheels are
# then force-reinstalled --no-deps (no extra) to guarantee their own
# modules win, and `pip check` guards against residual mismatches. No wheel
# staged → no-op, image already complete after the deps layer.
COPY .dockerignore *.wh[l] /tmp/ctx/
RUN if ls /tmp/ctx/*.whl >/dev/null 2>&1; then \
        echo "Overlaying locally-built osprey wheel (dev build)" \
        && whl="$(ls /tmp/ctx/osprey_framework-*.whl)" \
        && pip install --no-cache-dir "${whl}[virtual-accelerator]" /tmp/ctx/osprey_connectors-*.whl --only-binary pcaspy \
        && pip install --no-cache-dir --no-deps --force-reinstall /tmp/ctx/*.whl \
        && pip check ; \
    fi \
    && rm -rf /tmp/ctx

# The entrypoint module is configuration, not a bake-time constant: a
# facility may supply its own entrypoint module (e.g. one serving a
# file-backed manifest with no lattice) without rebuilding the image.
# `exec` replaces the shell so SIGTERM reaches python directly and
# entrypoint.py's shutdown handlers still run on `docker stop`.
# `:-` treats the compose passthrough's empty string the same as unset.
#
# EPICS_CAS_SERVER_PORT is derived here rather than set in the compose
# environment, because it has to track EPICS_CA_SERVER_PORT and cannot be
# allowed to drift from it: the CA *server* library reads the CAS variable and
# does not fall back to the client-side one, so a project that moves
# services.virtual_accelerator.port would otherwise publish the new port while
# the server kept binding 5064 — and a CA search reply carries the server's own
# port, so clients would be handed an address nothing listens on, with no
# useful error.
CMD ["/bin/sh", "-c", "export EPICS_CAS_SERVER_PORT=\"${EPICS_CAS_SERVER_PORT:-${EPICS_CA_SERVER_PORT:-5064}}\"; exec python -u -m ${VA_ENTRYPOINT_MODULE:-osprey.services.virtual_accelerator.entrypoint}"]

# Project metadata, kept as the final metadata-only layer so the shared deps
# cache chain above stays identical across projects (a per-project value here
# never invalidates the framework install below it).
ARG OSPREY_PROJECT_NAME=""
LABEL com.osprey.project=$OSPREY_PROJECT_NAME
