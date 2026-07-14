# Image for the Virtual Accelerator soft-IOC (osprey.services.virtual_accelerator):
# a PyAT-backed EPICS Channel Access server, reached over CA by an unmodified
# EPICSConnector (see src/osprey/services/virtual_accelerator/probe/README.md
# for the Phase-1 reachability gate this shape was proven against).
#
# osprey install strategy (matches the framework's --dev convention):
#   * dev builds  — `osprey deploy up --dev` drops a locally-built wheel into the
#     build context; if a *.whl is present we install that (so unreleased code
#     is included).
#   * prod builds — no wheel present, so install the pinned release from PyPI.
# Both paths install the `virtual-accelerator` extra on top of the framework's
# core deps (accelerator-toolbox is already a core dependency; the extra adds
# only softioc). Deliberately no pinned accelerator-toolbox/softioc versions
# here — the extra + core deps resolve them, so this Dockerfile can't drift
# from pyproject.toml's declared constraints.
#
# Architecture-agnostic: this image builds for the host's native architecture
# (no `--platform` pin). On arm64 the native deps (accelerator-toolbox, softioc,
# and softioc's epicscorelibs/pvxslibs deps) publish no manylinux_aarch64 wheels,
# so pip compiles them from sdist using the build toolchain installed below —
# purged in the same layer — for native execution with no x86 emulation. On
# x86_64 pip installs the prebuilt wheels as before. See the probe README's
# platform note for the historical amd64 rationale.
#
# This image is built locally by `osprey deploy up`; override with
# OSPREY_VA_IMAGE to use a prebuilt/published image.

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
        && pip install --no-cache-dir "${whl}[virtual-accelerator]" ; \
    elif [ -n "$OSPREY_VERSION" ]; then \
        echo "Installing osprey-framework[virtual-accelerator]==$OSPREY_VERSION from PyPI" \
        && pip install --no-cache-dir "osprey-framework[virtual-accelerator]==$OSPREY_VERSION" ; \
    else \
        echo "Installing osprey-framework[virtual-accelerator] (latest) from PyPI" \
        && pip install --no-cache-dir "osprey-framework[virtual-accelerator]" ; \
    fi \
    && apt-get purge -y build-essential python3-dev \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* /tmp/ctx

CMD ["python", "-m", "osprey.services.virtual_accelerator.entrypoint"]
