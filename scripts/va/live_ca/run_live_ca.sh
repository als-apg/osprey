#!/usr/bin/env bash
# Run the virtual accelerator's live Channel Access suites where they can
# actually run.
#
#   scripts/va/live_ca/run_live_ca.sh [PYTEST_TARGET...]
#
# tests/va/test_record_factory.py and tests/va/test_apply_fault.py stand a real
# pcaspy Channel Access server up in-process and drive it with a real pyepics
# client, and tests/va/test_facility_seam.py boots the whole serving assembly
# on both transports. pcaspy publishes manylinux x86_64 wheels only, so on a
# developer's Mac all of that skips. This builds a linux/amd64 container that
# installs the repo's own `virtual-accelerator` extra from the repo's own
# lockfile, mounts the worktree read-only, and runs the suites there under a
# gate that FAILS if anything skips (scripts/va/live_ca/gate.py) -- because a
# skipped live suite proves nothing.
#
# The extra now carries the whole serving stack -- pcaspy, lume-pva-apg and
# p4p -- so one image runs everything. There is no longer a separate PVA layer
# or a bind-mounted fork checkout: those existed only while lume-pva-apg was
# unpublished and had nothing to resolve by name.
#
# Nothing is published and no host port is claimed. The Channel Access server
# and its client share one process inside the container's own network
# namespace, so this cannot collide with a virtual accelerator already running
# on 5064. Do not add `-p` to the run below without re-reading that sentence.
#
# The image is rebuilt only when pyproject.toml or uv.lock changes; test and
# source edits are picked up from the mount with no rebuild. Set
# OSPREY_LIVE_CA_REBUILD=1 to force one.
#
# Exit status is the gate's: 0 only if the live suites ran and passed with
# nothing skipped.
#
# Environment:
#   OSPREY_VA_RUNTIME        force `docker` or `podman`.
#   OSPREY_LIVE_CA_REBUILD=1 rebuild the image even if it already exists.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PLATFORM="linux/amd64"

if [[ ! -f "${WORKTREE_ROOT}/uv.lock" ]]; then
    echo "FATAL: no uv.lock at ${WORKTREE_ROOT} -- is this the repo root?" >&2
    exit 1
fi

# The tag is a digest of everything that goes into the image: the two
# dependency files and the Containerfile itself. That makes "reuse it if it
# exists" safe rather than merely convenient -- bump the pcaspy floor, add a
# dependency, or edit a build step, and the tag changes, so a stale image
# cannot be silently reused under a name that no longer describes it. A fixed
# tag would also collide with any other image somebody happened to build under
# the same name.
#
# sha256sum on Linux, shasum on macOS -- neither is present on both.
if command -v sha256sum >/dev/null 2>&1; then
    DIGEST_CMD=(sha256sum)
else
    DIGEST_CMD=(shasum -a 256)
fi
BUILD_ID="$(cat "${WORKTREE_ROOT}/pyproject.toml" \
                "${WORKTREE_ROOT}/uv.lock" \
                "${SCRIPT_DIR}/Containerfile" | "${DIGEST_CMD[@]}" | cut -c1-12)"
IMAGE="osprey-va-live-ca:${BUILD_ID}"

# Container runtime. docker is preferred here, the reverse of
# scripts/va/probe_pcaspy/run_probe.sh: this image is cross-architecture on an
# arm64 developer machine, and Docker Desktop ships the amd64 emulation that
# makes that work out of the box.
RUNTIME="${OSPREY_VA_RUNTIME:-}"
if [[ -z "${RUNTIME}" ]]; then
    if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
        RUNTIME="docker"
    elif command -v podman >/dev/null 2>&1; then
        RUNTIME="podman"
        echo "WARNING: docker unavailable; using podman. It must be able to run" >&2
        echo "         ${PLATFORM} images, which needs qemu-user emulation on arm64." >&2
    else
        echo "FATAL: neither a working docker nor podman found on PATH" >&2
        exit 1
    fi
fi

echo "--- runtime: ${RUNTIME}, platform: ${PLATFORM} ---"

# The image needs exactly three files: pyproject.toml, uv.lock and README.md.
# They are staged into a scratch directory used as the build context, the same
# way scripts/va/run_va.sh stages its own -- the repo root would work as a
# context but also holds .git/, .venv/ and the worktrees, and would make every
# build re-tar gigabytes of content the image never reads. src/ and tests/
# arrive over the read-only mount at run time, not through the context.
if [[ "${OSPREY_LIVE_CA_REBUILD:-0}" == "1" ]] || \
   ! "${RUNTIME}" image inspect "${IMAGE}" >/dev/null 2>&1; then
    CONTEXT="$(mktemp -d)"
    trap 'rm -rf "${CONTEXT}"' EXIT
    cp "${WORKTREE_ROOT}/pyproject.toml" \
       "${WORKTREE_ROOT}/uv.lock" \
       "${WORKTREE_ROOT}/README.md" \
       "${CONTEXT}/"

    echo "--- building ${IMAGE} ---"
    "${RUNTIME}" build --platform "${PLATFORM}" \
        -t "${IMAGE}" \
        -f "${SCRIPT_DIR}/Containerfile" \
        "${CONTEXT}"

    # Explicitly, not only via the trap: the run below is an `exec`, which
    # replaces this shell and discards its traps. The trap covers the failure
    # paths; this covers the one that succeeds.
    rm -rf "${CONTEXT}"
    trap - EXIT
else
    echo "--- reusing ${IMAGE} (OSPREY_LIVE_CA_REBUILD=1 to rebuild) ---"
fi

# --pva unconditionally, not as a mode. The extra installs all three server
# roots, so the served-boot branch of tests/va/test_facility_seam.py is
# reachable in this one image -- and `--pva` is what makes the gate REQUIRE it,
# by asserting pcaspy, p4p and lume_pva_apg import before pytest starts. That
# seam test passes on either the served boot or a missing-server-module stop,
# so without the precondition a green here would not say which branch ran.
echo "--- running the live Channel Access + PVAccess gate ---"
exec "${RUNTIME}" run --rm --platform "${PLATFORM}" \
    -v "${WORKTREE_ROOT}:/work:ro" \
    "${IMAGE}" \
    python -u scripts/va/live_ca/gate.py --pva "$@"
