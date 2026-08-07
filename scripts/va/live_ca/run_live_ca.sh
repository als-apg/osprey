#!/usr/bin/env bash
# Run the virtual accelerator's live Channel Access suites where they can
# actually run.
#
#   scripts/va/live_ca/run_live_ca.sh [PYTEST_TARGET...]
#
# tests/va/test_record_factory.py and tests/va/test_apply_fault.py stand a real
# pcaspy Channel Access server up in-process and drive it with a real pyepics
# client. pcaspy publishes manylinux x86_64 wheels only, so on a developer's
# Mac those classes skip. This builds a linux/amd64 container that installs the
# repo's own `virtual-accelerator` extra from the repo's own lockfile, mounts
# the worktree read-only, and runs the suites there under a gate that FAILS if
# anything skips (scripts/va/live_ca/gate.py) -- because a skipped live suite
# proves nothing.
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

# The PVA layer gets its own digest, over its own Containerfile as well as the
# base's inputs. Reusing BUILD_ID alone would leave an edit to Containerfile.pva
# invisible to the tag, and the stale layer would be silently reused.
PVA_BUILD_ID="$(cat "${WORKTREE_ROOT}/pyproject.toml" \
                    "${WORKTREE_ROOT}/uv.lock" \
                    "${SCRIPT_DIR}/Containerfile" \
                    "${SCRIPT_DIR}/Containerfile.pva" | "${DIGEST_CMD[@]}" | cut -c1-12)"

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

# PVA layer. `serving/runner.py` imports lume_pva_apg and p4p alongside pcaspy,
# so the *served* boot in tests/va/test_facility_seam.py needs both on top of
# the CA venue. lume_pva_apg is not published yet, so there is nothing to
# install and nothing to guess at: point OSPREY_LUME_PVA_PATH at a checkout of
# the fork and this runs the composed suite set; leave it unset and it runs
# CA-only.
#
# Deliberately opt-in rather than probing likely sibling directories. A guessed
# path that silently misses would drop back to CA-only while looking like it
# had run everything, which is the same class of quiet false green this whole
# script exists to prevent.
if [[ -n "${OSPREY_LUME_PVA_PATH:-}" ]]; then
    FORK_PATH="$(cd "${OSPREY_LUME_PVA_PATH}" 2>/dev/null && pwd)" || {
        echo "FATAL: OSPREY_LUME_PVA_PATH=${OSPREY_LUME_PVA_PATH} is not a directory" >&2
        exit 1
    }
    if [[ ! -d "${FORK_PATH}/lume_pva_apg" ]]; then
        echo "FATAL: no lume_pva_apg/ package under ${FORK_PATH}" >&2
        exit 1
    fi

    PVA_IMAGE="${IMAGE%:*}-pva:${PVA_BUILD_ID}"
    if [[ "${OSPREY_LIVE_CA_REBUILD:-0}" == "1" ]] || \
       ! "${RUNTIME}" image inspect "${PVA_IMAGE}" >/dev/null 2>&1; then
        echo "--- building ${PVA_IMAGE} (PVA layer) ---"
        "${RUNTIME}" build --platform "${PLATFORM}" \
            -t "${PVA_IMAGE}" \
            --build-arg "BASE_IMAGE=${IMAGE}" \
            -f "${SCRIPT_DIR}/Containerfile.pva" \
            "${SCRIPT_DIR}"
    fi

    echo "--- running the composed CA+PVA gate (fork: ${FORK_PATH}) ---"
    exec "${RUNTIME}" run --rm --platform "${PLATFORM}" \
        -v "${WORKTREE_ROOT}:/work:ro" \
        -v "${FORK_PATH}:/fork:ro" \
        "${PVA_IMAGE}" \
        python -u scripts/va/live_ca/gate.py --pva "$@"
fi

echo "--- running the live Channel Access gate (CA only) ---"
echo "    the served-boot branch of tests/va/test_facility_seam.py is NOT"
echo "    exercised here -- set OSPREY_LUME_PVA_PATH=<lume-pva checkout> for that."
exec "${RUNTIME}" run --rm --platform "${PLATFORM}" \
    -v "${WORKTREE_ROOT}:/work:ro" \
    "${IMAGE}" \
    python -u scripts/va/live_ca/gate.py "$@"
