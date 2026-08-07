#!/usr/bin/env bash
# Launch the Virtual Accelerator container for tutorial/interactive use.
#
# Usage:
#   scripts/va/run_va.sh [DATA_DIR]
#
# DATA_DIR is your project's data/simulation/ DIRECTORY (never a single
# file -- see docker/virtual-accelerator/README.md for why). Defaults to the
# packaged control_assistant preset's own copy, so this runs with zero
# arguments out of the box; point it at a real project's data/simulation to
# use that project's channel_limits.json-scoped scenarios instead.
#
# Builds the image if it doesn't already exist (set OSPREY_VA_REBUILD=1 to
# force a rebuild, e.g. after editing src/osprey/services/virtual_accelerator/**,
# docker/virtual-accelerator/Containerfile, or the lume-pva-apg checkout the
# serving stack is built from). Runs in the foreground -- Ctrl-C (or
# `docker stop`) shuts the IOC down cleanly.
#
# The image is linux/amd64 (see the Containerfile for why), so on an Apple
# Silicon host it runs emulated.
#
# After it reports ready, point a project at it with:
#   control_system:
#     type: virtual_accelerator
#     facility: simulation   # the "Local Simulation" gateway preset, localhost:5064
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VA_DIR="${WORKTREE_ROOT}/docker/virtual-accelerator"
IMAGE="osprey-va-full:latest"
CONTAINER="osprey-va-tutorial"
CA_PORT="5064"

DEFAULT_DATA_DIR="${WORKTREE_ROOT}/src/osprey/templates/apps/control_assistant/data/simulation"
DATA_DIR_ARG="${1:-${DEFAULT_DATA_DIR}}"
if [[ ! -d "${DATA_DIR_ARG}" ]]; then
    echo "FATAL: ${DATA_DIR_ARG} is not a directory." >&2
    echo "DATA_DIR must be a project's data/simulation/ directory." >&2
    exit 1
fi
DATA_DIR="$(cd "${DATA_DIR_ARG}" && pwd)"
if [[ ! -f "${DATA_DIR}/machine.json" ]]; then
    echo "FATAL: no machine.json under ${DATA_DIR}" >&2
    echo "DATA_DIR must be a project's data/simulation/ directory." >&2
    exit 1
fi

RUNTIME="${OSPREY_VA_RUNTIME:-}"
if [[ -z "${RUNTIME}" ]]; then
    if command -v podman >/dev/null 2>&1 && \
       podman run --rm --name "osprey-va-tutorial-healthcheck-$$" busybox:latest true >/dev/null 2>&1; then
        RUNTIME="podman"
    elif command -v docker >/dev/null 2>&1; then
        RUNTIME="docker"
    else
        echo "FATAL: neither a working podman nor docker found on PATH" >&2
        exit 1
    fi
fi
echo "Using container runtime: ${RUNTIME}"

# Single cleanup/trap for both the staging dir (only created on a rebuild --
# see below) and the container, so neither leaks regardless of which path
# runs or where the script exits.
STAGING_DIR=""
cleanup() {
    "${RUNTIME}" rm -f "${CONTAINER}" >/dev/null 2>&1 || true
    [[ -n "${STAGING_DIR}" ]] && rm -rf "${STAGING_DIR}"
}
trap cleanup EXIT INT TERM

if [[ "${OSPREY_VA_REBUILD:-0}" == "1" ]] || ! "${RUNTIME}" image inspect "${IMAGE}" >/dev/null 2>&1; then
    STAGING_DIR="$(mktemp -d "${TMPDIR:-/tmp}/osprey-va-full-build.XXXXXX")"

    echo "--- Staging minimal build context at ${STAGING_DIR} ---"
    cp "${WORKTREE_ROOT}/pyproject.toml" "${WORKTREE_ROOT}/README.md" "${STAGING_DIR}/"
    mkdir -p "${STAGING_DIR}/src" "${STAGING_DIR}/docker/virtual-accelerator"
    cp -R "${WORKTREE_ROOT}/src/." "${STAGING_DIR}/src/"
    cp "${VA_DIR}/Containerfile" "${STAGING_DIR}/docker/virtual-accelerator/Containerfile"
    find "${STAGING_DIR}" -name "__pycache__" -type d -prune -exec rm -rf {} +

    # The serving stack is a sibling checkout, not a PyPI package: build its
    # wheel into the context so the image installs the fork you actually have.
    # `git rev-parse --git-common-dir` resolves to the MAIN checkout's .git even
    # from a worktree, so this finds the sibling from either.
    if [[ -z "${LUME_PVA_SRC:-}" ]]; then
        MAIN_CHECKOUT="$(cd "$(git -C "${WORKTREE_ROOT}" rev-parse --path-format=absolute --git-common-dir)/.." && pwd)"
        LUME_PVA_SRC="$(dirname "${MAIN_CHECKOUT}")/lume-pva"
    fi
    if [[ ! -f "${LUME_PVA_SRC}/pyproject.toml" ]]; then
        echo "FATAL: no lume-pva checkout at ${LUME_PVA_SRC}." >&2
        echo "Set LUME_PVA_SRC to the serving stack's source tree." >&2
        exit 1
    fi
    command -v uv >/dev/null 2>&1 || { echo "FATAL: uv is required to build the serving-stack wheel" >&2; exit 1; }
    echo "--- Building the serving-stack wheel from ${LUME_PVA_SRC} ---"
    # Writes only build/ and the generated _version.py inside that checkout,
    # both of which it ignores -- the same footprint as building it by hand.
    uv build --wheel --out-dir "${STAGING_DIR}" "${LUME_PVA_SRC}"

    # osprey's version comes from git (hatch-vcs) and the staged context has no
    # .git, so the host resolves it and passes it in; see the Containerfile.
    OSPREY_VERSION="$("${WORKTREE_ROOT}/.venv/bin/python" -c 'import osprey; print(osprey.__version__)')"

    echo "--- Building ${IMAGE} (linux/amd64) ---"
    "${RUNTIME}" build \
        --build-arg "OSPREY_VERSION=${OSPREY_VERSION}" \
        -t "${IMAGE}" -f "${STAGING_DIR}/docker/virtual-accelerator/Containerfile" "${STAGING_DIR}"
else
    echo "Reusing existing image ${IMAGE} (set OSPREY_VA_REBUILD=1 to force a rebuild)"
fi

"${RUNTIME}" rm -f "${CONTAINER}" >/dev/null 2>&1 || true

echo "--- Serving CA on localhost:${CA_PORT} (name-server mode); data dir: ${DATA_DIR} ---"
echo "--- On the host, connect with: ---"
echo "    export EPICS_CA_NAME_SERVERS=localhost:${CA_PORT}"
echo "    export EPICS_CA_AUTO_ADDR_LIST=NO"
echo "--- Ctrl-C stops the container. ---"

# The server port is passed explicitly, from the same CA_PORT the publish maps:
# a CA search reply carries the server's own port number, so a container bound
# to one port and published on another hands clients an unreachable address
# with no useful error. (The image derives EPICS_CAS_SERVER_PORT from this.)
"${RUNTIME}" run --rm --name "${CONTAINER}" \
    --platform linux/amd64 \
    -e "EPICS_CA_SERVER_PORT=${CA_PORT}" \
    -p "127.0.0.1:${CA_PORT}:${CA_PORT}/tcp" \
    -v "${DATA_DIR}:/data/simulation:ro" \
    "${IMAGE}"
