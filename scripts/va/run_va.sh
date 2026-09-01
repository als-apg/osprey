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
# The IOC has no default channel namespace and refuses to boot without one (see
# src/osprey/services/virtual_accelerator/entrypoint.py), so this script always
# names one: a DATA_DIR carrying its own channel_manifest.json -- what `osprey
# build` generates for a project -- is served as that project's namespace, and
# otherwise the run is the tutorial quick-start and gets the framework's
# packaged demo manifest paired with the tutorial lattice.
#
# Builds the image if it doesn't already exist (set OSPREY_VA_REBUILD=1 to
# force a rebuild, e.g. after editing src/osprey/services/virtual_accelerator/**,
# docker/virtual-accelerator/Containerfile, or the `virtual-accelerator` extra
# in pyproject.toml). Runs in the foreground -- Ctrl-C (or
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
# Whether the caller pointed this somewhere. Tracked as a flag rather than
# recovered later by comparing paths against the default, which a symlinked
# checkout would get wrong. It gates one message: pointing at a directory that
# turns out not to be a built project deserves a word, arriving at the packaged
# default does not.
DATA_DIR_GIVEN="no"
if [[ $# -gt 0 ]]; then
    DATA_DIR_GIVEN="yes"
fi
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

# `osprey sim apply` writes the active-scenario state under the deployment's
# DURABLE zone (`var/agent_data/simulation`) — never beside machine.json, since
# `data/` is build-owned and re-rendered. Mount it so the IOC sees a scenario
# switch; without it the IOC falls back to the state next to machine.json and
# never does.
#
# Found by walking up to the repo root (the directory holding profile.yml)
# rather than by counting `..` from DATA_DIR: the same data/simulation exists
# both in the source zone (<repo>/data) and in the render (<repo>/build/data),
# so a fixed number of parents is right for one and wrong for the other.
STATE_DIR=""
_search="$(cd "${DATA_DIR}" >/dev/null 2>&1 && pwd)"
while [[ -n "${_search}" && "${_search}" != "/" ]]; do
    if [[ -f "${_search}/profile.yml" ]]; then
        STATE_DIR="${_search}/var/agent_data/simulation"
        break
    fi
    _search="$(dirname "${_search}")"
done

STATE_MOUNT=()
if [[ -n "${STATE_DIR}" && -d "${STATE_DIR}" ]]; then
    STATE_MOUNT=(-v "${STATE_DIR}:/state/simulation:ro" -e "VA_STATE_DIR=/state/simulation")
else
    # Say so. The old gate skipped the mount in silence, so a wrong path and a
    # genuinely-absent state directory looked identical — and the symptom
    # (scenario switches ignored by the IOC) shows up far from here.
    echo "NOTE: no scenario-state directory mounted (${STATE_DIR:-no profile.yml above ${DATA_DIR}})." >&2
    echo "      The IOC will read the state beside machine.json and will not see" >&2
    echo "      \`osprey sim apply\` scenario switches. Run a scenario apply first," >&2
    echo "      or pass a DATA_DIR inside a deployment repo." >&2
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

# Single cleanup/trap for the build staging dir (only created on a rebuild --
# see below), the assembled demo data dir (only created for the quick-start)
# and the container, so none of them leaks regardless of which path runs or
# where the script exits.
STAGING_DIR=""
DEMO_DATA_DIR=""
cleanup() {
    "${RUNTIME}" rm -f "${CONTAINER}" >/dev/null 2>&1 || true
    [[ -n "${STAGING_DIR}" ]] && rm -rf "${STAGING_DIR}"
    [[ -n "${DEMO_DATA_DIR}" ]] && rm -rf "${DEMO_DATA_DIR}"
    return 0
}
trap cleanup EXIT INT TERM

if [[ "${OSPREY_VA_REBUILD:-0}" == "1" ]] || ! "${RUNTIME}" image inspect "${IMAGE}" >/dev/null 2>&1; then
    STAGING_DIR="$(mktemp -d "${TMPDIR:-/tmp}/osprey-va-full-build.XXXXXX")"

    echo "--- Staging minimal build context at ${STAGING_DIR} ---"
    cp "${WORKTREE_ROOT}/pyproject.toml" "${WORKTREE_ROOT}/README.md" "${STAGING_DIR}/"
    mkdir -p "${STAGING_DIR}/src" "${STAGING_DIR}/docker/virtual-accelerator"
    cp -R "${WORKTREE_ROOT}/src/." "${STAGING_DIR}/src/"
    cp -R "${WORKTREE_ROOT}/packages" "${STAGING_DIR}/packages"
    cp "${VA_DIR}/Containerfile" "${STAGING_DIR}/docker/virtual-accelerator/Containerfile"
    find "${STAGING_DIR}" -name "__pycache__" -type d -prune -exec rm -rf {} +

    # The serving stack needs no staging step: `lume-pva-apg[ca,pva]` is an
    # exact pin in osprey's `virtual-accelerator` extra, so the image resolves
    # it by name from PyPI along with everything else the extra carries. The
    # context is therefore exactly the five things copied above (packages/
    # carries the osprey-connectors workspace member the framework installs
    # from source).

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

# The channel namespace to serve, named explicitly. The IOC has no default one:
# the only namespace it could pick unasked is the framework's bundled demo
# namespace, and a container serving those addresses under a facility's name is
# indistinguishable, on the wire, from one serving the facility. So it refuses
# instead, and this script names one of exactly two things.
#
# A DATA_DIR carrying its own channel_manifest.json is a built project, and it
# is already in the layout the IOC reads: manifest and channel_limits.json
# beside machine.json, which is exactly what `osprey build` stages. Mount it as
# it stands and name the manifest relative to the mount, the same way the
# project's own .env names it. VA_LATTICE then defaults to `none`, because the
# only model this image could build is the framework's tutorial ring and that
# ring is not this facility's physics; export VA_LATTICE to override.
#
# Otherwise this is the tutorial quick-start. The packaged preset tree is NOT
# in that layout -- it carries no manifest at all (the framework's is package
# data) and keeps channel_limits.json one level up, at the data root -- so the
# same layout is assembled in a temp directory and that is what gets mounted.
# Assembled rather than overlaid with extra bind mounts because a bind mount
# INTO a read-only mount cannot create its own mountpoint (the runtime refuses
# with EROFS), and mounting the tree read-write to make room would leave the
# container able to write into the checkout.
VA_LATTICE_VALUE="${VA_LATTICE:-}"
MOUNT_DIR="${DATA_DIR}"
if [[ -f "${DATA_DIR}/channel_manifest.json" ]]; then
    : "${VA_LATTICE_VALUE:=none}"
    echo "--- Serving this project's own manifest: ${DATA_DIR}/channel_manifest.json ---"
else
    : "${VA_LATTICE_VALUE:=builtin}"
    # Say what is about to happen, because this branch REINTERPRETS the argument.
    # A DATA_DIR with no manifest beside machine.json is not a built project, so
    # it is being treated as a plain simulation tree for the demo -- the channels
    # served will be the framework's, not this directory's. Silence here would
    # leave someone who pointed at a half-built project believing they were
    # watching their own machine.
    if [[ "${DATA_DIR_GIVEN}" == "yes" ]]; then
        echo "NOTE: ${DATA_DIR} has no channel_manifest.json, so it is not a built" >&2
        echo "      project. Treating it as a demo data tree: the served channels" >&2
        echo "      will be the framework's packaged demo namespace, with only" >&2
        echo "      machine.json and scenarios/ taken from this directory. Run" >&2
        echo "      \`osprey build\` to serve this project's own channels instead." >&2
    fi

    VENV_PY="${WORKTREE_ROOT}/.venv/bin/python"
    if [[ ! -x "${VENV_PY}" ]]; then
        echo "FATAL: no worktree venv python at ${VENV_PY}." >&2
        echo "       It is what locates the packaged demo manifest (the manifest is" >&2
        echo "       package data, so its path comes from the installed osprey rather" >&2
        echo "       than from a path spelled here). Create the venv with \`uv sync\`," >&2
        echo "       or pass a built project's data/simulation, which needs no lookup." >&2
        exit 1
    fi
    # The manifest this checkout carries, located through the installed package
    # rather than by a relative path, so it is the same file the image was built
    # from however the tree is laid out.
    PACKAGED_MANIFEST="$("${VENV_PY}" -c \
        'from osprey.services.virtual_accelerator.manifest.paths import MANIFEST_OUTPUT
print(MANIFEST_OUTPUT)')"
    if [[ ! -f "${PACKAGED_MANIFEST}" ]]; then
        echo "FATAL: the packaged demo manifest is missing at ${PACKAGED_MANIFEST}." >&2
        echo "       Regenerate it with:" >&2
        echo "         uv run python -m osprey.services.virtual_accelerator.manifest.build" >&2
        exit 1
    fi
    # Drive limits for the demo come from the preset's data root, one level above
    # a simulation tree. Named as the reinterpretation it is: this is the demo
    # branch asking THIS directory to look like the preset, not a missing piece
    # of the framework's own demo assets.
    PRESET_LIMITS="$(cd "${DATA_DIR}/.." && pwd)/channel_limits.json"
    if [[ ! -f "${PRESET_LIMITS}" ]]; then
        echo "FATAL: serving ${DATA_DIR} as a demo data tree needs a" >&2
        echo "       channel_limits.json at its data root (${PRESET_LIMITS})," >&2
        echo "       the way the packaged preset lays one out. Without it the IOC" >&2
        echo "       would serve every setpoint unbounded. Pass a built project's" >&2
        echo "       data/simulation, or the packaged preset's, instead." >&2
        exit 1
    fi

    DEMO_DATA_DIR="$(mktemp -d "${TMPDIR:-/tmp}/osprey-va-demo-data.XXXXXX")"
    echo "--- Assembling the demo data dir at ${DEMO_DATA_DIR} ---"
    cp -R "${DATA_DIR}/." "${DEMO_DATA_DIR}/"
    cp "${PACKAGED_MANIFEST}" "${DEMO_DATA_DIR}/channel_manifest.json"
    cp "${PRESET_LIMITS}" "${DEMO_DATA_DIR}/channel_limits.json"
    MOUNT_DIR="${DEMO_DATA_DIR}"
fi
CHANNELS_FILE_VALUE="channel_manifest.json"
echo "--- Channel manifest: ${MOUNT_DIR}/${CHANNELS_FILE_VALUE} (VA_LATTICE=${VA_LATTICE_VALUE}) ---"

echo "--- Serving CA on localhost:${CA_PORT} (name-server mode); data dir: ${MOUNT_DIR} ---"
echo "--- On the host, connect with: ---"
echo "    export EPICS_CA_NAME_SERVERS=localhost:${CA_PORT}"
echo "    export EPICS_CA_AUTO_ADDR_LIST=NO"
echo "--- Ctrl-C stops the container. ---"

# The server port is passed explicitly, from the same CA_PORT the publish maps:
# a CA search reply carries the server's own port number, so a container bound
# to one port and published on another hands clients an unreachable address
# with no useful error. (The image derives EPICS_CAS_SERVER_PORT from this.)
# Both source variables are passed, never left to the IOC: it has no default
# namespace at all, and its lattice default is `none`. The packaged manifest
# and the tutorial ring are the two halves of one machine, so the quick-start
# asks for both together.
"${RUNTIME}" run --rm --name "${CONTAINER}" \
    --platform linux/amd64 \
    -e "EPICS_CA_SERVER_PORT=${CA_PORT}" \
    -e "VA_CHANNELS_FILE=${CHANNELS_FILE_VALUE}" \
    -e "VA_LATTICE=${VA_LATTICE_VALUE}" \
    -p "127.0.0.1:${CA_PORT}:${CA_PORT}/tcp" \
    -v "${MOUNT_DIR}:/data/simulation:ro" \
    ${STATE_MOUNT[@]+"${STATE_MOUNT[@]}"} \
    "${IMAGE}"
