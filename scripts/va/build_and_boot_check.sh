#!/usr/bin/env bash
# Validation gate for the full Virtual Accelerator image (task 3.6).
#
# Stages a minimal build context (pyproject.toml, README.md, src/,
# docker/virtual-accelerator/Containerfile -- never the repo root, which
# also contains .venv/.git/worktrees), builds the image, boots a container
# with the packaged control_assistant preset's data/simulation/ directory
# bind-mounted (a real machine.json, so the engine source has something to
# serve), waits for it to report ready, then confirms a live CA read
# succeeds. Exits 0 only if the image builds, boots to serving PVs within
# BOOT_TIMEOUT_SECS (<60s per the gate), and the caget succeeds.
#
# Idempotent: safe to re-run. Removes any prior gate container first.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VA_DIR="${WORKTREE_ROOT}/docker/virtual-accelerator"
IMAGE="osprey-va-full:latest"
CONTAINER="osprey-va-full-gate"
CA_PORT="5064"
BOOT_TIMEOUT_SECS=60
READY_LOG_MARKER="virtual accelerator IOC serving PVs"
GATE_PV="SR:DIAG:BPM:01:POSITION:X"  # pyat-coupled BPM readback: seeded at
    # boot by the physics bridge's initial closed-orbit solve, so it's
    # readable with no write required -- exercising the full
    # manifest -> records -> physics-bridge -> lattice chain.

DEFAULT_DATA_DIR="${WORKTREE_ROOT}/src/osprey/templates/apps/control_assistant/data/simulation"
DATA_DIR="${1:-${DEFAULT_DATA_DIR}}"
if [[ ! -f "${DATA_DIR}/machine.json" ]]; then
    echo "FATAL: no machine.json under ${DATA_DIR}" >&2
    exit 1
fi

# Container runtime: prefer podman, fall back to docker (same auto-detect as
# scripts/va/probe_build_and_caget.sh -- podman-machine storage has been
# broken host-wide independent of this feature at time of writing).
RUNTIME="${OSPREY_VA_RUNTIME:-}"
if [[ -z "${RUNTIME}" ]]; then
    if command -v podman >/dev/null 2>&1 && \
       podman run --rm --name "osprey-va-full-healthcheck-$$" busybox:latest true >/dev/null 2>&1; then
        RUNTIME="podman"
    elif command -v docker >/dev/null 2>&1; then
        RUNTIME="docker"
        if command -v podman >/dev/null 2>&1; then
            echo "WARNING: podman found but failed a basic container-run health check; falling back to docker" >&2
        fi
    else
        echo "FATAL: neither a working podman nor docker found on PATH" >&2
        exit 1
    fi
fi
echo "Using container runtime: ${RUNTIME}"

HOST_CAGET="${HOME}/EPICS/epics-base/bin/darwin-aarch64/caget"
VENV_PY="${WORKTREE_ROOT}/.venv/bin/python"

STAGING_DIR="$(mktemp -d "${TMPDIR:-/tmp}/osprey-va-full-build.XXXXXX")"
cleanup() {
    "${RUNTIME}" rm -f "${CONTAINER}" >/dev/null 2>&1 || true
    rm -rf "${STAGING_DIR}"
}
trap cleanup EXIT

# Idempotency: remove any prior gate container before starting.
"${RUNTIME}" rm -f "${CONTAINER}" >/dev/null 2>&1 || true

echo "--- Staging minimal build context at ${STAGING_DIR} ---"
cp "${WORKTREE_ROOT}/pyproject.toml" "${WORKTREE_ROOT}/README.md" "${STAGING_DIR}/"
mkdir -p "${STAGING_DIR}/src" "${STAGING_DIR}/docker/virtual-accelerator"
cp -R "${WORKTREE_ROOT}/src/." "${STAGING_DIR}/src/"
cp "${VA_DIR}/Containerfile" "${STAGING_DIR}/docker/virtual-accelerator/Containerfile"
find "${STAGING_DIR}" -name "__pycache__" -type d -prune -exec rm -rf {} +

echo "--- Building ${IMAGE} (linux/amd64) ---"
"${RUNTIME}" build --platform linux/amd64 --build-arg TARGET_PLATFORM=linux/amd64 \
    -t "${IMAGE}" -f "${STAGING_DIR}/docker/virtual-accelerator/Containerfile" "${STAGING_DIR}"

echo "--- Starting ${CONTAINER} (data dir: ${DATA_DIR}) ---"
"${RUNTIME}" run -d --name "${CONTAINER}" \
    -p "127.0.0.1:${CA_PORT}:${CA_PORT}/tcp" \
    -v "${DATA_DIR}:/data/simulation:ro" \
    "${IMAGE}" >/dev/null

echo "--- Waiting up to ${BOOT_TIMEOUT_SECS}s for PVs to serve ---"
boot_wait_start=${SECONDS}
deadline=$((SECONDS + BOOT_TIMEOUT_SECS))
booted=false
while [[ ${SECONDS} -lt ${deadline} ]]; do
    if "${RUNTIME}" logs "${CONTAINER}" 2>&1 | grep -q "${READY_LOG_MARKER}"; then
        booted=true
        break
    fi
    sleep 1
done
boot_elapsed=$((SECONDS - boot_wait_start))
if [[ "${booted}" != true ]]; then
    echo "FATAL: IOC did not report ready within ${BOOT_TIMEOUT_SECS}s" >&2
    "${RUNTIME}" logs "${CONTAINER}" >&2 || true
    exit 1
fi
echo "Booted after ${boot_elapsed} s (container start -> ready log line; excludes the image build above)"

echo "--- Reading ${GATE_PV} from host via CA name-server (TCP) ---"
export EPICS_CA_NAME_SERVERS="localhost:${CA_PORT}"
export EPICS_CA_AUTO_ADDR_LIST="NO"
export EPICS_CA_ADDR_LIST=""

if [[ -x "${HOST_CAGET}" ]]; then
    echo "Using host caget: ${HOST_CAGET}"
    if ! "${HOST_CAGET}" -w 10 "${GATE_PV}"; then
        echo "FATAL: caget failed to read ${GATE_PV}" >&2
        "${RUNTIME}" logs "${CONTAINER}" >&2 || true
        exit 1
    fi
elif [[ -x "${VENV_PY}" ]]; then
    echo "Host caget not found; falling back to pyepics via worktree venv"
    "${VENV_PY}" -c "
import epics
val = epics.caget('${GATE_PV}', timeout=10)
if val is None:
    raise SystemExit('FATAL: pyepics caget returned None for ${GATE_PV}')
print('${GATE_PV}', val)
"
else
    echo "FATAL: no host CA client available (no caget, no worktree venv python)" >&2
    exit 1
fi

echo "--- Gate PASSED ---"
