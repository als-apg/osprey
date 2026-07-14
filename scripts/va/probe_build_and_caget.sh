#!/usr/bin/env bash
# Phase-1 hard-gate script for the PyAT Virtual Accelerator probe.
#
# Builds the toy IOC probe image, runs it, and confirms the container's
# EPICS Channel Access PVs are readable from the host over real CA (not a
# mock), then cleans up the container. Exits 0 only if the host-side caget
# of PROBE:BPM:POSITION:X succeeds.
#
# Idempotent: safe to re-run. Removes any prior probe container first.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROBE_DIR="${SCRIPT_DIR}/../../src/osprey/services/virtual_accelerator/probe"
IMAGE="osprey-va-probe:latest"
CONTAINER="osprey-va-probe-gate"
CA_PORT="5064"
BOOT_TIMEOUT_SECS=20

# Container runtime: prefer podman, fall back to docker. Override with
# OSPREY_VA_RUNTIME=docker|podman if needed. If podman is present but its
# machine's storage layer can't actually mount a new container (a known
# failure mode on this host at probe-writing time -- overlay mount fails
# with "input/output error" for every new container, unrelated to this
# image), auto-fall-back to docker rather than fail the whole gate on an
# unrelated infra issue.
RUNTIME="${OSPREY_VA_RUNTIME:-}"
if [[ -z "${RUNTIME}" ]]; then
    if command -v podman >/dev/null 2>&1 && \
       podman run --rm --name "osprey-va-probe-healthcheck-$$" busybox:latest true >/dev/null 2>&1; then
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

# Host-side CA client: prefer real caget from EPICS base if present, else
# fall back to a pyepics one-liner using the worktree venv.
HOST_CAGET="${HOME}/EPICS/epics-base/bin/darwin-aarch64/caget"
WORKTREE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_PY="${WORKTREE_ROOT}/.venv/bin/python"

cleanup() {
    "${RUNTIME}" rm -f "${CONTAINER}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# Idempotency: remove any prior probe container before starting.
"${RUNTIME}" rm -f "${CONTAINER}" >/dev/null 2>&1 || true

echo "--- Building ${IMAGE} (native arch) ---"
"${RUNTIME}" build \
    -t "${IMAGE}" -f "${PROBE_DIR}/Containerfile" "${PROBE_DIR}"

echo "--- Starting ${CONTAINER} ---"
"${RUNTIME}" run -d --name "${CONTAINER}" \
    -p "127.0.0.1:${CA_PORT}:${CA_PORT}/tcp" \
    -p "127.0.0.1:${CA_PORT}:${CA_PORT}/udp" \
    "${IMAGE}" >/dev/null

echo "--- Waiting up to ${BOOT_TIMEOUT_SECS}s for PVs to serve ---"
deadline=$((SECONDS + BOOT_TIMEOUT_SECS))
booted=false
while [[ ${SECONDS} -lt ${deadline} ]]; do
    if "${RUNTIME}" logs "${CONTAINER}" 2>&1 | grep -q "probe IOC serving PVs"; then
        booted=true
        break
    fi
    sleep 1
done
if [[ "${booted}" != true ]]; then
    echo "FATAL: IOC did not report ready within ${BOOT_TIMEOUT_SECS}s" >&2
    "${RUNTIME}" logs "${CONTAINER}" >&2 || true
    exit 1
fi

echo "--- Reading PROBE:BPM:POSITION:X from host via CA name-server (TCP) ---"
export EPICS_CA_NAME_SERVERS="localhost:${CA_PORT}"
export EPICS_CA_AUTO_ADDR_LIST="NO"
export EPICS_CA_ADDR_LIST=""

if [[ -x "${HOST_CAGET}" ]]; then
    echo "Using host caget: ${HOST_CAGET}"
    if ! "${HOST_CAGET}" -w 10 PROBE:BPM:POSITION:X; then
        echo "FATAL: caget failed to read PROBE:BPM:POSITION:X" >&2
        "${RUNTIME}" logs "${CONTAINER}" >&2 || true
        exit 1
    fi
elif [[ -x "${VENV_PY}" ]]; then
    echo "Host caget not found; falling back to pyepics via worktree venv"
    "${VENV_PY}" -c "
import epics
val = epics.caget('PROBE:BPM:POSITION:X', timeout=10)
if val is None:
    raise SystemExit('FATAL: pyepics caget returned None for PROBE:BPM:POSITION:X')
print('PROBE:BPM:POSITION:X', val)
"
else
    echo "FATAL: no host CA client available (no caget, no worktree venv python)" >&2
    exit 1
fi

echo "--- Gate PASSED ---"
