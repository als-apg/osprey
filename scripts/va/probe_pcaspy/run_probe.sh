#!/usr/bin/env bash
# Phase-4 hard gate: does pcaspy's portable Channel Access server work under
# the transport the virtual accelerator container actually uses?
#
# Builds a minimal pcaspy server image, runs it with TCP-only publish, and
# drives it from the host with pyepics under CA name-server mode
# (EPICS_CA_NAME_SERVERS=<host>:<port>, EPICS_CA_AUTO_ADDR_LIST=NO). Three
# behaviours are scored independently, each with its own PASS/FAIL line:
#
#   1. transport        -- host caget/caput reach the served PVs at all
#   2. async-completion -- an `asyn: True` write, committed later via
#                          callbackPV, blocks a caput with put-completion
#   3. monitor-events   -- a camonitor subscription receives events the
#                          server posts on its own, with no client write
#
# A negative control then repoints the client at a dead name-server address
# and requires the PVs to become unreachable, so a green run cannot be an
# artifact of some other discovery path.
#
# Exits 0 only if all three behaviours and the negative control pass.
# Idempotent: safe to re-run.
#
# Environment:
#   PROBE_CA_PORT           host+container CA server port (default 5164).
#                           Must NOT be 5064 -- that port belongs to the
#                           running virtual-accelerator container.
#   PROBE_KEEP_CONTAINER=1  leave the container running after the gate, for
#                           the separate client-coexistence check.
#   OSPREY_VA_RUNTIME       force docker or podman.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
VENV_PY="${WORKTREE_ROOT}/.venv/bin/python"

PCASPY_VERSION="${PCASPY_VERSION:-0.8.1}"
# Native arch by default. Set to linux/amd64 to repeat the gate against the
# x86_64 wheels (the arch CI lanes use), which are built differently from the
# aarch64 ones.
PROBE_PLATFORM="${PROBE_PLATFORM:-}"
IMAGE="osprey-va-pcaspy-probe:${PCASPY_VERSION}${PROBE_PLATFORM:+-${PROBE_PLATFORM##*/}}"
CONTAINER="osprey-va-pcaspy-probe"
CA_PORT="${PROBE_CA_PORT:-5164}"
# A port with nothing listening on it, for the negative control.
DEAD_PORT="${PROBE_DEAD_PORT:-5165}"
BOOT_TIMEOUT_SECS=30

export PROBE_PV_PREFIX="${PROBE_PV_PREFIX:-PCASPY:}"
export PROBE_ASYNC_DELAY_S="${PROBE_ASYNC_DELAY_S:-2.0}"
export PROBE_TELEM_PERIOD_S="${PROBE_TELEM_PERIOD_S:-0.5}"

if [[ "${CA_PORT}" == "5064" ]]; then
    echo "FATAL: port 5064 belongs to the running virtual-accelerator container; pick another" >&2
    exit 1
fi

if [[ ! -x "${VENV_PY}" ]]; then
    echo "FATAL: worktree venv python not found at ${VENV_PY}" >&2
    echo "       run 'uv sync --extra dev --extra docs' in ${WORKTREE_ROOT} first" >&2
    exit 1
fi

# Container runtime: prefer podman, fall back to docker if podman cannot
# actually mount a container (a known storage fault on this host).
RUNTIME="${OSPREY_VA_RUNTIME:-}"
if [[ -z "${RUNTIME}" ]]; then
    if command -v podman >/dev/null 2>&1 && \
       podman run --rm --name "osprey-va-pcaspy-probe-healthcheck-$$" busybox:latest true >/dev/null 2>&1; then
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
echo "Using CA server port: ${CA_PORT} (TCP only)"

cleanup() {
    if [[ "${PROBE_KEEP_CONTAINER:-0}" == "1" ]]; then
        echo "PROBE_KEEP_CONTAINER=1 -- leaving ${CONTAINER} running on port ${CA_PORT}"
        return
    fi
    "${RUNTIME}" rm -f "${CONTAINER}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

"${RUNTIME}" rm -f "${CONTAINER}" >/dev/null 2>&1 || true

PLATFORM_ARGS=()
if [[ -n "${PROBE_PLATFORM}" ]]; then
    PLATFORM_ARGS=(--platform "${PROBE_PLATFORM}")
fi

echo "--- Building ${IMAGE} (pcaspy ${PCASPY_VERSION}, ${PROBE_PLATFORM:-native arch}) ---"
"${RUNTIME}" build ${PLATFORM_ARGS[@]+"${PLATFORM_ARGS[@]}"} --build-arg "PCASPY_VERSION=${PCASPY_VERSION}" \
    -t "${IMAGE}" -f "${SCRIPT_DIR}/Containerfile" "${SCRIPT_DIR}"

echo "--- Starting ${CONTAINER} (TCP-only publish, no UDP) ---"
"${RUNTIME}" run -d --name "${CONTAINER}" ${PLATFORM_ARGS[@]+"${PLATFORM_ARGS[@]}"} \
    -p "127.0.0.1:${CA_PORT}:${CA_PORT}/tcp" \
    -e "EPICS_CA_SERVER_PORT=${CA_PORT}" \
    -e "EPICS_CAS_SERVER_PORT=${CA_PORT}" \
    -e "PROBE_PV_PREFIX=${PROBE_PV_PREFIX}" \
    -e "PROBE_ASYNC_DELAY_S=${PROBE_ASYNC_DELAY_S}" \
    -e "PROBE_TELEM_PERIOD_S=${PROBE_TELEM_PERIOD_S}" \
    "${IMAGE}" >/dev/null

echo "--- Waiting up to ${BOOT_TIMEOUT_SECS}s for the server to report ready ---"
deadline=$((SECONDS + BOOT_TIMEOUT_SECS))
booted=false
while [[ ${SECONDS} -lt ${deadline} ]]; do
    if "${RUNTIME}" logs "${CONTAINER}" 2>&1 | grep -q "pcaspy probe server serving PVs"; then
        booted=true
        break
    fi
    sleep 1
done
if [[ "${booted}" != true ]]; then
    echo "FATAL: pcaspy server did not report ready within ${BOOT_TIMEOUT_SECS}s" >&2
    "${RUNTIME}" logs "${CONTAINER}" >&2 || true
    exit 1
fi
"${RUNTIME}" logs "${CONTAINER}" 2>&1 | sed 's/^/  server| /'

# --- Host CA client environment: name-server mode, no broadcast path --------
export EPICS_CA_NAME_SERVERS="localhost:${CA_PORT}"
export EPICS_CA_AUTO_ADDR_LIST="NO"
export EPICS_CA_ADDR_LIST=""

echo "--- Scoring the three behaviours from the host ---"
set +e
"${VENV_PY}" "${SCRIPT_DIR}/probe_client.py"
behaviors_rc=$?
set -e

echo "--- Negative control: same client, wrong name-server address (localhost:${DEAD_PORT}) ---"
set +e
EPICS_CA_NAME_SERVERS="localhost:${DEAD_PORT}" \
    "${VENV_PY}" "${SCRIPT_DIR}/probe_client.py" --expect-unreachable
negative_rc=$?
set -e

echo "--- Gate summary ---"
if [[ ${behaviors_rc} -eq 0 && ${negative_rc} -eq 0 ]]; then
    echo "GATE PASSED -- pcaspy portable CAS satisfies all three behaviours over CA name-server TCP transport"
    exit 0
fi
[[ ${behaviors_rc} -ne 0 ]] && echo "GATE FAILED -- at least one behaviour did not pass (see BEHAVIOR lines above)"
[[ ${negative_rc} -ne 0 ]] && echo "GATE FAILED -- negative control did not fail as required; the probe is not testing the transport it claims"
exit 1
