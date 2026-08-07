#!/usr/bin/env bash
# Validation gate for the full Virtual Accelerator image.
#
# Stages a minimal build context (pyproject.toml, README.md, src/,
# docker/virtual-accelerator/Containerfile -- never the repo root, which
# also contains .venv/.git/worktrees), builds the image, boots a container
# with the packaged control_assistant preset's data/simulation/ directory
# bind-mounted (a real machine.json, so the engine source has something to
# serve), waits for it to report ready, and then drives both served
# transports against it.
#
# What the gate asserts, and why each assertion is not vacuous:
#
#   1. The readiness line matches its whole contract -- marker AND a
#      positive channel count -- rather than just the marker, so a boot that
#      served nothing could not satisfy it.
#   2. A Channel Access read of a quiescent BPM returns zero. This is the
#      BASELINE, deliberately not an assertion of correctness: the tutorial
#      lattice's closed orbit with no corrector excited IS exactly zero, so
#      a read here cannot distinguish a working physics chain from an
#      unseeded PV. That is why (3) exists.
#   3. Exciting a corrector moves the BPMs off zero. THIS is the assertion
#      with teeth: only a live manifest -> records -> physics-bridge ->
#      lattice chain can move a reading, and an unseeded PV stays at zero
#      through the write. The setpoint is written with put-completion, so
#      the read that follows cannot race the solve.
#   4. The runner's own control PV is ABSENT. The name is read out of the
#      serving stack's own constant in the container rather than hardcoded,
#      because an absence check only means something when the name is one the
#      server WOULD serve under the opposite configuration -- and the policy
#      flag is asserted alongside it, so the step knows which configuration it
#      is testing. Paired with a positive control on a known-good PV, so
#      "absent" cannot be satisfied by a server that stopped answering.
#      SNAPSHOT is checked too but is explicitly only a forward regression
#      guard: no such PV exists in this stack, so its absence is evidence of
#      nothing by itself.
#   5. A PVA get of model_info returns the model's real variable roster.
#   6. A PVA put above the drive limit lands CLAMPED, on both views of the
#      address -- the value is checked back over PVA and over CA.
#   7. A refused PVA put moves nothing. Checked on the Channel Access view,
#      because that is the view that carries a real reading; see the step
#      itself for why the PVA view would make this check vacuous today.
#   8. The PVAccess port is not published to the host, while PVA is
#      confirmed live inside the container -- so "unpublished" is
#      distinguished from "not running".
#
# Exits 0 only if the image builds, boots to serving PVs within
# BOOT_TIMEOUT_SECS, and every assertion above holds.
#
# Idempotent: safe to re-run. Removes any prior gate container first.
#
# Environment:
#   OSPREY_VA_CA_PORT   Channel Access port to bind and publish. Defaults to
#                       5064, which is what a full certification run needs
#                       (it is the port the "Local Simulation" gateway preset
#                       and the compose template both name). Override it to
#                       run this gate on a host where 5064 is already held --
#                       a running demo stack, most often -- without stopping
#                       whatever holds it.
#   OSPREY_VA_RUNTIME   Container runtime. Auto-detected when unset.
#   LUME_PVA_SRC        Serving-stack checkout. Derived from the repo's
#                       sibling layout when unset.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VA_DIR="${WORKTREE_ROOT}/docker/virtual-accelerator"
IMAGE="osprey-va-full:latest"
CONTAINER="osprey-va-full-gate"
# 5064 is the number a full certification needs: the "Local Simulation"
# gateway preset and the compose template both name it, and the published
# port must equal the bound port (a CA search reply carries the server's own
# port). Overridable so this gate can run on a host where something else
# already holds 5064 -- the alternative being to stop that stack, which is an
# operator's decision and not this script's.
CA_PORT="${OSPREY_VA_CA_PORT:-5064}"
BOOT_TIMEOUT_SECS=60
READY_LOG_MARKER="virtual accelerator IOC serving PVs"

# The PVAccess server's port. Served inside the container only -- step 8
# asserts it is never published to the host.
PVA_PORT=5075

# A pyat-coupled BPM readback, and the corrector whose field moves it. Reading
# the BPM alone proves nothing: the tutorial lattice's closed orbit with no
# corrector excited is exactly zero, which is indistinguishable from a PV that
# was never seeded. So the gate writes ${EXCITE_PV} and requires the readings
# to move -- only a live manifest -> records -> physics-bridge -> lattice chain
# can do that.
GATE_PV="SR:DIAG:BPM:01:POSITION:X"
GATE_PV_2="SR:DIAG:BPM:02:POSITION:X"
EXCITE_PV="SR:MAG:HCM:01:CURRENT:SP"
EXCITE_VALUE="0.5"
# Floor for "this reading moved". The excitation above produces order 1e-6 m
# at both BPMs, so this sits ~100x below the signal and far above float noise.
MOVED_THRESHOLD_M="1e-8"

# ${EXCITE_PV}'s drive band is [-12, 12] (channel_limits.json). A put past the
# top of it must land clamped at the limit rather than being refused or taken
# literally.
DRIVE_HIGH="12.0"
OVER_DRIVE="20.0"

# The control PV the runner must not claim is NOT named here on purpose: step 4
# reads it out of the serving stack's own constant, in the container, so the
# check cannot drift into testing a name the stack no longer uses.

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

# The serving stack is a sibling checkout, not a PyPI package: build its wheel
# into the context so the image installs the fork you actually have. Same
# staging step as scripts/va/run_va.sh -- both callers must satisfy the
# Containerfile's build-context contract.
#
# This is a dev path with an end date: once the package is published, the image
# resolves it by name from osprey's `virtual-accelerator` extra and this block
# goes away. It is deliberately not a `git+https://` URL -- that would make the
# image track a pushed ref rather than the checkout in front of you, which is
# the property the source install exists to preserve, and would put the network
# on the build path.
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
# Built from a throwaway copy, never in place: that checkout is someone else's
# repository and this script must leave it untouched, but a build backend
# writes build/ and a generated _version.py into whatever tree it runs in.
# .git comes along because the version is derived from it; the copy is deleted
# before the image build so it never enters the context.
FORK_COPY="${STAGING_DIR}/.lume-pva-src"
rsync -a --exclude '.venv' --exclude 'build' --exclude 'dist' \
    --exclude '*.egg-info' --exclude '.pytest_cache' --exclude '.ruff_cache' \
    "${LUME_PVA_SRC}/" "${FORK_COPY}/"
uv build --wheel --out-dir "${STAGING_DIR}" "${FORK_COPY}"
rm -rf "${FORK_COPY}"

# osprey's version comes from git (hatch-vcs) and the staged context has no
# .git, so the host resolves it and passes it in; see the Containerfile.
OSPREY_VERSION="$("${VENV_PY}" -c 'import osprey; print(osprey.__version__)')"

echo "--- Building ${IMAGE} (linux/amd64) ---"
"${RUNTIME}" build \
    --build-arg "OSPREY_VERSION=${OSPREY_VERSION}" \
    -t "${IMAGE}" -f "${STAGING_DIR}/docker/virtual-accelerator/Containerfile" "${STAGING_DIR}"

echo "--- Starting ${CONTAINER} (data dir: ${DATA_DIR}) ---"
# The server port is passed explicitly, from the same CA_PORT the publish maps:
# a CA search reply carries the server's own port, so a container bound to one
# port and published on another hands clients an unreachable address with no
# useful error. (The image derives EPICS_CAS_SERVER_PORT from this.)
"${RUNTIME}" run -d --name "${CONTAINER}" \
    -e "EPICS_CA_SERVER_PORT=${CA_PORT}" \
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

echo "--- [1/8] Asserting the runner readiness line ---"
# The whole line is the contract, not just the marker: entrypoint.py writes it
# in one place as "<marker>: <N> channels", and N is len(records.all). Matching
# the marker alone would accept a boot that announced itself while serving an
# empty namespace, so the count is parsed out and required to be positive.
READY_LINE="$("${RUNTIME}" logs "${CONTAINER}" 2>&1 | grep -m1 "${READY_LOG_MARKER}" || true)"
SERVED_CHANNELS="$(printf '%s' "${READY_LINE}" |
    sed -n "s/.*${READY_LOG_MARKER}: \\([0-9][0-9]*\\) channels.*/\\1/p")"
if [[ -z "${SERVED_CHANNELS}" ]]; then
    echo "FATAL: readiness line did not match '<marker>: <N> channels'" >&2
    echo "  got: ${READY_LINE:-<no line matching the marker>}" >&2
    exit 1
fi
if [[ "${SERVED_CHANNELS}" -le 0 ]]; then
    echo "FATAL: the IOC announced readiness while serving ${SERVED_CHANNELS} channels" >&2
    exit 1
fi
echo "OK: ${READY_LINE}"

# Host CA client configuration: name-server (TCP) mode, which is the one
# host<->container arrangement proven to work across container runtimes. UDP
# broadcast discovery is deliberately not relied upon and not published.
export EPICS_CA_NAME_SERVERS="localhost:${CA_PORT}"
export EPICS_CA_AUTO_ADDR_LIST="NO"
export EPICS_CA_ADDR_LIST=""

# Every Channel Access assertion below runs through pyepics in the worktree
# venv rather than through the host's caget/caput binaries. One client, not
# two: these steps compare floats and assert on absence, and a single
# implementation is what keeps "did not move" and "does not exist" meaning the
# same thing in each of them. The venv is already a hard requirement above --
# OSPREY_VERSION is resolved through it -- so this adds no new dependency.
#
# EVERY read below passes use_monitor=False, and that is load-bearing rather
# than stylistic. pyepics' caget defaults to returning the value its monitor
# subscription last cached, which after a write is whatever arrived before the
# write did: the first draft of this gate read a setpoint back as 0 having just
# completed a put of 0.5, intermittently, because the monitor event had not been
# processed yet. Two distinct ways that corrupts a gate -- an assertion that a
# value MOVED fails at random, and, far worse, an assertion that a value did NOT
# move (step 7) passes for free, since a stale cache never moves. use_monitor=False
# forces a fresh one-shot read over the wire, which is what put-completion
# actually guarantees something about.
if [[ ! -x "${VENV_PY}" ]]; then
    echo "FATAL: no worktree venv python at ${VENV_PY}; the CA assertions need pyepics" >&2
    exit 1
fi
if ! "${VENV_PY}" -c "import epics" >/dev/null 2>&1; then
    echo "FATAL: pyepics is not importable in ${VENV_PY}" >&2
    exit 1
fi

ca_fail() {
    echo "FATAL: $1" >&2
    "${RUNTIME}" logs "${CONTAINER}" >&2 || true
    exit 1
}

# Run a python payload inside the container, with a guard against the payload
# never arriving. `python -` reading an empty stdin exits 0 and prints nothing,
# so an exec form that failed to forward the heredoc would let every step below
# "pass" having asserted precisely nothing -- the exact failure this gate exists
# to catch, hiding in the gate itself. Each payload therefore ends by printing
# an `OK:` line, and reaching the end without one is a failure.
#
# Extra `exec` arguments (the `-e VAR=value` pairs) are passed through; the
# payload is read from this function's own stdin.
in_container_py() {
    local out rc
    out="$("${RUNTIME}" exec -i "$@" "${CONTAINER}" python -)"
    rc=$?
    printf '%s\n' "${out}"
    if [[ ${rc} -ne 0 ]]; then
        return "${rc}"
    fi
    if ! printf '%s' "${out}" | grep -q '^OK: '; then
        echo "FATAL: the in-container payload printed no OK sentinel, so it did not run" >&2
        return 1
    fi
    return 0
}

echo "--- [2/8] Baseline: reading the quiescent BPMs over Channel Access ---"
# Not an assertion of correctness -- see the header. With no corrector
# excited the closed orbit is exactly zero, so this only establishes that CA
# answers at all and records the baseline step 3 measures movement against.
QUIESCENT="$("${VENV_PY}" - "${GATE_PV}" "${GATE_PV_2}" <<'PY'
import sys
import epics
vals = []
for pv in sys.argv[1:]:
    v = epics.caget(pv, timeout=15, use_monitor=False)
    if v is None:
        raise SystemExit(f"FATAL: no CA connection to {pv}")
    vals.append(f"{float(v):.12g}")
print(" ".join(vals))
PY
)" || ca_fail "could not read the BPMs over Channel Access"
echo "OK: quiescent ${GATE_PV}=$(echo "${QUIESCENT}" | cut -d' ' -f1)" \
     "${GATE_PV_2}=$(echo "${QUIESCENT}" | cut -d' ' -f2)"

echo "--- [3/8] Exciting ${EXCITE_PV} and requiring the BPMs to move ---"
# The assertion with teeth. put-completion (wait=True) is what makes the read
# safe to do immediately: the setpoint is asynchronous, so the put returns only
# once the physics hand-off behind it has finished and the readings it produced
# have been pushed. Without that this would race the solve and read staleness.
"${VENV_PY}" - "${EXCITE_PV}" "${EXCITE_VALUE}" "${MOVED_THRESHOLD_M}" \
    "${GATE_PV}" "${GATE_PV_2}" <<'PY' || ca_fail "the excitation did not move the BPM readings"
import sys
import epics

sp, value, threshold = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
bpms = sys.argv[4:]

before = {}
for pv in bpms:
    v = epics.caget(pv, timeout=15, use_monitor=False)
    if v is None:
        raise SystemExit(f"FATAL: no CA connection to {pv}")
    before[pv] = float(v)

if epics.caput(sp, value, wait=True, timeout=30) != 1:
    raise SystemExit(f"FATAL: put-completion never returned for {sp}")

readback = epics.caget(sp, timeout=15, use_monitor=False)
if readback is None or abs(float(readback) - value) > 1e-9:
    raise SystemExit(f"FATAL: {sp} reads {readback!r} after a completed put of {value}")

failures = []
for pv in bpms:
    v = epics.caget(pv, timeout=15, use_monitor=False)
    if v is None:
        raise SystemExit(f"FATAL: no CA connection to {pv}")
    moved = abs(float(v) - before[pv])
    status = "moved" if moved > threshold else "DID NOT MOVE"
    print(f"  {pv}: {before[pv]:.6g} -> {float(v):.6g}  (|delta|={moved:.3g}, {status})")
    if moved <= threshold:
        failures.append(pv)

if failures:
    raise SystemExit(
        "FATAL: "
        + ", ".join(failures)
        + f" did not move by more than {threshold:g} m when {sp} was driven to {value}. "
        "A reading that stays put through a corrector excitation is an unseeded PV, "
        "not a physics chain."
    )
print(f"OK: every BPM moved off its quiescent value when {sp} was excited")
PY

echo "--- [4/8] Asserting the runner claims no control PVs ---"
# The serving runner is configured with control_pvs off: nothing is served that
# the facility's channel manifest does not describe.
#
# An absence check is worthless unless the name being looked for is one the
# server WOULD serve under the opposite configuration. So the name is not
# hardcoded here -- it is read out of the serving stack's own
# RESET_CONTROL_PV constant, in the running container. If the fork renames its
# control PV, this check follows it instead of quietly testing a dead string
# forever. The deployed policy flag is read alongside it, so the step states
# what configuration it is actually asserting about rather than assuming.
CONTROL_FACTS="$("${RUNTIME}" exec -i "${CONTAINER}" python - <<'PY'
from lume_pva_apg.runner import RESET_CONTROL_PV
from osprey.services.virtual_accelerator.serving.write_path import RUNNER_CONFIG_POLICY

# Printed for the shell to consume; the assertions on these are below.
print(f"RESET_PV={RESET_CONTROL_PV}")
print(f"CONTROL_PVS={RUNNER_CONFIG_POLICY.get('control_pvs')!r}")
PY
)" || ca_fail "could not read the serving stack's control-PV configuration"

RESET_PV="$(printf '%s\n' "${CONTROL_FACTS}" | sed -n 's/^RESET_PV=//p')"
CONTROL_PVS="$(printf '%s\n' "${CONTROL_FACTS}" | sed -n 's/^CONTROL_PVS=//p')"
echo "  serving stack's control PV name: ${RESET_PV:-<none>}"
echo "  deployed control_pvs policy    : ${CONTROL_PVS:-<unknown>}"
if [[ -z "${RESET_PV}" ]]; then
    echo "FATAL: could not resolve the serving stack's control-PV name, so its absence" >&2
    echo "       from the served namespace would prove nothing" >&2
    exit 1
fi
if [[ "${CONTROL_PVS}" != "False" ]]; then
    echo "FATAL: control_pvs is ${CONTROL_PVS}, not False -- this deployment claims control" >&2
    echo "       PVs the channel manifest does not describe" >&2
    exit 1
fi

# ${RESET_PV} is the load-bearing assertion: that exact name IS served when
# control_pvs is on, so its absence here is evidence about this deployment's
# configuration and not about the name being unknown.
#
# SNAPSHOT is a different thing and is labelled as such: no such PV exists
# anywhere in this serving stack -- it went with the remote-input mode the fork
# dropped -- so its absence is NOT evidence about control_pvs and must not be
# read as any. It is kept purely as a forward regression guard, to fail if a
# future change reintroduces that name into the served namespace.
"${VENV_PY}" - "${EXCITE_PV}" "${RESET_PV}" "SNAPSHOT" <<'PY' || ca_fail "control-PV absence check failed"
import sys
import epics

control, reset_pv, regression_guard = sys.argv[1], sys.argv[2], sys.argv[3]

# Absence is only meaningful next to a positive control: without one, a server
# that had stopped answering altogether would satisfy every check below.
if epics.caget(control, timeout=15, use_monitor=False) is None:
    raise SystemExit(
        f"FATAL: positive control {control} did not answer, so this step cannot "
        "distinguish an absent control PV from a server that stopped responding"
    )
print(f"  positive control {control}: answers")


def served(pv: str) -> bool:
    # Short timeout deliberately: this waits for a connection that must never
    # come, and every second here is spent proving a negative.
    return epics.caget(pv, timeout=3, connection_timeout=3, use_monitor=False) is not None


if served(reset_pv):
    raise SystemExit(
        f"FATAL: {reset_pv} is served. It is the serving stack's own control PV, so "
        "this deployment is claiming a name the channel manifest does not describe."
    )
print(f"  {reset_pv}: absent, as required (load-bearing)")

if served(regression_guard):
    raise SystemExit(
        f"FATAL: {regression_guard} is served. No such PV exists in this serving stack, "
        "so something has reintroduced it into the served namespace."
    )
print(f"  {regression_guard}: absent (regression guard only -- proves nothing on its own)")

print("OK: no control PV is served outside the channel manifest")
PY

echo "--- [5/8] PVA get of model_info (in-container; PVA is not published) ---"
in_container_py <<'PY' || ca_fail "PVA get of model_info failed"
from p4p.client.thread import Context

ctx = Context("pva")
info = ctx.get("model_info", timeout=20)

variables = info["supported_variables"]
if len(variables) <= 0:
    raise SystemExit("FATAL: model_info describes no variables")

names = [entry["name"] for entry in variables]
print(f"  class       : {info['class']}")
print(f"  variables   : {len(names)}")
print(f"  first       : {names[0]}")
if len(set(names)) != len(names):
    raise SystemExit("FATAL: model_info lists duplicate variable names")
print("OK: model_info served over PVA with a populated variable roster")
PY

echo "--- [6/8] PVA put above the drive limit lands clamped, on both views ---"
in_container_py \
    -e "SP_PV=${EXCITE_PV}" -e "OVER_DRIVE=${OVER_DRIVE}" -e "DRIVE_HIGH=${DRIVE_HIGH}" \
    <<'PY' || ca_fail "PVA put was not clamped to the drive limit"
import os

from p4p.client.thread import Context

sp = os.environ["SP_PV"]
over = float(os.environ["OVER_DRIVE"])
high = float(os.environ["DRIVE_HIGH"])

ctx = Context("pva")
before = float(ctx.get(sp, timeout=15))
ctx.put(sp, over, timeout=30)
after = float(ctx.get(sp, timeout=15))

print(f"  {sp}: {before:g} -> put({over:g}) -> {after:g}")
if abs(after - high) > 1e-9:
    raise SystemExit(
        f"FATAL: a put of {over:g} left {sp} at {after:g}; the drive limit is {high:g}, "
        "so it should have landed clamped there"
    )
print(f"OK: the over-range put landed clamped at the drive limit ({high:g})")
PY

# The clamp must be visible on the OTHER view too. The address is served
# twice -- co-hosted on Channel Access, natively on PVA because it is also a
# model variable -- and a clamp that moved only the transport the client used
# would leave the two views disagreeing about the machine.
"${VENV_PY}" - "${EXCITE_PV}" "${DRIVE_HIGH}" <<'PY' || ca_fail "the clamped value did not reach the CA view"
import sys
import epics

sp, high = sys.argv[1], float(sys.argv[2])
v = epics.caget(sp, timeout=15, use_monitor=False)
if v is None:
    raise SystemExit(f"FATAL: no CA connection to {sp}")
print(f"  CA view of {sp}: {float(v):g}")
if abs(float(v) - high) > 1e-9:
    raise SystemExit(
        f"FATAL: the PVA put clamped to {high:g} but the Channel Access view reads "
        f"{float(v):g}; the two views of one address have diverged"
    )
print("OK: both views of the address carry the clamped value")
PY

echo "--- [7/8] A refused PVA put moves nothing ---"
# The refusal is checked on a read-only model variable, and the "did not move"
# half is asserted on the CHANNEL ACCESS view rather than the PVA one. That is
# deliberate: the value this step requires to hold still has to be one that
# could visibly have moved, and on Channel Access a BPM carries a real,
# non-zero reading pushed there by the physics bridge -- step 3 above is what
# put it there. The check is guarded accordingly below: it FAILS rather than
# passes if that reading is 0, so it can never degenerate into comparing one
# zero against another.
#
# Channel Access is the right instrument here whether or not the PVA view of a
# reading tracks it, so this step does not rest on which of those is true.
CA_BEFORE="$("${VENV_PY}" - "${GATE_PV}" <<'PY'
import sys
import epics
v = epics.caget(sys.argv[1], timeout=15, use_monitor=False)
if v is None:
    raise SystemExit("FATAL: no CA connection")
print(f"{float(v):.17g}")
PY
)" || ca_fail "could not read ${GATE_PV} before the refused put"

in_container_py -e "RO_PV=${GATE_PV}" <<'PY' || ca_fail "the read-only PVA put was not refused"
import os

from p4p.client.thread import Context

ro = os.environ["RO_PV"]
ctx = Context("pva")
try:
    ctx.put(ro, 999.0, timeout=15)
except Exception as exc:  # noqa: BLE001 - any refusal is the pass condition
    print(f"  put({ro}, 999.0) refused: {type(exc).__name__}: {exc}")
else:
    raise SystemExit(f"FATAL: a put to the read-only {ro} was ACCEPTED")
print("OK: the read-only variable refused the put")
PY

"${VENV_PY}" - "${GATE_PV}" "${CA_BEFORE}" <<'PY' || ca_fail "a refused put moved the served value"
import sys
import epics

pv, before = sys.argv[1], float(sys.argv[2])
v = epics.caget(pv, timeout=15, use_monitor=False)
if v is None:
    raise SystemExit(f"FATAL: no CA connection to {pv}")
after = float(v)
print(f"  CA view of {pv}: {before:.6g} -> {after:.6g}")
if before == 0.0:
    raise SystemExit(
        f"FATAL: {pv} read 0 before the refused put, so 'unchanged' would prove "
        "nothing here. The excitation in step 3 should have left it non-zero."
    )
if after != before:
    raise SystemExit(f"FATAL: a refused put moved {pv} from {before!r} to {after!r}")
if after == 999.0:
    raise SystemExit(f"FATAL: the refused value landed on {pv}")
print("OK: the refusal moved nothing any reader can see")
PY

echo "--- [8/8] The PVAccess port is not published to the host ---"
# Asserted against the container runtime's own port table, which is
# authoritative about what THIS container published. A host-side connect probe
# would not be: on a shared development host something unrelated may already
# be listening on ${PVA_PORT}, and this gate would then report a port leak
# that is not its container's. (Measured on the development host -- an
# unrelated process held ${PVA_PORT} while this gate ran clean.)
PORT_TABLE="$("${RUNTIME}" port "${CONTAINER}" 2>&1 || true)"
echo "  published: ${PORT_TABLE:-<nothing>}"
if printf '%s' "${PORT_TABLE}" | grep -Eq "(^|[^0-9])${PVA_PORT}([^0-9]|\$)"; then
    echo "FATAL: the PVAccess port ${PVA_PORT} is published to the host" >&2
    exit 1
fi
if ! printf '%s' "${PORT_TABLE}" | grep -Eq "(^|[^0-9])${CA_PORT}([^0-9]|\$)"; then
    echo "FATAL: the Channel Access port ${CA_PORT} is not in the container's port table" >&2
    exit 1
fi
# ...and PVA really is running behind that unpublished port, so this step
# distinguishes "not reachable from the host" from "never started".
in_container_py -e "PVA_PORT=${PVA_PORT}" <<'PY' || ca_fail "PVA is not listening inside the container"
import os
import socket

port = int(os.environ["PVA_PORT"])
with socket.socket() as s:
    s.settimeout(5)
    try:
        s.connect(("127.0.0.1", port))
    except OSError as exc:
        raise SystemExit(
            f"FATAL: nothing is listening on {port} inside the container ({exc}), so its "
            "absence from the host says nothing about PVA being served"
        ) from exc
print(f"OK: PVA is live on {port} inside the container and published nowhere")
PY

echo "--- Gate PASSED ---"
