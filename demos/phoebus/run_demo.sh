#!/usr/bin/env bash
#
# Launch the full OSPREY <-> Phoebus demo stack.
#
# Two modes, auto-selected:
#   * DESKTOP  (default when a display is present): Phoebus opens on your
#     desktop, exactly as before.
#   * ISOLATED (auto on headless Linux, e.g. appsdev2; force with ISOLATED=1):
#     Phoebus runs inside a private TigerVNC X server (Xvnc) on a dedicated
#     display, with a private user dir, a dedicated single-instance-server port,
#     dedicated demo Channel Access ports, and a PRE-FLIGHT COLLISION CHECK that
#     refuses to start if the bridge port, X display, the instance-server port,
#     the Xvnc RFB port, or the demo CA ports are already in use — so it can
#     never disturb another Phoebus / CA server already running on the host.
#     (Xvnc renders a real X11 window; a websockify+noVNC bridge on the RFB port
#     surfaces it in the OSPREY web terminal — see Task 3.2.)
#
# Starts:
#   1. the caproto soft IOC (demo_ioc.py) serving DEMO:* on loopback CA, and
#   2. a Phoebus product showing the demo panel (osprey_demo.bob), with the
#      in-JVM agent bridge listening on http://127.0.0.1:${BRIDGE_PORT}.
#
# Then it waits — leave it running and, in a SEPARATE terminal, start the OSPREY
# agent in your built project and ask it to run the Phoebus walkthrough:
#
#     cd <your phoebus-demo project> && claude
#     > run the phoebus walkthrough
#
# Press Ctrl-C to tear the stack down (kills only the demo's own processes).
#
# Environment overrides:
#   PHOEBUS_REPO   phoebus checkout (default: sibling ../phoebus)
#   PHOEBUS_JAR    product jar (default: autodetected under phoebus-product/target)
#   INSTANCE       demo instance number (default: 1). INSTANCE=2 launches a
#                  SECOND isolated Phoebus alongside the first: every
#                  per-instance default below shifts by (INSTANCE-1), the soft
#                  IOC is NOT started again (instance 1's is shared via the
#                  same loopback CA ports), and log files get a .i<N> suffix.
#   START_IOC      1/0 — start the demo soft IOC (default: 1 for INSTANCE=1,
#                  0 otherwise)
#   BRIDGE_PORT    bridge HTTP port (default: 7979 + INSTANCE-1)
#   JAVA_HOME      JDK 21 (default: Homebrew openjdk@21, then `java` on PATH;
#                  a shared Linux host, e.g. appsdev2, supplies its own JDK by
#                  setting this env var — no facility-specific path is baked in)
#   ISOLATED       1 = force isolated/headless mode, 0 = force desktop mode
#                  (default: auto — isolated on headless Linux)
#   OSPREY_PYTHON  python used for the demo IOC (needs caproto); overrides the
#                  default uv / ${OSPREY_ROOT}/.venv resolution
#   Isolated-mode knobs (defaults are the appsdev2 recon-chosen values; the
#   per-instance ones shift by INSTANCE-1):
#     DEMO_DISPLAY     X display for Xvnc            (default: :99 + INSTANCE-1)
#     DEMO_GEOMETRY    Xvnc framebuffer size WxH     (default: 1600x900) — the
#                      Phoebus window is pinned to fill it exactly, so the
#                      stream shows ONLY Phoebus (no bare X root around it)
#     RFB_PORT         Xvnc RFB port                 (default: 5977 + INSTANCE-1)
#     INSTANCE_PORT    Phoebus single-instance port  (default: 4079 + INSTANCE-1)
#     CA_SERVER_PORT   demo Channel Access server    (default: 5074, shared)
#     CA_REPEATER_PORT demo CA repeater              (default: 5075, shared)
#     PHOEBUS_USER_DIR private phoebus.user dir       (default: a fresh mktemp dir)
#   Web-stream knobs (isolated mode; the leg is skipped when websockify or the
#   vendored noVNC assets are absent — the bridge demo still works without it):
#     WEB_VNC_PORT     websockify HTTP+WS port        (default: 6080 + INSTANCE-1)
#     NOVNC_DIR        vendored noVNC client dir      (default: probe
#                      ${SCRIPT_DIR}/novnc, then ~/phoebus-remote-bridge-demo/novnc)
#     WEBSOCKIFY_CMD   command to run websockify      (default: `websockify` on
#                      PATH; e.g. "env PYTHONPATH=… python -m websockify")
#     XVNC_CMD         Xvnc binary — must be TigerVNC (default: `Xvnc` on PATH,
#                      except when that is a RealVNC symlink and the TigerVNC
#                      binary survives as Xvnc.conflict — RealVNC's package
#                      renames it — in which case the .conflict one is used.
#                      RealVNC's Xvnc ignores -geometry in virtual mode and
#                      overlays its own UI windows on the stream.)
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OSPREY_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
IOC="${SCRIPT_DIR}/demo_ioc.py"
PANEL="${SCRIPT_DIR}/panels/osprey_demo.bob"

: "${PHOEBUS_REPO:=$(cd "${OSPREY_ROOT}/.." && pwd)/phoebus}"

# Multi-instance support: INSTANCE=2 (3, …) runs an additional isolated Phoebus
# next to instance 1. All per-instance defaults shift by this offset; the demo
# soft IOC is shared (only instance 1 starts it — the rest are plain CA clients
# on the same loopback ports).
: "${INSTANCE:=1}"
case "${INSTANCE}" in (*[!0-9]*|'') echo "ERROR: INSTANCE must be a positive integer, got '${INSTANCE}'"; exit 2;; esac
[[ "${INSTANCE}" -ge 1 ]] || { echo "ERROR: INSTANCE must be >= 1, got '${INSTANCE}'"; exit 2; }
_IOFF=$(( INSTANCE - 1 ))
: "${START_IOC:=$(( INSTANCE == 1 ? 1 : 0 ))}"
LOG_SUFFIX=""; [[ "${INSTANCE}" != 1 ]] && LOG_SUFFIX=".i${INSTANCE}"

: "${BRIDGE_PORT:=$(( 7979 + _IOFF ))}"
BASE_URL="http://127.0.0.1:${BRIDGE_PORT}"

# --- JDK 21 autodetection (Homebrew, then JAVA_HOME env, then PATH) ------------
# Facility-neutral: no facility-specific JDK path is baked in here. A shared
# Linux host (e.g. appsdev2) supplies its JDK via the JAVA_HOME env var.
if [[ -z "${JAVA_HOME:-}" && -x /opt/homebrew/opt/openjdk@21/bin/java ]]; then
  JAVA_HOME=/opt/homebrew/opt/openjdk@21
fi
if [[ -n "${JAVA_HOME:-}" ]]; then
  export JAVA_HOME
  export PATH="${JAVA_HOME}/bin:${PATH}"
fi
command -v java >/dev/null 2>&1 || { echo "ERROR: no java found; set JAVA_HOME to a JDK 21"; exit 2; }

# --- mode selection: isolated/headless vs desktop ------------------------------
if [[ -z "${ISOLATED:-}" ]]; then
  # Safety-first on shared Linux hosts (e.g. appsdev2): default to ISOLATED so the
  # pre-flight guard + dedicated ports ALWAYS engage. A Linux desktop run must opt
  # out explicitly with ISOLATED=0. We key on the OS, NOT on DISPLAY — an `ssh -X`
  # session sets DISPLAY, so keying on it would silently disable the isolation.
  if [[ "$(uname -s)" == "Linux" ]]; then ISOLATED=1; else ISOLATED=0; fi
fi
# Isolated-mode knobs (appsdev2 recon-chosen, non-colliding values; per-instance
# ones shift by the INSTANCE offset — CA ports are shared across instances)
: "${DEMO_DISPLAY:=:$(( 99 + _IOFF ))}"
: "${DEMO_GEOMETRY:=1600x900}"
: "${RFB_PORT:=$(( 5977 + _IOFF ))}"
: "${INSTANCE_PORT:=$(( 4079 + _IOFF ))}"
: "${CA_SERVER_PORT:=5074}"
: "${CA_REPEATER_PORT:=5075}"
: "${WEB_VNC_PORT:=$(( 6080 + _IOFF ))}"

# --- Channel Access / PV Access: loopback only, or loopback + read-only gateway -
# Default: both the IOC (server) and Phoebus/OSPREY (clients) stay on the loopback.
# LIVE_READONLY=1 additionally points the CA/PVA *client* search path at the ALS
# read-only gateway (the same host ALS's own readOnlyPhoebus.sh uses). The gateway
# grants "read, no write" via CA access security on every channel, so real panels
# show live machine data while writes stay impossible — even for a human clicking
# buttons in the streamed noVNC panel, which bypasses all MCP-level write gates.
# Server-side interfaces (EPICS_CAS_*) stay loopback in EVERY mode: the demo IOC
# never announces itself off-box.
#
# READONLY_GATEWAY names a facility-specific gateway host, so it has no
# framework default here — it is supplied via the env var, or via an untracked
# demos/phoebus/local.env (sourced if present; not part of the repo, so each
# facility keeps its own gateway host out of version control). Only required
# when LIVE_READONLY=1 actually asks for it.
if [[ -z "${READONLY_GATEWAY:-}" && -f "${SCRIPT_DIR}/local.env" ]]; then
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/local.env"
fi
if [[ "${LIVE_READONLY:-0}" == 1 && -z "${READONLY_GATEWAY:-}" ]]; then
  echo "ERROR: LIVE_READONLY=1 requires READONLY_GATEWAY (facility read-only CA/PVA gateway host)."
  echo "       Set the env var, or create demos/phoebus/local.env with READONLY_GATEWAY=<host>."
  exit 2
fi
if [[ "${ISOLATED}" == 1 ]]; then
  _LOOP_CA_ENTRY="127.0.0.1:${CA_SERVER_PORT}"   # explicit port: stays valid in a mixed list
else
  _LOOP_CA_ENTRY="127.0.0.1"
fi
if [[ "${LIVE_READONLY:-0}" == 1 ]]; then
  export EPICS_CA_ADDR_LIST="${_LOOP_CA_ENTRY} ${READONLY_GATEWAY}:5064"
  export EPICS_PVA_ADDR_LIST="${READONLY_GATEWAY}"
else
  export EPICS_CA_ADDR_LIST=127.0.0.1
  # Scrub PVA too — the host login env points EPICS_PVA_ADDR_LIST at the
  # production subnets, which would leak any pva:// PV out of the sandbox.
  export EPICS_PVA_ADDR_LIST=127.0.0.1
fi
export EPICS_CA_AUTO_ADDR_LIST=NO
export EPICS_PVA_AUTO_ADDR_LIST=NO
export EPICS_CAS_INTF_ADDR_LIST=127.0.0.1
export EPICS_CAS_BEACON_ADDR_LIST=127.0.0.1
if [[ "${ISOLATED}" == 1 ]]; then
  # Dedicated CA ports so the demo IOC never touches the host's default CA
  # server (5064) or repeater (5065) — a hard isolation guarantee on a shared box.
  export EPICS_CA_SERVER_PORT="${CA_SERVER_PORT}"
  export EPICS_CA_REPEATER_PORT="${CA_REPEATER_PORT}"
fi

# Debug/test seam: print the resolved EPICS client/server env and exit before
# any pre-flight checks or process launches.
if [[ "${PRINT_EPICS_ENV:-0}" == 1 ]]; then
  env | grep -E '^EPICS_(CA|CAS|PVA)_' | sort
  exit 0
fi

IOC_PID=""
PHOEBUS_PID=""
XVNC_PID=""
WEBSOCKIFY_PID=""
RUNDIR=""
RUNDIR_CREATED=0
log() { printf '[demo] %s\n' "$*"; }

cleanup() {
  log "shutting down…"
  [[ -n "${PHOEBUS_PID}" ]] && kill "${PHOEBUS_PID}" 2>/dev/null
  [[ -n "${IOC_PID}" ]] && kill "${IOC_PID}" 2>/dev/null
  [[ -n "${WEBSOCKIFY_PID}" ]] && kill "${WEBSOCKIFY_PID}" 2>/dev/null
  [[ -n "${XVNC_PID}" ]] && kill "${XVNC_PID}" 2>/dev/null
  wait 2>/dev/null
  # Only remove a run dir WE created (mktemp); never a user-supplied PHOEBUS_USER_DIR.
  [[ "${RUNDIR_CREATED}" == 1 && -n "${RUNDIR}" && -d "${RUNDIR}" ]] && rm -rf "${RUNDIR}"
}
trap cleanup EXIT INT TERM

# --- helpers -------------------------------------------------------------------
# Read-only listener probes. Return 0 (true) if the port is already bound.
# NB: no `-H` on ss — some builds reject it and would silently emit nothing,
# turning this safety guard into a false "port free". The header line can't
# match a ":PORT<end-of-field>" pattern, so plain `ss` is safe to grep.
tcp_port_in_use() {  # $1 = port
  if command -v ss >/dev/null 2>&1; then
    ss -tln 2>/dev/null | grep -qE "[:.]${1}([[:space:]]|\$)"
  elif command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"${1}" -sTCP:LISTEN >/dev/null 2>&1
  else
    return 1  # can't probe -> don't block
  fi
}
udp_port_in_use() {  # $1 = port
  if command -v ss >/dev/null 2>&1; then
    ss -uln 2>/dev/null | grep -qE "[:.]${1}([[:space:]]|\$)"
  else
    return 1
  fi
}

# Pre-flight collision check (isolated mode only). Refuses to start — the hard
# guarantee that we never disturb another Phoebus / CA server / X display.
preflight_check() {
  local fail=0 dn="${DEMO_DISPLAY#:}"
  log "pre-flight collision check (isolated mode)…"
  if [[ -e "/tmp/.X11-unix/X${dn}" || -e "/tmp/.X${dn}-lock" ]]; then
    echo "ERROR: X display ${DEMO_DISPLAY} already in use"; fail=1
  fi
  tcp_port_in_use "${BRIDGE_PORT}"   && { echo "ERROR: bridge port ${BRIDGE_PORT} already in use"; fail=1; }
  tcp_port_in_use "${INSTANCE_PORT}" && { echo "ERROR: Phoebus instance-server port ${INSTANCE_PORT} already in use"; fail=1; }
  tcp_port_in_use "${RFB_PORT}"      && { echo "ERROR: Xvnc RFB port ${RFB_PORT} already in use"; fail=1; }
  # CA server binds both UDP (search) and TCP (virtual circuits) on its port.
  # Only checked when THIS instance starts the IOC — a secondary instance
  # (START_IOC=0) *expects* instance 1's IOC to hold these ports.
  if [[ "${START_IOC}" == 1 ]]; then
    { tcp_port_in_use "${CA_SERVER_PORT}" || udp_port_in_use "${CA_SERVER_PORT}"; } && { echo "ERROR: demo CA server port ${CA_SERVER_PORT} already in use"; fail=1; }
    udp_port_in_use "${CA_REPEATER_PORT}" && { echo "ERROR: demo CA repeater port ${CA_REPEATER_PORT} already in use"; fail=1; }
  fi
  [[ "${STREAM_ENABLED}" == 1 ]] && tcp_port_in_use "${WEB_VNC_PORT}" && { echo "ERROR: web-stream (websockify) port ${WEB_VNC_PORT} already in use"; fail=1; }
  if [[ "${fail}" == 1 ]]; then
    echo "ABORT: refusing to start — an isolated-runtime resource is occupied."
    echo "       (This guard exists so the demo can never disturb another running Phoebus/CA/X server.)"
    exit 3
  fi
  log "pre-flight OK — display ${DEMO_DISPLAY}, bridge ${BRIDGE_PORT}, -server ${INSTANCE_PORT}, RFB ${RFB_PORT}, CA ${CA_SERVER_PORT}/${CA_REPEATER_PORT} all free"
}

# --- preconditions -------------------------------------------------------------
command -v curl >/dev/null 2>&1 || { echo "ERROR: curl is required"; exit 2; }

# --- IOC interpreter: OSPREY_PYTHON > uv > ${OSPREY_ROOT}/.venv ------------------
# The demo IOC only needs caproto, so any python that can `import caproto` works.
# uv is absent on appsdev2 — point OSPREY_PYTHON at an existing venv there, e.g.
#   OSPREY_PYTHON=~/projects/osprey/.venv/bin/python
IOC_CMD=()
if [[ "${START_IOC}" != 1 ]]; then
  : # secondary instance — no IOC, no python needed
elif [[ -n "${OSPREY_PYTHON:-}" ]]; then
  "${OSPREY_PYTHON}" -c 'import caproto' 2>/dev/null \
    || { echo "ERROR: OSPREY_PYTHON=${OSPREY_PYTHON} cannot import caproto"; exit 2; }
  IOC_CMD=( "${OSPREY_PYTHON}" )
elif command -v uv >/dev/null 2>&1; then
  IOC_CMD=( uv run --project "${OSPREY_ROOT}" python )
elif [[ -x "${OSPREY_ROOT}/.venv/bin/python" ]] \
     && "${OSPREY_ROOT}/.venv/bin/python" -c 'import caproto' 2>/dev/null; then
  IOC_CMD=( "${OSPREY_ROOT}/.venv/bin/python" )
else
  echo "ERROR: cannot run the IOC — install uv, or set OSPREY_PYTHON to a python with caproto"
  exit 2
fi

# --- web-stream leg (optional): websockify + vendored noVNC ---------------------
# Serves the noVNC client over HTTP and bridges its WebSocket to the Xvnc RFB
# port, all on loopback. The OSPREY web terminal embeds it as the PHOEBUS panel
# (iframe /panel/phoebus/vnc.html; WS relayed by the terminal's /panel proxy).
WS_CMD=()
if [[ -n "${WEBSOCKIFY_CMD:-}" ]]; then
  # Intentionally word-split: the override may carry env vars + args.
  read -r -a WS_CMD <<< "${WEBSOCKIFY_CMD}"
elif command -v websockify >/dev/null 2>&1; then
  WS_CMD=( websockify )
fi
if [[ -z "${NOVNC_DIR:-}" ]]; then
  for _cand in "${SCRIPT_DIR}/novnc" "${HOME}/phoebus-remote-bridge-demo/novnc"; do
    [[ -f "${_cand}/vnc.html" ]] && { NOVNC_DIR="${_cand}"; break; }
  done
fi
STREAM_ENABLED=0
if [[ ${#WS_CMD[@]} -gt 0 && -n "${NOVNC_DIR:-}" && -f "${NOVNC_DIR}/vnc.html" ]]; then
  STREAM_ENABLED=1
fi
[[ -f "${PANEL}" ]] || { echo "ERROR: demo panel not found: ${PANEL}"; exit 2; }

PHOEBUS_JAR="${PHOEBUS_JAR:-$(ls "${PHOEBUS_REPO}"/phoebus-product/target/product-*.jar 2>/dev/null | head -1)}"
if [[ -z "${PHOEBUS_JAR}" || ! -f "${PHOEBUS_JAR}" ]]; then
  echo "ERROR: Phoebus product jar not found under ${PHOEBUS_REPO}/phoebus-product/target."
  echo "       Build it:  (cd ${PHOEBUS_REPO} && mvn -pl phoebus-product -am install -DskipTests)"
  echo "       Or set PHOEBUS_REPO / PHOEBUS_JAR."
  exit 2
fi

# --- isolated mode: pre-flight guard + private Xvnc X server -------------------
# Extra JVM args (before -jar) and Phoebus product args (after -jar). Empty in
# desktop mode; the ${arr[@]+...} expansion below is set -u / bash-3.2 safe.
JVM_EXTRA_ARGS=()
PHX_EXTRA_ARGS=()
PANEL_RES="file:${PANEL}"   # -resource wants a file: URL (canonical; matches prod usage)
if [[ "${ISOLATED}" == 1 ]]; then
  # Resolve the X server. It must be TigerVNC: RealVNC's Xvnc ignores -geometry
  # in virtual mode (it sizes the desktop itself, e.g. to a past viewer) and
  # paints vncserverui overlay windows into the framebuffer. RealVNC's package
  # also usurps /usr/bin/Xvnc and renames TigerVNC's binary to Xvnc.conflict —
  # prefer that survivor when the default resolves to RealVNC.
  if [[ -z "${XVNC_CMD:-}" ]]; then
    XVNC_CMD="Xvnc"
    _xvnc_path="$(command -v Xvnc 2>/dev/null || true)"
    if [[ -n "${_xvnc_path}" ]] && readlink -f "${_xvnc_path}" 2>/dev/null | grep -qi realvnc; then
      if [[ -x /usr/bin/Xvnc.conflict ]]; then
        XVNC_CMD="/usr/bin/Xvnc.conflict"
        log "Xvnc on PATH is RealVNC — using TigerVNC at ${XVNC_CMD} instead"
      else
        echo "ERROR: Xvnc on PATH is RealVNC (ignores -geometry, overlays its UI on the stream)."
        echo "       Point XVNC_CMD at a TigerVNC Xvnc binary."
        exit 2
      fi
    fi
  fi
  command -v "${XVNC_CMD}" >/dev/null 2>&1 || {
    echo "ERROR: Xvnc (TigerVNC) is required for isolated/headless mode."
    echo "       Install tigervnc-server, or run on a desktop with ISOLATED=0."
    exit 2; }

  preflight_check

  if [[ -n "${PHOEBUS_USER_DIR:-}" ]]; then
    RUNDIR="${PHOEBUS_USER_DIR}"; RUNDIR_CREATED=0
  else
    RUNDIR="$(mktemp -d "${TMPDIR:-/tmp}/osprey-phoebus-demo.XXXXXX")"; RUNDIR_CREATED=1
  fi
  mkdir -p "${RUNDIR}/user"

  # Pin the main Phoebus window to fill the Xvnc framebuffer exactly: with no
  # window manager on the display there are no decorations, so 0,0 x geometry
  # means the VNC stream shows ONLY Phoebus — no bare X root around it.
  # Phoebus restores stage bounds from ${phoebus.user}/.phoebus/memento on
  # startup (Locations.user() appends the ".phoebus" folder;
  # MementoHelper.restoreStage applies the bounds); geometry applies even though
  # the empty <pane/> restores no tabs. Re-seeded every launch, clobbering any
  # layout a previous run saved into a reused PHOEBUS_USER_DIR — for the demo
  # that reset is wanted.
  GEOM_W="${DEMO_GEOMETRY%x*}"; GEOM_H="${DEMO_GEOMETRY#*x}"
  case "${GEOM_W}${GEOM_H}" in (*[!0-9]*|'')
    echo "ERROR: DEMO_GEOMETRY must be WxH (e.g. 1600x900), got '${DEMO_GEOMETRY}'"; exit 2;;
  esac
  mkdir -p "${RUNDIR}/user/.phoebus"
  cat > "${RUNDIR}/user/.phoebus/memento" <<MEMENTO
<memento>
  <DockStage_MAIN x="0" y="0" width="${GEOM_W}" height="${GEOM_H}">
    <pane/>
  </DockStage_MAIN>
</memento>
MEMENTO

  export XAUTHORITY="${RUNDIR}/Xauthority"; : > "${XAUTHORITY}"
  _cookie="$(mcookie 2>/dev/null || head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n')"
  xauth -f "${XAUTHORITY}" add "${DEMO_DISPLAY}" . "${_cookie}" >/dev/null 2>&1

  log "starting Xvnc (${XVNC_CMD}) on ${DEMO_DISPLAY} (${DEMO_GEOMETRY}, RFB 127.0.0.1:${RFB_PORT}, loopback, no window manager)…"
  "${XVNC_CMD}" "${DEMO_DISPLAY}" -geometry "${DEMO_GEOMETRY}" -depth 24 -rfbport "${RFB_PORT}" \
       -SecurityTypes None -localhost -auth "${XAUTHORITY}" \
       >"${RUNDIR}/xvnc.log" 2>&1 &
  XVNC_PID=$!
  sleep 2
  kill -0 "${XVNC_PID}" 2>/dev/null || { echo "ERROR: Xvnc failed to start; see ${RUNDIR}/xvnc.log"; exit 1; }
  export DISPLAY="${DEMO_DISPLAY}"

  if [[ "${STREAM_ENABLED}" == 1 ]]; then
    log "starting websockify on 127.0.0.1:${WEB_VNC_PORT} (noVNC: ${NOVNC_DIR}, RFB target 127.0.0.1:${RFB_PORT})…"
    "${WS_CMD[@]}" --web "${NOVNC_DIR}" 127.0.0.1:"${WEB_VNC_PORT}" 127.0.0.1:"${RFB_PORT}" \
      >"/tmp/osprey_demo_websockify${LOG_SUFFIX}.log" 2>&1 &
    WEBSOCKIFY_PID=$!
    sleep 1
    kill -0 "${WEBSOCKIFY_PID}" 2>/dev/null || {
      echo "ERROR: websockify failed to start; see /tmp/osprey_demo_websockify${LOG_SUFFIX}.log"; exit 1; }
  else
    log "web-stream leg skipped (websockify or noVNC assets not found) — bridge demo still fully functional"
  fi

  # The bridge's listen port is a Phoebus preference (default 7979) — pass the
  # instance's BRIDGE_PORT through a per-run settings file, or every instance
  # would listen on 7979 regardless of the port we probe.
  printf 'org.phoebus.applications.bridge.web/port=%s\n' "${BRIDGE_PORT}" \
    > "${RUNDIR}/settings.ini"

  # -Dprism.order=sw : GTK-glass software pipeline (a real X11 window Xvnc can
  #                    stream — NOT Monocle, which draws no window).
  # -Dphoebus.user / -Duser.home : keep ALL writes inside the private RUNDIR.
  # -server ${INSTANCE_PORT} : dedicated single-instance server, so -resource
  #                    forwarding can never target another Phoebus on the host.
  JVM_EXTRA_ARGS=( -Dprism.order=sw
                   -Dphoebus.user="${RUNDIR}/user"
                   -Duser.home="${RUNDIR}" )
  PHX_EXTRA_ARGS=( -server "${INSTANCE_PORT}" -settings "${RUNDIR}/settings.ini" )
  log "isolated runtime: DISPLAY=${DEMO_DISPLAY}, user dir=${RUNDIR}/user, -server ${INSTANCE_PORT}"
fi

# --- 1. soft IOC ---------------------------------------------------------------
if [[ "${START_IOC}" == 1 ]]; then
  log "starting soft IOC (DEMO:*) …"
  "${IOC_CMD[@]}" "${IOC}" >/tmp/osprey_demo_ioc.log 2>&1 &
  IOC_PID=$!
  sleep 4
  kill -0 "${IOC_PID}" 2>/dev/null || { echo "ERROR: IOC exited; see /tmp/osprey_demo_ioc.log"; exit 1; }
  log "IOC up (pid ${IOC_PID})"
else
  log "soft IOC leg skipped (INSTANCE=${INSTANCE}, START_IOC=${START_IOC}) — sharing instance 1's IOC on CA ${CA_SERVER_PORT}"
fi

# --- 2. Phoebus + bridge -------------------------------------------------------
# -Djca.use_env=true makes Phoebus's pure-Java CA honor the EPICS_CA_* env vars
# above (by default it ignores them and uses auto_addr_list=true / LAN broadcast,
# so it never probes the loopback-bound soft IOC). With it, Phoebus searches
# 127.0.0.1 and connects to DEMO:*.
#
# Launch with an explicit classpath (jar + sibling lib/*) instead of -jar: the
# manifest Class-Path pins the JavaFX jars of the BUILD platform, so a product
# built on macOS never finds the Linux natives on appsdev2. With -cp, whatever
# platform jars actually sit in lib/ define the runtime set (missing manifest
# entries are ignored per the JAR spec), so swapping lib/ jars is sufficient.
PHOEBUS_LIB="$(dirname "${PHOEBUS_JAR}")/lib"
if [[ -d "${PHOEBUS_LIB}" ]]; then
  PHOEBUS_LAUNCH=( -cp "${PHOEBUS_JAR}:${PHOEBUS_LIB}/*" org.phoebus.product.Launcher )
else
  PHOEBUS_LAUNCH=( -jar "${PHOEBUS_JAR}" )
fi
log "launching Phoebus with the demo panel …"
java -Djca.use_env=true ${JVM_EXTRA_ARGS[@]+"${JVM_EXTRA_ARGS[@]}"} \
     "${PHOEBUS_LAUNCH[@]}" ${PHX_EXTRA_ARGS[@]+"${PHX_EXTRA_ARGS[@]}"} \
     -resource "${PANEL_RES}" \
     >"/tmp/osprey_demo_phoebus${LOG_SUFFIX}.log" 2>&1 &
PHOEBUS_PID=$!

log "waiting for the bridge at ${BASE_URL}/displays …"
ready=0
for _ in $(seq 1 60); do
  if curl -fsS "${BASE_URL}/displays" >/dev/null 2>&1; then ready=1; break; fi
  kill -0 "${PHOEBUS_PID}" 2>/dev/null || { echo "ERROR: Phoebus exited; see /tmp/osprey_demo_phoebus${LOG_SUFFIX}.log"; exit 1; }
  sleep 2
done
[[ "${ready}" == 1 ]] || { echo "ERROR: bridge never became ready"; exit 1; }

cat <<EOF

[demo]  ✓ Stack is UP$( [[ "${INSTANCE}" != 1 ]] && echo " (instance ${INSTANCE})" ).
[demo]    • Soft IOC   : DEMO:* on Channel Access (127.0.0.1$( [[ "${ISOLATED}" == 1 ]] && echo ":${CA_SERVER_PORT}" ))$( [[ "${START_IOC}" != 1 ]] && echo "  [shared — started by instance 1]" )
[demo]    • Phoebus    : showing osprey_demo.bob$( [[ "${ISOLATED}" == 1 ]] && echo " (isolated: DISPLAY ${DEMO_DISPLAY} @ ${DEMO_GEOMETRY}, -server ${INSTANCE_PORT})" )
[demo]    • Bridge     : ${BASE_URL}$( [[ "${ISOLATED}" == 1 ]] && echo "
[demo]    • VNC stream : 127.0.0.1:${RFB_PORT}  (display ${DEMO_DISPLAY})" )$( [[ "${STREAM_ENABLED}" == 1 ]] && echo "
[demo]    • noVNC      : http://127.0.0.1:${WEB_VNC_PORT}/vnc.html?autoconnect=1&resize=scale
[demo]                   (the OSPREY web terminal's PHOEBUS$( [[ "${INSTANCE}" != 1 ]] && echo "${INSTANCE}" ) panel embeds this via /panel/phoebus$( [[ "${INSTANCE}" != 1 ]] && echo "${INSTANCE}" ))" )
[demo]
[demo]  Next: in a SEPARATE terminal, open your built phoebus-demo project and run:
[demo]      claude
[demo]      > run the phoebus walkthrough
[demo]
[demo]  Quick manual check (this terminal works too):
[demo]      curl -s ${BASE_URL}/perceive?display=active | jq '.widgets[].name'
[demo]
[demo]  Press Ctrl-C to tear down the stack.
EOF

wait "${PHOEBUS_PID}"
