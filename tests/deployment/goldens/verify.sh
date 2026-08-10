#!/usr/bin/env bash
# =============================================================================
# Demo Facility — post-deploy health check
# =============================================================================
# osprey-scaffold: deploy/verify
# osprey-version: OSPREY_VERSION
#
# Emitted by `osprey deploy scaffold` into the profile's project/ mirror, which
# carries it to <project>/scripts/verify.sh in every build. `osprey deploy up`
# runs it there automatically once the containers are up; you can also run it
# by hand from the project root:
#
#   ./scripts/verify.sh                    # every probe
#   ./scripts/verify.sh services           # one group
#
# ALWAYS exits 0. Verification is advisory: a failed probe tells an operator
# where to look, and must never be the reason a deploy is reported as failed.
# Do not add `exit 1` here — `osprey deploy up` ignores the exit code anyway,
# so the only thing a nonzero exit buys is a misleading log line.
#
# No `set -e`: one probe timing out must not skip the ones after it.
# =============================================================================
set -uo pipefail

GREEN=$'\033[32m'; RED=$'\033[31m'; DIM=$'\033[90m'; BOLD=$'\033[1m'; RESET=$'\033[0m'

# Probe groups, selectable as arguments. Default is all of them. Not named
# GROUPS: bash owns that name (the caller's group ids), and assigning to it
# silently does nothing — a script that tried would run no probes at all and
# still exit 0.
PROBE_GROUPS="${*:-services web}"

# ── Probe helpers ────────────────────────────────────────────────────────────

# An HTTP endpoint that answers. Used for anything speaking HTTP.
probe_http() {
  local label="$1" url="$2"
  if curl -sf --max-time 5 -o /dev/null "$url"; then
    printf '  %s✓%s %s\n' "$GREEN" "$RESET" "$label"
  else
    printf '  %s✗%s %s — no response from %s\n' "$RED" "$RESET" "$label" "$url"
  fi
}

# A TCP listener. The virtual accelerator serves EPICS Channel Access, not
# HTTP, so a connect is as far as a probe can go without an EPICS client.
probe_tcp() {
  local label="$1" host="$2" port="$3"
  if python3 -c "import socket,sys; s=socket.socket(); s.settimeout(3); \
sys.exit(s.connect_ex(('$host', $port)))" 2>/dev/null; then
    printf '  %s✓%s %s\n' "$GREEN" "$RESET" "$label"
  else
    printf '  %s✗%s %s — nothing listening on %s:%s\n' "$RED" "$RESET" "$label" "$host" "$port"
  fi
}

wants() { case " $PROBE_GROUPS " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

# ── Deployed services ────────────────────────────────────────────────────────
if wants services; then
  printf '\n%s── Services ──%s\n\n' "$BOLD" "$RESET"
  probe_tcp  'virtual-accelerator: Channel Access on 5064' 127.0.0.1 5064
  probe_http 'openobserve: telemetry store on 5080'        http://127.0.0.1:5080/healthz
  # A bare GET to an MCP endpoint is not a request it answers, so this stops at
  # the listener — the same thing the container's own healthcheck settles for.
  probe_tcp  'facility-mcp: machine status on 8200'        127.0.0.1 8200
fi

# ── Web terminal tier ────────────────────────────────────────────────────────
# One container per roster user behind nginx. A user whose terminal is down is
# a person who cannot work, so each is named rather than summarized.
if wants web; then
  printf '\n%s── Web terminals ──%s\n\n' "$BOLD" "$RESET"
  probe_http 'nginx: landing page on 9080' http://127.0.0.1:9080/
  probe_http 'alice (readonly) on 9091'    http://127.0.0.1:9091/
  probe_http 'bob (readwrite) on 9092'     http://127.0.0.1:9092/
fi

printf '\n%sProbes are advisory — a failure here does not mean the deploy failed.%s\n\n' \
  "$DIM" "$RESET"
exit 0
