#!/usr/bin/env bash
# =============================================================================
# ${config.facility.name} — verify.sh
# =============================================================================
#
# Post-deploy advisory health check. Runs automatically at the end of
# deploy.sh; can also be run standalone:
#
#   ./scripts/verify.sh                           # full check
#   ./scripts/verify.sh epics archiver            # restrict to specific categories
#
# IMPORTANT: this script ALWAYS exits 0. Verification is advisory — a failed
# probe tells you something to look at, but never blocks a deploy. If you
# want a deploy to abort on health failure, wrap the call in deploy.sh;
# don't add `exit 1` here. Operations learned the hard way that letting
# verify.sh fail the deploy created perverse incentives to mask problems
# rather than investigate them.
#
# No -e: individual probe failures must not abort the rest of the suite.
# The main() wrapper is required so the per-user / per-sidecar FOR-each
# expansions can use `local` (bash only allows `local` inside functions).
# =============================================================================
set -uo pipefail

HEALTH_URL="http://localhost:${config.ports.integration_tests}/checks"
TIMEOUT=30
POLL_INTERVAL=2

BOLD="\033[1m"
GREEN="\033[32m"
YELLOW="\033[33m"
RED="\033[31m"
DIM="\033[90m"
RESET="\033[0m"

# IF MODULE web_terminals.enabled
# ── Per-user web-terminal writability probe ──────────────────────────
# Called from the FOR-each expansion below. Function form so `local` works.
probe_web_terminal() {
  local user="$1"
  local container="${config.facility.prefix}-web-${user}"
  if ! ${config.runtime.engine} container exists "$container" 2>/dev/null; then
    echo -e "  ${DIM}-${RESET} ${DIM}${container}: not running${RESET}"
    return
  fi
  local probe_out
  probe_out=$(${config.runtime.engine} exec -u 1000:1000 "$container" bash -c '
    p1=/app/${config.facility.prefix}-assistant/_agent_data/.writeprobe
    p2=/data/claude-config/.writeprobe
    touch "$p1" && rm "$p1" && touch "$p2" && rm "$p2" && echo OK
  ' 2>&1)
  if [[ "$probe_out" == "OK" ]]; then
    echo -e "  ${GREEN}✓${RESET} ${container}: dispatch can write to _agent_data + claude-config"
  else
    echo -e "  ${RED}✗${RESET} ${container}: write failed — ${probe_out}"
  fi
}
# END IF

# IF MODULE event_dispatcher.enabled
# ── Per-sidecar health probe ─────────────────────────────────────────
probe_sidecar() {
  local port="$1"
  local sc_health
  sc_health=$(curl -sf --max-time 3 "http://localhost:${port}/health" 2>/dev/null)
  if [[ -n "$sc_health" ]]; then
    local total pending
    total=$(echo "$sc_health" | python3 -c \
      "import sys,json; d=json.load(sys.stdin); print(d.get('total_runs',0))" 2>/dev/null)
    pending=$(echo "$sc_health" | python3 -c \
      "import sys,json; d=json.load(sys.stdin); print(d.get('pending_runs',0))" 2>/dev/null)
    echo -e "  ${GREEN}✓${RESET} Sidecar :${port}: ${total} runs (${pending} pending)"
  else
    echo -e "  ${DIM}-${RESET} ${DIM}Sidecar :${port}: not reachable${RESET}"
  fi
}
# END IF

main() {
  # ── Parse args into ?categories= query string ────────────────────────
  if [[ $# -gt 0 ]]; then
    local cats
    cats=$(IFS=,; echo "$*")
    HEALTH_URL="${HEALTH_URL}?categories=${cats}"
  fi

  # IF MODULE web_terminals.enabled
  # ── Web-terminal named-volume writability ──────────────────────────
  # _agent_data and claude-config are both named volumes. On rootless podman
  # with NFS home dirs, it's easy for UID mapping to go wrong and the dispatch
  # user to lose write access silently. Probe from inside each container.
  echo -e "\n${BOLD}── Web Terminal Volume Writability ──${RESET}\n"
  # FOR user in modules.web_terminals.users
  probe_web_terminal "${user}"
  # END FOR
  echo ""
  # END IF

  # IF MODULE event_dispatcher.enabled
  # ── Dispatch pipeline pre-checks ───────────────────────────────────
  echo -e "\n${BOLD}── Dispatch Pipeline ──${RESET}\n"

  local disp_health triggers pool_running
  disp_health=$(curl -sf --max-time 5 \
    "http://localhost:${config.modules.event_dispatcher.port}/health" 2>/dev/null)
  if [[ -n "$disp_health" ]]; then
    triggers=$(echo "$disp_health" | python3 -c \
      "import sys,json; d=json.load(sys.stdin); print(d.get('trigger_count','?'))" 2>/dev/null)
    pool_running=$(echo "$disp_health" | python3 -c \
      "import sys,json; d=json.load(sys.stdin); p=d.get('pool',{}); \
       print(f\"{p.get('running',0)}/{p.get('max','?')}\")" 2>/dev/null)
    echo -e "  ${GREEN}✓${RESET} Dispatcher: ${triggers} trigger(s), pool ${pool_running}"
  else
    echo -e "  ${RED}✗${RESET} Dispatcher: not reachable (localhost:${config.modules.event_dispatcher.port})"
  fi

  # Sidecar ports: sidecar_port_base .. sidecar_port_base + sidecar_count - 1
  # FOR i in 0..${config.modules.event_dispatcher.sidecar_count - 1}
  probe_sidecar "$((${config.modules.event_dispatcher.sidecar_port_base} + ${i}))"
  # END FOR
  echo ""
  # END IF

  # IF MODULE web_terminals.enabled
  # ── Nginx landing page ─────────────────────────────────────────────
  echo -e "\n${BOLD}── Web Terminal Landing ──${RESET}\n"
  if curl -sf --max-time 5 -o /dev/null \
       "http://localhost:${config.modules.web_terminals.nginx_port}/" 2>/dev/null; then
    echo -e "  ${GREEN}✓${RESET} Nginx: landing page reachable (localhost:${config.modules.web_terminals.nginx_port})"
  else
    echo -e "  ${RED}✗${RESET} Nginx: landing page NOT reachable"
  fi
  echo ""
  # END IF

  # IF MODULE shared_disk.enabled
  # ── Shared-disk mount probe ────────────────────────────────────────
  # Confirm at least one service in services_to_mount actually sees the
  # mount — catches silent mount failures where the container starts but
  # the bind didn't apply.
  echo -e "\n${BOLD}── Shared Disk ──${RESET}\n"
  local probe_service probe_container
  probe_service="${config.modules.shared_disk.services_to_mount[0]}"
  probe_container="${config.facility.prefix}-mcp-${probe_service}"
  if ${config.runtime.engine} exec "$probe_container" \
       test -d "${config.modules.shared_disk.container_path}" 2>/dev/null; then
    echo -e "  ${GREEN}✓${RESET} ${probe_container}: ${config.modules.shared_disk.container_path} present"
  else
    echo -e "  ${RED}✗${RESET} ${probe_container}: ${config.modules.shared_disk.container_path} NOT mounted"
  fi
  echo ""
  # END IF

  # ── Wait for integration-tests /health endpoint ────────────────────
  # Poll /health (fast, no side effects) before hitting /checks, so we don't
  # emit a noisy error while containers are still starting up.
  local elapsed=0
  while ! curl -sf -o /dev/null \
            "http://localhost:${config.ports.integration_tests}/health" 2>/dev/null; do
    if [[ $elapsed -ge $TIMEOUT ]]; then
      echo -e "${YELLOW}⚠${RESET} integration-tests /health did not respond within ${TIMEOUT}s — skipping full report"
      return 0
    fi
    if [[ $elapsed -eq 0 ]]; then
      echo "Waiting for integration-tests /health endpoint..."
    fi
    sleep "$POLL_INTERVAL"
    elapsed=$((elapsed + POLL_INTERVAL))
  done

  # ── Fetch full health report from integration-tests MCP server ─────
  local health_json
  health_json=$(curl -sf --max-time 300 "$HEALTH_URL" 2>/dev/null)
  if [[ -z "$health_json" ]]; then
    echo -e "${YELLOW}⚠${RESET} Empty response from $HEALTH_URL — skipping formatted report"
    return 0
  fi

  # ── Format with Python ─────────────────────────────────────────────
  # Inline Python is used here (rather than jq or shell parsing) because the
  # report structure is nested and we need grouping by category. jq works but
  # is not universally present on RHEL-like hosts; python3 always is.
  export HEALTH_JSON="$health_json"
  python3 << 'PYEOF'
import json, os
from collections import OrderedDict

data = json.loads(os.environ["HEALTH_JSON"])
results = data["results"]

GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
DIM    = "\033[90m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

ICONS = {
    "ok":      f"{GREEN}\u2713{RESET}",
    "warning": f"{YELLOW}\u26a0{RESET}",
    "error":   f"{RED}\u2717{RESET}",
    "skip":    f"{DIM}-{RESET}",
}

groups = OrderedDict()
for r in results:
    groups.setdefault(r["category"], []).append(r)

print(f"\n\u2500\u2500\u2500 {BOLD}Integration Test Results{RESET} " + "\u2500" * 26 + "\n")

for cat, checks in groups.items():
    print(f"{BOLD}{cat}{RESET}")
    for c in checks:
        icon = ICONS.get(c["status"], "?")
        msg = c["message"]
        if c["status"] == "skip":
            line = f"  {icon} {DIM}{msg}{RESET}"
        else:
            line = f"  {icon} {msg}"
            val = c.get("value", "")
            lat = c.get("latency_ms", 0)
            if val:
                line += f"  [{val}]"
            if lat:
                line += f"  ({lat}ms)"
        print(line)
    print()

ok        = data.get("ok", 0)
total     = data.get("total", 0)
warnings  = data.get("warnings", 0)
errors    = data.get("errors", 0)
skipped   = total - ok - warnings - errors

parts = []
if warnings: parts.append(f"{warnings} warning{'s' if warnings != 1 else ''}")
if errors:   parts.append(f"{errors} error{'s' if errors != 1 else ''}")
if skipped:  parts.append(f"{skipped} skipped")
extra = f" ({', '.join(parts)})" if parts else ""
summary = f"{ok}/{total} checks passed{extra}"

color = RED if errors else (YELLOW if warnings else GREEN)
print("\u2500" * 52)
print(f"{color}{summary}{RESET}")
PYEOF
}

main "$@"
exit 0
