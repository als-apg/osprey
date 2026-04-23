#!/usr/bin/env bash
# =============================================================================
# ${config.facility.name} — deploy.sh
# =============================================================================
#
# Pulls the latest code + pre-built container images and (re)starts everything.
# Idempotent: run it as many times as you like.
#
# Usage:
#   ./scripts/deploy.sh           # Pull images + restart changed containers
#   ./scripts/deploy.sh --clean   # compose down first, then pull + start
#   ./scripts/deploy.sh --nuke    # Remove ALL containers, images, volumes — full rebuild
#
# The main() wrapper is REQUIRED. `git pull` rewrites this file mid-run, and
# bash reads scripts lazily by byte offset. Without the wrapper, byte offsets
# shift after the pull and bash skips or mis-reads sections.
# =============================================================================

set -euo pipefail

# IF MODULE web_terminals.enabled
# ── Per-user CLAUDE.md seed (helper function so `local` is legal) ─────
# claude-config is a named volume; on first mount it's empty and owned by
# root. chown it to the dispatch user (so osprey web can write session
# state) and stream the per-user CLAUDE.md in via podman exec. Idempotent
# across redeploys — this replaces the file every time.
seed_web_terminal() {
  local user="$1"
  local container="${config.facility.prefix}-web-${user}"
  if ! ${config.runtime.engine} container exists "$container" 2>/dev/null; then
    echo "  (skipped ${user}: container not ready)"
    return
  fi
  cat docker/web-terminal-context/base.md \
      docker/web-terminal-context/"${user}".md 2>/dev/null \
    | ${config.runtime.engine} exec -u 0 -i "$container" sh -c '
        chown dispatch:dispatch /data/claude-config
        cat > /data/claude-config/CLAUDE.md
        chown dispatch:dispatch /data/claude-config/CLAUDE.md
      '
}
# END IF

# IF registry.external_projects
# ── External-project pull (helper) ────────────────────────────────────
# Uses indirect env lookup (`${!var:-}`) so an unset deploy token doesn't
# trip `set -u`. A warning beats a cryptic abort.
pull_external_image() {
  local name="$1" url_with_image="$2" token_var="$3"
  local token="${!token_var:-}"
  if [[ -n "$token" ]]; then
    echo "Pulling ${name} image (external registry)..."
    ${config.runtime.engine} pull --creds "deploy:${token}" "$url_with_image"
  else
    echo "WARNING: ${token_var} not set — ${name} image may be stale"
  fi
}
# END IF

main() {
  local CLEAN=false NUKE=false
  for arg in "$@"; do
    case "$arg" in
      --clean) CLEAN=true ;;
      --nuke)  NUKE=true ;;
      -h|--help)
        sed -n '1,20p' "$0"
        exit 0
        ;;
    esac
  done

  # ── Compose file list from facility-config.yml ────────────────────────
  # Built as an array so the expansion survives spaces in paths and respects
  # -f <path> pairing. The FOR-each renderer emits one line per compose file.
  local -a COMPOSE_FILES=()
  # FOR each in runtime.compose_files
  COMPOSE_FILES+=(-f "${each}")
  # END FOR

  # ── Step 1: Update local repo ─────────────────────────────────────────
  # Everything else (compose files, scripts, nginx config, triggers) comes
  # from this pull. If this fails, bail — we don't want to deploy stale code.
  echo "──────────────────────────────────────────────────────────────────"
  echo "Step 1: Pulling latest code from ${config.gitlab.remote_name}/${config.gitlab.default_branch}..."
  echo "──────────────────────────────────────────────────────────────────"
  # IF network.http_proxy
  # GitLab host must bypass the proxy — without NO_PROXY the pull goes
  # through the corporate proxy and fails auth.
  NO_PROXY=${config.gitlab.host} git pull --ff-only ${config.gitlab.remote_name} ${config.gitlab.default_branch}
  # ELSE
  git pull --ff-only ${config.gitlab.remote_name} ${config.gitlab.default_branch}
  # END IF

  # ── Step 2: Source .env.production ────────────────────────────────────
  # Values in .env.production are unquoted (podman env_file format). Re-quote
  # them before `eval` so multi-word values (e.g., EPICS_CA_ADDR_LIST) survive
  # word-splitting in the shell that consumes them here.
  if [[ ! -f .env.production ]]; then
    echo "ERROR: .env.production not found. Copy .env.template and fill in secrets:"
    echo "         cp .env.template .env.production"
    echo "         chmod 600 .env.production"
    echo "         \$EDITOR .env.production"
    exit 1
  fi
  set -a
  eval "$(sed '/^#/d;/^$/d;s/=\(.*\)/="\1"/' .env.production)"
  set +a

  # IF MODULE shared_disk.enabled
  # ── Step 2b: Pre-flight shared-disk check ────────────────────────────
  # Containers that bind-mount a missing host path will still START but fail
  # at first read with an obscure error. Check upfront so the failure mode
  # is "deploy aborts clearly" rather than "services are up but broken."
  if [[ ! -d "${config.modules.shared_disk.host_path}" ]]; then
    echo "ERROR: shared_disk.host_path does not exist on this server:"
    echo "         ${config.modules.shared_disk.host_path}"
    echo "Mount the filesystem (check /etc/fstab) or correct modules.shared_disk.host_path."
    exit 1
  fi
  # END IF

  # ── Step 3: Teardown (only if --clean or --nuke) ──────────────────────
  if $NUKE; then
    echo
    echo "══════════════════════════════════════════════════════════════════"
    echo "=== NUKE: removing all containers, images, volumes, networks ==="
    echo "══════════════════════════════════════════════════════════════════"
    ${config.runtime.compose_command} "${COMPOSE_FILES[@]}" down --volumes || true
    # Compose down occasionally leaves stragglers when containers were
    # created outside the expected labels. Force-remove by project label.
    local remaining
    remaining=$(${config.runtime.engine} ps -a \
      --filter "label=io.podman.compose.project=$(basename "$(pwd)")" -q 2>/dev/null || true)
    if [[ -n "$remaining" ]]; then
      echo "Compose down left containers behind — force-removing..."
      echo "$remaining" | xargs ${config.runtime.engine} rm -f 2>/dev/null || true
    fi
    # Scope image removal to our project_path to avoid nuking sibling
    # external_projects images that happen to share the registry host.
    echo "Removing project images..."
    ${config.runtime.engine} images --format '{{.Repository}}:{{.Tag}}' \
      | grep -E "^${config.registry.url}/" \
      | xargs -r ${config.runtime.engine} rmi -f 2>/dev/null || true
    ${config.runtime.engine} network prune -f 2>/dev/null || true
    echo "Nuke complete. Rebuilding from scratch."
  elif $CLEAN; then
    echo
    echo "Stopping all containers (--clean)..."
    ${config.runtime.compose_command} "${COMPOSE_FILES[@]}" down || true
  fi

  # ── Step 4: Registry login + pull external images ────────────────────
  echo
  echo "──────────────────────────────────────────────────────────────────"
  echo "Step 4: Logging in to registry and pulling images..."
  echo "──────────────────────────────────────────────────────────────────"

  # The renderer substitutes ${env.X} → $X, so after scaffolding the line
  # below becomes e.g. `echo "$ALS_GITLAB_TOKEN" | ...`. The .env.production
  # source step above has already exported the variable, so bash finds it.
  echo "${env.${config.gitlab.token_env_var}}" \
    | ${config.runtime.engine} login --username deploy --password-stdin ${config.registry.url%%/*}

  # IF registry.external_projects
  # External projects are hosted in sibling GitLab projects with their own
  # registries. The main deploy token does not have cross-project registry
  # access, so pull each explicitly using its dedicated deploy token.
  # Without this, `compose pull` silently skips them on auth failure and you
  # end up running stale images — deploying with --nuke won't fix it either.
  # FOR each in registry.external_projects
  pull_external_image "${each.name}" "${each.url}/${each.image}" "${each.token_env_var}"
  # END FOR
  # END IF

  # ── Step 5: Pull + bring up ──────────────────────────────────────────
  ${config.runtime.compose_command} "${COMPOSE_FILES[@]}" pull
  ${config.runtime.compose_command} "${COMPOSE_FILES[@]}" up -d

  # IF MODULE web_terminals.enabled
  # ── Step 5b: Seed per-user CLAUDE.md into each web-terminal volume ───
  echo
  echo "Seeding per-user CLAUDE.md into claude-config volumes..."
  # FOR user in modules.web_terminals.users
  seed_web_terminal "${user}"
  # END FOR
  # END IF

  # ── Step 6: Advisory verify ──────────────────────────────────────────
  echo
  echo "Containers started. Running post-deploy verification (advisory)..."
  local SCRIPT_DIR
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  # Run verify.sh; its `exit 0` at the end makes this guard belt-and-suspenders
  # rather than a silent failure mask. If verify.sh ever returns non-zero it
  # means the script itself broke (not a probe failure), and we want to see it.
  "$SCRIPT_DIR/verify.sh"
}

main "$@"
