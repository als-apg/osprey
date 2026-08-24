#!/bin/sh
#
# OSPREY container entrypoint.
#
# Three steps, in this order and no other:
#
#   1. Regen      Re-render the Claude Code artifacts that config.yml drives,
#                 and only those that have actually drifted.
#   2. Restore    Put volume-owned scaffold bodies back into the render, so the
#                 agent runs the operator's claimed artifacts rather than the
#                 framework's originals.
#   3. Hand back  Return the state zone to the `osprey` user, because steps 1
#                 and 2 wrote into it as root.
#   4. Drop       Hand the container's real command to the unprivileged
#                 `osprey` user and get out of the way.
#
# The ordering is the point. Steps 1 and 2 write into the render, which this
# image makes root-owned so that nothing the agent can reach may rewrite the
# files that decide what the agent is allowed to do. Only root can perform
# them, and only before the server starts — so they happen here, once, and the
# process that serves requests never has the privilege to repeat them. A
# container started with `--user osprey` skips both and says so, rather than
# failing halfway through a partial write.
#
# Both steps fail open: a regen or restore that raises is reported and the
# container still starts. A container that will not boot because an artifact
# could not be re-rendered is strictly worse than one running slightly stale
# artifacts and saying so in its logs. The privilege drop is the opposite —
# a missing `gosu` is fatal, because continuing would run the agent as root,
# which is the one outcome this entrypoint exists to prevent.
#
# POSIX sh on purpose — this runs as PID 1 in a slim image and has no bash.
set -eu

# ── configuration ────────────────────────────────────────────────────────────

# The render this entrypoint maintains is the directory the script sits in:
# `osprey build` emits it beside config.yml and .claude/, and the image copies
# the deployment repo in verbatim. Deriving the path from $0 rather than baking
# one in means the same script is correct at any /app/<name>, and there is no
# second place that has to be kept in step with the Dockerfile's COPY target.
RENDER_DIR=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd -P)

# The durable state zone, one level up from the render: the render is
# `<repo>/build` and `var/` is its sibling, which is how every runtime reader
# resolves it (utils.workspace.repo_root_for_config). The image chowns this
# whole tree to `osprey` at build time; the hand-back below restores that after
# root has written into it.
STATE_DIR=$(dirname -- "$RENDER_DIR")/var

# The interpreter that runs the maintenance step. The image installs osprey
# into its system Python (/usr/local/bin/python in the python:*-slim base), not
# into the render's .venv — a container render has none.
PYTHON="${OSPREY_ENTRYPOINT_PYTHON:-python}"

# ── logging ──────────────────────────────────────────────────────────────────

log() {
    printf '%s [osprey-entrypoint] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

die() {
    log "FATAL: $*"
    exit 1
}

# ── startup maintenance ──────────────────────────────────────────────────────
# Regen and restore are Python-side operations on framework internals, so they
# run in one interpreter rather than two: importing osprey is the expensive part
# and doing it twice would add seconds to every container start for nothing.
# Each step carries its own try/except — including its import — so a step that
# cannot even load does not take the other one down with it.
#
# The restore is deliberately the shared `restore_scaffold_bodies`, not a
# reimplementation. Its refusal to install a reserved path lives in that
# function, which means the bare-host path and this root-privileged one are
# gated by the same code; a private copy here would be a second gate to keep in
# step, and the one that runs as root is the worst place to discover a drift.

run_maintenance() {
    "$PYTHON" - "$RENDER_DIR" <<'PY'
import sys
from datetime import datetime, timezone
from pathlib import Path

render_dir = Path(sys.argv[1])


def log(message):
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"{stamp} [osprey-entrypoint] {message}", flush=True)


# 1. Drift regen. A dry run decides whether anything is re-rendered at all, so
#    a container whose config.yml has not changed rewrites no artifact and does
#    not disturb settings.json's mtime — the signal the SessionStart drift hook
#    reads. regen_if_drift keeps that contract, including the in-sync utime
#    stamp; calling regenerate_claude_code directly would not.
try:
    from osprey.cli.templates.manager import TemplateManager

    changed = TemplateManager().regen_if_drift(render_dir)
    if changed:
        log(f"regenerated {len(changed)} stale Claude Code artifact(s): {', '.join(changed)}")
    else:
        log("Claude Code artifacts are in sync with config.yml")
except Exception as exc:  # noqa: BLE001 — never block the container on regen
    log(f"WARNING: Claude Code artifact regen failed ({exc!r}); continuing with what is on disk")

# 2. Scaffold restore. The render comes back image-fresh on every container
#    recreation while the operator's claimed artifact bodies live on the
#    claude-config volume. Without this the agent would run the framework's
#    originals while the gallery displayed the operator's, and nothing would
#    report the divergence. A no-op when there is no ownership store to read.
try:
    from osprey.interfaces.web_terminal.scaffold_gallery_service import (
        restore_scaffold_bodies,
    )

    restored = restore_scaffold_bodies(render_dir)
    if restored:
        log(f"restored {len(restored)} user-owned artifact(s): {', '.join(sorted(restored))}")
    else:
        log("no user-owned artifact bodies to restore")
except Exception as exc:  # noqa: BLE001 — never block the container on restore
    log(f"WARNING: scaffold restore failed ({exc!r}); the image's own artifacts stay in place")
PY
}

# ── main ─────────────────────────────────────────────────────────────────────

main() {
    [ "$#" -gt 0 ] || die "no command to run; the image's CMD supplies one (e.g. osprey web ...)"

    log "starting: render $RENDER_DIR, command: $*"

    # Already unprivileged — someone ran the image with `--user`. Neither
    # maintenance step can write a root-owned render, and gosu cannot drop to a
    # user it is not root to become, so do the one useful thing left: run the
    # command. Loud, because the artifacts this would have refreshed are now
    # whatever the image happens to carry.
    if [ "$(id -u)" -ne 0 ]; then
        log "WARNING: running as uid $(id -u), not root; skipping the startup regen"
        log "         and scaffold restore, and running the command directly."
        log "         Derived artifacts will be whatever this image was built with."
        exec "$@"
    fi

    # Checked before the maintenance step, not after: without gosu the only
    # ways out are running the agent as root or refusing to start, and refusing
    # to start is the answer. Say so before spending time on work whose only
    # consumer is the process that is not going to launch.
    command -v gosu > /dev/null 2>&1 \
        || die "gosu is not installed; refusing to start, because the alternative is running the agent as root"
    id osprey > /dev/null 2>&1 \
        || die "no 'osprey' user in this image; refusing to start rather than run the agent as root"

    if command -v "$PYTHON" > /dev/null 2>&1; then
        run_maintenance || log "WARNING: startup maintenance exited non-zero; continuing to the privilege drop"
    else
        log "WARNING: no '$PYTHON' interpreter on PATH; skipping the startup regen and scaffold restore"
    fi

    # Hand the state zone back before dropping. Both maintenance steps above
    # run as root and both WRITE into `var/`: the scaffold restore's
    # reserved-path refusals are appended to `var/audit/protected-writes.jsonl`,
    # and on a fresh deployment root is the first writer, so the file is created
    # root-owned 0644. The server then runs as `osprey` and cannot append to it
    # — and the refusal recorder never raises, so every refusal the running app
    # makes after that is dropped in silence. An audit log only root can write
    # is worse than none, because it looks like one.
    #
    # Scoped to the whole zone rather than that one file: `var/` is the agent
    # user's tree by construction (the image chowns it wholesale), so root
    # leaving it that way is the invariant. Any future root-run startup step
    # that writes UNDER var/ inherits this; a step that writes the claude-config
    # volume or $HOME would need its own hand-back, because neither is here.
    #
    # But only the paths that are actually wrong: `chown -R` would rewrite an
    # operator's bind-mounted storage under var/ on every single start, and
    # deliberate foreign ownership there is a choice, not damage. `find ! -user`
    # touches only what root left behind. Fails open like the maintenance steps
    # — a container that will not start is worse than one whose audit log needs
    # a manual chown.
    if [ -d "$STATE_DIR" ]; then
        find "$STATE_DIR" ! -user osprey -exec chown osprey:osprey {} + 2> /dev/null \
            || log "WARNING: could not hand $STATE_DIR back to the osprey user; the app may be unable to write it"
    fi

    log "dropping privileges to the osprey user"
    exec gosu osprey "$@"
}

main "$@"
