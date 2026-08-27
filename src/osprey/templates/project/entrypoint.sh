#!/bin/sh
#
# OSPREY container entrypoint.
#
# Five steps, in this order and no other:
#
#   1. Regen      Re-render the Claude Code artifacts that config.yml drives,
#                 and only those that have actually drifted.
#   2. Restore    Put volume-owned scaffold bodies back into the render, so the
#                 agent runs the operator's claimed artifacts rather than the
#                 framework's originals.
#   3. Join       Give this image an /etc/group entry for the group that owns
#                 each bind-mounted directory the render named, and add
#                 `osprey` to it — because `gosu` re-derives the dropped
#                 process's supplementary groups from that file and discards
#                 whatever the runtime granted the initial process.
#   4. Hand back  Return the state zone to the `osprey` user, because steps 1
#                 and 2 wrote into it as root.
#   5. Drop       Hand the container's real command to the unprivileged
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
# Every diagnostic goes to stderr, without exception. This entrypoint runs in
# front of whatever command the image was given, and that command's stdout is
# its own: `docker run <image> whoami` must print `osprey` and nothing else,
# and anything that reads a container command's output — a probe, a version
# query, a JSON payload — breaks the moment a progress line is prepended to it.
# stderr keeps the diagnostics visible in `docker logs`, which interleaves both
# streams, while leaving the command's stdout untouched.

log() {
    printf '%s [osprey-entrypoint] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" >&2
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

# OSPREY_AUDIT_WRITER is a PER-COMMAND prefix on the interpreter invocation,
# never a shell `export`. It marks the records this root phase emits so they
# land in the writer's `maintenance` file rather than the app's — one uid per
# file, which is what makes an audit trail readable off disk. An `export` here
# would set it for the whole shell, and this function runs in that same shell:
# the marker would still be set at `exec gosu` below, gosu preserves the
# environment across the drop, and every record the app wrote afterwards would
# claim to come from this phase. The prefix scopes it to the one command that
# should carry it; the `unset` before the drop is the belt to this brace.
run_maintenance() {
    OSPREY_AUDIT_WRITER=maintenance "$PYTHON" - "$RENDER_DIR" <<'PY'
import sys
from datetime import datetime, timezone
from pathlib import Path

render_dir = Path(sys.argv[1])


def log(message):
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"{stamp} [osprey-entrypoint] {message}", file=sys.stderr, flush=True)


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

# ── mounted-group membership ─────────────────────────────────────────────────
# `gosu` re-derives the dropped process's supplementary groups from this
# image's /etc/group. A group the RUNTIME granted — compose's `group_add:` —
# reaches the container's initial (root) process and is discarded at the drop,
# so it grants the process that actually serves requests nothing here. What
# `group_add:` still covers is every topology where that drop never happens:
# a container started with `--user`, which takes main()'s non-root branch and
# execs the command directly, so the runtime's grant is the serving process's
# only membership — and any image that runs no entrypoint of ours at all. This
# step is the grant that survives `gosu` in framework-built images started the
# normal way. The two mechanisms cover different processes; neither replaces
# the other.
#
# What needs it: the audit subdir and the facility bundle are BIND MOUNTS owned
# by a host group (setgid 2770, the group inherited from the host and never
# invented). The dropped process reaches them through MEMBERSHIP, not
# ownership — which is the whole reason the ownership sweep below leaves the
# audit mount alone. So the group has to exist in /etc/group and `osprey` has
# to be in it before the drop, and this is the only place that can be true.
#
# Which directories: exactly the ones the render NAMED, never guessed. This
# script is POSIX sh and reads no config; each service's compose
# `environment:` carries OSPREY_AUDIT_DIR (its own audit subdir) and, where a
# bundle is mounted, OSPREY_FACILITY_BUNDLE_DIR. Iterating what those name and
# skipping any that is unset or not a directory is what lets one script serve a
# web terminal, a dispatch worker with no bundle mount, and a bare
# `docker run` with neither, without a single special case.
#
# The gid is STATted off the mounted directory rather than passed in as env: a
# gid chosen at render time is the host's gid THEN, while the mount's is the
# only one that is true at start — and there is no second place to keep in step.

#: The gids joined so far this run. The audit subdir and the bundle commonly
#: share a group, and joining it twice is noise at best.
JOINED_GIDS=''

join_mounted_group() {
    _var=$1
    _dir=$2

    # Unset is an ordinary topology, not an error: a dispatch worker mounts no
    # bundle, a bare `docker run` mounts neither. A variable that NAMES a path
    # this container does not have is worth one line, because that is the shape
    # a mistyped or dropped bind takes — but it is still not fatal.
    [ -n "$_dir" ] || return 0
    if [ ! -d "$_dir" ]; then
        log "$_var=$_dir is not a directory in this container; no group to join"
        return 0
    fi

    if ! _gid=$(stat -c '%g' "$_dir" 2> /dev/null); then
        _gid=''
    fi
    case "$_gid" in
        '' | *[!0-9]*)
            log "WARNING: could not read the owning group of $_var=$_dir;"
            log "         the osprey user may be unable to write it after the drop."
            return 0
            ;;
    esac

    # Refuse the root group and the system range, loudly and without failing
    # the boot. Joining a gid below 100 would hand the agent's user the root
    # group — or a system group like `disk` or `shadow` — on the strength of a
    # mount's metadata alone, so a mount that presents one degrades to
    # non-group access instead: the container still starts, the writes fail
    # visibly, and nobody's privileges grew.
    #
    # This is a PRIVILEGE floor, not a remap detector. An ownership remap
    # (Docker Desktop's file sharing does this) is one common way a bind comes
    # to look root-owned inside the container, and the warning says so because
    # that is the fix the operator needs — but a remap surfaces just as often
    # ABOVE the floor, as nobody/nogroup (65534) or the 32-bit overflow gid,
    # and those pass this branch. They are caught one level down instead:
    # `groupadd` refuses a gid over GID_MAX and the join warns, which is why
    # that warning names the case too.
    if [ "$_gid" -lt 100 ]; then
        log "WARNING: $_var=$_dir is owned by gid $_gid — the root group or a system"
        log "         group. REFUSING to add osprey to it. A bind mount that looks"
        log "         root-owned inside the container usually means the host's"
        log "         ownership was remapped (Docker Desktop does this), not that"
        log "         gid $_gid is the group the deployment meant to share. Writes to"
        log "         that path will fail after the privilege drop; fix the host"
        log "         directory's group rather than granting this one."
        return 0
    fi

    for _seen in $JOINED_GIDS; do
        if [ "$_seen" = "$_gid" ]; then
            return 0
        fi
    done

    # An /etc/group entry for the gid, under whatever name it already has:
    # `gosu` carries gids, and the name exists only because /etc/group has a
    # column for one. `getent` is the lookup because this image resolves groups
    # from files alone, which is the same file gosu itself parses.
    if _entry=$(getent group "$_gid" 2> /dev/null) && [ -n "$_entry" ]; then
        _group=${_entry%%:*}
        # A line whose name column is empty would reach usermod as
        # `--groups ""`, and the failure below would read as a usermod bug
        # rather than as the corrupt group file it is.
        if [ -z "$_group" ]; then
            log "WARNING: /etc/group has a malformed entry for gid $_gid ($_var=$_dir);"
            log "         the osprey user will not be able to write that mount."
            return 0
        fi
    else
        # groupadd/usermod, matching the Dockerfile's groupadd/useradd — the
        # `passwd` tools are in the base image; `adduser` need not be.
        _group="osprey-mount-$_gid"
        if ! groupadd --gid "$_gid" "$_group" > /dev/null 2>&1; then
            log "WARNING: no group for gid $_gid and could not create one ($_var=$_dir);"
            log "         the osprey user will not be able to write that mount. A gid"
            log "         above GID_MAX (65534 = nobody/nogroup, or 4294967295) is"
            log "         refused by groupadd and means the mount's ownership was"
            log "         remapped, not that a group is missing."
            return 0
        fi
    fi

    # Idempotent by construction: `--append` re-adds an existing member without
    # complaint, so a restarted container repeats this step harmlessly.
    if usermod --append --groups "$_group" osprey > /dev/null 2>&1; then
        # Recorded only now that the membership exists, so a second directory
        # sharing this gid is skipped as "already joined" only when there is
        # something to skip — a failed join must not silence the next one.
        JOINED_GIDS="$JOINED_GIDS $_gid"

        # Membership is the mechanism, not the goal: a 2700 directory, a
        # read-only bind and an id-mapped mount all accept the usermod above
        # and still refuse the write. Ask the question the audit trail
        # actually depends on, as the user that will be asking it — `gosu` is
        # already known present (main() dies without it) and re-reads
        # /etc/group, so this sees the membership just granted. Fail-open like
        # every other arm here: the container boots either way, but its log
        # says which of the two it is rather than asserting a capability it
        # never tested.
        if gosu osprey test -w "$_dir" 2> /dev/null; then
            log "joined osprey to group $_group (gid $_gid) for $_var=$_dir"
        else
            log "WARNING: osprey is in group $_group (gid $_gid) but still cannot write"
            log "         $_var=$_dir. Check the directory's permission bits (it has to"
            log "         be group-writable, 2770) and that the bind is not read-only;"
            log "         records for this identity will be dropped, silently, because"
            log "         the writer never raises."
        fi
    else
        log "WARNING: could not add osprey to group $_group (gid $_gid) for $_var;"
        log "         it will not be able to write $_dir after the privilege drop."
    fi
}

join_mounted_groups() {
    JOINED_GIDS=''
    join_mounted_group OSPREY_AUDIT_DIR "${OSPREY_AUDIT_DIR:-}"
    join_mounted_group OSPREY_FACILITY_BUNDLE_DIR "${OSPREY_FACILITY_BUNDLE_DIR:-}"
}

# ── state-zone hand-back ─────────────────────────────────────────────────────
# Hand the state zone back before dropping. Both maintenance steps run as root
# and both WRITE into `var/`: the scaffold restore's reserved-path refusals are
# appended to the audit ledger — under `var/audit/<identity>/maintenance.jsonl`,
# because the marker above routes everything this phase records there — and on a
# fresh deployment root is the first writer, so the file is created root-owned
# 0644. The server then runs as `osprey` and records its own refusals under the
# surface that decided; the marker is what keeps the two out of one file, since
# the app user could not append to a root-owned one and the refusal recorder
# never raises, so every refusal after that would be dropped in silence. An
# audit log only root can write is worse than none, because it looks like one.
#
# Scoped to the whole zone rather than that one file: `var/` is the agent
# user's tree by construction (the image chowns it wholesale), so root leaving
# it that way is the invariant. Any future root-run startup step that writes
# UNDER var/ inherits this; a step that writes the claude-config volume or
# $HOME would need its own hand-back, because neither is here.
#
# But only the paths that are actually wrong: `chown -R` would rewrite an
# operator's bind-mounted storage under var/ on every single start, and
# deliberate foreign ownership there is a choice, not damage. `find ! -user
# osprey` narrows it to what root left behind — with the caveat that it
# matches ALL foreign ownership, so any other deliberately foreign-owned path
# under var/ needs a prune of its own, exactly as the audit subdir has one.
# Fails open like the maintenance steps —
# a container that will not start is worse than one whose audit log needs a
# manual chown.
#
# ONE deliberate exception, pruned by name: the BIND-MOUNTED AUDIT SUBDIR that
# OSPREY_AUDIT_DIR points at (`var/audit/<identity>/`, the operator's host
# directory, setgid 2770). It is excluded because the dropped process reaches
# it through GROUP MEMBERSHIP — the join above — and never through ownership,
# so there is nothing to hand back. Chowning it would rewrite a host directory
# the operator owns on every start, break host-side `osprey reset
# --purge-audit`, and blur the one property that makes the trail legible: root
# and `osprey` never share ownership of a file in the audit zone, so the uid on
# a record is evidence of who wrote it. The prune covers the subdir's contents
# too, which is the point — root's `maintenance` records stay root's.
hand_back_state_zone() {
    [ -d "$STATE_DIR" ] || return 0

    # Positional parameters are function-local in POSIX sh, so this builds the
    # optional prune expression without touching main()'s "$@" (the command).
    set --
    if [ -n "${OSPREY_AUDIT_DIR:-}" ] && [ -d "${OSPREY_AUDIT_DIR:-}" ]; then
        set -- -path "$OSPREY_AUDIT_DIR" -prune -o
    fi

    find "$STATE_DIR" "$@" ! -user osprey -exec chown osprey:osprey {} + 2> /dev/null \
        || log "WARNING: could not hand $STATE_DIR back to the osprey user; the app may be unable to write it"
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

    # Groups before the drop, because `gosu` reads /etc/group and not this
    # process's supplementary groups. Fails open like the maintenance steps,
    # and for the same reason: a mount whose group cannot be joined is a
    # container that logs a warning and cannot write one directory, while a
    # container that refuses to start over it cannot write anything at all.
    join_mounted_groups \
        || log "WARNING: the mounted-group step exited non-zero; continuing to the privilege drop"

    hand_back_state_zone

    # Belt to the per-command prefix on the maintenance invocation: nothing
    # below this line may carry the writer marker across the drop, because the
    # app's records must not claim to have come from the root phase. Harmless
    # if it was never set — `unset` on an unset name succeeds even under -u.
    unset OSPREY_AUDIT_WRITER

    log "dropping privileges to the osprey user"
    exec gosu osprey "$@"
}

main "$@"
