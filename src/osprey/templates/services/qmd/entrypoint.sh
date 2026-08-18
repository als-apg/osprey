#!/bin/sh
#
# qmd search sidecar entrypoint.
#
# Three phases, in this order and no other:
#
#   1. Startup pass   Bring the index up to date -- a full rebuild if the
#                     embedder that produced the stored vectors is not the
#                     embedder baked into this image, an incremental pass
#                     otherwise.
#   2. Serve          Start the qmd MCP daemon and the forwarder that owns the
#                     routable port, but only after the startup pass has
#                     finished AND the index is provably non-empty.
#   3. Update loop    Re-index when a corpus root's marker file advances, and
#                     on a slower interval as a fallback sweep.
#
# The ordering is the safety property: nothing outside the container can reach
# a half-built index, because the port that reaches it does not exist yet.
#
# POSIX sh on purpose -- this runs as PID 1 in a slim image and has no bash.
set -eu

# ── configuration ────────────────────────────────────────────────────────────
# Everything the deployment can vary. The image supplies HOME, XDG_CACHE_HOME,
# OSPREY_QMD_MODEL_DIR, OSPREY_QMD_MODEL_IDENTITY_FILE and the QMD_*_MODEL pins;
# those are deliberately not re-set here, because they are how qmd finds the
# models baked into the image.

# Writable state: the qmd config and the SQLite index both live here. This is
# the directory the deployment should back with a named volume. It is kept
# separate from the model cache so that persisting the index does not shadow
# the 2.1 GB of baked models.
STATE_DIR="${OSPREY_QMD_STATE_DIR:-/var/lib/qmd}"

# Rendered collection config, mounted read-only by the deployment. Copied into
# place on every start, so editing the deployment's config and restarting is
# enough to change what gets indexed.
INDEX_CONFIG="${OSPREY_QMD_INDEX_CONFIG:-/etc/qmd/index.yml}"

# Fallback for running the image without a rendered config: a whitespace- or
# comma-separated list of `name=/path` pairs. Paths containing spaces are not
# supported here; use the rendered config for those.
COLLECTIONS_SPEC="${OSPREY_QMD_COLLECTIONS:-}"

# Port split. qmd hardcodes `httpServer.listen(port, "localhost")` -- there is
# no --host flag and no env override -- so the daemon only ever answers on a
# loopback address inside the container, and which family "localhost" resolves
# to depends on the environment. Either way it is unreachable from outside its
# own network namespace. INTERNAL_PORT is where the daemon actually listens;
# PORT is the routable port the forwarder owns and clients connect to.
PORT="${OSPREY_QMD_PORT:-8180}"
INTERNAL_PORT="${OSPREY_QMD_INTERNAL_PORT:-8181}"

# Fallback sweep interval. The marker file is the primary freshness trigger --
# a no-op scan costs ~12.5 s at 135,000 documents, so a short interval would
# leave the loop permanently busy discovering nothing.
UPDATE_INTERVAL="${OSPREY_QMD_UPDATE_INTERVAL:-300}"

# How often the loop looks at the marker files. This is the real freshness
# latency for a producer that touches its marker, so it is much shorter than
# the sweep interval; the check itself is a stat() per corpus root.
POLL_INTERVAL="${OSPREY_QMD_MARKER_POLL_INTERVAL:-5}"

# Marker filename, looked for at the root of every indexed corpus. Producers
# touch it after writing; the sidecar never deletes it, because deleting would
# race a producer that is mid-write.
MARKER_NAME="${OSPREY_QMD_MARKER_NAME:-.qmd-touch}"

# How long to wait for the daemon's /health to come up before giving up.
HEALTH_TIMEOUT="${OSPREY_QMD_HEALTH_TIMEOUT:-120}"

# Escape hatch: force the full rebuild path even when the embedder matches.
FORCE_REINDEX="${OSPREY_QMD_FORCE_REINDEX:-0}"

CONFIG_FILE="$STATE_DIR/.qmd/index.yml"
DB_FILE="$STATE_DIR/.qmd/index.sqlite"
IDENTITY_STAMP="$STATE_DIR/.qmd/osprey-embed-identity"

QMD_PID=""
SOCAT_PID=""
STAGE_PID=""
HEARTBEAT_PID=""

# ── logging ──────────────────────────────────────────────────────────────────

log() {
    printf '%s [qmd-sidecar] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

die() {
    log "FATAL: $*"
    exit 1
}

# Seconds as a human duration, so a 41-minute build reads as "41m12s" rather
# than as a four-digit number an operator has to divide in their head.
fmt_duration() {
    _d=$1
    if [ "$_d" -lt 60 ]; then
        printf '%ds' "$_d"
    else
        printf '%dm%02ds' $((_d / 60)) $((_d % 60))
    fi
}

# ── shutdown ─────────────────────────────────────────────────────────────────
# `docker stop` sends SIGTERM to PID 1 only, so every long-running child has to
# be killed explicitly. Long stages also run as background jobs rather than in
# the foreground: a POSIX shell defers trap handlers until the current
# foreground command returns, and a full index build is a 40-minute foreground
# command. Backgrounding them is what makes the container stoppable mid-build.

stop_child() {
    _pid=$1
    [ -n "$_pid" ] || return 0
    kill -TERM "$_pid" 2>/dev/null || true
    wait "$_pid" 2>/dev/null || true
}

on_term() {
    log "signal received; shutting down"
    stop_child "$HEARTBEAT_PID"
    # Forwarder first: stop accepting new connections before the daemon behind
    # it goes away, so in-flight clients see a closed port rather than a hang.
    stop_child "$SOCAT_PID"
    stop_child "$QMD_PID"
    stop_child "$STAGE_PID"
    log "stopped"
    exit 0
}

trap on_term TERM INT

# ── stage helpers ────────────────────────────────────────────────────────────

# Run a command as a background job and wait for it, so SIGTERM is handled
# while it runs. Returns the command's exit status.
run_stage() {
    "$@" &
    STAGE_PID=$!
    set +e
    wait "$STAGE_PID"
    _rc=$?
    set -e
    STAGE_PID=""
    return $_rc
}

# Periodic "still working" line for the stages that can run for tens of
# minutes. Without it a full rebuild looks indistinguishable from a hang.
start_heartbeat() {
    _label=$1
    (
        _t0=$(date +%s)
        while :; do
            sleep 120
            _elapsed=$(( $(date +%s) - _t0 ))
            log "  ... $_label still running ($(fmt_duration "$_elapsed") elapsed)"
        done
    ) &
    HEARTBEAT_PID=$!
}

stop_heartbeat() {
    stop_child "$HEARTBEAT_PID"
    HEARTBEAT_PID=""
}

# ── index inspection ─────────────────────────────────────────────────────────
# `qmd status` and `qmd collection list` are the only surfaces that report
# these counts before the MCP daemon is up, and the counts have to be known
# before the daemon is up: the whole point of the assertion is to refuse to
# serve an empty index. Both parsers fail towards zero, so an output format
# change reads as "cannot prove the index is populated" -- which is the safe
# direction.

status_field() {
    # $1 = status output, $2 = field label ("Total", "Vectors")
    _n=$(printf '%s\n' "$1" | awk -v want="$2:" '$1 == want { print $2; exit }')
    case "$_n" in
        '' | *[!0-9]*) printf '0' ;;
        *) printf '%s' "$_n" ;;
    esac
}

# Emit "<name> <file-count>" per configured collection.
collection_counts() {
    qmd collection list 2>/dev/null | awk '
        /^[A-Za-z0-9._-]+ \(qmd:\/\// { name = $1; next }
        $1 == "Files:" && name != "" { print name, $2; name = "" }
    '
}

# Values of every `path:` key in the qmd config. Only collections carry one, so
# this is the list of corpus roots to watch for marker files.
corpus_roots() {
    [ -f "$CONFIG_FILE" ] || return 0
    awk '
        /^[[:space:]]+path:[[:space:]]*/ {
            sub(/^[[:space:]]*path:[[:space:]]*/, "")
            gsub(/^["'"'"']|["'"'"']$/, "")
            if (length($0) > 0) print
        }
    ' "$CONFIG_FILE"
}

# ── phase 1: startup pass ────────────────────────────────────────────────────

prepare_state() {
    mkdir -p "$STATE_DIR/.qmd"
    cd "$STATE_DIR"

    # qmd switches to project-local mode -- config *and* index inside
    # ./.qmd/ -- when it finds .qmd/index.yml while walking up from the
    # working directory. The file has to exist before the first qmd
    # invocation, or that invocation writes to the global paths instead and
    # the index lands outside the volume.
    if [ -f "$INDEX_CONFIG" ]; then
        log "installing rendered collection config from $INDEX_CONFIG"
        cp "$INDEX_CONFIG" "$CONFIG_FILE"
    elif [ ! -f "$CONFIG_FILE" ]; then
        printf 'collections: {}\n' > "$CONFIG_FILE"
    fi

    log "state directory: $STATE_DIR (config $CONFIG_FILE, index $DB_FILE)"
}

# Declare collections from OSPREY_QMD_COLLECTIONS for deployments that do not
# mount a rendered config.
add_env_collections() {
    [ -n "$COLLECTIONS_SPEC" ] || return 0

    _existing=$(collection_counts | awk '{ print $1 }')
    # shellcheck disable=SC2020  # both separators map to newline on purpose
    for _spec in $(printf '%s' "$COLLECTIONS_SPEC" | tr ', ' '\n\n'); do
        _name=${_spec%%=*}
        _path=${_spec#*=}
        if [ -z "$_name" ] || [ -z "$_path" ] || [ "$_name" = "$_spec" ]; then
            log "WARNING: ignoring malformed collection spec '$_spec' (want name=/path)"
            continue
        fi
        if printf '%s\n' "$_existing" | grep -qx "$_name"; then
            log "collection '$_name' already configured; leaving it alone"
        elif [ ! -d "$_path" ]; then
            log "WARNING: collection '$_name' path '$_path' is not a directory; skipping"
        else
            # Path first, name as a flag. The reversed order does not error:
            # it creates a collection pointed at a directory named after the
            # collection, indexes zero files, and reports success.
            log "adding collection '$_name' -> $_path"
            run_stage qmd collection add "$_path" --name "$_name" \
                || log "WARNING: 'qmd collection add' failed for '$_name'; continuing"
        fi
    done
}

# The embedder that produced the stored vectors, versus the one baked into this
# image. A mismatch means every vector in the index is in a different embedding
# space and comparing them is meaningless, so the index has to be rebuilt.
decide_build_mode() {
    _want="${OSPREY_QMD_EMBED_MODEL_ID:-}"
    [ -n "$_want" ] || die "no embedder identity available (OSPREY_QMD_EMBED_MODEL_ID is unset and ${OSPREY_QMD_MODEL_IDENTITY_FILE:-the identity file} was not readable); refusing to index against an unknown embedder"

    if [ ! -f "$DB_FILE" ]; then
        BUILD_MODE=full
        BUILD_REASON="no index exists yet at $DB_FILE"
    elif [ "$FORCE_REINDEX" = "1" ]; then
        BUILD_MODE=full
        BUILD_REASON="OSPREY_QMD_FORCE_REINDEX=1"
    elif [ ! -f "$IDENTITY_STAMP" ]; then
        BUILD_MODE=full
        BUILD_REASON="index exists but records no embedder identity"
    else
        _have=$(cat "$IDENTITY_STAMP" 2>/dev/null || printf '')
        if [ "$_have" = "$_want" ]; then
            BUILD_MODE=incremental
            BUILD_REASON="embedder unchanged ($_want)"
        else
            BUILD_MODE=full
            BUILD_REASON="embedder changed: index was built with '$_have', image supplies '$_want'"
        fi
    fi
    EMBED_IDENTITY=$_want
}

announce_full_build() {
    log "FULL REINDEX required: $BUILD_REASON"
    log "  Every document will be re-scanned and every embedding recomputed."
    log "  This is slow by nature: at ALS scale (~135,000 documents) it measured"
    log "  41 minutes -- 17 for the keyword index, 24 for the embeddings."
    log "  The search endpoint on port $PORT stays CLOSED until it finishes."
    log "  That is deliberate, not a hang: serving a half-built index would"
    log "  return confidently wrong results. Consumers fall back to substring"
    log "  search while this runs. Progress is reported every 2 minutes."
}

run_startup_pass() {
    decide_build_mode

    if [ "$BUILD_MODE" = full ]; then
        announce_full_build
        # Drop the whole database rather than re-embedding in place: a changed
        # embedder can change the vector dimension, and a clean rebuild has no
        # way to leave stale vectors behind.
        rm -f "$DB_FILE" "$DB_FILE-wal" "$DB_FILE-shm"
    else
        log "incremental startup pass: $BUILD_REASON"
    fi

    add_env_collections

    _pass_t0=$(date +%s)

    # `qmd update` rescans every collection and refreshes the keyword index.
    # It logs and skips individual files it cannot read, so a single bad file
    # does not abort the batch; a non-zero exit here is not fatal on its own,
    # because a previously-built index may still be perfectly serviceable.
    # The document assertion below is the actual gate.
    log "scanning collections ('qmd update')"
    start_heartbeat "collection scan"
    run_stage qmd update || log "WARNING: 'qmd update' exited non-zero; continuing to the index assertion"
    stop_heartbeat

    # `qmd update` deliberately does not embed -- it reports how many hashes
    # need vectors and leaves them pending. Without this step the index would
    # serve keyword hits and silently return nothing for vector searches.
    log "computing embeddings ('qmd embed')"
    start_heartbeat "embedding"
    run_stage qmd embed || log "WARNING: 'qmd embed' exited non-zero; continuing to the index assertion"
    stop_heartbeat

    _pass_elapsed=$(( $(date +%s) - _pass_t0 ))
    log "startup pass finished in $(fmt_duration "$_pass_elapsed")"

    assert_index_populated
    printf '%s\n' "$EMBED_IDENTITY" > "$IDENTITY_STAMP"
}

# The fail-closed gate. A /health-only check would pass for a sidecar whose
# collections point at empty or wrong directories -- qmd reports success for
# that case and serves zero results forever. Refusing to start is louder and
# strictly more useful than serving nothing.
assert_index_populated() {
    _status=$(qmd status 2>&1 || true)
    _total=$(status_field "$_status" Total)
    _vectors=$(status_field "$_status" Vectors)

    log "index reports $_total document(s), $_vectors vector(s)"

    for _name in $(collection_counts | awk '$2 == 0 { print $1 }'); do
        log "WARNING: collection '$_name' indexed 0 files. Either its corpus is"
        log "         genuinely empty and will fill in later, or it points at the"
        log "         wrong directory. Check its 'path:' in $CONFIG_FILE."
    done

    if [ "$_total" -eq 0 ]; then
        log "$_status"
        die "index is empty after the startup pass; refusing to serve. Nothing would be findable, and a healthy-looking empty sidecar is worse than a sidecar that will not start. Check the collection paths above and the corpus mounts."
    fi

    if [ "$BUILD_MODE" = full ] && [ "$_vectors" -eq 0 ]; then
        die "full rebuild produced $_total document(s) but 0 embeddings; refusing to serve, because every vector search would return nothing. Check the 'qmd embed' output above."
    fi
}

# ── phase 2: serve ───────────────────────────────────────────────────────────

start_daemon() {
    log "starting qmd MCP daemon on internal loopback:$INTERNAL_PORT"
    qmd mcp --http --port "$INTERNAL_PORT" &
    QMD_PID=$!

    # qmd hardcodes listen(port, "localhost"), and which loopback family that
    # binds is environment-dependent: Node resolves localhost to ::1 on some
    # images/hosts and to 127.0.0.1 on others (observed: 127.0.0.1 under
    # rootless podman on RHEL-family hosts, where the [::1]-only probe made
    # the sidecar die at HEALTH_TIMEOUT and crash-loop the container). Probe
    # both families and remember the one that answers for the forwarder.
    DAEMON_TARGET=""
    _waited=0
    while :; do
        if ! kill -0 "$QMD_PID" 2>/dev/null; then
            QMD_PID=""
            die "qmd daemon exited during startup; see its output above"
        fi
        for _a in "[::1]" "127.0.0.1"; do
            if curl -fsS -o /dev/null --max-time 5 "http://$_a:$INTERNAL_PORT/health" 2>/dev/null; then
                DAEMON_TARGET="$_a"
                log "daemon healthy after ${_waited}s on $_a"
                return 0
            fi
        done
        [ "$_waited" -lt "$HEALTH_TIMEOUT" ] \
            || die "qmd daemon did not answer /health on [::1] or 127.0.0.1 within ${HEALTH_TIMEOUT}s"
        sleep 1
        _waited=$((_waited + 1))
    done
}

start_forwarder() {
    # qmd answers on the container's loopback only, which is unreachable from
    # any other container. socat owns the routable port and forwards to the
    # loopback family the health probe found answering. The security posture is
    # unchanged -- qmd itself still never listens on a routable address; only
    # this forwarder does, inside the container.
    log "publishing MCP endpoint on 0.0.0.0:$PORT (POST /mcp, GET /health) -> $DAEMON_TARGET:$INTERNAL_PORT"
    if [ "$DAEMON_TARGET" = "127.0.0.1" ]; then
        socat "TCP4-LISTEN:$PORT,fork,reuseaddr" "TCP4:127.0.0.1:$INTERNAL_PORT" &
    else
        socat "TCP4-LISTEN:$PORT,fork,reuseaddr" "TCP6:[::1]:$INTERNAL_PORT" &
    fi
    SOCAT_PID=$!
}

# ── phase 3: update loop ─────────────────────────────────────────────────────
# Marker state is in memory only. It is not persisted, and does not need to be:
# a restart re-runs the startup pass, which indexes everything the markers
# would have asked for.

MARKER_STATE=""

marker_mtime() {
    stat -c %Y "$1" 2>/dev/null || printf ''
}

marker_last_seen() {
    printf '%s\n' "$MARKER_STATE" | awk -F '\t' -v p="$1" '$2 == p { print $1; exit }'
}

marker_record() {
    MARKER_STATE=$(printf '%s\n' "$MARKER_STATE" | awk -F '\t' -v p="$1" '$2 != p')
    MARKER_STATE=$(printf '%s\n%s\t%s\n' "$MARKER_STATE" "$2" "$1")
}

# Record the current marker values without acting on them. The startup pass has
# already indexed whatever they were signalling.
#
# Every temporary below carries a per-function prefix. POSIX sh has no local
# variables, so a bare name here would silently overwrite the caller's variable
# of the same name -- and a marker mtime overwriting the update loop's clock is
# the kind of bug that shows up as "the fallback sweep never runs", with no
# error anywhere.
seed_markers() {
    # Corpus roots are split on whitespace, so a mount path containing a space
    # is not watched. Bind-mount targets are ours to choose; keep them plain.
    for _seed_root in $(corpus_roots); do
        log "watching $_seed_root/$MARKER_NAME"
        _seed_mtime=$(marker_mtime "$_seed_root/$MARKER_NAME")
        if [ -n "$_seed_mtime" ]; then
            marker_record "$_seed_root/$MARKER_NAME" "$_seed_mtime"
        fi
    done
}

# Returns 0 if any marker advanced past the value last acted on. Every marker
# is recorded before returning, so one update covers all of them.
markers_advanced() {
    _mk_hit=1
    for _mk_root in $(corpus_roots); do
        _mk_marker="$_mk_root/$MARKER_NAME"
        _mk_now=$(marker_mtime "$_mk_marker")
        [ -n "$_mk_now" ] || continue
        _mk_was=$(marker_last_seen "$_mk_marker")
        if [ -z "$_mk_was" ] || [ "$_mk_now" -gt "$_mk_was" ]; then
            log "marker advanced: $_mk_marker"
            marker_record "$_mk_marker" "$_mk_now"
            _mk_hit=0
        fi
    done
    return $_mk_hit
}

# One update tick. Never fatal: a corpus that is briefly unreadable, or a file
# that vanishes mid-scan, must not take the search endpoint down. The daemon
# keeps serving the last good index and the next tick tries again.
run_update() {
    _upd_reason=$1
    _upd_t0=$(date +%s)
    log "update triggered by $_upd_reason"
    run_stage qmd update || log "WARNING: 'qmd update' exited non-zero; keeping the previous index and continuing"
    run_stage qmd embed || log "WARNING: 'qmd embed' exited non-zero; new documents are keyword-searchable but not vector-searchable until the next tick"
    log "update finished in $(fmt_duration "$(( $(date +%s) - _upd_t0 ))")"
}

update_loop() {
    _loop_last_sweep=$(date +%s)

    # Updates run inline, so a sweep that outlives its own interval simply
    # delays the next one. There is no way for two of them to overlap.
    while :; do
        # Slept a second at a time, as a backgrounded job, so that SIGTERM is
        # acted on within a second instead of at the end of the poll interval.
        _loop_slept=0
        while [ "$_loop_slept" -lt "$POLL_INTERVAL" ]; do
            sleep 1 &
            STAGE_PID=$!
            set +e
            wait "$STAGE_PID"
            set -e
            STAGE_PID=""
            _loop_slept=$((_loop_slept + 1))
        done

        if ! kill -0 "$QMD_PID" 2>/dev/null; then
            QMD_PID=""
            die "qmd daemon exited; stopping so the orchestrator restarts the container"
        fi
        if ! kill -0 "$SOCAT_PID" 2>/dev/null; then
            SOCAT_PID=""
            die "port forwarder exited; the endpoint on $PORT is dead, stopping"
        fi

        _loop_now=$(date +%s)
        if markers_advanced; then
            run_update "corpus marker"
            _loop_last_sweep=$(date +%s)
        elif [ $((_loop_now - _loop_last_sweep)) -ge "$UPDATE_INTERVAL" ]; then
            run_update "fallback sweep (${UPDATE_INTERVAL}s)"
            _loop_last_sweep=$(date +%s)
        fi
    done
}

# ── main ─────────────────────────────────────────────────────────────────────

main() {
    # The image records what it baked -- qmd's version and the embedder's
    # identity -- in a shell-sourceable file rather than in the environment, so
    # that `docker inspect` on a running container cannot disagree with what is
    # actually on disk. Sourcing it here makes both available to every phase.
    # An identity explicitly set in the environment wins over the file, so a
    # deployment can pin one without rebuilding the image. Sourcing would
    # otherwise silently overwrite it, and an ignored override that changes
    # whether a 40-minute rebuild happens is not a thing to discover in
    # production.
    _env_embed_id="${OSPREY_QMD_EMBED_MODEL_ID:-}"
    if [ -n "${OSPREY_QMD_MODEL_IDENTITY_FILE:-}" ] && [ -f "$OSPREY_QMD_MODEL_IDENTITY_FILE" ]; then
        # shellcheck disable=SC1090
        . "$OSPREY_QMD_MODEL_IDENTITY_FILE"
    fi
    if [ -n "$_env_embed_id" ]; then
        OSPREY_QMD_EMBED_MODEL_ID="$_env_embed_id"
    fi

    log "qmd sidecar starting (qmd ${OSPREY_QMD_VERSION:-unknown})"

    # `qmd pull` is never invoked, here or anywhere else in this image. It
    # HEAD-checks the model registry and, when the registry is unreachable --
    # air-gapped site, blocked egress, upstream outage -- treats a missing
    # remote etag as "stale", deletes the baked model files, and then fails to
    # re-download them. The container does not recover without an image
    # rebuild. `update`, `embed` and `mcp` all resolve models from the cache
    # and are safe.

    command -v qmd > /dev/null 2>&1 || die "qmd is not on PATH"
    command -v socat > /dev/null 2>&1 || die "socat is not installed; the loopback forwarder cannot start"

    prepare_state
    run_startup_pass
    seed_markers
    start_daemon
    start_forwarder
    log "ready: serving on port $PORT; watching markers every ${POLL_INTERVAL}s, sweeping every ${UPDATE_INTERVAL}s"
    update_loop
}

main "$@"
