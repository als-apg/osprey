#!/bin/sh
# First-boot provisioning for the Nextcloud + Talk e2e fixture image
# (Dockerfile.nextcloud_talk). Copied into BOTH the "post-installation" and
# "before-starting" entrypoint hook folders, so a fresh container provisions
# itself and a restarted one re-asserts the same state.
#
# The official entrypoint executes hook scripts through its own run_as()
# helper, which means this already runs as www-data — `php occ` is invoked
# directly, with no su dance.
set -eu

OCC="php /var/www/html/occ"
READY_MARKER=/var/lib/talk-fixture/ready

# Never advertise readiness while re-provisioning.
rm -f "$READY_MARKER"

# "before-starting" also fires on containers that were never installed (someone
# cleared the admin env vars), and there a no-op is correct: occ would fail and
# take the whole entrypoint down with it.
#
# But occ exits nonzero for plenty of other reasons — a stale/half-populated
# volume, an unreadable config.php, maintenance mode, a PHP fatal — and those
# must NOT be read as "not installed". Booting apache Talk-less and marker-less
# turns a real breakage into an opaque health-wait timeout, with a log line that
# actively misdirects whoever debugs it: the same wasted e2e session the
# assertion further down exists to prevent. So the exit status and the payload
# are judged separately, and only "exited 0 AND reported installed:false" is a
# silent no-op.
#
# Matching stays substring/grep-based on purpose: on a genuinely uninstalled
# instance occ prepends a "Nextcloud is not installed - only a limited number of
# commands are available" banner BEFORE the JSON, so decoding the whole stream
# would choke on the one case this branch exists to handle.
status_rc=0
status_output=$($OCC status --output=json 2>&1) || status_rc=$?

if [ "$status_rc" -ne 0 ]; then
    echo "[talk-fixture] FATAL: 'occ status' exited ${status_rc}. This instance is broken," >&2
    echo "[talk-fixture] not merely uninstalled - refusing to boot without Talk." >&2
    printf '%s\n' "$status_output" >&2
    exit "$status_rc"
fi

status_json=$(printf '%s' "$status_output" | tr -d ' ')

if printf '%s' "$status_json" | grep -q '"installed":false'; then
    echo "[talk-fixture] Nextcloud is not installed; skipping Talk provisioning."
    exit 0
fi

if ! printf '%s' "$status_json" | grep -q '"installed":true'; then
    echo "[talk-fixture] FATAL: 'occ status' reported no install state at all." >&2
    printf '%s\n' "$status_output" >&2
    exit 1
fi

echo "[talk-fixture] Enabling Talk (spreed)..."
$OCC app:enable spreed

# Onboarding overlay is pure noise for an API-driven fixture, and is not
# guaranteed to be shipped in every Nextcloud build.
$OCC app:disable firstrunwizard || true

# app:enable already fails loudly, but assert the end state too: reporting ready
# with Talk half-provisioned is the one failure mode that would waste an entire
# e2e debugging session.
if ! $OCC app:list --output=json \
    | php -r 'exit(isset(json_decode(stream_get_contents(STDIN), true)["enabled"]["spreed"]) ? 0 : 1);'; then
    echo "[talk-fixture] FATAL: spreed is not in the enabled app list after app:enable." >&2
    exit 1
fi

# Marker content is the provisioned Talk version, so a failing e2e run can tell
# "wrong Talk" apart from "Talk missing" straight from the container.
#
# The version is captured and checked BEFORE anything is written, because a plain
# redirect creates the marker before the command produces output: a
# config:app:get that printed nothing (or only a newline) but exited 0 would
# leave a present-but-blank marker, so the container would be genuinely ready
# while the marker's whole diagnostic purpose had silently evaporated.
#
# The capture is deliberately NOT piped into tr: a pipeline reports the LAST
# command's status, so `occ ... | tr` would mask an occ failure that set -eu
# should be killing the boot over. Strip in a second step instead, and test the
# stripped value rather than the file size — whitespace-only output is a blank
# marker too, just not a zero-byte one.
talk_version=$($OCC config:app:get spreed installed_version)
talk_version=$(printf '%s' "$talk_version" | tr -d '[:space:]')
if [ -z "$talk_version" ]; then
    echo "[talk-fixture] FATAL: could not read Talk's installed_version; refusing to write a blank marker." >&2
    exit 1
fi

# tmp+mv so the marker is never observable in a half-written state: the
# HEALTHCHECK tests only for its existence, so a partial file would read as ready.
printf '%s\n' "$talk_version" > "$READY_MARKER.tmp"
mv "$READY_MARKER.tmp" "$READY_MARKER"

echo "[talk-fixture] Talk provisioned: $(cat "$READY_MARKER")"
