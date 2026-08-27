#!/bin/sh
# =============================================================================
# Demo Facility — OSPREY boot hook
# =============================================================================
# osprey-scaffold: deploy/boot-hook
# osprey-version: OSPREY_VERSION
#
# Emitted by `osprey scaffold systemd` beside the user unit, for one situation:
# a deploy host whose home directory is an NFS or autofs mount.
#
# A lingering `systemd --user` manager starts at boot, before that mount lands.
# It resolves its unit search path once, finds nothing under the not-yet-
# mounted home, and never looks again — so `systemctl --user status
# osprey.service` reads `not-found` after every reboot and the stack sits
# there until somebody runs `daemon-reload` by hand.
#
# When systemd manages the mount itself — an fstab entry, or a .mount or
# .automount unit — a `RequiresMountsFor` drop-in on `user@<uid>.service`
# orders the manager after it, and needs root. A home served by the autofs
# daemon has no mount unit for that drop-in to order against, so there it
# changes nothing, and this script is the fix rather than the fallback. It
# waits for the home, the deployment and the user manager to appear, then
# reloads the unit files and starts the unit. Wire it into the account's own
# crontab — all of these lines, pasted whole:
#
#   crontab -e
#   SHELL=/bin/sh
#   HOME=/
#   @reboot d=/tmp/osprey-boot-hook.$(id -u); mkdir -m 700 "$d" 2>/dev/null; if [ -d "$d" ] && [ ! -L "$d" ] && [ -O "$d" ]; then log=/tmp/osprey-boot-hook.$(id -u)/boot.log; else log=/dev/null; fi; echo "$(date) osprey-boot-hook: cron fired" >> "$log"; n=0; until [ -x /srv/osprey/demo-facility/scripts/osprey-boot-hook.sh ] || [ $n -ge 120 ]; do sleep 5; n=$((n+1)); done; if [ -x /srv/osprey/demo-facility/scripts/osprey-boot-hook.sh ]; then exec /srv/osprey/demo-facility/scripts/osprey-boot-hook.sh; fi; echo "$(date) osprey-boot-hook: gave up, /srv/osprey/demo-facility/scripts/osprey-boot-hook.sh never appeared" | tee -a "$log"
#
# None of them is optional, and the job is deliberately not a bare `@reboot`
# with this script's path. cron changes into the crontab's HOME before it runs
# a job, and on this host the home is not there yet when cron starts, so the
# job dies before anything runs — silently, with no mail. `HOME=/` gives cron
# a directory that exists; `SHELL=/bin/sh` is the shell the job is written
# for, whatever an existing crontab set above. Then `sh` has to read this
# script, which sits on the same late mount, so the job — which lives in the
# crontab, on the local disk — first notes that cron fired it in
# `/tmp/osprey-boot-hook.$(id -u)/boot.log`, waits for this file to become readable, and only
# then runs it. Put the lines last in the crontab: every job below them runs
# from `/` and sees HOME=/ in its environment, so a job body that expands
# $HOME breaks silently unless it starts with its own
# `export HOME=<the real home>`.
#
# Everything this script prints lands in that same log, on the local disk
# rather than the home, so a boot on which the home never came still says
# how far things got; cron mails the account the same lines if the host
# delivers mail. Running the script again by hand is harmless: an already
# active unit is left alone.
#
# Deployment how-to: https://als-apg.github.io/osprey/how-to/deploy-a-facility.html
#
# Re-run `osprey scaffold systemd` after the repo moves or OSPREY is
# reinstalled elsewhere — the paths below are absolute. The marker line above
# is what makes re-emission safe, so a file without it is treated as
# hand-written and left alone unless you pass --force.
# =============================================================================
set -u

# cron ran this with HOME=/ (see the header), so the real home goes back first:
# the unit file lives under it, and on this host it is the mount everything
# else waits on. Written in as a full path like the repo and the executable —
# until the home is mounted nothing on this host can be asked where it is.
HOME="/home/osprey"
export HOME

# The one line that makes a cron job able to talk to the user manager at all.
# `@reboot` runs with no session and no bus address, and `systemctl --user`
# with no XDG_RUNTIME_DIR does not fail loudly — it just fails. Every hook of
# this kind that "does nothing" is missing this.
XDG_RUNTIME_DIR="/run/user/$(id -u)"
export XDG_RUNTIME_DIR

# Proof of launch, before any wait, on a disk that is there at boot. Appended:
# the crontab job wrote the line before this one, and the two together say
# whether cron fired, whether this file was reachable, and how far it got.
# The directory, not a bare file, because /tmp is shared and the name is
# predictable: appending would follow a symlink anyone could have planted
# there. Only a real directory this account owns is used; otherwise nothing
# is logged rather than something written where a stranger pointed.
LOG_DIR="/tmp/osprey-boot-hook.$(id -u)"
mkdir -m 700 "$LOG_DIR" 2>/dev/null
if [ -d "$LOG_DIR" ] && [ ! -L "$LOG_DIR" ] && [ -O "$LOG_DIR" ]; then
  LOG="/tmp/osprey-boot-hook.$(id -u)/boot.log"
else
  LOG=/dev/null
fi
if ! printf 'osprey-boot-hook: launched %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" >> "$LOG" 2>/dev/null; then
  LOG=/dev/null
fi

# Seconds between attempts, and the budget shared by every wait below. Past
# the budget the hook gives up and says which thing never showed, rather than
# holding a cron slot open forever.
POLL_SECONDS=5
TOTAL_WAIT_SECONDS=600
WAITED=0

say() { printf 'osprey-boot-hook: %s\n' "$1" | tee -a "$LOG"; }
die() { printf 'osprey-boot-hook: %s\n' "$1" | tee -a "$LOG" >&2; exit 1; }

# Block until a path exists, spending from the shared budget. POSIX sh has no
# `local`, so WAITED is deliberately global: the budget covers the whole boot,
# not each path in turn.
wait_for() {
  while [ ! -e "$1" ]; do
    if [ "$WAITED" -ge "$TOTAL_WAIT_SECONDS" ]; then
      die "gave up after ${TOTAL_WAIT_SECONDS}s waiting for $2 ($1) — the unit was not started"
    fi
    sleep "$POLL_SECONDS"
    WAITED=$((WAITED + POLL_SECONDS))
  done
  say "$2 is there: $1"
}

# The home carries the unit file, and on this kind of host it is the mount
# everything else is waiting on. On an autofs home the test itself is what
# triggers the mount.
wait_for "$HOME" "the home directory"

# Both absolute, and both may live under that same mount: the unit's
# WorkingDirectory, and the executable it runs.
wait_for "/srv/osprey/demo-facility" "the deployment repo"
wait_for "/usr/local/bin/osprey" "the osprey executable"

# The runtime directory is created by the user manager itself, so it appearing
# is the signal that there is a manager to talk to — checking for the mount
# alone would race a manager that has not started yet.
wait_for "$XDG_RUNTIME_DIR" "the user manager's runtime directory"

# Re-resolve the unit search path, which is the whole point: the manager read
# it once, before the home was there.
if ! systemctl --user daemon-reload; then
  die "systemctl --user daemon-reload failed — the manager is up but not answering this account"
fi
say "reloaded the user manager's unit files"

if systemctl --user is-active --quiet osprey.service; then
  say "osprey.service is already active — nothing to start"
  exit 0
fi

if ! systemctl --user start osprey.service; then
  die "systemctl --user start osprey.service failed — see: systemctl --user status osprey.service"
fi
say "started osprey.service"
