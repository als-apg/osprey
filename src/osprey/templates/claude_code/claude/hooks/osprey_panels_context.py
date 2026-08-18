#!/usr/bin/env python3
"""
---
name: Panels Context
description: Injects the web surface (simple/expert), panel inventory and open-tile state into the agent context at session start
summary: Agent learns which web UX it serves, what panels exist and what is on screen without a list_panels round-trip
event: SessionStart
---

## Flow

```
SessionStart ──► read OSPREY_WEB_UX (set by the web terminal per surface)
                     │
                     ▼
                 read stdin JSON ──► session_id (absent/malformed ─► None)
                     │
                     ▼
                 GET /api/panels (web terminal)
                     │
                     ├──► build [ux line?] + [inventory?] + [tile line?]
                     │         │
                     │         ▼
                     │    emit additionalContext JSON (any part present) ─► exit 0
                     │    (no env var AND web terminal down/empty) ─► silent ─► exit 0
                     │
                     └──► session_id known? ─► overwrite the workspace snapshot
                                                (tempdir/osprey-workspace-<id>.json)
```

## Details

Injects three independent pieces of session context as ``additionalContext``:

1. **Web surface** — ``OSPREY_WEB_UX`` (``simple`` | ``expert``) marks which web
   UI surface launched this session (the operator chat sets ``simple``, the PTY
   terminal ``expert`` — see the web terminal's session launchers). In the
   simple UX the workspace starts hidden until an artifact exists, so the
   simple line instructs the agent to ``open_panel("artifacts")`` whenever it
   produces something the operator should see. Emitted even when the panel
   inventory is unavailable — the surface is known from the env alone.
2. **Panel inventory** — fetched from the web terminal so the agent knows which
   panels exist, their labels, rail membership, and the active tab — without a
   ``list_panels`` tool round-trip.
3. **Open tiles** — which panels are actually on screen, in spatial reading
   order, e.g. ``Open tiles: [lattice, artifacts], active artifacts.`` Tiles are
   reported by the browser, so the line is omitted entirely whenever occupancy
   is *unknown* rather than empty (see below) — an unobserved workspace is never
   described as an empty one.

## Workspace snapshot

Besides the context lines, this hook seeds the *workspace snapshot* — the file
``osprey_workspace_delta.py`` (UserPromptSubmit) diffs against to report what
moved between turns.  It is JSON at::

    os.path.join(tempfile.gettempdir(), f"osprey-workspace-{session_id}.json")

with exactly three keys::

    {
      "tiles":   ["lattice", "artifacts"] | null,  # on-screen, reading order;
                                                   # null = occupancy UNKNOWN
      "active":  "artifacts" | null,               # active tab id
      "visible": ["lattice", "artifacts", "bluesky"]  # launcher-rail membership
    }

``tiles`` is null whenever the server's answer does not actually describe what
is on screen — ``open_tiles`` absent, ``open_tiles_age_s`` null (nobody has ever
reported), or ``open_tiles_dock`` false (the reporting client has no dock, so it
knows the rail but not the tiles, and its report carries a *numeric* age).
Unknown and empty are different claims, and only one of them is honest.  Both
hooks apply this rule identically — see :func:`_occupancy_is_known`.

The seed is **unconditional**: SessionStart re-fires on resume and on ``/clear``
with the same session id, and a stale snapshot would make the first delta line
of the resumed session describe changes the agent already knows about.  It is
written only when a ``session_id`` is available *and* the live state was
fetched — with no live state there is nothing honest to record, and the delta
hook reseeds from a missing snapshot on its own.

``session_id`` comes from the hook's stdin JSON and is restricted to
``[A-Za-z0-9_-]`` before it reaches a path join; anything else is treated as
absent.  ``osprey_workspace_delta.py`` applies the same rule, so both hooks
agree on the file name.

stdlib-only — must NOT import osprey, yaml, requests, or any third-party lib.
Uses only ``json``, ``os``, ``sys``, ``tempfile``, ``urllib.request``,
``urllib.error``.  The SessionStart hook runs under ``python3`` (possibly system
Python 3.9, not the venv interpreter), so this file must be 3.9-safe.

Fails open on any error — stays silent and exits 0.  A down web terminal is not
a session-blocking condition.
"""

import json
import os
import sys
import tempfile
import urllib.error
import urllib.request

#: Characters a session id may contain before it is joined into a file name.
_SESSION_ID_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")


def _read_session_id(stream=None):
    """Pull ``session_id`` out of the hook's stdin JSON payload.

    Args:
        stream: Text stream to read the payload from. Defaults to ``sys.stdin``.

    Returns:
        The session id, or None when stdin is unreadable, is not a JSON object,
        carries no session id, or carries one with characters that have no
        business in a file name. The caller then skips all snapshot work and
        still emits whatever context the environment alone supports.
    """
    try:
        raw = (stream if stream is not None else sys.stdin).read()
    except Exception:
        return None
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    session_id = payload.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        return None
    if not set(session_id) <= _SESSION_ID_CHARS:
        return None
    return session_id


def _snapshot_path(session_id):
    """Return the workspace-snapshot path for ``session_id``.

    Both this hook and ``osprey_workspace_delta.py`` derive the path this way;
    changing it here without changing it there breaks the delta line.
    """
    return os.path.join(tempfile.gettempdir(), f"osprey-workspace-{session_id}.json")


def _occupancy_is_known(data):
    """True when the payload's tile list really describes what is on screen.

    There are three ways occupancy is unknown, and none of them may be reported
    as an empty workspace:

    * ``open_tiles`` absent or not a list — a web terminal predating the field.
    * ``open_tiles_age_s`` explicitly null — the field exists, but no client has
      ever reported a layout.
    * ``open_tiles_dock`` false — the reporting client has no dock, so it knows
      the launcher rail but cannot see tiles. This case carries a *numeric* age,
      so the age check alone would read it as a trustworthy empty workspace.

    A missing or null ``open_tiles_dock`` says nothing either way and is left to
    the other two checks; only an explicit ``false`` forces unknown.
    """
    if not isinstance(data.get("open_tiles"), list):
        return False
    if "open_tiles_age_s" in data and data.get("open_tiles_age_s") is None:
        return False
    return data.get("open_tiles_dock") is not False


def _workspace_state(data):
    """Reduce a ``/api/panels`` payload to the three snapshot facets.

    Args:
        data: Decoded ``GET /api/panels`` response.

    Returns:
        A dict with ``tiles`` (on-screen ids in reading order, or None when
        occupancy is unknown per :func:`_occupancy_is_known`), ``active``
        (active tab id or None) and ``visible`` (launcher-rail membership).
        Every field is defended against a payload of the wrong shape.
    """
    tiles = data.get("open_tiles")
    if _occupancy_is_known(data):
        tiles = [t for t in tiles if isinstance(t, str)]
    else:
        tiles = None

    visible = data.get("visible")
    visible = [v for v in visible if isinstance(v, str)] if isinstance(visible, list) else []

    active = data.get("active")
    if not isinstance(active, str):
        active = None

    return {"tiles": tiles, "active": active, "visible": visible}


def _write_snapshot(path, state):
    """Overwrite the snapshot at ``path`` with ``state``.

    Written to a sibling temp file and renamed so the delta hook can never read
    a half-written file. Returns True on success, False on any failure — the
    snapshot is an optimization, never a reason to disturb a session start.
    """
    tmp_path = f"{path}.tmp"
    try:
        with open(tmp_path, "w") as handle:
            json.dump(state, handle)
        os.replace(tmp_path, path)
        return True
    except Exception:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        return False


def _build_tile_line(state):
    """Describe what is on screen, or None when no client has reported tiles.

    Silence is the honest answer for an unreported workspace: the browser may
    not have connected yet, and "no tiles are open" would be a claim the server
    cannot back up.
    """
    tiles = state.get("tiles")
    if tiles is None:
        return None
    active = state.get("active")
    if not tiles:
        return "Open tiles: [] (nothing on screen besides the terminal)."
    listed = ", ".join(tiles)
    active_part = f"active {active}" if active else "no active tile"
    return f"Open tiles: [{listed}], {active_part}."


def _build_ux_context(ux):
    """Describe the web surface this session serves, from OSPREY_WEB_UX.

    Returns None for absent/unknown values (fail-open: sessions launched
    outside the web terminal carry no marker and get no surface line).
    """
    if ux == "simple":
        return (
            "This session serves the web UI's SIMPLE surface: the operator sees "
            "only the chat, and the WORKSPACE panel stays hidden until there is "
            "something in it. Whenever you produce an artifact the operator "
            "should see (a plot, report, page, or other file), call "
            'open_panel("artifacts") to bring up the WORKSPACE panel so it '
            "appears next to the chat."
        )
    if ux == "expert":
        return "This session serves the web UI's EXPERT surface (full terminal + workspace layout)."
    return None


def _build_inventory(data):
    """Build a concise human-readable inventory string from /api/panels response.

    Returns None when there are no panels to report.
    """
    labels = data.get("labels", {})
    visible_ids = set(data.get("visible", []))
    active = data.get("active")

    parts = []
    for pid in data.get("enabled", []):
        label = labels.get(pid, pid.upper())
        membership = "on rail" if pid in visible_ids else "off rail"
        parts.append(f"{label} (id={pid}, {membership})")
    for cp in data.get("custom", []):
        cid = cp.get("id", "")
        if not cid:
            continue
        label = cp.get("label", cid.upper())
        membership = "on rail" if cid in visible_ids else "off rail"
        parts.append(f"{label} (id={cid}, {membership})")

    if not parts:
        return None

    panel_list = ", ".join(parts)
    active_part = f"Active tab: {active}." if active else "No active tab."
    return (
        f"Web terminal panels (right pane tabs): {panel_list}. "
        f"{active_part} "
        "Put one in front of the operator with open_panel(id) and take it off "
        "screen again with close_panel(id). Rail membership is separate: "
        "add_panel_to_rail(id)/remove_panel_from_rail(id) control whether the "
        "operator can launch a panel in one click. An off-rail panel is already "
        "running — open_panel puts it on the rail and on screen in one step. "
        "Add an ad-hoc URL tab with register_panel(...)."
    )


def main():
    try:
        ux_context = _build_ux_context(os.environ.get("OSPREY_WEB_UX"))
        session_id = _read_session_id()

        port = os.environ.get("OSPREY_WEB_PORT", "8087")
        host = "127.0.0.1"
        base = f"http://{host}:{port}"

        inventory = None
        tile_line = None
        try:
            req = urllib.request.urlopen(f"{base}/api/panels", timeout=2)
            data = json.loads(req.read())
            inventory = _build_inventory(data)
            state = _workspace_state(data)
            tile_line = _build_tile_line(state)
            if session_id:
                # Unconditional: SessionStart re-fires on resume and /clear with
                # the same id, and the delta hook must start from what is on
                # screen now, not from what was there before the resume.
                _write_snapshot(_snapshot_path(session_id), state)
        except Exception:
            # Web terminal down or unreachable — the surface line (env-only)
            # still applies; only the live-state parts are dropped.
            pass

        parts = [p for p in (ux_context, inventory, tile_line) if p]
        if not parts:
            return 0

        sys.stdout.write(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "SessionStart",
                        "additionalContext": " ".join(parts),
                    }
                }
            )
        )
        return 0
    except Exception:
        # Last-resort fail-open: never block a session start.
        return 0


if __name__ == "__main__":
    sys.exit(main())
