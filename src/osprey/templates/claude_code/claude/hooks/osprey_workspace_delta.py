#!/usr/bin/env python3
"""
---
name: Workspace Delta
description: Tells the agent what moved in the web workspace since its last turn
summary: One terse line when tiles, the active tab or rail membership changed — silence when nothing did
event: UserPromptSubmit
---

## Flow

```
UserPromptSubmit ──► read stdin JSON ──► session_id (absent/malformed ─► exit 0, silent)
                          │
                          ▼
                     GET /api/panels ──► (down/unreachable) ─► exit 0, silent
                          │
                          ▼
                     load snapshot (tempdir/osprey-workspace-<id>.json)
                          │
              ┌───────────┴────────────┐
        missing/corrupt              loaded
              │                        │
              ▼                        ▼
        reseed, emit nothing     diff {tiles, active, visible}
              │                        │
              ▼               ┌────────┴────────┐
           exit 0        unchanged            changed
                              │                  │
                              ▼                  ▼
                        exit 0, silent    emit one additionalContext
                        (no output)       line, rewrite snapshot ─► exit 0
```

## Details

The SessionStart sibling (``osprey_panels_context.py``) tells the agent what the
workspace looked like when the session began.  Between turns a human can close a
tile, drag one somewhere else, or switch tabs — none of which the agent would
otherwise see without spending a ``list_panels`` round-trip on every turn.  This
hook spends nothing when nothing moved.

**Silence is the default.**  The hook writes to stdout only when the workspace
genuinely differs from the snapshot, and then only one line naming what moved::

    Workspace changed: lattice tile closed; active now artifacts.

Everything else exits 0 with byte-empty stdout: an unchanged workspace, a
missing or corrupt snapshot (reseeded first, so the *next* turn diffs cleanly),
a session id it cannot read, and a web terminal that is down.  A down terminal
is never reported as an empty workspace — an unknown workspace and an empty one
are different claims, and only one of them is honest here.

The same applies to tiles specifically: the browser reports them, so occupancy
may be unknown while the rail state is perfectly well known — nobody has ever
reported (``open_tiles_age_s`` null), or the reporting client has no dock
(``open_tiles_dock`` false) and can see the rail but not the tiles.  When tiles
go from known to unknown — the last dock-capable browser closed — the hook says
nothing about tiles and keeps the last known list in the snapshot, so a
returning browser reporting the same layout does not read as a change.

## Terminal-API authorization

The ``GET /api/panels`` call carries ``Authorization: Bearer
<OSPREY_PANEL_TOKEN>`` whenever that variable holds a non-blank value in the
environment the hook inherits — the web terminal exports it into the agent it
launches.  When it is unset, empty or whitespace-only the header is omitted
entirely rather than sent blank, matching how the server-side
``mcp_server.http._panel_auth_headers`` reads the same carrier.  Either way the call stays fail-open: a refused fetch (401
included) reads as an unknown workspace, so the hook stays silent.

## Snapshot contract

Reads and rewrites the file seeded by ``osprey_panels_context.py``::

    os.path.join(tempfile.gettempdir(), f"osprey-workspace-{session_id}.json")

    {
      "tiles":   ["lattice", "artifacts"] | null,  # on-screen, reading order;
                                                   # null = occupancy UNKNOWN
      "active":  "artifacts" | null,               # active tab id
      "visible": ["lattice", "artifacts", "bluesky"]  # launcher-rail membership
    }

The schema is documented in full in the sibling hook, which owns it.  Both hooks
restrict ``session_id`` to ``[A-Za-z0-9_-]`` before joining it into a path, and
both derive the file name the same way — change one and the other stops seeing
the same file.

stdlib-only — must NOT import osprey, yaml, requests, or any third-party lib.
Uses only ``json``, ``os``, ``sys``, ``tempfile``, ``urllib.request``,
``urllib.error``.  Runs under ``python3`` (possibly system Python 3.9, not the
venv interpreter), so this file must be 3.9-safe.

Fails open on any error — stays silent and exits 0.  A user prompt is never
blocked by workspace bookkeeping.
"""

import json
import os
import sys
import tempfile
import urllib.error
import urllib.request

#: Characters a session id may contain before it is joined into a file name.
#: Must match ``osprey_panels_context.py`` — the two hooks share the file.
_SESSION_ID_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")


def _read_session_id(stream=None):
    """Pull ``session_id`` out of the hook's stdin JSON payload.

    Args:
        stream: Text stream to read the payload from. Defaults to ``sys.stdin``.

    Returns:
        The session id, or None when stdin is unreadable, is not a JSON object,
        carries no session id, or carries one with characters that have no
        business in a file name. Without an id there is no snapshot to diff
        against, so the hook simply stays silent.
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
    """Return the workspace-snapshot path for ``session_id``."""
    return os.path.join(tempfile.gettempdir(), f"osprey-workspace-{session_id}.json")


def _load_snapshot(path):
    """Read the previous workspace state, or None when there is none to trust.

    A missing file (first prompt of a session started outside the web terminal),
    a half-written one, or one carrying anything other than the three-key object
    all read as None, which the caller answers by reseeding silently.
    """
    try:
        with open(path) as handle:
            data = json.load(handle)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    if not all(key in data for key in ("tiles", "active", "visible")):
        return None
    return _normalize_state(data)


def _normalize_state(data):
    """Coerce a snapshot or live payload into the three diffed facets."""
    tiles = data.get("tiles")
    tiles = [t for t in tiles if isinstance(t, str)] if isinstance(tiles, list) else None

    visible = data.get("visible")
    visible = [v for v in visible if isinstance(v, str)] if isinstance(visible, list) else []

    active = data.get("active")
    if not isinstance(active, str):
        active = None

    return {"tiles": tiles, "active": active, "visible": visible}


def _occupancy_is_known(data):
    """True when the payload's tile list really describes what is on screen.

    Kept identical to the sibling hook's copy — the two must agree or a session
    would be told about a "change" that is only a difference in how the two
    hooks read the same server. Occupancy is unknown when ``open_tiles`` is
    absent or not a list, when ``open_tiles_age_s`` is explicitly null (nobody
    has ever reported), or when ``open_tiles_dock`` is false (the reporting
    client has no dock, so it knows the rail but cannot see tiles — and its
    report carries a *numeric* age, which the age check alone would trust).
    """
    if not isinstance(data.get("open_tiles"), list):
        return False
    if "open_tiles_age_s" in data and data.get("open_tiles_age_s") is None:
        return False
    return data.get("open_tiles_dock") is not False


def _workspace_state(data):
    """Reduce a ``GET /api/panels`` response to the three snapshot facets.

    ``tiles`` is None whenever occupancy is unknown per
    :func:`_occupancy_is_known`. Unknown is not the same as empty, and the
    difference decides whether this hook says anything about tiles at all.
    """
    tiles = data.get("open_tiles") if _occupancy_is_known(data) else None
    return _normalize_state(
        {"tiles": tiles, "active": data.get("active"), "visible": data.get("visible")}
    )


def _write_snapshot(path, state):
    """Overwrite the snapshot at ``path`` with ``state``.

    Written to a sibling temp file and renamed, so a hook reading concurrently
    never sees a half-written file. Returns True on success, False on any
    failure — a snapshot that cannot be written costs one redundant line next
    turn, which is not worth disturbing a prompt over.
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


def _merge_state(previous, current):
    """Build the state to persist, carrying forward facts that went unknown.

    Only ``tiles`` can go from known to unknown (the last browser closed). The
    last known list is kept so a browser returning with the same layout reads as
    no change rather than as a reopened workspace.
    """
    merged = dict(current)
    if merged["tiles"] is None:
        merged["tiles"] = previous.get("tiles")
    return merged


def _describe_tiles(previous, current):
    """Describe how the on-screen tiles moved, in the order a reader wants them.

    Returns a list of clause strings (possibly empty). Nothing is said when the
    tile set is unknown on either side: a first report cannot be phrased as a
    transition, and a workspace that stopped being reported has not changed —
    it stopped being observed.
    """
    before, after = previous.get("tiles"), current.get("tiles")
    if after is None:
        return []
    if before is None:
        listed = ", ".join(after) if after else "none"
        return [f"tiles now [{listed}]"]

    before_set, after_set = set(before), set(after)
    clauses = []
    opened = [t for t in after if t not in before_set]
    closed = [t for t in before if t not in after_set]
    if opened:
        clauses.append(f"{', '.join(opened)} {_tile_noun(opened)} opened")
    if closed:
        clauses.append(f"{', '.join(closed)} {_tile_noun(closed)} closed")
    if not clauses and before != after:
        clauses.append(f"tiles rearranged to [{', '.join(after)}]")
    return clauses


def _tile_noun(ids):
    return "tile" if len(ids) == 1 else "tiles"


def _describe_active(previous, current):
    """Describe an active-tab change, or nothing when it held still."""
    before, after = previous.get("active"), current.get("active")
    if before == after:
        return []
    if after is None:
        return ["no tile active now"]
    return [f"active now {after}"]


def _describe_visible(previous, current):
    """Describe launcher-rail membership changes as a set difference.

    Rail order carries no meaning the agent can act on, so only membership is
    compared — a reordered rail is not a change worth a line.
    """
    before, after = set(previous.get("visible") or []), set(current.get("visible") or [])
    clauses = []
    added = sorted(after - before)
    removed = sorted(before - after)
    if added:
        clauses.append(f"{', '.join(added)} added to the launcher rail")
    if removed:
        clauses.append(f"{', '.join(removed)} removed from the launcher rail")
    return clauses


def _build_delta_line(previous, current):
    """Build the one-line change summary, or None when nothing moved."""
    clauses = (
        _describe_tiles(previous, current)
        + _describe_active(previous, current)
        + _describe_visible(previous, current)
    )
    if not clauses:
        return None
    return f"Workspace changed: {'; '.join(clauses)}."


def _panels_target(url):
    """Build what ``urlopen`` should be handed for a terminal-API GET.

    The web terminal exports ``OSPREY_PANEL_TOKEN`` into the environment of the
    agent it launches, so a hook running under that agent inherits the token
    without reading any file or importing anything.

    Args:
        url: Absolute URL of the terminal-API endpoint to fetch.

    Returns:
        A ``urllib.request.Request`` carrying ``Authorization: Bearer <token>``
        when the variable holds a non-blank value; otherwise *url* unchanged,
        so no ``Authorization`` header is sent at all. A blank bearer would be
        a credential claim this hook cannot back, and the token being absent is
        exactly the unauthenticated read every caller here already expected.
        Whitespace-only counts as absent for the same reason the empty string
        does — an uninterpolated compose variable arrives as ``""`` and a
        hand-edited one can arrive as ``"   "`` — and it is how
        ``mcp_server.http._panel_auth_headers`` reads the same carrier, so both
        sides agree on what "no token" means.
    """
    token = (os.environ.get("OSPREY_PANEL_TOKEN") or "").strip()
    if not token:
        return url
    return urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})


def _fetch_state():
    """Fetch live workspace state, or None when the web terminal is unreachable."""
    port = os.environ.get("OSPREY_WEB_PORT", "8087")
    try:
        target = _panels_target(f"http://127.0.0.1:{port}/api/panels")
        response = urllib.request.urlopen(target, timeout=2)
        return _workspace_state(json.loads(response.read()))
    except Exception:
        return None


def main():
    try:
        session_id = _read_session_id()
        if not session_id:
            return 0

        current = _fetch_state()
        if current is None:
            # Web terminal down: the workspace is unknown, not unchanged and not
            # empty. Say nothing and leave the snapshot alone.
            return 0

        path = _snapshot_path(session_id)
        previous = _load_snapshot(path)
        if previous is None:
            # No trustworthy baseline — seed one so the next turn can diff.
            _write_snapshot(path, current)
            return 0

        line = _build_delta_line(previous, current)
        if line is None:
            return 0

        _write_snapshot(path, _merge_state(previous, current))
        sys.stdout.write(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "UserPromptSubmit",
                        "additionalContext": line,
                    }
                }
            )
        )
        return 0
    except Exception:
        # Last-resort fail-open: never block a user prompt.
        return 0


if __name__ == "__main__":
    sys.exit(main())
