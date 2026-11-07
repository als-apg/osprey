#!/usr/bin/env python3
"""
---
name: Workspace Delta
description: Tells the agent what moved in the web workspace since its last turn
summary: One terse line when tiles, the panels they show, the active panel or rail membership changed — silence when nothing did
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
workspace looked like when the session began.  Between turns a human can open a
panel into a tile (replacing the panel it showed), close a tile, drag a panel or
a tile to a new place, or make another panel the active one — none of which the
agent would otherwise see without spending a ``list_panels`` round-trip on every
turn.  This hook spends nothing when nothing moved.

**Silence is the default.**  The hook writes to stdout only when the workspace
genuinely differs from the snapshot, and then only one line naming what moved::

    Workspace changed: ariel replaced okf in its tile; active now ariel.

Each tile shows one panel, and tiles are compared panel by panel, the
terminal's included: a tile opened or closed (``bluesky tile opened``,
``okf tile closed``), a tile that kept its id while its panel left the
screen and another came on (``ariel replaced okf in its tile``), a panel
now in a tile with another id (``okf moved to a new tile``), and — only
when nothing else moved — a reordering.  Ids are reused after a close, so a
tile closed and a new one opened under its id between two turns reads as a
replacement too; see :func:`_describe_tiles`.

Everything else exits 0 with byte-empty stdout: an unchanged workspace, a
missing or corrupt snapshot (reseeded first, so the *next* turn diffs cleanly),
a session id it cannot read, and a web terminal that is down.  A down terminal
is never reported as an empty workspace — an unknown workspace and an empty one
are different claims, and only one of them is honest here.

The same applies to tiles specifically: the browser reports them, so occupancy
may be unknown while the rail state is perfectly well known — nobody has ever
reported (``layout`` null), or the reporting client has no dock
(``layout_dock`` false) and can see the rail but not the tiles.  When tiles
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
      "tiles": [                                   # on-screen, reading order;
        {"tile": "t1", "panel": "terminal"},       # each tile shows one panel
        {"tile": "t2", "panel": "artifacts"}
      ] | null,                                    # null = occupancy UNKNOWN
      "popouts": ["ariel"],                        # panels in their own windows
      "active":  "artifacts" | null,               # active panel id
      "visible": ["okf", "artifacts", "bluesky"]  # launcher-rail membership
    }

The schema is documented in full in the sibling hook, which owns it.  Both hooks
restrict ``session_id`` to ``[A-Za-z0-9_-]`` before joining it into a path, and
both derive the file name the same way — change one and the other stops seeing
the same file.

stdlib-only — must NOT import osprey, yaml, requests, or any third-party lib.
Uses only ``json``, ``os``, ``pathlib``, ``sys``, ``tempfile``,
``urllib.request``, ``urllib.error`` and the sibling stdlib-only helper
``osprey_safety_file``.  Runs under ``python3`` (possibly system Python 3.9, not the
venv interpreter), so this file must be 3.9-safe.

That contract is why the web terminal's port is the one number written out
here rather than looked up: ``osprey.port_layout`` is where every framework
port lives, and this file may not import it.  The literal is the ``web`` slot
at the layout's default base — what ``default_port('web', 0)`` returns — and it
is a *fallback*, reached only when ``OSPREY_WEB_PORT`` is unset, which is the
plain ``claude`` session run beside a default-base ``osprey web``.  A
deployment that moved its block exports the variable, so the literal is never
what such a deployment dials.  It carries the ``osprey:not-a-port`` marker so
the retired-number lint reads it as the stated exception it is; move it if the
layout's ``web`` offset or default base ever moves.

Fails open on any error — stays silent and exits 0.  A user prompt is never
blocked by workspace bookkeeping.
"""

import json
import os
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# The stdlib-only file helper the snapshot is written through, a sibling script.
# Without it no snapshot is written, which costs one redundant line next turn.
try:
    import osprey_safety_file as _safety_file
except Exception:  # pragma: no cover - render without the helper
    _safety_file = None  # type: ignore[assignment]  # the sibling is absent

#: Characters a session id may contain before it is joined into a file name.
#: Must match ``osprey_panels_context.py`` — the two hooks share the file.
_SESSION_ID_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")

#: The web terminal's port when ``OSPREY_WEB_PORT`` names none.  Written out
#: rather than looked up because this file may not import osprey; the module
#: docstring states the contract and why the literal is the honest fallback.
#: Must match ``osprey_panels_context.py`` — the two hooks dial one terminal.
_DEFAULT_WEB_PORT = "10100"  # osprey:not-a-port — stdlib-only hook contract (see module docstring); equals default_port('web', 0)


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


def _normalize_tiles(tiles):
    """Coerce a snapshot's ``tiles`` into the tile list, or None when unknown.

    Anything but a list of well-formed ``{"tile", "panel"}`` tiles is unknown.
    ``[]`` is an empty dock: every pane is in a window of its own.
    """
    if not isinstance(tiles, list):
        return None
    out = []
    for tile in tiles:
        if not isinstance(tile, dict):
            return None
        panel = tile.get("panel")
        if not isinstance(panel, str) or not panel:
            return None
        tile_id = tile.get("tile")
        out.append({"tile": tile_id if isinstance(tile_id, str) else None, "panel": panel})
    return out


def _normalize_state(data):
    """Coerce a snapshot or live payload into the three diffed facets."""
    tiles = _normalize_tiles(data.get("tiles"))

    visible = data.get("visible")
    visible = [v for v in visible if isinstance(v, str)] if isinstance(visible, list) else []

    active = data.get("active")
    if not isinstance(active, str):
        active = None

    popouts = data.get("popouts")
    if isinstance(popouts, list):
        popouts = sorted({key for key in popouts if isinstance(key, str) and key})
    else:
        popouts = None if tiles is None else []

    return {"tiles": tiles, "popouts": popouts, "active": active, "visible": visible}


#: Deepest split nesting the hook will walk. A real dock is a few levels deep;
#: anything past this is a payload of the wrong shape, read as unknown.
_MAX_LAYOUT_DEPTH = 32


def _layout_tiles(layout):
    """Flatten a dock layout tree into its tiles in reading order.

    A layout node is a tile ``{tile, tabs, showing, ...}`` or a split ``{split,
    items, ...}``; tiles are read depth-first, each split's items in order — the
    order ``dock_engine.layout_keys`` uses on the server. A tile shows one
    panel, the one its ``tabs`` lists; the terminal's tile (key ``"terminal"``)
    is kept like any other.

    Args:
        layout: The ``layout`` value of a ``GET /api/panels`` response.

    Returns:
        A list of ``{"tile": id|None, "panel": key}`` dicts, or None when
        ``layout`` is not a well-formed tree (a non-dict, a tile whose ``tabs``
        is not a list of one non-empty string, a split without items, nesting past
        :data:`_MAX_LAYOUT_DEPTH`). The server refuses a tile listing several
        panels, so such a tile is a payload of the wrong shape as well. Sizes,
        split directions and ``showing`` are dropped: the context lines name
        what is where, not how big it is. The tile id is kept for the delta
        hook, which reads a tile that kept its id while its panel changed as a
        replacement; ids are reused after a close, so it is never shown.
    """
    tiles = []

    def walk(node, depth):
        if depth > _MAX_LAYOUT_DEPTH or not isinstance(node, dict):
            return False
        if "split" in node:
            items = node.get("items")
            if not isinstance(items, list) or not items:
                return False
            return all(walk(item, depth + 1) for item in items)
        tabs = node.get("tabs")
        if not isinstance(tabs, list) or len(tabs) != 1:
            return False
        panel = tabs[0]
        if not isinstance(panel, str) or not panel:
            return False
        tile = node.get("tile")
        tiles.append({"tile": tile if isinstance(tile, str) else None, "panel": panel})
        return True

    return tiles if walk(layout, 0) else None


def _occupancy_is_known(data):
    """True when the payload's layout really describes what is on screen.

    Kept identical to the sibling hook's copy — the two must agree or a session
    would be told about a "change" that is only a difference in how the two
    hooks read the same server.

    The web terminal publishes the current dock client's dock layout as
    ``layout`` (None when no dock reports). Occupancy is unknown, and must never
    be reported as an empty workspace, when:

    * ``layout`` is absent or not an object — nobody has reported a dock layout,
      or a web terminal predating the field answered.
    * ``layout_dock`` is false — the reporting client has no dock, so it
      knows the launcher rail but cannot see tiles.

    A missing or null ``layout_dock`` says nothing either way; only an
    explicit ``false`` forces unknown. A layout that is present but malformed is
    caught one step later, by :func:`_layout_tiles` returning None.

    One null layout is known: a ``layout_dock: true`` report. That dock said it
    is empty (its panes in windows of their own, named by ``popouts``, or
    closed), which is a workspace, not silence.
    """
    layout = data.get("layout")
    if layout is None:
        return data.get("layout_dock") is True
    if not isinstance(layout, dict):
        return False
    return data.get("layout_dock") is not False


def _workspace_state(data):
    """Reduce a ``GET /api/panels`` response to the three snapshot facets.

    ``tiles`` is the server's ``layout`` flattened by :func:`_layout_tiles`, or
    None whenever occupancy is unknown per :func:`_occupancy_is_known`. Unknown
    is not the same as empty, and the difference decides whether this hook says
    anything about tiles at all.
    """
    tiles = None
    if _occupancy_is_known(data):
        tiles = [] if data.get("layout") is None else _layout_tiles(data.get("layout"))
    return _normalize_state(
        {
            "tiles": tiles,
            "popouts": data.get("popouts") if tiles is not None else None,
            "active": data.get("active"),
            "visible": data.get("visible"),
        }
    )


def _write_snapshot(path, state):
    """Overwrite the snapshot at ``path`` with ``state``.

    Replaced whole through ``osprey_safety_file``, so a hook reading
    concurrently never sees a half-written file. Returns True on success, False
    on any failure or a missing helper — a snapshot that cannot be written costs
    one redundant line next turn, which is not worth disturbing a prompt over.
    """
    if _safety_file is None:
        return False
    try:
        _safety_file.atomic_replace(
            Path(path), json.dumps(state).encode("utf-8"), mode=0o644, sync="best_effort"
        )
    except Exception:
        return False
    return True


def _merge_state(previous, current):
    """Build the state to persist, carrying forward facts that went unknown.

    Only ``tiles`` can go from known to unknown (the last browser closed). The
    last known list is kept so a browser returning with the same layout reads as
    no change rather than as a reopened workspace.
    """
    merged = dict(current)
    if merged["tiles"] is None:
        merged["tiles"] = previous.get("tiles")
        merged["popouts"] = previous.get("popouts")
    return merged


def _describe_tiles(previous, current):
    """Describe how tiles and the panels they show moved, in the order a reader wants them.

    Each tile shows one panel. Clauses come as: tiles opened, panels that
    replaced another in a tile that kept its id, panels moved to a new tile
    (their tile's id changed), tiles closed, and — only when nothing else
    changed — a reordering. A tile whose id names no tile on the other side is
    opened or closed; the id is the only identity a tile has across turns, and
    since ids are reused after a close, a tile closed and another opened under
    its id reads as a replacement too. Returns a list of clause strings
    (possibly empty). Nothing is said when the tiles are unknown on either
    side: a first report cannot be phrased as a transition, and a workspace that
    stopped being reported has not changed — it stopped being observed.
    """
    before, after = previous.get("tiles"), current.get("tiles")
    if after is None:
        return []
    if before is None:
        return [f"tiles now {_render_tiles(after) if after else 'none'}"]

    tile_before = {tile["panel"]: tile["tile"] for tile in before}
    tile_after = {tile["panel"]: tile["tile"] for tile in after}
    opened = [tile for tile in after if tile["panel"] not in tile_before]
    closed = [tile for tile in before if tile["panel"] not in tile_after]

    # A tile that kept its id while its panel left the screen and another came on.
    left = {tile["tile"]: tile["panel"] for tile in closed if tile["tile"] is not None}
    replaced = {}
    for tile in opened:
        if tile["tile"] in left:
            replaced[tile["panel"]] = left.pop(tile["tile"])
    gone = set(replaced.values())

    new_tiles = [tile["panel"] for tile in opened if tile["panel"] not in replaced]
    moved = [
        tile["panel"]
        for tile in after
        if tile["panel"] in tile_before
        and None not in (tile["tile"], tile_before[tile["panel"]])
        and tile["tile"] != tile_before[tile["panel"]]
    ]
    gone_tiles = [tile["panel"] for tile in closed if tile["panel"] not in gone]

    clauses = []
    if new_tiles:
        clauses.append(f"{', '.join(new_tiles)} {_noun(new_tiles, 'tile')} opened")
    clauses.extend(f"{new} replaced {old} in its tile" for new, old in replaced.items())
    if moved:
        clauses.append(f"{', '.join(moved)} moved to {_noun(moved, 'a new tile', 'new tiles')}")
    if gone_tiles:
        clauses.append(f"{', '.join(gone_tiles)} {_noun(gone_tiles, 'tile')} closed")

    if not clauses and [t["panel"] for t in before] != [t["panel"] for t in after]:
        clauses.append(f"tiles rearranged to {_render_tiles(after)}")
    return clauses


def _noun(items, noun, plural=None):
    if len(items) == 1:
        return noun
    return plural if plural is not None else f"{noun}s"


def _render_tile(tile):
    """One tile as ``[panel]``: the one panel it shows."""
    return f"[{tile['panel']}]"


def _render_tiles(tiles):
    """Every tile in reading order, comma-separated (see :func:`_render_tile`)."""
    return ", ".join(_render_tile(tile) for tile in tiles)


def _describe_popouts(previous, current):
    """Describe panels that left for, or came back from, a window of their own.

    Like the tiles, unknown on either side says nothing: a pop-out set is known
    exactly when the tiles are.
    """
    before, after = previous.get("popouts"), current.get("popouts")
    if before is None or after is None:
        return []
    clauses = []
    out = [key for key in after if key not in before]
    back = [key for key in before if key not in after]
    if out:
        clauses.append(f"{', '.join(out)} popped out to {_noun(out, 'a window', 'windows')}")
    if back:
        clauses.append(f"{', '.join(back)} back from {_noun(back, 'its window', 'their windows')}")
    return clauses


def _describe_active(previous, current):
    """Describe an active-panel change, or nothing when it held still."""
    before, after = previous.get("active"), current.get("active")
    if before == after:
        return []
    if after is None:
        return ["no panel active now"]
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
        + _describe_popouts(previous, current)
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
    port = os.environ.get("OSPREY_WEB_PORT", _DEFAULT_WEB_PORT)
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
