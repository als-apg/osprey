#!/usr/bin/env python3
"""
---
name: Workspace Delta
description: Tells the agent what moved in the web workspace since its last turn
summary: One terse line when tiles, their tabs, the active tab or rail membership changed — silence when nothing did
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
tab or a whole tile, drag a tab into another tile or out into a new one, or switch
the tab a tile shows — none of which the agent would
otherwise see without spending a ``list_panels`` round-trip on every turn.  This
hook spends nothing when nothing moved.

**Silence is the default.**  The hook writes to stdout only when the workspace
genuinely differs from the snapshot, and then only one line naming what moved::

    Workspace changed: okf tab moved to [bluesky, okf (showing)]; active now okf.

Tiles are compared tab by tab, the terminal included: a whole tile opened or
closed (``bluesky tile opened``, ``okf tile closed``), a tab opened into or
closed out of a tile that remains (``bluesky tab opened in [...]``, ``ariel tab
closed``), a tab moved to another tile or a new one (``... moved to a new
tile``), a tile now showing another of its tabs (``artifacts tab now
showing``), and — only when nothing else moved — a reordering.  A tile has no
identity across turns (ids are reused after a close), so earlier and later
tiles are paired by the tabs they keep; see :func:`_match_tiles`.

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
        {"tile": "t1", "tabs": ["terminal"], "showing": "terminal"},
        {"tile": "t2", "tabs": ["okf", "artifacts"], "showing": "artifacts"}
      ] | null,                                    # null = occupancy UNKNOWN
      "popouts": ["ariel"],                        # panels in their own windows
      "active":  "artifacts" | null,               # active tab id
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

    Anything but a non-empty list of well-formed tiles is unknown. A live
    layout always holds at least one tile, so ``[]`` is never a real current
    state.
    """
    if not isinstance(tiles, list) or not tiles:
        return None
    out = []
    for tile in tiles:
        if not isinstance(tile, dict):
            return None
        tabs = tile.get("tabs")
        if not isinstance(tabs, list) or not tabs:
            return None
        if not all(isinstance(key, str) and key for key in tabs):
            return None
        showing = tile.get("showing")
        if showing not in tabs:
            showing = tabs[0]
        tile_id = tile.get("tile")
        out.append(
            {
                "tile": tile_id if isinstance(tile_id, str) else None,
                "tabs": list(tabs),
                "showing": showing,
            }
        )
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
    order ``dock_engine.layout_keys`` uses on the server. The terminal is a tab
    like any other (key ``"terminal"``) and is kept.

    Args:
        layout: The ``layout`` value of a ``GET /api/panels`` response.

    Returns:
        A list of ``{"tile": id|None, "tabs": [key, ...], "showing": key}``
        dicts, or None when
        ``layout`` is not a well-formed tree (a non-dict, a tile without string
        tabs, a split without items, nesting past :data:`_MAX_LAYOUT_DEPTH`).
        A ``showing`` that is missing or names no tab of its tile falls back to
        the first tab, the engine's own default. Sizes and split directions are
        dropped: the context lines name what is where, not how big it is. The
        tile id is kept only as a tie-break for the delta hook's tile matching —
        ids are reused after a close, so it is never shown or trusted alone.
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
        if not isinstance(tabs, list) or not tabs:
            return False
        if not all(isinstance(key, str) and key for key in tabs):
            return False
        showing = node.get("showing")
        if showing not in tabs:
            showing = tabs[0]
        tile = node.get("tile")
        tiles.append(
            {
                "tile": tile if isinstance(tile, str) else None,
                "tabs": list(tabs),
                "showing": showing,
            }
        )
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
    """
    if not isinstance(data.get("layout"), dict):
        return False
    return data.get("layout_dock") is not False


def _workspace_state(data):
    """Reduce a ``GET /api/panels`` response to the three snapshot facets.

    ``tiles`` is the server's ``layout`` flattened by :func:`_layout_tiles`, or
    None whenever occupancy is unknown per :func:`_occupancy_is_known`. Unknown
    is not the same as empty, and the difference decides whether this hook says
    anything about tiles at all.
    """
    tiles = _layout_tiles(data.get("layout")) if _occupancy_is_known(data) else None
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


def _match_tiles(before, after, survivors):
    """Pair each earlier tile with the later tile that continues it.

    A tile has no identity the hook can trust across turns — ids are reused
    after a close — so tiles are matched by the tabs they keep: the pair sharing
    the most surviving tabs first, a shared tile id breaking a tie (the engine
    keeps the source tile's id when a tab is dragged out of it), then reading
    order. Each tile is matched at most once; a tile sharing no surviving tab
    with any other is unmatched (closed before, new after).

    Returns:
        ``{before_index: after_index}``.
    """
    candidates = []
    for bi, old in enumerate(before):
        kept = survivors.intersection(old["tabs"])
        for ai, new in enumerate(after):
            overlap = len(kept.intersection(new["tabs"]))
            if overlap:
                same_id = old["tile"] is not None and old["tile"] == new["tile"]
                candidates.append((-overlap, 0 if same_id else 1, bi, ai))
    match, taken = {}, set()
    for _, _, bi, ai in sorted(candidates):
        if bi not in match and ai not in taken:
            match[bi] = ai
            taken.add(ai)
    return match


def _group(keys, where):
    """Group ``keys`` by tile index, keeping first-appearance order."""
    groups: dict[object, list] = {}
    for key in keys:
        groups.setdefault(where[key], []).append(key)
    return groups


def _tile_name(tile):
    """A tile as a reader names it: its key alone when it holds one tab."""
    return tile["tabs"][0] if len(tile["tabs"]) == 1 else _render_tile(tile)


def _describe_tiles(previous, current):
    """Describe how tiles and their tabs moved, in the order a reader wants them.

    Clauses come as: tabs opened (a whole new tile, or a tab added to a tile),
    tabs moved to another tile, tabs closed (a whole tile, or a tab out of a
    tile that remains), tiles now showing another tab, and — only when nothing
    else changed — a reordering. Returns a list of clause strings (possibly
    empty). Nothing is said when the tiles are unknown on either side: a first
    report cannot be phrased as a transition, and a workspace that stopped being
    reported has not changed — it stopped being observed.
    """
    before, after = previous.get("tiles"), current.get("tiles")
    if after is None:
        return []
    if before is None:
        return [f"tiles now {_render_tiles(after)}"]

    where_before = {key: i for i, tile in enumerate(before) for key in tile["tabs"]}
    where_after = {key: i for i, tile in enumerate(after) for key in tile["tabs"]}
    after_order = [key for tile in after for key in tile["tabs"]]
    before_order = [key for tile in before for key in tile["tabs"]]
    survivors = set(where_before) & set(where_after)
    match = _match_tiles(before, after, survivors)
    continued = set(match.values())

    opened = [key for key in after_order if key not in where_before]
    moved = [
        key
        for key in after_order
        if key in survivors and match.get(where_before[key]) != where_after[key]
    ]
    closed = [key for key in before_order if key not in where_after]

    clauses = []

    new_tiles, into = [], []
    for ai, keys in _group(opened, where_after).items():
        if ai not in continued and len(keys) == len(after[ai]["tabs"]):
            new_tiles.append(_tile_name(after[ai]))
        else:
            into.append(
                f"{', '.join(keys)} {_noun(keys, 'tab')} opened in {_render_tile(after[ai])}"
            )
    if new_tiles:
        clauses.append(f"{', '.join(new_tiles)} {_noun(new_tiles, 'tile')} opened")
    clauses.extend(into)

    for ai, keys in _group(moved, where_after).items():
        destination = _render_tile(after[ai]) if ai in continued else "a new tile"
        clauses.append(f"{', '.join(keys)} {_noun(keys, 'tab')} moved to {destination}")

    gone_tiles, out_of = [], []
    for bi, keys in _group(closed, where_before).items():
        if len(keys) == len(before[bi]["tabs"]):
            gone_tiles.append(_tile_name(before[bi]))
        else:
            out_of.append(f"{', '.join(keys)} {_noun(keys, 'tab')} closed")
    if gone_tiles:
        clauses.append(f"{', '.join(gone_tiles)} {_noun(gone_tiles, 'tile')} closed")
    clauses.extend(out_of)

    showing = [
        after[ai]["showing"]
        for bi, ai in sorted(match.items(), key=lambda pair: pair[1])
        if after[ai]["showing"] != before[bi]["showing"]
        and where_before.get(after[ai]["showing"]) == bi
        and len(after[ai]["tabs"]) > 1
    ]
    if showing:
        clauses.append(f"{', '.join(showing)} {_noun(showing, 'tab')} now showing")

    if not clauses and [t["tabs"] for t in before] != [t["tabs"] for t in after]:
        clauses.append(f"tiles rearranged to {_render_tiles(after)}")
    return clauses


def _noun(items, noun, plural=None):
    if len(items) == 1:
        return noun
    return plural if plural is not None else f"{noun}s"


def _render_tile(tile):
    """One tile as ``[a]``, or ``[a, b (showing)]`` when it holds several tabs.

    A one-tab tile can only show that tab, so it carries no marker.
    """
    tabs = tile["tabs"]
    if len(tabs) == 1:
        return f"[{tabs[0]}]"
    listed = ", ".join(f"{key} (showing)" if key == tile["showing"] else key for key in tabs)
    return f"[{listed}]"


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
    """Describe an active-tab change, or nothing when it held still."""
    before, after = previous.get("active"), current.get("active")
    if before == after:
        return []
    if after is None:
        return ["no tab active now"]
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
