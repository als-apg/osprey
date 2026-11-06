#!/usr/bin/env python3
"""
---
name: Focus State Validator
description: Injects the prompting session's gallery focus and pins, filtered to that session and to artifacts that still exist
summary: Keeps one session's focus and pins out of every other session's prompt, and drops ghost-artifact references
event: UserPromptSubmit
---

## Flow

```
stdin ──► UserPromptSubmit payload ── session_id missing / unparsable ─► print nothing
              │
              ▼
   Read <agent-data-root>/focus_state.txt
   keep `artifact:` lines whose trailing (session=<id>) marker names this session
              │
              ▼
   Read <agent-data-root>/artifacts/artifacts.json
   ├─ unreadable ─► keep the focus lines, add no pins, check no ids
   └─ readable   ─► drop focus lines whose id is gone;
                    add `pinned:` per entry pinned with metadata.pinned_in == session
              │
              ▼
   Strip the markers ─► print under [Gallery Focus] (nothing if no line) ─► exit 0
```

## Details

Contract C7. The gallery writes one ``artifact:`` line per transcript slot,
each ending in that transcript's ``(session=<id>)`` marker; pins are not in the
file, so this hook reads them from the artifact index, where
``ArtifactStore.set_pinned`` records the pinning session in
``metadata.pinned_in``. A pin needs no gallery route to reach the prompt.

Filtering happens before any output, on every path. An unmarked line, a line
marked for another session, and a pin recorded for no session are never
printed. Titles are re-sanitised the way the writer does it — control
characters and newlines collapse to a space, ``(id=`` and ``(session=`` lose
their ``=`` — so no title can end a line or spell a marker.

Fails open on any error by printing *nothing*, never the raw file, and exits 0
so the user prompt is never blocked.
"""

# A hook can be executed by whatever bare ``python3`` is on PATH — on macOS that
# is still 3.9, which parses no PEP 604 union in an evaluated annotation. Making
# annotations lazy keeps this module importable there; without it the hook dies
# at import and its whole check goes dark.
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from osprey_hook_log import repo_agent_data_root

_HEADER = "[Gallery Focus]"

#: A writer focus line: ``<body ending in (id=<id>)> (session=<transcript>)``,
#: the marker being the line's last token.
_FOCUS_LINE = re.compile(
    r"^(?P<body>\s*artifact:.*\(id=(?P<id>[^)\s]+)\))\s+\(session=(?P<session>[^)\s]+)\)\s*$"
)

# Must equal the title sanitiser in osprey.interfaces.artifacts.app (_focus_title).
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]+")
_MARKER_OPENER = re.compile(r"\((id|session)=")


def _read_text(path: Path) -> str:
    try:
        return path.read_text()
    except OSError:
        return ""


def _session_id(raw_stdin: str) -> str | None:
    """The payload's ``session_id``, or ``None`` when missing or unparsable."""
    try:
        payload = json.loads(raw_stdin)
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    sid = payload.get("session_id")
    if not isinstance(sid, str) or not sid.strip():
        return None
    return sid.strip()


def _safe_title(title: object) -> str:
    text = title if isinstance(title, str) else ""
    return _MARKER_OPENER.sub(r"(\1 ", _CONTROL_CHARS.sub(" ", text))


def _load_entries(artifacts_index: Path) -> list[dict] | None:
    """The index's entries, or ``None`` when the index cannot be read."""
    try:
        data = json.loads(artifacts_index.read_text())
    except (OSError, ValueError):
        return None
    entries = data.get("entries") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        return None
    return [e for e in entries if isinstance(e, dict) and isinstance(e.get("id"), str)]


def _focus_lines(raw: str, session_id: str, valid_ids: set[str] | None) -> list[tuple[str, str]]:
    """(id, line) for each focus line marked for *session_id*, marker stripped.

    *valid_ids* ``None`` means the index is unreadable: no stale-id check.
    """
    kept: list[tuple[str, str]] = []
    for line in raw.splitlines():
        m = _FOCUS_LINE.match(line)
        if m is None or m.group("session") != session_id:
            continue
        if valid_ids is not None and m.group("id") not in valid_ids:
            continue
        kept.append((m.group("id"), m.group("body")))
    return kept


def _pin_lines(entries: list[dict], session_id: str, exclude: set[str]) -> list[str]:
    """One ``pinned:`` line per entry pinned in *session_id*, except *exclude*."""
    lines: list[str] = []
    for e in entries:
        if e.get("pinned") is not True or e["id"] in exclude:
            continue
        meta = e.get("metadata")
        if not isinstance(meta, dict) or meta.get("pinned_in") != session_id:
            continue
        lines.append(f'  pinned:   "{_safe_title(e.get("title"))}" (id={e["id"]})')
    return lines


def render(session_id: str, focus_raw: str, entries: list[dict] | None) -> str:
    """The text injected for *session_id* (contract C7), ``""`` when none."""
    valid_ids = None if entries is None else {e["id"] for e in entries}
    focus = _focus_lines(focus_raw, session_id, valid_ids)
    lines = [line for _, line in focus]
    if entries is not None:
        lines += _pin_lines(entries, session_id, {aid for aid, _ in focus})
    if not lines:
        return ""
    return _HEADER + "\n" + "\n".join(lines) + "\n"


def main() -> int:
    try:
        session_id = _session_id(sys.stdin.read())
        if session_id is None:
            return 0

        # The root the gallery writes both files under: agent_data.base_dir,
        # anchored on the repo rather than the render this hook runs in.
        agent_data = Path(repo_agent_data_root())
        focus_raw = _read_text(agent_data / "focus_state.txt")
        entries = _load_entries(agent_data / "artifacts" / "artifacts.json")

        sys.stdout.write(render(session_id, focus_raw, entries))
        return 0
    except Exception:
        # Last-resort fail-open: print nothing, never block a user prompt.
        return 0


if __name__ == "__main__":
    sys.exit(main())
