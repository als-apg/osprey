#!/usr/bin/env python3
"""
---
name: Focus State Validator
description: Strips stale artifact IDs from focus_state.txt before injecting into the user prompt
summary: Prevents ghost-artifact references after MCP-driven deletes
event: UserPromptSubmit
---

## Flow

```
stdin ──► (ignored — this hook reads files, not stdin)
              │
              ▼
   Read <agent-data-root>/focus_state.txt
              │
              ▼
   Read <agent-data-root>/artifacts/artifacts.json
              │
              ▼
   Build set of valid artifact IDs
              │
              ▼
   For each line: extract (id=...) ─► keep if id is valid
              │
              ▼
   Print cleaned content (or empty) ─► exit 0
```

## Details

Defense-in-depth backstop for the artifact-delete listener pattern: even if
some delete callsite ever forgets to refresh focus_state.txt, this hook
silently drops stale entries before the agent sees them.

Fails open on any parse/IO error — prints the original file (or empty) and
exits 0 so the user prompt is never blocked.
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
from osprey_hook_log import get_repo_root

_ID_PATTERN = re.compile(r"\(id=([^)]+)\)")

# The framework DEFAULT agent-data root, imported rather than spelled out here
# so the two cannot drift apart. It does not follow a project that overrides
# `agent_data.base_dir` — this hook reads two files the artifact gallery
# writes, and under an overridden root it would look in the wrong directory,
# find no focus file, strip nothing, and report success. The fallback covers a
# hook running with osprey off the path, the one case where guessing beats
# crashing.
try:
    from osprey.utils.workspace import DEFAULT_AGENT_DATA_BASE_DIR as _AGENT_DATA_ROOT
except Exception:  # pragma: no cover - hooks must never crash the agent
    _AGENT_DATA_ROOT = "var/agent_data"


def _read_text(path: Path) -> str:
    try:
        return path.read_text()
    except OSError:
        return ""


def _load_valid_ids(artifacts_index: Path) -> set[str] | None:
    """Return the set of currently valid artifact IDs, or ``None`` on error."""
    try:
        data = json.loads(artifacts_index.read_text())
    except (OSError, ValueError):
        return None
    entries = data.get("entries") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        return None
    return {e["id"] for e in entries if isinstance(e, dict) and "id" in e}


def _clean(raw: str, valid_ids: set[str]) -> str:
    """Drop any line referencing an unknown artifact id."""
    lines = raw.splitlines()
    if not lines:
        return ""

    header_idx = 0 if lines[0].strip() == "[Gallery Focus]" else -1
    body_lines = lines[header_idx + 1 :] if header_idx >= 0 else lines

    kept: list[str] = []
    for line in body_lines:
        m = _ID_PATTERN.search(line)
        if m is None:
            # Lines without an id pattern are pass-through (defensive — the
            # current writer always includes one, but future formats might not).
            kept.append(line)
            continue
        if m.group(1) in valid_ids:
            kept.append(line)

    if not kept:
        return ""

    if header_idx >= 0:
        return "[Gallery Focus]\n" + "\n".join(kept) + "\n"
    return "\n".join(kept) + ("\n" if raw.endswith("\n") else "")


def main() -> int:
    try:
        # The REPO root, not the render this hook runs in: the gallery writes
        # both files below through ArtifactStore, which anchors on the config's
        # project_root. Anchored on the render, this hook found no focus file,
        # stripped nothing and reported success — a silent no-op.
        agent_data = Path(get_repo_root()) / _AGENT_DATA_ROOT
        focus_file = agent_data / "focus_state.txt"
        artifacts_index = agent_data / "artifacts" / "artifacts.json"

        raw = _read_text(focus_file)
        if not raw.strip():
            return 0

        valid_ids = _load_valid_ids(artifacts_index)
        if valid_ids is None:
            # Index unreadable — fail open with the original contents.
            sys.stdout.write(raw)
            return 0

        sys.stdout.write(_clean(raw, valid_ids))
        return 0
    except Exception:
        # Last-resort fail-open: never block a user prompt.
        return 0


if __name__ == "__main__":
    sys.exit(main())
