#!/usr/bin/env python3
"""
---
name: Notebook Cache Invalidation
description: Invalidates cached notebook HTML after NotebookEdit so the gallery re-renders
summary: Tracks notebook edits as workspace artifacts
event: PostToolUse
tools: NotebookEdit
---

## Flow

```
stdin ──► Parse JSON
              │
              ▼
         Has notebook_path? ──NO──► EXIT
              │
             YES
              │
              ▼
         Locate _notebook_cache/
         {stem}_rendered.html
              │
              ▼
         Cached file exists? ──NO──► EXIT
              │
             YES
              │
              ▼
         Delete cached HTML
              │
              ▼
         EXIT
```

## Details

Lightweight utility hook with no safety implications. When Claude edits a
notebook via `NotebookEdit`, the gallery's cached HTML rendering becomes
stale. This hook deletes it so the next gallery view triggers a fresh render.
"""

import json
import sys
from pathlib import Path


def main():
    try:
        hook_input = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        sys.exit(0)

    tool_input = hook_input.get("tool_input", {})
    notebook_path = tool_input.get("notebook_path", "")

    if not notebook_path:
        sys.exit(0)

    # Invalidate cached HTML for this notebook
    try:
        nb_path = Path(notebook_path)
        # Check in the artifact cache directory
        cache_dir = nb_path.parent / "_notebook_cache"
        cached_html = cache_dir / f"{nb_path.stem}_rendered.html"
        if cached_html.exists():
            cached_html.unlink()
    except Exception:
        pass  # Never block on cache invalidation failure

    sys.exit(0)


if __name__ == "__main__":
    main()
