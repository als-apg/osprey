#!/usr/bin/env python3
"""
---
name: Panels Context
description: Injects the panel inventory into the agent context at session start
summary: Agent learns what panels exist and their state without a list_panels round-trip
event: SessionStart
---

## Flow

```
SessionStart ──► GET /api/panels (web terminal)
                     │
                     ▼
   build inventory string from enabled + custom panels
                     │
                     ▼
   emit additionalContext JSON ──────────────────────► exit 0
   (web terminal down / parse failure) ─► silent ────► exit 0
```

## Details

Fetches the panel inventory from the web terminal at session start and injects
it as ``additionalContext`` so the agent knows which panels exist, their labels,
visibility state, and the active tab — without a ``list_panels`` tool round-trip.

stdlib-only — must NOT import osprey, yaml, requests, or any third-party lib.
Uses only ``json``, ``os``, ``sys``, ``urllib.request``, ``urllib.error``.
The SessionStart hook runs under ``python3`` (possibly system Python 3.9, not
the venv interpreter), so this file must be 3.9-safe.

Fails open on any error — stays silent and exits 0.  A down web terminal is not
a session-blocking condition.
"""

import json
import os
import sys
import urllib.error
import urllib.request


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
        shown = "shown" if pid in visible_ids else "hidden"
        parts.append(f"{label} (id={pid}, {shown})")
    for cp in data.get("custom", []):
        cid = cp.get("id", "")
        if not cid:
            continue
        label = cp.get("label", cid.upper())
        shown = "shown" if cid in visible_ids else "hidden"
        parts.append(f"{label} (id={cid}, {shown})")

    if not parts:
        return None

    panel_list = ", ".join(parts)
    active_part = f"Active tab: {active}." if active else "No active tab."
    return (
        f"Web terminal panels (right pane tabs): {panel_list}. "
        f"{active_part} "
        "You can reveal/conceal these with show_panel(id)/hide_panel(id) and "
        "add an ad-hoc URL tab with register_panel(...). "
        "Hidden panels are launched but their tab is not shown until you show_panel them."
    )


def main():
    try:
        port = os.environ.get("OSPREY_WEB_PORT", "8087")
        host = "127.0.0.1"
        base = f"http://{host}:{port}"

        try:
            req = urllib.request.urlopen(f"{base}/api/panels", timeout=2)
            data = json.loads(req.read())
        except Exception:
            # Web terminal down or unreachable — fail open, no output.
            return 0

        inventory = _build_inventory(data)
        if not inventory:
            return 0

        sys.stdout.write(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "SessionStart",
                        "additionalContext": inventory,
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
