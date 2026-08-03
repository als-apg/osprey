#!/usr/bin/env python3
"""
---
name: Panels Context
description: Injects the web surface (simple/expert) and panel inventory into the agent context at session start
summary: Agent learns which web UX it serves and what panels exist without a list_panels round-trip
event: SessionStart
---

## Flow

```
SessionStart ──► read OSPREY_WEB_UX (set by the web terminal per surface)
                     │
                     ▼
                 GET /api/panels (web terminal)
                     │
                     ▼
   build [ux line?] + [inventory?] from env + enabled/custom panels
                     │
                     ▼
   emit additionalContext JSON (any part present) ───► exit 0
   (no env var AND web terminal down/empty) ─► silent ► exit 0
```

## Details

Injects two independent pieces of session context as ``additionalContext``:

1. **Web surface** — ``OSPREY_WEB_UX`` (``simple`` | ``expert``) marks which web
   UI surface launched this session (the operator chat sets ``simple``, the PTY
   terminal ``expert`` — see the web terminal's session launchers). In the
   simple UX the workspace starts hidden until an artifact exists, so the
   simple line instructs the agent to ``show_panel("artifacts")`` whenever it
   produces something the operator should see. Emitted even when the panel
   inventory is unavailable — the surface is known from the env alone.
2. **Panel inventory** — fetched from the web terminal so the agent knows which
   panels exist, their labels, visibility state, and the active tab — without a
   ``list_panels`` tool round-trip.

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
            'show_panel("artifacts") to bring up the WORKSPACE panel so it '
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
        ux_context = _build_ux_context(os.environ.get("OSPREY_WEB_UX"))

        port = os.environ.get("OSPREY_WEB_PORT", "8087")
        host = "127.0.0.1"
        base = f"http://{host}:{port}"

        inventory = None
        try:
            req = urllib.request.urlopen(f"{base}/api/panels", timeout=2)
            inventory = _build_inventory(json.loads(req.read()))
        except Exception:
            # Web terminal down or unreachable — the surface line (env-only)
            # still applies; only the inventory is dropped.
            pass

        parts = [p for p in (ux_context, inventory) if p]
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
