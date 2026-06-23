#!/usr/bin/env python3
"""
---
name: Config Drift Guard
description: Warns at session start when config.yml has drifted from the rendered Claude Code artifacts
summary: Catch-all so a hand-edited config.yml (or raw `claude` launch) never silently runs stale safety settings
event: SessionStart
---

## Flow

```
SessionStart ──► stat config.yml & .claude/settings.json
                     │
                     ▼
   (a) config.yml mtime > settings.json mtime? ──► general drift
                     │
                     ▼
   (b) config writes_enabled  vs  channel_write in deny? ──► precise drift
                     │
                     ▼
   drift?  ── yes ─► warn (stderr + additionalContext) ─► exit 0
           ── no  ─► silent ─────────────────────────────► exit 0
```

## Details

`config.yml` is a build-time input: safety-critical fields (notably the
``control_system.writes_enabled`` kill-switch, which bakes
``mcp__controls__channel_write`` into ``settings.json``'s ``permissions.deny``)
only take effect once the artifacts are re-rendered by ``osprey build`` /
``osprey claude regen``. The web and CLI entry points now auto-regenerate, but a
hand-edited config.yml launched via raw ``claude`` would otherwise run stale
settings with no signal. This hook is that signal.

Warn-only by design — it never mutates artifacts and never blocks the session
(writes stay blocked until a proper regen, which is the safe direction). It is
stdlib-only and must NOT import osprey or PyYAML: under a raw ``claude`` launch
the project venv's interpreter is not guaranteed to be the one running this hook,
so YAML is matched with a regex and JSON is read with the stdlib ``json`` module.

Fails open on any IO/parse error — stays silent and exits 0.
"""

import json
import os
import re
import sys
from pathlib import Path

# Section-blind on purpose: `writes_enabled` exists only under `control_system:` in
# OSPREY config.yml, so the first line-anchored match is unambiguous. This is a
# warn-only signal — even a mis-parse cannot weaken the write-block gate.
_WRITES_ENABLED = re.compile(r"^\s*writes_enabled:\s*(true|false)\b", re.MULTILINE | re.IGNORECASE)
_CHANNEL_WRITE = "mcp__controls__channel_write"

_REGEN_HINT = "run 'osprey claude regen' and restart for changes to take effect"


def _writes_enabled_in_config(config_path: Path) -> bool | None:
    """Best-effort parse of control_system.writes_enabled (None if undeterminable)."""
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError:
        return None
    match = _WRITES_ENABLED.search(text)
    if match is None:
        return None
    return match.group(1).lower() == "true"


def _channel_write_denied(settings_path: Path) -> bool | None:
    """Whether channel_write is in permissions.deny (None if undeterminable)."""
    try:
        data = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    perms = data.get("permissions")
    if not isinstance(perms, dict):  # e.g. hand-edited {"permissions": null}
        return None
    deny = perms.get("deny", [])
    if not isinstance(deny, list):
        return None
    return _CHANNEL_WRITE in deny


def _detect_drift(config_path: Path, settings_path: Path) -> str | None:
    """Return a drift warning message, or None when artifacts look in sync."""
    if not config_path.exists() or not settings_path.exists():
        return None

    # (b) Targeted, high-signal check: does the kill-switch match the config?
    writes_enabled = _writes_enabled_in_config(config_path)
    denied = _channel_write_denied(settings_path)
    if writes_enabled is not None and denied is not None:
        if writes_enabled and denied:
            return (
                "config.yml sets control_system.writes_enabled: true, but the agent "
                f"config still blocks writes — {_REGEN_HINT}."
            )
        if not writes_enabled and not denied:
            return (
                "config.yml sets control_system.writes_enabled: false, but the agent "
                f"config still permits writes — {_REGEN_HINT}."
            )

    # (a) General signal: config edited more recently than the rendered artifacts.
    try:
        if config_path.stat().st_mtime > settings_path.stat().st_mtime:
            return f"config.yml changed since the agent config was last generated — {_REGEN_HINT}."
    except OSError:
        return None

    return None


def main() -> int:
    try:
        project_dir = Path(os.environ.get("CLAUDE_PROJECT_DIR", "."))
        config_path = project_dir / "config.yml"
        settings_path = project_dir / ".claude" / "settings.json"

        message = _detect_drift(config_path, settings_path)
        if not message:
            return 0

        warning = f"⚠ {message}"
        # Visible in the session transcript ...
        sys.stderr.write(warning + "\n")
        # ... and injected as context so the agent can relay it to the operator.
        sys.stdout.write(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "SessionStart",
                        "additionalContext": warning,
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
