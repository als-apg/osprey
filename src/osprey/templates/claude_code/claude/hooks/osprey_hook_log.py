"""Shared hook logging utility. No external dependencies."""

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path


def get_hook_input():
    """Read and return hook input JSON from stdin. Returns {} on failure."""
    try:
        return json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return {}


def get_project_dir(hook_input):
    """Single deterministic source: hook_input['cwd']."""
    return hook_input.get("cwd", "")


def log_hook(hook_name, hook_input, status="ok", detail=""):
    """Append one line to data/hooks/activity.log if OSPREY_HOOK_DEBUG is set."""
    if not os.environ.get("OSPREY_HOOK_DEBUG"):
        return
    cwd = get_project_dir(hook_input)
    if not cwd:
        return
    log_dir = Path(cwd) / "data" / "hooks"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S")
    tool = hook_input.get("tool_name", "-")
    line = f"{ts} [{hook_name}] tool={tool} status={status}"
    if detail:
        line += f" {detail}"
    try:
        with open(log_dir / "activity.log", "a") as f:
            f.write(line + "\n")
    except OSError:
        pass
