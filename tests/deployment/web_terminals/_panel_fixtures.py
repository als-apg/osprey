"""Hand-written renders of the ``knob`` application's panel, for the shell-gate tests."""

from __future__ import annotations

import json

import yaml

from osprey.agent_runner.tool_names import DENY_DEFAULTS
from tests._renders import tier_render

_TOKEN = "KNOB_LAUNCH_TOKEN"


def _write_render_project(tmp_path, name: str, project_config: dict, *, shell_open: bool) -> str:
    """Write a render's ``config.yml`` and shipped ``.claude/settings.json``."""
    project_dir = tmp_path / "profiles" / name
    project_dir.mkdir(parents=True)
    (project_dir / "config.yml").write_text(
        yaml.safe_dump({"project_name": name, **project_config}), encoding="utf-8"
    )
    deny = [entry for entry in DENY_DEFAULTS if not shell_open or entry != "Bash"]
    (project_dir / ".claude").mkdir()
    (project_dir / ".claude" / "settings.json").write_text(
        json.dumps({"permissions": {"allow": [], "deny": deny, "ask": []}}), encoding="utf-8"
    )
    return f"profiles/{name}"


def _roster(tmp_path, renders: dict[str, tuple[dict, bool]]) -> dict:
    """One user per render; ``renders`` maps a render name to (config, shell_open)."""
    names = sorted(renders)
    return {
        "facility": {"prefix": "als"},
        "registry": {"url": "registry.example.org"},
        "deploy": {"fqdn": "deploy.example.org"},
        "modules": {
            "web_terminals": {
                "enabled": True,
                "users": [
                    {"name": f"user{index}", "index": index, "tier": name}
                    for index, name in enumerate(names)
                ],
                "default_tier": names[0],
                "renders": {
                    name: tier_render(
                        name,
                        name,
                        project_path=_write_render_project(
                            tmp_path, name, project, shell_open=shell_open
                        ),
                    )
                    for name, (project, shell_open) in renders.items()
                },
            }
        },
    }


def _panel_render(writes: str, *, routes=(("GET", "/state"),), token: bool = False) -> dict:
    """A rendered config showing the ``knob`` application's panel."""
    block = {
        "label": "Knob",
        "url": "http://127.0.0.1:8100",
        "path": "/",
        "app": "knob",
        "agent_routes": [{"method": method, "path": path} for method, path in routes],
    }
    if token:
        block["launch_token_env"] = _TOKEN
    return {
        "control_system": {"writes": writes},
        "web": {"panels": {"knob": block}},
        "web_panels": ["knob"],
    }
