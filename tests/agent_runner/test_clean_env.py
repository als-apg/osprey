"""``build_clean_env`` leaves hook debugging to the hooks themselves.

Hooks read ``hooks.debug`` from ``config.yml`` on every run, so the toggle takes
effect without respawning the agent. An ``OSPREY_HOOK_DEBUG`` derived from that
key at spawn time would freeze it for the life of the process.
"""

from __future__ import annotations

import yaml

from osprey.agent_runner.clean_env import build_clean_env


def test_does_not_set_debug_from_config(tmp_path, monkeypatch):
    (tmp_path / "config.yml").write_text(yaml.dump({"hooks": {"debug": True}}))
    monkeypatch.delenv("OSPREY_CONFIG", raising=False)
    monkeypatch.delenv("OSPREY_HOOK_DEBUG", raising=False)

    env = build_clean_env(project_cwd=str(tmp_path))

    assert "OSPREY_HOOK_DEBUG" not in env


def test_manual_env_var_still_passes_through(tmp_path, monkeypatch):
    """An operator who exports the variable keeps it."""
    (tmp_path / "config.yml").write_text(yaml.dump({"hooks": {"debug": False}}))
    monkeypatch.delenv("OSPREY_CONFIG", raising=False)
    monkeypatch.setenv("OSPREY_HOOK_DEBUG", "1")

    env = build_clean_env(project_cwd=str(tmp_path))

    assert env["OSPREY_HOOK_DEBUG"] == "1"
