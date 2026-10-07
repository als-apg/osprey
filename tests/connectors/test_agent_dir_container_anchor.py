"""Where ``get_agent_dir`` anchors when the configured project root is not here.

A container runs with a ``project_root`` written on the host, so the configured
path frequently does not exist inside the image. What to do then used to be a
guess: try ``/app``, ``/pipelines``, ``/jupyter`` in turn and take the first that
happens to hold an agent-data directory. Two of those names belong to an
execution method that no longer ships, and the one that remains never fires in
the shipped layout — the image puts the project at ``/app/<project>/``, not at
``/app``.

``CONFIG_FILE`` already names the config this process actually loaded, and its
directory IS the project root by construction. These tests hold the fallback to
that, and to the cwd-with-a-warning it always had when even that is unset.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey_connectors import config as connectors_config


@pytest.fixture
def agent_dir(monkeypatch):
    """Call ``get_agent_dir`` against a config whose project root is elsewhere."""

    def _call(project_root: str, raw: dict | None = None):
        class _Config:
            raw_config = raw or {}

            def get(self, key, default=None):
                return {"project_root": project_root, "file_paths": {}}.get(key, default)

        monkeypatch.setattr(connectors_config, "_get_config", lambda: _Config())
        return Path(connectors_config.get_agent_dir("api_calls_dir"))

    return _call


def test_config_file_names_the_project_root(agent_dir, monkeypatch, tmp_path):
    project = tmp_path / "facility"
    project.mkdir()
    (project / "config.yml").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("CONFIG_FILE", str(project / "config.yml"))

    resolved = agent_dir(str(tmp_path / "does-not-exist"))

    assert resolved.is_relative_to(project)
    assert resolved.name == "api_calls_dir"


def test_without_config_file_it_falls_back_to_the_cwd_with_a_warning(
    agent_dir, monkeypatch, tmp_path, caplog
):
    """With nothing to anchor on, the path resolves under the cwd, and says so.

    No container root is guessed: ``/app`` is not a project root in the shipped
    layout, and never was one this could prove. Resolving under the cwd pins
    that as well as any list of forbidden prefixes would.
    """
    monkeypatch.delenv("CONFIG_FILE", raising=False)
    monkeypatch.chdir(tmp_path)

    with caplog.at_level("WARNING"):
        resolved = agent_dir(str(tmp_path / "does-not-exist"))

    # resolve(): on macOS tmp_path lives under /var, a symlink to /private/var.
    assert resolved.is_relative_to(tmp_path.resolve())
    assert not resolved.is_relative_to(tmp_path / "does-not-exist")
    assert resolved.name == "api_calls_dir"
    assert "Falling back to relative path resolution" in caplog.text
