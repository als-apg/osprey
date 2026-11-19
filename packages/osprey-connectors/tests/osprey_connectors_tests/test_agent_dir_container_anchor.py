"""Where the agent-data root anchors when the configured project root is not here.

A container runs with a ``project_root`` written on the host, so the configured
path frequently does not exist inside the image. What to do then used to be a
guess: try ``/app``, ``/pipelines``, ``/jupyter`` in turn and take the first that
happens to hold an agent-data directory. Two of those names belong to an
execution method that no longer ships, and the one that remains never fires in
the shipped layout — the image puts the project at ``/app/<project>/``, not at
``/app``.

The loaded config's own repo root names the project root by construction: the
file this process read, not an environment variable that may name another.
:attr:`~osprey_connectors.config.LoadedConfig.agent_data_root` derives from it.
"""

from __future__ import annotations

from pathlib import Path

from osprey_connectors import config as connectors_config


def _config(directory: Path, project_root: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "config.yml"
    path.write_text(f"project_root: {project_root}\n", encoding="utf-8")
    return path


def test_the_loaded_configs_repo_names_the_project_root(monkeypatch, tmp_path):
    project = tmp_path / "facility"
    monkeypatch.setenv("OSPREY_CONFIG", str(_config(project, tmp_path / "does-not-exist")))

    resolved = connectors_config.current_config().agent_data_root

    assert resolved.is_relative_to(project)
    assert not resolved.is_relative_to(tmp_path / "does-not-exist")


def test_a_config_file_export_naming_another_project_is_not_the_anchor(monkeypatch, tmp_path):
    """No container root is guessed, and no stray ``CONFIG_FILE`` is believed either."""
    project = tmp_path / "facility"
    other = tmp_path / "other"
    monkeypatch.setenv("OSPREY_CONFIG", str(_config(project, tmp_path / "does-not-exist")))
    monkeypatch.setenv("CONFIG_FILE", str(_config(other, tmp_path / "does-not-exist")))

    resolved = connectors_config.current_config().agent_data_root

    assert resolved.is_relative_to(project)
    assert not resolved.is_relative_to(other)
