"""The registry serves environment files beside entry-point plugins."""

from pathlib import Path

import pytest

from osprey.services.optimizer.environments.connector_env import ConnectorEnvironment
from osprey.services.optimizer.services import plugin_registry
from osprey.services.optimizer.services.plugin_registry import EnvironmentRegistry

ENV = """
name: {name}
variables:
  A:SP: {{min: 0, max: 1}}
observables:
  obj: {{channel: OBJ}}
"""


@pytest.fixture
def envdir(tmp_path, monkeypatch):
    d = tmp_path / "environments"
    d.mkdir()
    monkeypatch.setenv("OPTIMIZER_ENVIRONMENTS_DIR", str(d))
    monkeypatch.setattr(plugin_registry, "entry_points", lambda group: [])
    EnvironmentRegistry.reset()
    monkeypatch.setattr(plugin_registry, "TemplateService", lambda **_: _NoTemplates())
    yield d
    EnvironmentRegistry.reset()


class _NoTemplates:
    def scan_templates(self):
        return []

    def load_template_config(self, name):
        raise FileNotFoundError(name)


def test_environment_files_are_listed_and_loaded(envdir: Path):
    (envdir / "a.yml").write_text(ENV.format(name="alpha"))
    registry = EnvironmentRegistry()
    infos = registry.list_environments()
    assert [(i.name, i.source) for i in infos] == [("alpha", "file")]
    env = registry.load_environment("alpha", {"control_target": "va"})
    assert isinstance(env, ConnectorEnvironment)
    assert list(env.variables) == ["A:SP"]
    assert registry.get_environment_class("alpha") is ConnectorEnvironment


def test_a_name_declared_twice_fails_discovery(envdir: Path):
    (envdir / "a.yml").write_text(ENV.format(name="alpha"))
    (envdir / "b.yaml").write_text(ENV.format(name="alpha"))
    with pytest.raises(ValueError, match="declared twice"):
        EnvironmentRegistry().discover_environments()


def test_a_name_from_both_a_file_and_a_plugin_fails_discovery(envdir: Path, monkeypatch):
    (envdir / "a.yml").write_text(ENV.format(name="alpha"))
    plugin = type("EP", (), {"name": "alpha"})()
    monkeypatch.setattr(plugin_registry, "entry_points", lambda group: [plugin])
    with pytest.raises(ValueError, match="both a file and a plugin"):
        EnvironmentRegistry().discover_environments()


def test_a_broken_file_fails_discovery_instead_of_vanishing(envdir: Path):
    (envdir / "bad.yml").write_text("name: bad\nvariables: {}\nobservables: {}\n")
    with pytest.raises(ValueError):
        EnvironmentRegistry().discover_environments()


def test_no_directory_means_no_file_environments(envdir: Path, monkeypatch):
    monkeypatch.setenv("OPTIMIZER_ENVIRONMENTS_DIR", str(envdir / "missing"))
    assert EnvironmentRegistry().list_environments() == []


@pytest.mark.usefixtures("envdir")
def test_unknown_name_is_a_key_error():
    with pytest.raises(KeyError):
        EnvironmentRegistry().load_environment("nope", {})
