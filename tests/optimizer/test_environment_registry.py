"""The registry serves environment files beside entry-point plugins."""

from pathlib import Path

import pytest

from osprey_optimizer.environments.connector_env import ConnectorEnvironment
from osprey_optimizer.services import plugin_registry
from osprey_optimizer.services.plugin_registry import EnvironmentRegistry

ENV = """
name: {name}
variables:
  A:SP: {{min: 0, max: 1, step_size: 0.1}}
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
    yield d
    EnvironmentRegistry.reset()


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


PRESETS = """
presets:
  narrow:
    description: A within a tenth of its range
    variables: [A:SP]
    variable_overrides:
      A:SP: {window_fraction: 0.1}
    algorithm: upper_confidence_bound
    n_iterations: 15
    objective_name: obj
"""


def test_a_file_offers_its_presets(envdir: Path):
    (envdir / "a.yml").write_text(ENV.format(name="alpha") + PRESETS)
    presets = EnvironmentRegistry().get_presets("alpha")
    assert list(presets) == ["narrow"]
    assert presets["narrow"].variables == ["A:SP"]
    assert presets["narrow"].variable_overrides["A:SP"].window_fraction == 0.1


def test_a_file_without_presets_offers_none(envdir: Path):
    (envdir / "a.yml").write_text(ENV.format(name="alpha"))
    assert EnvironmentRegistry().get_presets("alpha") == {}


@pytest.mark.parametrize(
    ("preset", "match"),
    [
        pytest.param("{variables: [B:SP]}", "unknown variable", id="unknown_variable"),
        pytest.param(
            "{variable_overrides: {B:SP: {window_abs: 1}}}",
            "unknown variable",
            id="unknown_override",
        ),
        pytest.param(
            "{objective_name: nope}", "not a selectable observable", id="unknown_objective"
        ),
        pytest.param(
            "{variables: [A:SP], variable_overrides: {C:SP: {window_abs: 1}}}",
            "leaves out",
            id="override_outside_its_variables",
        ),
    ],
)
def test_a_preset_naming_what_the_file_lacks_fails_discovery(envdir: Path, preset: str, match: str):
    (envdir / "a.yml").write_text(ENV.format(name="alpha") + f"presets:\n  p: {preset}\n")
    with pytest.raises(ValueError, match=match):
        EnvironmentRegistry().discover_environments()


def _plugin(presets):
    """An entry point named ``plug`` whose class offers ``presets``."""
    members = dict.fromkeys(plugin_registry._REQUIRED_PROTOCOL_MEMBERS)
    cls = type("Plugin", (), {**members, "PRESETS": presets})
    return type("EP", (), {"name": "plug", "load": staticmethod(lambda: cls)})()


@pytest.mark.usefixtures("envdir")
def test_a_plugin_offers_presets_from_its_class(monkeypatch):
    plugin = _plugin({"quick": {"variables": ["x"], "n_iterations": 5}})
    monkeypatch.setattr(plugin_registry, "entry_points", lambda group: [plugin])
    presets = EnvironmentRegistry().get_presets("plug")
    assert presets["quick"].variables == ["x"]
    assert presets["quick"].n_iterations == 5


@pytest.mark.usefixtures("envdir")
def test_a_plugin_with_a_bad_preset_is_refused(monkeypatch):
    plugin = _plugin({"q": {"n_iterations": 0}})
    monkeypatch.setattr(plugin_registry, "entry_points", lambda group: [plugin])
    with pytest.raises(ValueError, match="invalid preset"):
        EnvironmentRegistry().get_presets("plug")


@pytest.mark.usefixtures("envdir")
def test_presets_of_an_unknown_environment_is_a_key_error():
    with pytest.raises(KeyError):
        EnvironmentRegistry().get_presets("nope")
