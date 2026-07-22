"""Unit tests for :mod:`osprey.health.loader`.

The loader is the synchronous phase of the health web view's refresh cycle. It
shares record-assembly with the CLI (:mod:`osprey.health.records`) but adds the
persistent-process behaviors the CLI never needs: config-path resolution per
cycle, an mtime/size change gate that avoids both YAML re-reads and ``os.environ``
mutation when nothing moved, edit observation between cycles, and a degraded
fallback to default settings so a broken config never stalls the scheduler.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from osprey.health.config import HealthSettings
from osprey.health.loader import HealthConfigLoader, LoadedHealthConfig

_VALID_CONFIG = """\
project_name: test_project
models:
  python_code_generator:
    provider: mock
    model_id: mock-model
"""


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


@pytest.fixture(autouse=True)
def _restore_environ():
    """Snapshot/restore ``os.environ`` — the loader mutates it via ``.env`` loads."""
    saved = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    proj = tmp_path / "proj"
    _write(proj / "config.yml", _VALID_CONFIG)
    return proj


# --------------------------------------------------------------------------- #
# Config-path resolution (config_path=None)
# --------------------------------------------------------------------------- #


class TestPathResolution:
    def test_explicit_override_is_used(self, project):
        loader = HealthConfigLoader(project / "config.yml")
        result = loader.load()
        assert result.config_ok is True
        assert result.expanded["project_name"] == "test_project"

    def test_none_resolves_osprey_config_env(self, project, monkeypatch):
        monkeypatch.setenv("OSPREY_CONFIG", str(project / "config.yml"))
        # Run from an unrelated cwd to prove resolution came from the env var.
        monkeypatch.chdir(project.parent)
        loader = HealthConfigLoader(None)
        result = loader.load()
        assert result.config_ok is True
        assert result.expanded["project_name"] == "test_project"

    def test_none_resolves_cwd_config(self, project, monkeypatch):
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)
        monkeypatch.chdir(project)
        loader = HealthConfigLoader(None)
        result = loader.load()
        assert result.config_ok is True


# --------------------------------------------------------------------------- #
# Return contract and record assembly
# --------------------------------------------------------------------------- #


class TestReturnContract:
    def test_returns_loaded_health_config_tuple(self, project):
        result = HealthConfigLoader(project / "config.yml").load()
        assert isinstance(result, LoadedHealthConfig)
        # Positional unpacking matches the documented field order.
        records, extra_rows, settings, expanded, control_system, config_ok = result
        assert isinstance(records, list) and records
        assert extra_rows == []
        assert isinstance(settings, HealthSettings)
        assert expanded is not None
        assert control_system == {}
        assert config_ok is True

    def test_records_assembled_via_health_records(self, project):
        result = HealthConfigLoader(project / "config.yml").load()
        names = {r.name for r in result.records}
        # Core categories are always present on a healthy config.
        assert {"configuration", "file_system", "providers"} <= names

    def test_control_system_mapping_extracted(self, tmp_path):
        _write(
            tmp_path / "config.yml",
            _VALID_CONFIG + "control_system:\n  type: mock\n",
        )
        result = HealthConfigLoader(tmp_path / "config.yml").load()
        assert result.control_system == {"type": "mock"}


# --------------------------------------------------------------------------- #
# Degraded path (missing / broken config)
# --------------------------------------------------------------------------- #


class TestDegradedDefaults:
    def test_missing_config_degrades_to_default_settings(self, tmp_path):
        result = HealthConfigLoader(tmp_path / "config.yml").load()
        assert result.config_ok is False
        # Settings are never None — default cadence keeps the scheduler alive.
        assert result.settings.suite_timeout_s == 30.0
        assert result.settings.interval_s == 60.0
        # The report still assembles: config-dependent core categories degrade to
        # skip records rather than vanishing.
        assert result.records
        assert result.extra_rows == []

    def test_broken_yaml_never_raises(self, tmp_path):
        _write(tmp_path / "config.yml", "invalid: yaml: content:\n")
        result = HealthConfigLoader(tmp_path / "config.yml").load()  # must not raise
        assert result.config_ok is False
        assert result.settings.suite_timeout_s == 30.0
        assert result.settings.interval_s == 60.0

    def test_invalid_health_section_degrades(self, tmp_path):
        _write(
            tmp_path / "config.yml",
            "project_name: bad\nhealth:\n  suite_timeout_s: not-a-number\n",
        )
        result = HealthConfigLoader(tmp_path / "config.yml").load()
        assert result.config_ok is False
        assert result.settings.suite_timeout_s == 30.0


# --------------------------------------------------------------------------- #
# mtime/size gate: caching, no env mutation on unchanged, edit observation
# --------------------------------------------------------------------------- #


class TestChangeGate:
    def test_unchanged_returns_cached_object(self, project):
        loader = HealthConfigLoader(project / "config.yml")
        first = loader.load()
        second = loader.load()
        # No disk change → the same cached result object, no re-assembly.
        assert first is second

    def test_no_env_mutation_when_env_unchanged(self, project):
        (project / ".env").write_text("OSPREY_LOADER_CANARY=from-env\n")
        loader = HealthConfigLoader(project / "config.yml")

        loader.load()
        assert os.environ.get("OSPREY_LOADER_CANARY") == "from-env"

        # Remove the canary and reload without touching any file: the loader must
        # NOT re-run load_dotenv, so the canary stays absent.
        del os.environ["OSPREY_LOADER_CANARY"]
        loader.load()
        assert "OSPREY_LOADER_CANARY" not in os.environ

    def test_env_change_reloads_dotenv(self, project):
        env_path = project / ".env"
        env_path.write_text("OSPREY_LOADER_CANARY=v1\n")
        loader = HealthConfigLoader(project / "config.yml")
        loader.load()
        assert os.environ["OSPREY_LOADER_CANARY"] == "v1"

        # Rewrite .env with a later mtime; the next cycle must observe it.
        env_path.write_text("OSPREY_LOADER_CANARY=v2\n")
        _bump_mtime(env_path)
        loader.load()
        assert os.environ["OSPREY_LOADER_CANARY"] == "v2"

    def test_edit_observed_between_loads(self, project):
        config_path = project / "config.yml"
        loader = HealthConfigLoader(config_path)
        first = loader.load()
        assert first.expanded["project_name"] == "test_project"

        config_path.write_text("project_name: edited_project\n")
        _bump_mtime(config_path)
        second = loader.load()
        assert second is not first
        assert second.expanded["project_name"] == "edited_project"


def _bump_mtime(path: Path) -> None:
    """Force a strictly-later mtime so the change gate fires deterministically."""
    st = path.stat()
    later = st.st_mtime + 10
    os.utime(path, (later, later))
