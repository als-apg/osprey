"""Where a lookup that names no config finds one, and what happens when it cannot.

A process asks "where is my config" in two ways: :func:`resolve_config_path`
names the rendered file, and the default singleton behind
:func:`get_config_value` loads one for every lookup that names no path. They
have to give one answer. A project directory keeps its render at
``build/config.yml``, and a process started there with neither variable set
used to load the render through the first and nothing through the second, so a
connector built from "the config" was built from an empty one.

The factory's no-argument forms then turned that load failure into the
simulator, whose own refusal says to run ``osprey build``. These tests hold
them to reporting the load failure instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey_connectors import types
from osprey_connectors.config import ConfigBuilder, default_config_path, get_config_value
from osprey_connectors.factory import ConnectorFactory
from osprey_connectors.workspace import load_osprey_config, resolve_config_path

ARCHIVER_URL = "https://archiver.example.org:8443"


def write_config(path: Path, archiver_type: str = types.EPICS_ARCHIVER) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"archiver:\n  type: {archiver_type}\n  settings:\n    url: {ARCHIVER_URL}\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def project(tmp_path, monkeypatch) -> Path:
    """A project directory to stand in, holding nothing yet."""
    root = tmp_path / "project"
    root.mkdir()
    monkeypatch.chdir(root)
    return root


class RecordingArchiver:
    """Stands in for a facility archiver: records the settings it was connected with."""

    settings: dict | None = None

    async def connect(self, config: dict) -> None:
        type(self).settings = config


@pytest.fixture
def facility_archiver(monkeypatch) -> type[RecordingArchiver]:
    RecordingArchiver.settings = None
    monkeypatch.setitem(
        ConnectorFactory._archiver_connectors, types.EPICS_ARCHIVER, RecordingArchiver
    )
    return RecordingArchiver


def test_a_lookup_in_a_project_directory_reads_the_render_under_build(project):
    rendered = write_config(project / "build" / "config.yml")

    assert get_config_value("archiver.type") == types.EPICS_ARCHIVER
    assert Path(default_config_path()) == rendered


def test_the_default_config_is_the_one_resolve_config_path_names(project):
    write_config(project / "build" / "config.yml")

    builder = ConfigBuilder()

    assert builder.config_path == resolve_config_path()
    assert builder.raw_config == load_osprey_config()


@pytest.mark.usefixtures("project")
def test_a_lookup_reads_the_config_osprey_config_names(tmp_path, monkeypatch):
    elsewhere = write_config(tmp_path / "mounted" / "config.yml")
    monkeypatch.setenv("OSPREY_CONFIG", str(elsewhere))

    assert get_config_value("archiver.settings.url") == ARCHIVER_URL
    assert Path(default_config_path()) == elsewhere


def test_config_file_outranks_the_resolved_path(project, tmp_path, monkeypatch):
    write_config(project / "build" / "config.yml", types.MOCK_ARCHIVER)
    handed = write_config(tmp_path / "handed" / "config.yml")
    monkeypatch.setenv("CONFIG_FILE", str(handed))

    assert get_config_value("archiver.type") == types.EPICS_ARCHIVER


def test_a_config_at_the_directory_root_is_still_found(project):
    flat = write_config(project / "config.yml")

    assert get_config_value("archiver.type") == types.EPICS_ARCHIVER
    assert Path(default_config_path()) == flat


def test_no_config_anywhere_names_both_places_it_looked(project):
    with pytest.raises(FileNotFoundError, match="No config.yml found") as caught:
        ConfigBuilder()

    message = str(caught.value)
    assert str(project / "build" / "config.yml") in message
    assert str(project / "config.yml") in message


@pytest.mark.usefixtures("project")
def test_a_missing_osprey_config_target_is_named(tmp_path, monkeypatch):
    missing = tmp_path / "not-mounted" / "config.yml"
    monkeypatch.setenv("OSPREY_CONFIG", str(missing))

    with pytest.raises(FileNotFoundError, match="OSPREY_CONFIG") as caught:
        ConfigBuilder()

    assert str(missing) in str(caught.value)


@pytest.mark.asyncio
async def test_the_archiver_built_with_no_argument_is_the_renders(project, facility_archiver):
    write_config(project / "build" / "config.yml")

    connector = await ConnectorFactory.create_archiver_connector()

    assert isinstance(connector, facility_archiver)
    assert facility_archiver.settings == {"url": ARCHIVER_URL}


@pytest.mark.asyncio
@pytest.mark.usefixtures("project")
async def test_no_config_refuses_the_archiver_instead_of_building_the_mock(caplog):
    with pytest.raises(FileNotFoundError, match="No config.yml found"):
        await ConnectorFactory.create_archiver_connector()

    assert types.MOCK_ARCHIVER not in caplog.text


@pytest.mark.usefixtures("project")
def test_no_config_refuses_the_control_system_instead_of_building_the_simulator():
    with pytest.raises(FileNotFoundError, match="No config.yml found"):
        ConnectorFactory.build_control_system_connector()


@pytest.mark.asyncio
async def test_a_render_with_no_archiver_section_still_builds_the_mock(project, monkeypatch):
    (project / "build").mkdir()
    (project / "build" / "config.yml").write_text("system:\n  timezone: UTC\n", encoding="utf-8")
    built: list[dict] = []

    class Mock:
        async def connect(self, config: dict) -> None:
            built.append(config)

    monkeypatch.setitem(ConnectorFactory._archiver_connectors, types.MOCK_ARCHIVER, Mock)

    connector = await ConnectorFactory.create_archiver_connector()

    assert isinstance(connector, Mock)
    assert built == [{}]
