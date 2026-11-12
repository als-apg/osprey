"""Tests for the ``pv_directory`` section resolver and its context keys.

Replay files come from the test's own project directory and from a monkeypatched
demo directory, never from the packaged replay.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from osprey.services.pv_directory import _connectors, demo, resolve
from osprey.services.pv_directory.resolve import (
    KAFKA_CLIENT_MISSING,
    NO_SOURCE_WARNING,
    PREDICATES,
    ResolvedSection,
    SectionError,
    predicates,
    resolve_section,
)

_SIMULATOR = {"services": {"virtual_accelerator": {"port": 5064}}}
_NO_SIMULATOR = {"control_system": {"type": "epics"}}
_KAFKA = {"bootstrap_servers": "alarm.example:9092", "topic": "Accelerator"}


@pytest.fixture
def demo_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A demo directory holding ``example-alarms.ndjson`` in place of the packaged one."""
    directory = tmp_path / "packaged"
    directory.mkdir()
    (directory / "example-alarms.ndjson").write_text('{"feed": "stopped"}\n', encoding="utf-8")
    monkeypatch.setattr(demo, "_packaged_dir", lambda: directory)
    return directory


@pytest.fixture
def kafka_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(resolve, "_kafka_client_installed", lambda: True)


def _config(section: Any, base: dict[str, Any] | None = None) -> dict[str, Any]:
    return {**(base if base is not None else _SIMULATOR), "pv_directory": section}


def _facility_like() -> dict[str, Any]:
    return {
        "type": "channelfinder",
        "settings": {"url": "https://cf.example/ChannelFinder"},
        "description_property": "Description",
        "alarm_log": {"type": "phoebus_alarm_logger", "settings": {"url": "https://al.example"}},
        "alarm_state": {"type": "phoebus_alarm_server", "settings": dict(_KAFKA)},
        "put_log": {"type": "caputlog", "settings": {"url": "https://es.example"}},
    }


class TestNoSection:
    def test_absent_is_none(self) -> None:
        assert resolve_section(dict(_SIMULATOR)) is None

    def test_null_is_none(self) -> None:
        assert resolve_section(_config(None)) is None

    def test_no_section_has_every_key_false(self) -> None:
        assert predicates(None) == dict.fromkeys(PREDICATES, False)


class TestValidSections:
    @pytest.mark.usefixtures("demo_dir")
    def test_va_only(self) -> None:
        resolved = resolve_section(_config({"alarm_state": {"replay_file": "demo:example-alarms"}}))

        assert resolved is not None
        assert resolved.type is None
        assert resolved.replay_file == "demo:example-alarms"
        assert resolved.warnings == ()
        assert predicates(resolved) == {
            "pv_directory_configured": True,
            "pv_directory_alarm_log": False,
            "pv_directory_alarm_state": True,
            "pv_directory_alarm_config": True,
            "pv_directory_put_log": False,
        }

    def test_empty_section_on_the_simulator_baseline(self) -> None:
        """No ``control_system`` section is the simulator baseline: no warning."""
        resolved = resolve_section({"pv_directory": {}})

        assert resolved == ResolvedSection(section={})

    def test_no_simulator_warns_once_and_resolves(self) -> None:
        resolved = resolve_section(_config({}, _NO_SIMULATOR))

        assert resolved is not None
        assert resolved.warnings == (NO_SOURCE_WARNING,)
        assert predicates(resolved)["pv_directory_configured"] is True

    @pytest.mark.usefixtures("kafka_client")
    def test_facility_like(self) -> None:
        resolved = resolve_section(_config(_facility_like(), _NO_SIMULATOR))

        assert resolved is not None
        assert (resolved.type, resolved.alarm_log, resolved.alarm_state, resolved.put_log) == (
            "channelfinder",
            "phoebus_alarm_logger",
            "phoebus_alarm_server",
            "caputlog",
        )
        assert resolved.warnings == ()
        assert all(predicates(resolved).values())

    def test_alarm_log_alone_turns_on_alarm_config(self) -> None:
        section = {"alarm_log": {"type": "phoebus_alarm_logger", "settings": {"url": "https://x"}}}

        keys = predicates(resolve_section(_config(section)))

        assert keys["pv_directory_alarm_config"] is True
        assert keys["pv_directory_alarm_state"] is False

    @pytest.mark.usefixtures("kafka_client")
    def test_type_plus_project_replay(self, tmp_path: Path) -> None:
        (tmp_path / "alarms").mkdir()
        (tmp_path / "alarms" / "feed.ndjson").write_text("", encoding="utf-8")
        section = _facility_like()
        section["alarm_state"]["replay_file"] = "alarms/feed.ndjson"

        resolved = resolve_section(_config(section), project_dir=tmp_path)

        assert resolved is not None
        assert resolved.alarm_state == "phoebus_alarm_server"
        assert resolved.replay_file == "alarms/feed.ndjson"

    def test_project_replay_unchecked_without_project_dir(self) -> None:
        resolved = resolve_section(_config({"alarm_state": {"replay_file": "missing.ndjson"}}))

        assert resolved is not None
        assert resolved.replay_file == "missing.ndjson"

    def test_kafka_without_client_warns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(resolve, "_kafka_client_installed", lambda: False)

        resolved = resolve_section(_config(_facility_like()))

        assert resolved is not None
        assert resolved.warnings == (f"pv_directory.alarm_state: {KAFKA_CLIENT_MISSING}",)

    @pytest.mark.usefixtures("demo_dir")
    def test_replay_without_client_does_not_warn(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(resolve, "_kafka_client_installed", lambda: False)

        resolved = resolve_section(_config({"alarm_state": {"replay_file": "demo:example-alarms"}}))

        assert resolved is not None
        assert resolved.warnings == ()

    def test_floor_connectors_warn_once_without_consulting_the_simulator(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(_connectors, "DIRECTORY_CONNECTORS_AVAILABLE", False)
        monkeypatch.setattr(_connectors, "has_simulator", None)

        resolved = resolve_section(_config({}, _NO_SIMULATOR))

        assert resolved is not None
        assert resolved.warnings == (f"pv_directory: {_connectors.FLOOR_REASON}",)


class TestRefusals:
    @pytest.mark.parametrize(
        ("section", "key"),
        [
            ({"type": "channelfinder"}, "pv_directory.settings.url"),
            ({"type": "channelfinder", "settings": {"url": " "}}, "pv_directory.settings.url"),
            (
                {"alarm_log": {"type": "phoebus_alarm_logger"}},
                "pv_directory.alarm_log.settings.url",
            ),
            ({"put_log": {"type": "caputlog"}}, "pv_directory.put_log.settings.url"),
        ],
    )
    def test_type_without_url(self, section: dict[str, Any], key: str) -> None:
        with pytest.raises(SectionError) as raised:
            resolve_section(_config(section))

        assert raised.value.key == key

    @pytest.mark.parametrize("missing", ["bootstrap_servers", "topic"])
    def test_kafka_without_broker_or_topic(self, missing: str) -> None:
        settings = {name: value for name, value in _KAFKA.items() if name != missing}
        section = {"alarm_state": {"type": "phoebus_alarm_server", "settings": settings}}

        with pytest.raises(SectionError) as raised:
            resolve_section(_config(section))

        assert raised.value.key == f"pv_directory.alarm_state.settings.{missing}"

    @pytest.mark.parametrize("value", ["demo:no-such-replay", "demo:", "demo:../escape"])
    @pytest.mark.usefixtures("demo_dir")
    def test_demo_replay_that_does_not_resolve(self, value: str) -> None:
        with pytest.raises(SectionError) as raised:
            resolve_section(_config({"alarm_state": {"replay_file": value}}))

        assert raised.value.key == "pv_directory.alarm_state.replay_file"

    def test_project_replay_that_does_not_resolve(self, tmp_path: Path) -> None:
        with pytest.raises(SectionError) as raised:
            resolve_section(
                _config({"alarm_state": {"replay_file": "missing.ndjson"}}),
                project_dir=tmp_path,
            )

        assert raised.value.key == "pv_directory.alarm_state.replay_file"

    @pytest.mark.parametrize(
        ("section", "key"),
        [
            ({"type": "simulator_view", "settings": {"url": "x"}}, "pv_directory.type"),
            ({"type": "alarm_replay", "settings": {"url": "x"}}, "pv_directory.type"),
            ({"alarm_state": {"type": "alarm_replay"}}, "pv_directory.alarm_state.type"),
            ({"alarm_log": {"type": "simulator_view"}}, "pv_directory.alarm_log.type"),
        ],
    )
    def test_type_naming_an_internal_builtin(self, section: dict[str, Any], key: str) -> None:
        with pytest.raises(SectionError) as raised:
            resolve_section(_config(section))

        assert raised.value.key == key

    @pytest.mark.parametrize(
        ("section", "key"),
        [
            ("on", "pv_directory"),
            ({"alarm_state": "demo:example-alarms"}, "pv_directory.alarm_state"),
            ({"type": "channelfinder", "settings": "https://x"}, "pv_directory.settings"),
        ],
    )
    def test_block_that_is_not_a_mapping(self, section: Any, key: str) -> None:
        with pytest.raises(SectionError) as raised:
            resolve_section(_config(section))

        assert raised.value.key == key


class TestReplayFile:
    def test_demo_name_resolves_into_the_packaged_directory(self, demo_dir: Path) -> None:
        path = demo.resolve_replay_file("demo:example-alarms", None)

        assert path == demo_dir / "example-alarms.ndjson"

    def test_relative_path_joins_the_project_directory(self, tmp_path: Path) -> None:
        assert demo.resolve_replay_file("a/b.ndjson", tmp_path) == tmp_path / "a" / "b.ndjson"


class TestInternalTypesPin:
    def test_constants_equal_the_connectors(self) -> None:
        pytest.importorskip("osprey_connectors.service_targets")
        from osprey_connectors import factory, service_targets

        kafka_rows = {
            row.name
            for row in factory._DIRECTORY_BUILTINS
            if row.module_path == "osprey_connectors.alarm_state._phoebus_kafka"
        }

        assert resolve.SIMULATOR_VIEW == service_targets._SIMULATOR_VIEW
        assert resolve.ALARM_REPLAY == service_targets._ALARM_REPLAY
        assert kafka_rows == {resolve.KAFKA_ALARM_STATE}
