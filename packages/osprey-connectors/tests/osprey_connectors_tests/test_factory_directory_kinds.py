"""The connector factory builds the four directory kinds from a BackendSpec.

The panel and the directory MCP server never call ``register_builtin_connectors``,
so each create call has to find its builtins with nothing registered beforehand:
that is proved in a fresh interpreter, where no earlier test can have done it.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from typing import Any

import pytest

from osprey_connectors import factory
from osprey_connectors.alarm_state._phoebus_kafka import PhoebusAlarmState
from osprey_connectors.directory import ChannelDirectory
from osprey_connectors.factory import (
    _BUILTIN_ALARM_LOGS,
    _BUILTIN_ALARM_STATES,
    _BUILTIN_PUT_LOGS,
    _BUILTIN_PV_DIRECTORIES,
    _DIRECTORY_BUILTINS,
    BUILTIN_CONNECTORS,
    ConnectorFactory,
    register_builtin_connectors,
)
from osprey_connectors.service_answers import BackendSpec
from osprey_connectors_tests._harness import isolated_connector_registries

#: Written out rather than read from the table: a dropped row must still fail.
_EVERY_BUILTIN = (
    ("channelfinder", "directory", "create_pv_directory_connector", "ChannelFinderDirectory"),
    ("simulator_view", "directory", "create_pv_directory_connector", "SimulatorDirectory"),
    ("phoebus_alarm_logger", "alarm_log", "create_alarm_log_connector", "PhoebusAlarmLogger"),
    ("phoebus_alarm_server", "alarm_state", "create_alarm_state_connector", "PhoebusAlarmState"),
    ("alarm_replay", "alarm_state", "create_alarm_state_connector", "ReplayAlarmState"),
    ("caputlog", "put_log", "create_put_log_connector", "CaPutLog"),
)

_CREATE_BY_KIND = {
    "directory": "create_pv_directory_connector",
    "alarm_log": "create_alarm_log_connector",
    "alarm_state": "create_alarm_state_connector",
    "put_log": "create_put_log_connector",
}


def _stem(kind: str) -> str:
    """The factory's method and registry stem for *kind*; the directory kind keeps its old one."""
    return "pv_directory" if kind == "directory" else kind


def _spec(kind: str, type_name: str, **settings: Any) -> BackendSpec:
    return BackendSpec(
        kind=kind,  # type: ignore[arg-type]
        type=type_name,
        target="live",
        settings={"url": "https://directory.example.org", **settings},
        rehearsal=False,
    )


class _OutOfTreeBackend(ChannelDirectory):
    """Records what the factory handed its constructor."""

    def __init__(self, spec, *, project_dir=None, **seams):
        super().__init__(spec, project_dir=project_dir)
        self.seams = seams

    async def search(self, query, deadline):
        raise NotImplementedError

    async def count(self, query, deadline):
        raise NotImplementedError

    async def describe(self, name, deadline):
        raise NotImplementedError

    async def property_names(self, deadline):
        raise NotImplementedError

    async def property_values(self, name, deadline):
        raise NotImplementedError

    async def probe(self, deadline):
        raise NotImplementedError


@pytest.mark.parametrize(
    ("type_name", "kind", "create", "class_name"),
    _EVERY_BUILTIN,
    ids=[r[0] for r in _EVERY_BUILTIN],
)
def test_a_builtin_resolves_in_a_fresh_interpreter(type_name, kind, create, class_name):
    # aiokafka is blocked for every row: an optional client imported at module
    # scope would take the whole table down on a machine without the extra.
    code = textwrap.dedent(
        f"""
        import asyncio
        import sys

        sys.modules["aiokafka"] = None

        from osprey_connectors.factory import ConnectorFactory
        from osprey_connectors.service_answers import BackendSpec

        assert not ConnectorFactory._{_stem(kind)}_connectors
        spec = BackendSpec(
            kind={kind!r},
            type={type_name!r},
            target="live",
            settings={{"url": "https://directory.example.org"}},
            rehearsal=False,
        )
        backend = asyncio.run(ConnectorFactory.{create}(spec))
        print(type(backend).__name__)
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == class_name


def test_the_table_is_exactly_the_written_out_builtins():
    assert {(row.name, row.kind, row.class_name) for row in _DIRECTORY_BUILTINS} == {
        (type_name, kind, class_name) for type_name, kind, _, class_name in _EVERY_BUILTIN
    }


@pytest.mark.parametrize(
    ("names", "kind"),
    [
        (_BUILTIN_PV_DIRECTORIES, "directory"),
        (_BUILTIN_ALARM_LOGS, "alarm_log"),
        (_BUILTIN_ALARM_STATES, "alarm_state"),
        (_BUILTIN_PUT_LOGS, "put_log"),
    ],
    ids=["directory", "alarm_log", "alarm_state", "put_log"],
)
def test_each_builtin_tuple_is_its_kinds_rows(names, kind):
    assert set(names) == {row.name for row in _DIRECTORY_BUILTINS if row.kind == kind}
    with isolated_connector_registries(clear=True):
        register_builtin_connectors()

        assert set(names) == set(getattr(ConnectorFactory, f"_{_stem(kind)}_connectors"))


def test_the_directory_table_stays_private_and_apart():
    assert "_DIRECTORY_BUILTINS" not in factory.__all__
    assert "_DirectoryBuiltin" not in factory.__all__
    assert {row.kind for row in BUILTIN_CONNECTORS} == {"control_system", "archiver"}
    assert not {row.name for row in BUILTIN_CONNECTORS} & {row.name for row in _DIRECTORY_BUILTINS}


def test_registration_heals_a_cleared_kind_and_never_replaces():
    with isolated_connector_registries():
        ConnectorFactory._put_log_connectors.clear()
        ConnectorFactory.register_pv_directory("channelfinder", _OutOfTreeBackend)

        register_builtin_connectors()

        assert set(_BUILTIN_PUT_LOGS) <= set(ConnectorFactory._put_log_connectors)
        assert ConnectorFactory._pv_directory_connectors["channelfinder"] is _OutOfTreeBackend


def test_the_harness_restores_the_new_registries():
    register_builtin_connectors()
    before = dict(ConnectorFactory._alarm_log_connectors)

    with isolated_connector_registries(clear=True):
        assert not ConnectorFactory._alarm_log_connectors
        ConnectorFactory.register_alarm_log("mine", _OutOfTreeBackend)

    assert ConnectorFactory._alarm_log_connectors == before


async def test_a_dotted_class_resolves_for_every_kind():
    with isolated_connector_registries():
        for kind, create in _CREATE_BY_KIND.items():
            spec = _spec(kind, f"{__name__}._OutOfTreeBackend")

            backend = await getattr(ConnectorFactory, create)(spec)

            assert isinstance(backend, _OutOfTreeBackend)
            assert backend.spec is spec


@pytest.mark.parametrize(("kind", "create"), _CREATE_BY_KIND.items(), ids=_CREATE_BY_KIND)
async def test_an_unknown_type_raises_naming_it(kind, create):
    with pytest.raises(ValueError, match=rf"Unknown {kind} type: 'no_such_service'"):
        await getattr(ConnectorFactory, create)(_spec(kind, "no_such_service"))


async def test_a_dotted_type_that_does_not_import_raises_naming_it():
    with pytest.raises(ValueError, match="no_such_module_here"):
        await ConnectorFactory.create_put_log_connector(_spec("put_log", "no_such_module_here.Cls"))


async def test_a_dotted_type_with_no_such_class_raises_naming_it():
    spec = _spec("alarm_log", f"{__name__}.NoSuchClass")

    with pytest.raises(ValueError, match="NoSuchClass"):
        await ConnectorFactory.create_alarm_log_connector(spec)


async def test_a_seam_is_passed_only_when_it_is_set(tmp_path):
    with isolated_connector_registries():
        ConnectorFactory.register_pv_directory("mine", _OutOfTreeBackend)
        marker = object()

        backend = await ConnectorFactory.create_pv_directory_connector(
            _spec("directory", "mine"),
            project_dir=tmp_path,
            transport=marker,
            consumer_factory=None,
        )

    assert backend.project_dir == tmp_path
    assert backend.seams == {"transport": marker}


async def test_transport_reaches_an_http_backend():
    class _Transport:
        sent: list[tuple[str, str]] = []

        async def request(self, method, path, **_request):
            self.sent.append((method, path))
            return [{"name": "iocName"}, {"name": "recordType"}]

    transport = _Transport()
    backend = await ConnectorFactory.create_pv_directory_connector(
        _spec("directory", "channelfinder"), transport=transport
    )

    assert await backend.property_names(1.0) == ["iocName", "recordType"]
    assert transport.sent


async def test_consumer_factory_reaches_the_kafka_backend():
    def consumer_factory(_settings):
        raise AssertionError("the constructor does no I/O")

    backend = await ConnectorFactory.create_alarm_state_connector(
        _spec("alarm_state", "phoebus_alarm_server"), consumer_factory=consumer_factory
    )

    assert isinstance(backend, PhoebusAlarmState)
    assert backend._consumer_factory is consumer_factory
