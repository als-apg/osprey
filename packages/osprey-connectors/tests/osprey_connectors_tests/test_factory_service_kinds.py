"""The connector factory builds the four service kinds from a BackendSpec.

One kind-keyed pair serves them: ``register_service(kind, name, cls)`` and
``create_service(spec)``, which reads ``spec.kind``. The panel and the directory
MCP server never call ``register_builtin_connectors``, so each create call has to
find its builtins with nothing registered beforehand: that is proved in a fresh
interpreter, where no earlier test can have done it.
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
    _SERVICE_BUILTINS,
    BUILTIN_CONNECTORS,
    ConnectorFactory,
    register_builtin_connectors,
)
from osprey_connectors.service_answers import BackendSpec
from osprey_connectors_tests._harness import isolated_connector_registries

KINDS = ("directory", "alarm_state", "alarm_log", "put_log")

#: Written out rather than read from the table: a dropped row must still fail.
_EVERY_BUILTIN = (
    ("channelfinder", "directory", "ChannelFinderDirectory"),
    ("simulator_view", "directory", "SimulatorDirectory"),
    ("phoebus_alarm_logger", "alarm_log", "PhoebusAlarmLogger"),
    ("phoebus_alarm_server", "alarm_state", "PhoebusAlarmState"),
    ("alarm_replay", "alarm_state", "ReplayAlarmState"),
    ("caputlog", "put_log", "CaPutLog"),
)


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
    ("type_name", "kind", "class_name"), _EVERY_BUILTIN, ids=[r[0] for r in _EVERY_BUILTIN]
)
def test_a_builtin_resolves_in_a_fresh_interpreter(type_name, kind, class_name):
    # aiokafka is blocked for every row: an optional client imported at module
    # scope would take the whole table down on a machine without the extra.
    code = textwrap.dedent(
        f"""
        import asyncio
        import sys

        sys.modules["aiokafka"] = None

        from osprey_connectors.factory import ConnectorFactory
        from osprey_connectors.service_answers import BackendSpec

        assert not ConnectorFactory._service_backends[{kind!r}]
        spec = BackendSpec(
            kind={kind!r},
            type={type_name!r},
            target="live",
            settings={{"url": "https://directory.example.org"}},
            rehearsal=False,
        )
        backend = asyncio.run(ConnectorFactory.create_service(spec))
        print(type(backend).__name__)
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == class_name


def test_the_table_is_exactly_the_written_out_builtins():
    assert {(row.name, row.kind, row.class_name) for row in _SERVICE_BUILTINS} == set(
        _EVERY_BUILTIN
    )


def test_the_registry_has_one_entry_per_kind():
    assert set(ConnectorFactory._service_backends) == set(KINDS)


@pytest.mark.parametrize("kind", KINDS)
def test_registration_fills_each_kind_with_its_rows(kind):
    with isolated_connector_registries(clear=True):
        register_builtin_connectors()

        assert set(ConnectorFactory._service_backends[kind]) == {
            name for name, row_kind, _ in _EVERY_BUILTIN if row_kind == kind
        }


def test_the_service_table_stays_private_and_apart():
    assert "_SERVICE_BUILTINS" not in factory.__all__
    assert "_ServiceBuiltin" not in factory.__all__
    assert {row.kind for row in BUILTIN_CONNECTORS} == {"control_system", "archiver"}
    assert not {row.name for row in BUILTIN_CONNECTORS} & {row.name for row in _SERVICE_BUILTINS}


@pytest.mark.parametrize("kind", KINDS)
def test_register_service_reaches_its_kinds_registry(kind):
    with isolated_connector_registries():
        ConnectorFactory.register_service(kind, "mine", _OutOfTreeBackend)

        assert ConnectorFactory._service_backends[kind]["mine"] is _OutOfTreeBackend


def test_register_service_refuses_an_unknown_kind_naming_the_four():
    with pytest.raises(ValueError, match="pv_directory") as raised:
        ConnectorFactory.register_service("pv_directory", "mine", _OutOfTreeBackend)

    for kind in KINDS:
        assert kind in str(raised.value)


async def test_create_service_refuses_an_unknown_kind_naming_the_four():
    with pytest.raises(ValueError, match="logbook") as raised:
        await ConnectorFactory.create_service(_spec("logbook", "channelfinder"))

    for kind in KINDS:
        assert kind in str(raised.value)


def test_registration_heals_a_cleared_kind_and_never_replaces():
    with isolated_connector_registries():
        ConnectorFactory._service_backends["put_log"].clear()
        ConnectorFactory.register_service("directory", "channelfinder", _OutOfTreeBackend)

        register_builtin_connectors()

        assert set(ConnectorFactory._service_backends["put_log"]) == {"caputlog"}
        assert ConnectorFactory._service_backends["directory"]["channelfinder"] is (
            _OutOfTreeBackend
        )


def test_the_harness_restores_the_service_registries():
    register_builtin_connectors()
    before = {kind: dict(names) for kind, names in ConnectorFactory._service_backends.items()}

    with isolated_connector_registries(clear=True):
        assert not any(ConnectorFactory._service_backends.values())
        ConnectorFactory.register_service("alarm_log", "mine", _OutOfTreeBackend)

    assert ConnectorFactory._service_backends == before


@pytest.mark.parametrize("kind", KINDS)
async def test_a_dotted_class_resolves_for_every_kind(kind):
    with isolated_connector_registries():
        spec = _spec(kind, f"{__name__}._OutOfTreeBackend")

        backend = await ConnectorFactory.create_service(spec)

        assert isinstance(backend, _OutOfTreeBackend)
        assert backend.spec is spec
        assert ConnectorFactory._service_backends[kind][spec.type] is _OutOfTreeBackend


@pytest.mark.parametrize("kind", KINDS)
async def test_an_unknown_type_raises_naming_it(kind):
    with pytest.raises(ValueError, match=rf"Unknown {kind} type: 'no_such_service'"):
        await ConnectorFactory.create_service(_spec(kind, "no_such_service"))


async def test_a_dotted_type_that_does_not_import_raises_naming_it():
    with pytest.raises(ValueError, match="no_such_module_here"):
        await ConnectorFactory.create_service(_spec("put_log", "no_such_module_here.Cls"))


async def test_a_dotted_type_with_no_such_class_raises_naming_it():
    with pytest.raises(ValueError, match="NoSuchClass"):
        await ConnectorFactory.create_service(_spec("alarm_log", f"{__name__}.NoSuchClass"))


async def test_a_seam_is_passed_only_when_it_is_set(tmp_path):
    with isolated_connector_registries():
        ConnectorFactory.register_service("directory", "mine", _OutOfTreeBackend)
        marker = object()

        backend = await ConnectorFactory.create_service(
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
    backend = await ConnectorFactory.create_service(
        _spec("directory", "channelfinder"), transport=transport
    )

    assert await backend.property_names(1.0) == ["iocName", "recordType"]
    assert transport.sent


async def test_consumer_factory_reaches_the_kafka_backend():
    def consumer_factory(_settings):
        raise AssertionError("the constructor does no I/O")

    backend = await ConnectorFactory.create_service(
        _spec("alarm_state", "phoebus_alarm_server"), consumer_factory=consumer_factory
    )

    assert isinstance(backend, PhoebusAlarmState)
    assert backend._consumer_factory is consumer_factory
