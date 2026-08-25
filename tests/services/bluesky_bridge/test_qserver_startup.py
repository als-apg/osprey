"""Unit tests for the queueserver RE-worker startup script.

No queueserver process, no Redis, no IOC and no Tiled server is involved: the
startup script is a plain module, so its three products — the device set, the
plan namespace, and the document-plane subscriptions — are each built and
inspected directly.

Two doubles carry the suite. ``FakeConnector`` stands in for the OSPREY
control-system connector so real ``ConnectorSettable``/``ConnectorReadable``
instances can be built without Channel Access (the same idiom
``test_connector_devices.py`` uses). ``FakeRunEngine`` stands in for the
RunEngine wherever only ``subscribe`` matters; the one test that needs a real
event loop uses a real ``RunEngine``.

The namespace-shape test runs the actual queueserver namespace scanner
(``existing_plans_and_devices_from_nspace``) over the namespace this module
builds. That is the contract that matters — a plan the scanner does not
recognize is silently invisible to the manager, not an error — and
``bluesky_queueserver`` is a hard dependency of ``bluesky-queueserver-api``,
which OSPREY depends on directly, so the check always runs.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from osprey.services.bluesky_bridge import plan_loader, qserver_startup
from osprey.services.bluesky_bridge.devices.connector import ConnectorReadable, ConnectorSettable

_PLAN_DIRS_ENV = "BLUESKY_PLAN_DIRS"
_PLAN_MODULE_ENV = "BLUESKY_PLAN_MODULE"

_CORRECTOR_ENTRY = {
    "name": "corrector_01",
    "setpoint": "SR:MAG:HCM:01:CUR:SP",
    "readback": "SR:MAG:HCM:01:CUR:RB",
}
"""The suite's stock settable entry, as it appears in a device file."""

_BPM_ENTRY = {"name": "bpm_01", "pv": "SR:DIAG:BPM:01:X:RB"}
"""The suite's stock readable entry."""


def _write_devices_file(tmp_path: Path, document: Any, name: str = "devices.yaml") -> Path:
    """Write ``document`` as a worker device file and return the path to it."""
    path = tmp_path / name
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def _devices_env(tmp_path: Path, document: Any, name: str = "devices.yaml") -> dict[str, str]:
    """The worker env for a device file holding ``document``.

    The file's presence is the substrate switch, so the env this returns is the
    whole of what turns a worker from browse-only into one that builds devices.
    """
    return {qserver_startup.DEVICES_FILE_ENV: str(_write_devices_file(tmp_path, document, name))}


def _stock_devices_env(tmp_path: Path) -> dict[str, str]:
    """The worker env for a device file holding one settable and one readable."""
    return _devices_env(tmp_path, {"settables": [_CORRECTOR_ENTRY], "readables": [_BPM_ENTRY]})


class FakeChannelValue:
    """Stand-in for ``osprey.connectors.control_system.base.ChannelValue``."""

    def __init__(self, value: Any) -> None:
        self.value = value
        self.timestamp = time.time()


class FakeConnector:
    """A minimal async double for ``ControlSystemConnector``."""

    def __init__(self) -> None:
        self.reads: list[str] = []
        self.writes: list[tuple[str, Any]] = []

    async def read_channel(self, address: str) -> FakeChannelValue:
        self.reads.append(address)
        return FakeChannelValue(0.0)

    async def write_channel_checked(self, address: str, value: Any, **_: Any) -> None:
        self.writes.append((address, value))


class FakeRunEngine:
    """Records every callback subscribed to it."""

    def __init__(self) -> None:
        self.subscriptions: list[Any] = []

    def subscribe(self, callback: Any) -> int:
        self.subscriptions.append(callback)
        return len(self.subscriptions)


@pytest.fixture(autouse=True)
def _isolated_plan_loader(monkeypatch: pytest.MonkeyPatch):
    """Every test gets a clean loader cache and no leftover layer env vars."""
    monkeypatch.delenv(_PLAN_DIRS_ENV, raising=False)
    monkeypatch.delenv(_PLAN_MODULE_ENV, raising=False)
    plan_loader.reset_facility_plans()
    yield
    plan_loader.reset_facility_plans()


def _plan_source(name: str) -> str:
    """A catalog plan file that records what it was called with.

    Both of its channel fields declare a role — one movable, on a nested model
    reached through a list, and one readable list at the top level — so this is
    also the fixture that shows a fully declared plan resolving normally.

    ``build_plan`` resolves those names against the mapping it is handed and
    returns a generator yielding one message carrying the resolved devices, the
    keys of the mapping it was offered, and the validated ``PARAMS`` instance —
    enough to assert on kwargs reconstruction, on device resolution, and on
    which devices the wrapper let through, without a RunEngine.
    """
    return (
        "from pydantic import BaseModel, Field\n\n"
        "from osprey.services.bluesky_bridge.plan_fields import (\n"
        "    MovableChannel,\n"
        "    ReadableChannels,\n"
        ")\n\n\n"
        "PLAN_METADATA = {\n"
        f'    "name": {name!r},\n'
        '    "description": "A sample catalog plan.",\n'
        '    "writes": True,\n'
        "}\n\n\n"
        "class Axis(BaseModel):\n"
        "    setpoint: MovableChannel\n"
        "    start: float\n"
        "    stop: float\n"
        "    points: int = Field(..., ge=2)\n\n\n"
        "class PARAMS(BaseModel):\n"
        "    monitors: ReadableChannels = Field(..., min_length=1)\n"
        "    axes: list[Axis] = Field(..., min_length=1)\n\n\n"
        "def build_plan(devices, params):\n"
        "    resolved = [devices[a.setpoint] for a in params.axes]\n"
        "    resolved += [devices[m] for m in params.monitors]\n"
        "    def _gen():\n"
        "        yield {'devices': resolved, 'offered': sorted(devices), 'params': params}\n"
        "    return _gen()\n"
    )


def _undeclared_field_plan_source(name: str) -> str:
    """A catalog plan naming one channel through a field that declares no role.

    ``corrector`` is declared movable; ``knob`` is a bare ``str`` that happens
    to hold a channel name. Both are resolved by ``build_plan``, so one plan
    file exercises both sides of the filter at once.
    """
    return (
        "from pydantic import BaseModel\n\n"
        "from osprey.services.bluesky_bridge.plan_fields import MovableChannel\n\n\n"
        "PLAN_METADATA = {\n"
        f'    "name": {name!r},\n'
        '    "description": "A plan reaching for an undeclared channel.",\n'
        '    "writes": True,\n'
        "}\n\n\n"
        "class PARAMS(BaseModel):\n"
        "    corrector: MovableChannel\n"
        "    knob: str\n\n\n"
        "def build_plan(devices, params):\n"
        "    resolved = [devices[params.corrector], devices[params.knob]]\n"
        "    def _gen():\n"
        "        yield {'devices': resolved}\n"
        "    return _gen()\n"
    )


def _write_plan_dir(tmp_path: Path, name: str = "sample_scan", source: Any = None) -> Path:
    """A one-file facility-tier plan directory exposing ``name``."""
    directory = tmp_path / "plans"
    directory.mkdir(parents=True, exist_ok=True)
    build_source = _plan_source if source is None else source
    (directory / f"{name}.py").write_text(build_source(name))
    return directory


def _catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str = "sample_scan",
    source: Any = None,
) -> dict:
    """Load a catalog containing exactly the one sample plan named ``name``."""
    monkeypatch.setattr(plan_loader, "_SHIPPED_PLANS_DIR", tmp_path / "no-shipped-dir")
    monkeypatch.setenv(_PLAN_DIRS_ENV, str(_write_plan_dir(tmp_path, name, source)))
    plans = dict(plan_loader.get_facility_plans().plans)
    assert name in plans, f"the sample plan did not load: {sorted(plans)}"
    return plans


def _sample_kwargs() -> dict[str, Any]:
    """JSON-shaped queue-item kwargs for the sample plan (nested models included)."""
    return {
        "monitors": ["bpm_01"],
        "axes": [{"setpoint": "corrector_01", "start": 0.0, "stop": 1.0, "points": 3}],
    }


# ---------------------------------------------------------------------------
# Device construction from the device file
# ---------------------------------------------------------------------------


def test_no_devices_file_env_builds_no_devices_and_does_not_raise() -> None:
    assert asyncio.run(qserver_startup.build_devices(env={})) == {}


def test_a_device_file_nothing_points_at_builds_no_devices(tmp_path: Path) -> None:
    """The env var, not the file lying on disk, is what enables the substrate.

    A deploy that stages a device file but never mounts its path into the
    worker's env is browse-only — the worker has no way to find the file and
    must not go looking for one.
    """
    _write_devices_file(tmp_path, {"settables": [_CORRECTOR_ENTRY], "readables": [_BPM_ENTRY]})

    assert asyncio.run(qserver_startup.build_devices(env={})) == {}


def test_devices_file_naming_a_missing_path_warns_and_builds_no_devices(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    env = {qserver_startup.DEVICES_FILE_ENV: str(tmp_path / "not-mounted.yaml")}

    with caplog.at_level("WARNING"):
        devices = asyncio.run(qserver_startup.build_devices(env=env))

    assert devices == {}
    # A path that names nothing is a mount that did not happen, so it is worth
    # a warning rather than the quiet info an unset var gets.
    assert "cannot read as a file" in caplog.text


def test_devices_file_naming_a_directory_warns_and_builds_no_devices(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A bind mount that lands a directory where the file should be is the
    classic compose typo, and it must not read as an enabled substrate."""
    directory = tmp_path / "devices.yaml"
    directory.mkdir()
    env = {qserver_startup.DEVICES_FILE_ENV: str(directory)}

    with caplog.at_level("WARNING"):
        devices = asyncio.run(qserver_startup.build_devices(env=env))

    assert devices == {}
    assert "cannot read as a file" in caplog.text


def test_unreadable_devices_file_warns_and_builds_no_devices(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A file the worker's uid cannot open is disabled, not half-enabled."""
    env = _stock_devices_env(tmp_path)
    path = Path(env[qserver_startup.DEVICES_FILE_ENV])
    path.chmod(0o000)
    if os.access(path, os.R_OK):  # pragma: no cover - only when running as root
        pytest.skip("this uid can read a mode-000 file; permissions are not enforced here")

    try:
        with caplog.at_level("WARNING"):
            devices = asyncio.run(qserver_startup.build_devices(env=env))
    finally:
        path.chmod(0o644)

    assert devices == {}
    assert "cannot read as a file" in caplog.text


def test_devices_file_with_no_entries_warns_and_builds_no_devices(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    env = _devices_env(tmp_path, {})

    with caplog.at_level("WARNING"):
        devices = asyncio.run(qserver_startup.build_devices(env=env))

    assert devices == {}
    assert "this worker will expose no plans" in caplog.text


def test_build_devices_builds_connector_mediated_devices_from_the_device_file(
    tmp_path: Path,
) -> None:
    connector = FakeConnector()
    env = _stock_devices_env(tmp_path)

    devices = asyncio.run(qserver_startup.build_devices(env=env, connector=connector))

    assert sorted(devices) == ["bpm_01", "corrector_01"]
    assert isinstance(devices["corrector_01"], ConnectorSettable)
    assert isinstance(devices["bpm_01"], ConnectorReadable)
    # Every device delegates to the one connector handed in — no raw Channel
    # Access is constructed anywhere in this path.
    assert devices["corrector_01"]._osprey_connector is connector
    assert devices["bpm_01"]._osprey_connector is connector


def test_address_named_devices_build_and_stay_visible_to_queueserver(tmp_path: Path) -> None:
    """A device named by its own channel address survives the whole worker path.

    A device file may name each device after the address it drives or reads, so
    the worker namespace is keyed by colon-bearing names that are not Python
    identifiers. Nothing in this path may quietly drop them: the plan
    functions' identifier filter applies to plan names only, and the manager's
    namespace scan — which decides what the manager will accept in a queue item
    — reports devices by their namespace key.
    """
    from bluesky_queueserver.manager.profile_ops import existing_plans_and_devices_from_nspace

    connector = FakeConnector()
    setpoint = "SR:MAG:HCM:01:CURRENT:SP"
    readback = "SR:MAG:HCM:01:CURRENT:RB"
    bpm = "SR:DIAG:BPM:01:POSITION:X"
    env = _devices_env(
        tmp_path,
        {
            "settables": [{"name": setpoint, "setpoint": setpoint, "readback": readback}],
            "readables": [{"name": bpm, "pv": bpm}],
        },
    )

    devices = asyncio.run(qserver_startup.build_devices(env=env, connector=connector))

    assert sorted(devices) == [bpm, setpoint]
    assert isinstance(devices[setpoint], ConnectorSettable)
    assert isinstance(devices[bpm], ConnectorReadable)
    # ophyd-async takes the colon name verbatim — it is also the event-data key.
    assert devices[setpoint].name == setpoint

    namespace = qserver_startup.build_namespace(
        env={}, run_engine=FakeRunEngine(), devices=devices, plans={}
    )
    _plans, existing_devices, _, _ = existing_plans_and_devices_from_nspace(nspace=namespace)

    assert {setpoint, bpm} <= set(existing_devices)


def test_built_devices_read_through_the_connector(tmp_path: Path) -> None:
    connector = FakeConnector()
    env = _devices_env(tmp_path, {"readables": [_BPM_ENTRY]})
    devices = asyncio.run(qserver_startup.build_devices(env=env, connector=connector))

    asyncio.run(devices["bpm_01"].read())

    assert connector.reads == ["SR:DIAG:BPM:01:X:RB"]


def test_a_readables_only_device_file_builds_that_half_alone(tmp_path: Path) -> None:
    """Both sections are optional: a file listing only readables is a valid
    monitoring-only worker, not an empty one."""
    env = _devices_env(tmp_path, {"readables": [_BPM_ENTRY]})

    devices = asyncio.run(qserver_startup.build_devices(env=env, connector=FakeConnector()))

    assert sorted(devices) == ["bpm_01"]


# ---------------------------------------------------------------------------
# Connector selection
# ---------------------------------------------------------------------------


def test_control_system_type_falls_back_to_mock_when_config_is_unreadable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import osprey.utils.config as config_module

    def _raise(*_: Any, **__: Any) -> Any:
        raise FileNotFoundError("no project config context")

    monkeypatch.setattr(config_module, "get_config_value", _raise)

    assert qserver_startup.resolve_control_system_type() == "mock"


@pytest.mark.parametrize("control_system_type", ["virtual_accelerator", "epics"])
def test_epics_like_types_get_a_gateway_less_type_config(control_system_type: str) -> None:
    config = qserver_startup.build_connector_config(control_system_type)

    assert config["type"] == control_system_type
    assert config["connector"][control_system_type] == {"timeout": 5.0}
    assert "gateways" not in config["connector"][control_system_type]


def test_other_types_are_forwarded_through_untouched() -> None:
    assert qserver_startup.build_connector_config("mock") == {
        "type": "mock",
        "connector": {"mock": {}},
    }


# ---------------------------------------------------------------------------
# Plan namespace construction
# ---------------------------------------------------------------------------


def test_each_catalog_plan_becomes_a_named_generator_function(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plans = _catalog(tmp_path, monkeypatch)

    functions = qserver_startup.build_plan_functions({}, plans)

    assert sorted(functions) == ["sample_scan"]
    plan_function = functions["sample_scan"]
    # Queueserver's namespace scan only recognizes generator functions as
    # plans; anything else is silently invisible to the manager.
    assert inspect.isgeneratorfunction(plan_function)
    assert plan_function.__name__ == "sample_scan"
    assert plan_function.__doc__ == "A sample catalog plan."


def test_plan_namespace_is_loaded_from_the_plan_dirs_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(plan_loader, "_SHIPPED_PLANS_DIR", tmp_path / "no-shipped-dir")
    monkeypatch.setenv(_PLAN_DIRS_ENV, str(_write_plan_dir(tmp_path, "env_sourced_scan")))

    functions = qserver_startup.build_plan_functions({})

    assert "env_sourced_scan" in functions


def test_plan_function_reconstructs_the_params_model_from_json_kwargs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plans = _catalog(tmp_path, monkeypatch)
    devices = {"corrector_01": object(), "bpm_01": object()}

    plan_function = qserver_startup.build_plan_functions(devices, plans)["sample_scan"]
    message = next(plan_function(**_sample_kwargs()))

    params = message["params"]
    assert params.__class__ is plans["sample_scan"].schema
    assert params.monitors == ["bpm_01"]
    # The nested JSON object came back as the plan's own nested model, with
    # its declared types — not as the raw dict the queue item carried.
    assert params.axes[0].setpoint == "corrector_01"
    assert params.axes[0].points == 3
    assert isinstance(params.axes[0].start, float)


def test_plan_function_resolves_device_names_against_the_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plans = _catalog(tmp_path, monkeypatch)
    corrector = object()
    detector = object()
    devices = {"corrector_01": corrector, "bpm_01": detector}

    plan_function = qserver_startup.build_plan_functions(devices, plans)["sample_scan"]
    message = next(plan_function(**_sample_kwargs()))

    assert message["devices"] == [corrector, detector]


def test_plan_function_rejects_kwargs_that_fail_the_params_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plans = _catalog(tmp_path, monkeypatch)
    plan_function = qserver_startup.build_plan_functions({}, plans)["sample_scan"]

    bad_kwargs = _sample_kwargs()
    bad_kwargs["axes"][0]["points"] = 1  # schema requires >= 2

    with pytest.raises(ValidationError):
        next(plan_function(**bad_kwargs))


def test_unknown_device_name_names_the_device_and_what_the_worker_has(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plans = _catalog(tmp_path, monkeypatch)
    plan_function = qserver_startup.build_plan_functions({"bpm_01": object()}, plans)["sample_scan"]

    with pytest.raises(KeyError) as excinfo:
        next(plan_function(**_sample_kwargs()))

    message = str(excinfo.value)
    assert "corrector_01" in message
    assert "bpm_01" in message


def test_plan_whose_name_is_not_an_identifier_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    plans = _catalog(tmp_path, monkeypatch)
    plans["not an identifier"] = plans["sample_scan"]

    with caplog.at_level("WARNING"):
        functions = qserver_startup.build_plan_functions({}, plans)

    assert sorted(functions) == ["sample_scan"]
    assert "not an identifier" in caplog.text


# ---------------------------------------------------------------------------
# The declared contract: what a plan is allowed to resolve
# ---------------------------------------------------------------------------
#
# A plan's params declare, field by field, which channels it moves and which it
# reads. That declaration is also its bound: the wrapper hands `build_plan`
# only the declared channels, so nothing else the worker happens to hold is
# reachable from inside a run.


def test_a_plan_receives_only_the_channels_its_params_declare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plans = _catalog(tmp_path, monkeypatch)
    corrector = object()
    monitor = object()
    devices = {
        "corrector_01": corrector,
        "bpm_01": monitor,
        # Built by this worker, named by nothing in these params.
        "corrector_02": object(),
        "bpm_02": object(),
    }

    plan_function = qserver_startup.build_plan_functions(devices, plans)["sample_scan"]
    message = next(plan_function(**_sample_kwargs()))

    # Both roles come through — the movable one from a nested model reached
    # through a list, the readable one from a top-level list.
    assert message["offered"] == ["bpm_01", "corrector_01"]
    assert message["devices"] == [corrector, monitor]


def test_a_channel_named_by_an_undeclared_field_is_not_resolvable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A role-less field's channel is out of reach even when the worker has it.

    This is the whole point of declaring: a plan reaches what its contract
    names, and a device it never declared is absent from the mapping however
    many devices the worker built. The failure says so — including that the
    device does exist here — rather than reading as a missing device.
    """
    plans = _catalog(tmp_path, monkeypatch, "undeclared_scan", source=_undeclared_field_plan_source)
    devices = {"corrector_01": object(), "bpm_01": object()}

    plan_function = qserver_startup.build_plan_functions(devices, plans)["undeclared_scan"]

    with pytest.raises(KeyError) as excinfo:
        next(plan_function(corrector="corrector_01", knob="bpm_01"))

    message = str(excinfo.value)
    assert "bpm_01" in message
    assert "do not declare as a movable or readable channel" in message
    # And it still names what this worker actually has, which is what tells an
    # author the device is there and the declaration is what is missing.
    assert "corrector_01" in message


def test_filtering_leaves_the_wrapper_annotations_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Narrowing the device mapping must not disturb the manager's signature read.

    The wrapper's rebound ``__annotations__`` are what let upstream build this
    plan's validation model at all (see the PEP 563 pin below); they are a
    property of the wrapper, not of the devices it closes over.
    """
    plans = _catalog(tmp_path, monkeypatch)

    plan_function = qserver_startup.build_plan_functions({"bpm_01": object()}, plans)["sample_scan"]

    assert plan_function.__annotations__ == {"kwargs": Any, "return": Iterator[Any]}
    assert inspect.signature(plan_function).parameters["kwargs"].kind is (
        inspect.Parameter.VAR_KEYWORD
    )


def test_the_wrapper_adds_nothing_to_the_plans_message_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wrapper narrows devices; it stamps no metadata of its own.

    Run identity is stamped on the enqueue path, as the queue item's metadata.
    A wrapper that also injected metadata here would be a second, competing
    source of it, so what it yields is exactly what ``build_plan`` yielded —
    nothing prepended, nothing appended.
    """
    plans = _catalog(tmp_path, monkeypatch)
    devices = {"corrector_01": object(), "bpm_01": object()}

    plan_function = qserver_startup.build_plan_functions(devices, plans)["sample_scan"]
    messages = list(plan_function(**_sample_kwargs()))

    assert len(messages) == 1
    assert sorted(messages[0]) == ["devices", "offered", "params"]


# ---------------------------------------------------------------------------
# Namespace assembly
# ---------------------------------------------------------------------------


def test_namespace_holds_the_run_engine_the_devices_and_the_plans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plans = _catalog(tmp_path, monkeypatch)
    run_engine = FakeRunEngine()
    devices = {"corrector_01": object(), "bpm_01": object()}

    namespace = qserver_startup.build_namespace(
        env={}, run_engine=run_engine, devices=devices, plans=plans
    )

    assert namespace["RE"] is run_engine
    assert namespace["corrector_01"] is devices["corrector_01"]
    assert callable(namespace["sample_scan"])


def test_a_device_less_worker_registers_no_plans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    plans = _catalog(tmp_path, monkeypatch)
    run_engine = FakeRunEngine()

    with caplog.at_level("WARNING"):
        namespace = qserver_startup.build_namespace(
            env={}, run_engine=run_engine, devices={}, plans=plans
        )

    # Browse-only: exposing plans here would advertise executable work that
    # could only fail at run time against devices that were never built.
    assert list(namespace) == ["RE"]
    assert "registering no plans" in caplog.text


def test_namespace_builds_devices_on_the_run_engine_loop_when_none_are_supplied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The async device build must be awaited on the RunEngine's own loop.

    Devices bind their I/O to whichever loop awaits them, so building them on
    a throwaway loop would leave the run driving devices bound to a loop it
    never touches. Uses a real ``RunEngine`` (its loop runs in a background
    thread from construction) rather than a double, since the loop is the
    point.
    """
    from bluesky import RunEngine

    plans = _catalog(tmp_path, monkeypatch)
    connector = FakeConnector()
    env = _stock_devices_env(tmp_path)
    monkeypatch.setattr(
        qserver_startup,
        "create_connector",
        lambda: asyncio.sleep(0, result=connector),
    )

    run_engine = RunEngine()
    namespace = qserver_startup.build_namespace(env=env, run_engine=run_engine, plans=plans)

    assert isinstance(namespace["corrector_01"], ConnectorSettable)
    assert namespace["corrector_01"]._osprey_connector is connector
    assert callable(namespace["sample_scan"])


def test_queueserver_recognizes_every_plan_and_device_in_the_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The manager's own namespace scan is the contract that decides visibility.

    A plan queueserver does not recognize is silently absent from the allowed
    list rather than an error, so this asserts against the real scanner
    (``bluesky_queueserver`` ships as a hard dependency of
    ``bluesky-queueserver-api``) instead of re-implementing its rules.
    """
    from bluesky_queueserver.manager.profile_ops import existing_plans_and_devices_from_nspace

    plans = _catalog(tmp_path, monkeypatch)
    connector = FakeConnector()
    env = _stock_devices_env(tmp_path)
    devices = asyncio.run(qserver_startup.build_devices(env=env, connector=connector))
    namespace = qserver_startup.build_namespace(
        env={}, run_engine=FakeRunEngine(), devices=devices, plans=plans
    )

    existing_plans, existing_devices, _, _ = existing_plans_and_devices_from_nspace(
        nspace=namespace
    )

    assert "sample_scan" in existing_plans
    assert existing_plans["sample_scan"]["properties"]["is_generator"] is True
    assert {"corrector_01", "bpm_01"} <= set(existing_devices)


# ---------------------------------------------------------------------------
# Document plane
# ---------------------------------------------------------------------------


def test_no_tiled_uri_means_no_tiled_writer() -> None:
    assert qserver_startup.build_tiled_writer(env={}) is None


def test_tiled_writer_is_built_from_the_uri_and_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    import bluesky.callbacks.tiled_writer as tiled_writer_module

    calls: list[tuple[str, Any]] = []

    class _FakeTiledWriter:
        @staticmethod
        def from_uri(uri: str, api_key: str | None = None) -> str:
            calls.append((uri, api_key))
            return "writer"

    monkeypatch.setattr(tiled_writer_module, "TiledWriter", _FakeTiledWriter)

    writer = qserver_startup.build_tiled_writer(
        env={"BLUESKY_TILED_URI": "http://tiled:8000", "BLUESKY_TILED_API_KEY": "secret"}
    )

    assert writer == "writer"
    assert calls == [("http://tiled:8000", "secret")]


def test_no_publish_address_means_no_publisher() -> None:
    assert qserver_startup.build_zmq_publisher(env={}) is None


def test_publisher_carries_the_client_curve_config(monkeypatch: pytest.MonkeyPatch) -> None:
    import bluesky.callbacks.zmq as zmq_module

    captured: dict[str, Any] = {}

    class _FakePublisher:
        def __init__(self, address: str, *, curve_config: Any = None) -> None:
            captured["address"] = address
            captured["curve_config"] = curve_config

    monkeypatch.setattr(zmq_module, "Publisher", _FakePublisher)

    qserver_startup.build_zmq_publisher(
        env={
            "BLUESKY_ZMQ_PUBLISH_ADDR": "tcp://bridge:5567",
            "BLUESKY_ZMQ_CURVE_SECRET_KEY": "/keys/worker.key_secret",
            "BLUESKY_ZMQ_CURVE_SERVER_PUBLIC_KEY": "/keys/bridge.key",
        }
    )

    assert captured["address"] == "tcp://bridge:5567"
    assert captured["curve_config"].secret_path == "/keys/worker.key_secret"
    assert captured["curve_config"].server_public_key == "/keys/bridge.key"


@pytest.mark.parametrize(
    "curve_env",
    [
        {"BLUESKY_ZMQ_CURVE_SECRET_KEY": "/keys/worker.key_secret"},
        {"BLUESKY_ZMQ_CURVE_SERVER_PUBLIC_KEY": "/keys/bridge.key"},
    ],
)
def test_half_configured_curve_auth_is_refused(curve_env: dict[str, str]) -> None:
    """Publishing in the clear because one env var was misspelled is exactly
    what this key auth exists to prevent, so a half-set pair raises rather
    than falling back to an unauthenticated socket."""
    env = {"BLUESKY_ZMQ_PUBLISH_ADDR": "tcp://bridge:5567", **curve_env}

    with pytest.raises(ValueError, match="must be set together"):
        qserver_startup.build_zmq_publisher(env=env)


def test_only_configured_document_streams_are_subscribed(monkeypatch: pytest.MonkeyPatch) -> None:
    import bluesky.callbacks.zmq as zmq_module

    monkeypatch.setattr(zmq_module, "Publisher", lambda address, **_: "publisher")
    run_engine = FakeRunEngine()

    subscribed = qserver_startup.subscribe_document_callbacks(
        run_engine, env={"BLUESKY_ZMQ_PUBLISH_ADDR": "tcp://bridge:5567"}
    )

    assert sorted(subscribed) == ["zmq"]
    assert len(run_engine.subscriptions) == 1


def test_a_document_callback_that_raises_degrades_instead_of_aborting_the_run(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import bluesky.callbacks.zmq as zmq_module

    def _explode(name: str, doc: dict[str, Any]) -> None:
        raise RuntimeError("proxy is gone")

    monkeypatch.setattr(zmq_module, "Publisher", lambda address, **_: _explode)
    run_engine = FakeRunEngine()

    subscribed = qserver_startup.subscribe_document_callbacks(
        run_engine, env={"BLUESKY_ZMQ_PUBLISH_ADDR": "tcp://bridge:5567"}
    )
    callback = run_engine.subscriptions[0]

    with caplog.at_level("ERROR"):
        callback("start", {"uid": "abc"})
        callback("event", {"seq_num": 1})

    assert subscribed["zmq"].degraded is True
    assert "degraded" in caplog.text


def test_a_publisher_that_cannot_be_built_never_fails_the_environment_open(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import bluesky.callbacks.zmq as zmq_module

    def _raise(address: str, **_: Any) -> Any:
        raise RuntimeError("cannot bind")

    monkeypatch.setattr(zmq_module, "Publisher", _raise)
    run_engine = FakeRunEngine()

    with caplog.at_level("ERROR"):
        subscribed = qserver_startup.subscribe_document_callbacks(
            run_engine, env={"BLUESKY_ZMQ_PUBLISH_ADDR": "tcp://bridge:5567"}
        )

    assert subscribed == {}
    assert run_engine.subscriptions == []
    assert "could not build the zmq document callback" in caplog.text


# ---------------------------------------------------------------------------
# Production-execution-context pins.
#
# Everything above imports this module NORMALLY. In production it is never
# imported at all: it is the RE manager's `--startup-script`, which upstream
# `exec`s. Two whole-stack defects lived in exactly that gap -- green here,
# dead in a container -- so these two pins assert the properties that only the
# exec'd-as-a-script context can violate. Both were found by
# `tests/e2e/test_bluesky_queue_e2e.py`, against real containers.
# ---------------------------------------------------------------------------


def test_startup_script_contains_no_relative_imports_anywhere() -> None:
    """A relative import here takes the whole worker environment down.

    `load_startup_script` runs `exec(code, nspace, nspace)` with `__name__`
    patched to `"__main__"` and NO `__package__`, so `from .devices import ...`
    raises `ImportError: attempted relative import with no known parent
    package` -- and the manager reports only "Failed to start RE Worker
    environment", leaving the deploy with no devices and no plans.

    An AST walk over the WHOLE file, not a grep and not an import-time check:
    the three that shipped were lazy imports nested inside functions, invisible
    to both. Note the file was HALF converted when the defect landed -- its
    `if __name__ == "__main__":` guard already used the absolute form -- so
    "the convention exists in this file" is not evidence that every site
    follows it.
    """
    import ast

    source = Path(qserver_startup.__file__).read_text(encoding="utf-8")
    offenders = [
        f"line {node.lineno}: from {'.' * node.level}{node.module or ''} import "
        + ", ".join(alias.name for alias in node.names)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.level > 0
    ]
    assert not offenders, (
        "qserver_startup.py is exec'd as a script with no package context, so it "
        "may never use a relative import -- spell these absolutely "
        "(`from osprey.services.bluesky_bridge... import`):\n  " + "\n  ".join(offenders)
    )


def test_plan_function_annotations_are_real_objects_not_pep563_strings() -> None:
    """The manager builds each plan's validation model off `inspect.signature`.

    This module uses `from __future__ import annotations`, so every annotation
    it writes is a STRING. Upstream wraps a VAR_KEYWORD parameter's annotation
    as `typing.Dict[str, <annotation>]` and hands it to
    `pydantic.create_model`; a string there is an unresolvable forward
    reference in pydantic's namespace, and the manager then refuses EVERY
    enqueue with "`Model` is not fully defined; you should define `Any`" -- a
    failure that reads like a plan problem and is really an
    annotation-representation problem. The wrapper therefore rebinds
    `__annotations__` to real objects.

    Asserted on the objects rather than on `inspect.signature`'s repr: the repr
    of a resolved annotation is not stable across Python versions, while "is
    not a string" is exactly the property upstream needs.

    THIS FILE COVERS ONE FACTORY. The same defect then shipped again in the
    OTHER one (`session_upload._make_session_plan`, which hands session plans
    to the same manager for the same reason), because fixing the site a failure
    points at is not the same as fixing the pattern. The enumerating pin that
    covers BOTH — and forces a third to declare itself — is
    ``test_session_upload.py::test_no_plan_wrapper_ships_pep563_string_annotations``.
    Keep this one as the local guard; add new factories THERE.
    """
    import pydantic

    plans = plan_loader.get_facility_plans().plans
    assert plans, "no catalog plans to build a wrapper from"
    spec = plans[sorted(plans)[0]]

    plan_function = qserver_startup._make_plan_function(spec, devices={})

    annotations = plan_function.__annotations__
    assert annotations, "the wrapper carries no annotations at all"
    for name, annotation in annotations.items():
        assert not isinstance(annotation, str), (
            f"{name!r} is the PEP 563 string {annotation!r}; upstream cannot build a "
            "pydantic model from it and every enqueue would be refused"
        )

    # The property upstream actually exercises, end to end: the VAR_KEYWORD
    # annotation must survive being parameterized into a mapping generic and
    # handed to `create_model`. Upstream spells that `typing.Dict[str, ...]`;
    # `dict[str, ...]` is the same object at runtime and is what this repo's
    # lint allows.
    kwargs_param = inspect.signature(plan_function).parameters["kwargs"]
    model = pydantic.create_model("Model", kwargs=(dict[str, kwargs_param.annotation], {}))
    assert model(kwargs={"anything": 1}).kwargs == {"anything": 1}


# ---------------------------------------------------------------------------
# Read-only pre-flight: preview_plan
# ---------------------------------------------------------------------------
#
# The preview walks a plan's message stream with no RunEngine, so these tests
# assert two things at once: that the reported trajectory is the one the plan
# would drive, and that walking it drives nothing — `FakeConnector` records
# every read and write the device layer would perform.


def _trajectory_plan_source(name: str) -> str:
    """A catalog plan whose message stream drives one movable channel.

    Targets are numpy scalars, as every stock bluesky scan's are: the preview
    has to reduce them to plain JSON numbers on the way out.
    """
    return (
        "import numpy\n"
        "from bluesky.utils import Msg\n"
        "from pydantic import BaseModel, Field\n\n"
        "from osprey.services.bluesky_bridge.plan_fields import (\n"
        "    MovableChannel,\n"
        "    ReadableChannel,\n"
        ")\n\n\n"
        "PLAN_METADATA = {\n"
        f'    "name": {name!r},\n'
        '    "description": "A sample catalog plan that moves one channel.",\n'
        '    "writes": True,\n'
        "}\n\n\n"
        "class PARAMS(BaseModel):\n"
        "    movable: MovableChannel\n"
        "    readable: ReadableChannel\n"
        "    targets: list[float] = Field(..., min_length=1)\n\n\n"
        "def build_plan(devices, params):\n"
        "    movable = devices[params.movable]\n"
        "    readable = devices[params.readable]\n"
        "    def _gen():\n"
        "        yield Msg('open_run', None)\n"
        "        for target in params.targets:\n"
        "            yield Msg('set', movable, numpy.float64(target))\n"
        "            yield Msg('trigger', readable)\n"
        "            yield Msg('read', readable)\n"
        "        yield Msg('close_run', None)\n"
        "    return _gen()\n"
    )


def _trajectory_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str = "trajectory_scan"
) -> tuple[dict[str, Any], FakeConnector]:
    """A worker namespace holding the moving plan and the real devices it drives."""
    monkeypatch.setattr(plan_loader, "_SHIPPED_PLANS_DIR", tmp_path / "no-shipped-dir")
    directory = tmp_path / "preview-plans"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.py").write_text(_trajectory_plan_source(name))
    monkeypatch.setenv(_PLAN_DIRS_ENV, str(directory))

    plans = dict(plan_loader.get_facility_plans().plans)
    assert name in plans, f"the sample plan did not load: {sorted(plans)}"

    connector = FakeConnector()
    devices = asyncio.run(
        qserver_startup.build_devices(env=_stock_devices_env(tmp_path), connector=connector)
    )
    namespace: dict[str, Any] = dict(devices)
    namespace.update(qserver_startup.build_plan_functions(devices, plans))
    return namespace, connector


def _trajectory_kwargs(targets: tuple[float, ...] = (0.0, 0.5, 1.0)) -> dict[str, Any]:
    return {"movable": "corrector_01", "readable": "bpm_01", "targets": list(targets)}


def test_preview_lists_the_moves_a_plan_would_make_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace, _connector = _trajectory_namespace(tmp_path, monkeypatch)

    preview = qserver_startup.preview_plan_in_namespace(
        "trajectory_scan", _trajectory_kwargs(), namespace
    )

    assert preview["ok"] is True
    assert preview["plan"] == "trajectory_scan"
    # Only the moves, in plan order — the reads and the run boundaries the same
    # stream carries change nothing and are not reported.
    assert preview["moves"] == [
        {"channel": "corrector_01", "target": 0.0},
        {"channel": "corrector_01", "target": 0.5},
        {"channel": "corrector_01", "target": 1.0},
    ]
    assert preview["total_moves"] == 3
    assert preview["truncated"] is False


def test_preview_counts_every_move_past_the_cap_and_flags_truncation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cap bounds the payload, never the count: the total stays exact."""
    namespace, _connector = _trajectory_namespace(tmp_path, monkeypatch)
    kwargs = _trajectory_kwargs(targets=(0.0, 1.0, 2.0, 3.0, 4.0))

    preview = qserver_startup.preview_plan_in_namespace("trajectory_scan", kwargs, namespace, cap=2)

    assert preview["ok"] is True
    assert [move["target"] for move in preview["moves"]] == [0.0, 1.0]
    assert preview["total_moves"] == 5
    assert preview["truncated"] is True
    assert preview["move_cap"] == 2
    # The shipped cap the bridge route and the approval prompt are bounded by.
    # Deliberately small: the approval prompt names only a few moves from each
    # end, and the end-to-end proof of this same contract has to walk a plan of
    # just over this many moves, so the number is what that walk costs.
    assert qserver_startup.PREVIEW_MOVE_CAP == 200


def test_preview_payload_is_json_serializable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The response crosses a 0MQ hop, so nothing exotic may survive in it."""
    import json

    namespace, _connector = _trajectory_namespace(tmp_path, monkeypatch)

    preview = qserver_startup.preview_plan_in_namespace(
        "trajectory_scan", _trajectory_kwargs(), namespace
    )

    assert json.loads(json.dumps(preview)) == preview
    # The plan yielded numpy scalars; what comes back is plain Python.
    for move in preview["moves"]:
        assert type(move["target"]) is float
        assert type(move["channel"]) is str


def test_preview_reports_kwargs_the_params_schema_rejects_without_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace, _connector = _trajectory_namespace(tmp_path, monkeypatch)
    bad_kwargs = _trajectory_kwargs()
    bad_kwargs["targets"] = []  # the schema requires at least one

    preview = qserver_startup.preview_plan_in_namespace("trajectory_scan", bad_kwargs, namespace)

    assert preview["ok"] is False
    assert "targets" in preview["error"]
    # A failed preview describes no trajectory at all, so the total keeps one
    # meaning: the exact move count of a walk that completed.
    assert preview["moves"] == []
    assert preview["total_moves"] == 0
    assert preview["truncated"] is False


def test_preview_reports_a_device_this_worker_never_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace, _connector = _trajectory_namespace(tmp_path, monkeypatch)
    kwargs = _trajectory_kwargs()
    kwargs["movable"] = "corrector_99"

    preview = qserver_startup.preview_plan_in_namespace("trajectory_scan", kwargs, namespace)

    assert preview["ok"] is False
    assert "corrector_99" in preview["error"]


def test_preview_of_an_unknown_plan_names_what_the_worker_has(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace, _connector = _trajectory_namespace(tmp_path, monkeypatch)

    preview = qserver_startup.preview_plan_in_namespace("no_such_scan", {}, namespace)

    assert preview["ok"] is False
    assert "no_such_scan" in preview["error"]
    assert "trajectory_scan" in preview["error"]


def test_preview_refuses_a_namespace_entry_that_is_not_a_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only plans are previewable — the entry point cannot call anything else.

    The worker namespace also holds this module's own module-level functions
    (upstream ``exec``s the startup script into it), so "callable and in the
    namespace" would be a far wider surface than a pre-flight needs.
    """
    namespace, _connector = _trajectory_namespace(tmp_path, monkeypatch)
    calls: list[str] = []
    namespace["not_a_plan"] = lambda: calls.append("called")

    preview = qserver_startup.preview_plan_in_namespace("not_a_plan", {}, namespace)

    assert preview["ok"] is False
    assert calls == []


def test_previewing_a_plan_never_reads_or_writes_a_single_channel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The safety property the whole pre-flight rests on.

    These are the real connector-mediated devices, so a walk that executed any
    part of the plan would show up as a connector read or write.
    """
    namespace, connector = _trajectory_namespace(tmp_path, monkeypatch)

    preview = qserver_startup.preview_plan_in_namespace(
        "trajectory_scan", _trajectory_kwargs(), namespace
    )

    assert preview["total_moves"] == 3
    assert connector.writes == []
    assert connector.reads == []


def test_preview_plan_reads_the_live_worker_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``preview_plan`` resolves against this module's globals.

    In production that dict IS the worker namespace: upstream ``exec``s the
    startup script into it and the ``__main__`` guard updates it with the
    devices and the plan functions, so the devices a preview resolves are the
    connected, mock-free ones the run itself would use.
    """
    namespace, connector = _trajectory_namespace(tmp_path, monkeypatch)
    for key, value in namespace.items():
        monkeypatch.setitem(qserver_startup.__dict__, key, value)

    preview = qserver_startup.preview_plan("trajectory_scan", _trajectory_kwargs())

    assert preview["ok"] is True
    assert preview["total_moves"] == 3
    assert connector.writes == []


def test_preview_plan_is_a_function_the_manager_executes_not_a_plan_it_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Upstream classifies namespace callables by whether they are generators.

    A generator function is a plan — handed to the RunEngine, and refused
    outright by ``prepare_function``; a plain function is a callable
    ``function_execute`` invokes directly. The pre-flight has to be the second,
    and the namespace it is looked up in is the one ``exec``ing this module
    produces: module globals, updated with the devices and plans.
    """
    from bluesky_queueserver.manager.profile_ops import (
        existing_plans_and_devices_from_nspace,
        prepare_function,
    )

    plan_namespace, _connector = _trajectory_namespace(tmp_path, monkeypatch)
    namespace = dict(vars(qserver_startup))
    namespace.update(plan_namespace)

    assert not inspect.isgeneratorfunction(qserver_startup.preview_plan)
    prepared = prepare_function(
        func_info={"name": "preview_plan", "user_group": "primary"},
        nspace=namespace,
        user_group_permissions=None,
    )
    assert prepared["callable"] is qserver_startup.preview_plan

    # And it is never advertised as a plan the queue could be asked to run.
    existing_plans, _devices, _plans_ns, _devices_ns = existing_plans_and_devices_from_nspace(
        nspace=namespace
    )
    assert "trajectory_scan" in existing_plans
    assert "preview_plan" not in existing_plans
