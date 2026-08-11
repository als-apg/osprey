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
import time
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from osprey.services.bluesky_bridge import plan_loader, qserver_startup
from osprey.services.bluesky_bridge.devices.connector import ConnectorReadable, ConnectorSettable

_PLAN_DIRS_ENV = "BLUESKY_PLAN_DIRS"
_PLAN_MODULE_ENV = "BLUESKY_PLAN_MODULE"


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

    ``build_plan`` resolves two device names against the namespace and returns
    a generator yielding one message carrying the resolved devices and the
    validated ``PARAMS`` instance — enough to assert on kwargs reconstruction
    and device resolution without a RunEngine.
    """
    return (
        "from pydantic import BaseModel, Field\n\n\n"
        "PLAN_METADATA = {\n"
        f'    "name": {name!r},\n'
        '    "description": "A sample catalog plan.",\n'
        '    "category": "accelerator",\n'
        '    "required_devices": ["setpoint", "detector"],\n'
        '    "writes": True,\n'
        "}\n\n\n"
        "class Axis(BaseModel):\n"
        "    setpoint: str\n"
        "    start: float\n"
        "    stop: float\n"
        "    num_points: int = Field(..., ge=2)\n\n\n"
        "class PARAMS(BaseModel):\n"
        "    detectors: list[str] = Field(..., min_length=1)\n"
        "    axes: list[Axis] = Field(..., min_length=1)\n\n\n"
        "def build_plan(devices, params):\n"
        "    resolved = [devices[a.setpoint] for a in params.axes]\n"
        "    resolved += [devices[d] for d in params.detectors]\n"
        "    def _gen():\n"
        "        yield {'devices': resolved, 'params': params}\n"
        "    return _gen()\n"
    )


def _write_plan_dir(tmp_path: Path, name: str = "sample_scan") -> Path:
    """A one-file facility-tier plan directory exposing ``name``."""
    directory = tmp_path / "plans"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.py").write_text(_plan_source(name))
    return directory


def _catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str = "sample_scan") -> dict:
    """Load a catalog containing exactly the one sample plan named ``name``."""
    monkeypatch.setattr(plan_loader, "_SHIPPED_PLANS_DIR", tmp_path / "no-shipped-dir")
    monkeypatch.setenv(_PLAN_DIRS_ENV, str(_write_plan_dir(tmp_path, name)))
    return dict(plan_loader.get_facility_plans().plans)


def _sample_kwargs() -> dict[str, Any]:
    """JSON-shaped queue-item kwargs for the sample plan (nested models included)."""
    return {
        "detectors": ["bpm_01"],
        "axes": [{"setpoint": "corrector_01", "start": 0.0, "stop": 1.0, "num_points": 3}],
    }


# ---------------------------------------------------------------------------
# Device construction from the substrate env
# ---------------------------------------------------------------------------


def test_no_substrate_env_builds_no_devices_and_does_not_raise() -> None:
    assert asyncio.run(qserver_startup.build_devices(env={})) == {}


def test_substrate_off_ignores_a_populated_pv_list() -> None:
    env = {
        "BLUESKY_EPICS_MOTORS": "corrector_01=SR:MAG:HCM:01:CUR:SP|SR:MAG:HCM:01:CUR:RB",
        "BLUESKY_EPICS_DETECTORS": "bpm_01=SR:DIAG:BPM:01:X:RB",
    }
    assert asyncio.run(qserver_startup.build_devices(env=env)) == {}


def test_substrate_on_with_empty_pv_lists_warns_and_builds_no_devices(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING"):
        devices = asyncio.run(qserver_startup.build_devices(env={"BLUESKY_EPICS_SUBSTRATE": "1"}))

    assert devices == {}
    assert "name no devices" in caplog.text


def test_build_devices_builds_connector_mediated_devices_from_the_pv_lists() -> None:
    connector = FakeConnector()
    env = {
        "BLUESKY_EPICS_SUBSTRATE": "1",
        "BLUESKY_EPICS_MOTORS": "corrector_01=SR:MAG:HCM:01:CUR:SP|SR:MAG:HCM:01:CUR:RB",
        "BLUESKY_EPICS_DETECTORS": "bpm_01=SR:DIAG:BPM:01:X:RB",
    }

    devices = asyncio.run(qserver_startup.build_devices(env=env, connector=connector))

    assert sorted(devices) == ["bpm_01", "corrector_01"]
    assert isinstance(devices["corrector_01"], ConnectorSettable)
    assert isinstance(devices["bpm_01"], ConnectorReadable)
    # Every device delegates to the one connector handed in — no raw Channel
    # Access is constructed anywhere in this path.
    assert devices["corrector_01"]._osprey_connector is connector
    assert devices["bpm_01"]._osprey_connector is connector


def test_address_named_devices_build_and_stay_visible_to_queueserver() -> None:
    """A device named by its own channel address survives the whole worker path.

    ``substrate_devices`` names each device after the address it drives or
    reads, so the worker namespace is keyed by colon-bearing names that are not
    Python identifiers. Nothing in this path may quietly drop them: the plan
    functions' identifier filter applies to plan names only, and the manager's
    namespace scan — which decides what the manager will accept in a queue item
    — reports devices by their namespace key.
    """
    from bluesky_queueserver.manager.profile_ops import existing_plans_and_devices_from_nspace

    connector = FakeConnector()
    setpoint = "SR:MAG:HCM:01:CURRENT:SP"
    readback = "SR:MAG:HCM:01:CURRENT:RB"
    bpm = "SR:DIAG:BPM:01:POSITION:X"
    env = {
        "BLUESKY_EPICS_SUBSTRATE": "1",
        "BLUESKY_EPICS_MOTORS": f"{setpoint}={setpoint}|{readback}",
        "BLUESKY_EPICS_DETECTORS": f"{bpm}={bpm}",
    }

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


def test_built_devices_read_through_the_connector() -> None:
    connector = FakeConnector()
    env = {
        "BLUESKY_EPICS_SUBSTRATE": "1",
        "BLUESKY_EPICS_DETECTORS": "bpm_01=SR:DIAG:BPM:01:X:RB",
    }
    devices = asyncio.run(qserver_startup.build_devices(env=env, connector=connector))

    asyncio.run(devices["bpm_01"].read())

    assert connector.reads == ["SR:DIAG:BPM:01:X:RB"]


@pytest.mark.parametrize("value", ["1", "true", "TRUE", " yes ", "on"])
def test_substrate_flag_accepts_the_usual_truthy_spellings(value: str) -> None:
    assert qserver_startup.is_substrate_enabled({"BLUESKY_EPICS_SUBSTRATE": value}) is True


@pytest.mark.parametrize("value", ["", "false", "0", "maybe"])
def test_substrate_flag_never_guesses_at_on(value: str) -> None:
    assert qserver_startup.is_substrate_enabled({"BLUESKY_EPICS_SUBSTRATE": value}) is False


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
    assert params.detectors == ["bpm_01"]
    # The nested JSON object came back as the plan's own nested model, with
    # its declared types — not as the raw dict the queue item carried.
    assert params.axes[0].setpoint == "corrector_01"
    assert params.axes[0].num_points == 3
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
    bad_kwargs["axes"][0]["num_points"] = 1  # schema requires >= 2

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
    env = {
        "BLUESKY_EPICS_SUBSTRATE": "1",
        "BLUESKY_EPICS_MOTORS": "corrector_01=SR:MAG:HCM:01:CUR:SP",
        "BLUESKY_EPICS_DETECTORS": "bpm_01=SR:DIAG:BPM:01:X:RB",
    }
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
    env = {
        "BLUESKY_EPICS_SUBSTRATE": "1",
        "BLUESKY_EPICS_MOTORS": "corrector_01=SR:MAG:HCM:01:CUR:SP",
        "BLUESKY_EPICS_DETECTORS": "bpm_01=SR:DIAG:BPM:01:X:RB",
    }
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
