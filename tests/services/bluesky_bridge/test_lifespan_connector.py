"""Tests for the bridge's single long-lived OSPREY connector (task 2.1, 3.4).

`app.py`'s `_lifespan` hook constructs exactly one OSPREY connector — via
`ConnectorFactory.create_control_system_connector`, built from the project's
`control_system.type` (task 3.4; fail-safe default `"mock"`) — whenever the
EPICS substrate is enabled (`_is_epics_substrate_enabled()`) and the
bluesky stack is importable, holds it as the module-level `_connector`
singleton for the process's whole lifetime, and disconnects it exactly once
on shutdown. The connector IS wired into the runner factory
(`_epics_runner_factory` builds devices via `connector_devices.build_devices`,
connector-mediated) — device construction is exercised here only insofar as
it touches connector construct/hold/disconnect; the mock connector's
readbacks do not track setpoints, so a scan does not run to completion
against it (mock mode is for browsing/UI, not for running plans).

Exercised here:

- Lifespan constructs exactly one connector and disconnects it exactly once
  on shutdown (spy `ConnectorFactory.create_control_system_connector`).
- With `control_system.type=virtual_accelerator`, the `type_config` passed is
  the gateway-less recipe (no "gateways" key) — byte-identical to the
  pre-task-3.4 hardcoded config.
- With `control_system.type` unset/unreadable, the resolved type — and the
  `type_config` built from it — is `"mock"` (fail-safe default).
- The real factory path, with the exact gateway-less `type_config` from the
  validated construction recipe, yields an `isinstance(connector, EPICSConnector)`
  instance for the `virtual_accelerator` type — no CA server needed, since a
  gateway-less config skips all CA I/O in `connect()`.
- Import-cleanliness: importing the bridge core app module does not pull
  `ophyd_async` or `pyepics` into `sys.modules` (mirrors `test_app_import_clean.py`'s
  subprocess pattern — an in-process check can't be trusted once other tests in
  this suite have already imported these packages).
"""

from __future__ import annotations

import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from osprey.services.bluesky_bridge import app as app_module
from osprey.services.bluesky_bridge.app import app, set_runner_factory
from osprey.services.bluesky_bridge.plan_runner import FakePlanRunner
from osprey.services.bluesky_bridge.runs import registry

_SUBSTRATE_ENV = "BLUESKY_EPICS_SUBSTRATE"
_DEMO_ENV = "BLUESKY_DEMO_RUNNER"
_MOTORS_ENV = "BLUESKY_EPICS_MOTORS"
_DETECTORS_ENV = "BLUESKY_EPICS_DETECTORS"
_TILED_URI_ENV = "BLUESKY_TILED_URI"
_TILED_API_KEY_ENV = "BLUESKY_TILED_API_KEY"


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch: pytest.MonkeyPatch):
    """Every test gets a clean flag set, registry, runner factory, and connector global."""
    for var in (
        _SUBSTRATE_ENV,
        _DEMO_ENV,
        _MOTORS_ENV,
        _DETECTORS_ENV,
        _TILED_URI_ENV,
        _TILED_API_KEY_ENV,
    ):
        monkeypatch.delenv(var, raising=False)
    registry._runs.clear()
    set_runner_factory(FakePlanRunner)
    app_module._connector = None
    yield
    registry._runs.clear()
    set_runner_factory(FakePlanRunner)
    app_module._connector = None


# =========================================================================
# Lifespan constructs exactly one connector, disconnects it exactly once
# =========================================================================


class _SpyConnector:
    """A minimal async-`disconnect`-only stand-in for `EPICSConnector`."""

    def __init__(self) -> None:
        self.disconnect_calls = 0

    async def disconnect(self) -> None:
        self.disconnect_calls += 1


def _patch_control_system_type(monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
    """Patch `osprey.utils.config.get_config_value` so `control_system.type`
    resolves to `value` (or falls through to the caller-supplied default when
    `value` is `None`, simulating "key absent" / "no project config").

    Same convention `test_startup_assertion.py`'s `_patch_config` uses:
    `_resolve_control_system_type` does its own
    `from osprey.utils.config import get_config_value` inside the function
    body, so patching the underlying `osprey.utils.config` attribute takes
    effect on the next call.
    """

    def fake_get_config_value(key: str, default=None):
        if key == "control_system.type":
            return value if value is not None else default
        return default

    monkeypatch.setattr("osprey.utils.config.get_config_value", fake_get_config_value)


def test_lifespan_constructs_one_connector_and_disconnects_once_on_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("bluesky")
    pytest.importorskip("ophyd_async")

    from osprey.connectors.factory import ConnectorFactory

    construct_calls: list[dict] = []
    spy_connector = _SpyConnector()

    async def fake_create_control_system_connector(config):
        construct_calls.append(config)
        return spy_connector

    monkeypatch.setattr(
        ConnectorFactory,
        "create_control_system_connector",
        fake_create_control_system_connector,
    )

    # Task 3.4: the connector type now comes from `control_system.type` — set
    # it explicitly to `virtual_accelerator` so this test still exercises the
    # VA path (the bridge's fail-safe default, absent config, is `mock`).
    _patch_control_system_type(monkeypatch, "virtual_accelerator")
    monkeypatch.setenv(_SUBSTRATE_ENV, "true")
    monkeypatch.setenv(_MOTORS_ENV, "mot1=TEST:MOTOR:01:SP")

    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        # Constructed exactly once, and held as the module-level singleton
        # while the app is up.
        assert len(construct_calls) == 1
        assert app_module.get_connector() is spy_connector
        assert spy_connector.disconnect_calls == 0

    # Disconnected exactly once on shutdown, and the singleton is cleared.
    assert spy_connector.disconnect_calls == 1
    assert app_module.get_connector() is None

    # The `type_config` used is gateway-less (no "gateways" key) — the
    # validated construction recipe from Task 0.2 that lets the
    # compose-inherited EPICS_CA_NAME_SERVERS survive untouched. This is
    # byte-identical to the config the bridge always built before task 3.4
    # made the type configurable.
    assert construct_calls == [
        {
            "type": "virtual_accelerator",
            "connector": {"virtual_accelerator": {"timeout": 5.0}},
        }
    ]


def test_lifespan_builds_mock_connector_when_control_system_type_is_mock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Task 3.4: `control_system.type=mock` builds a minimal mock `type_config`
    (no CA/gateways) instead of the `virtual_accelerator` config — the
    connector-mediated devices are then constructed against the mock
    connector using the real corrector/BPM channel names. This is
    construction-only: the mock connector's readbacks do not track
    setpoints, so a settle-verified scan would not complete against it
    (mock mode is for browsing/UI, not for running plans).
    """
    pytest.importorskip("bluesky")
    pytest.importorskip("ophyd_async")

    from osprey.connectors.factory import ConnectorFactory

    construct_calls: list[dict] = []
    spy_connector = _SpyConnector()

    async def fake_create_control_system_connector(config):
        construct_calls.append(config)
        return spy_connector

    monkeypatch.setattr(
        ConnectorFactory,
        "create_control_system_connector",
        fake_create_control_system_connector,
    )

    _patch_control_system_type(monkeypatch, "mock")
    monkeypatch.setenv(_SUBSTRATE_ENV, "true")
    monkeypatch.setenv(_MOTORS_ENV, "mot1=TEST:MOTOR:01:SP")

    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert app_module.get_connector() is spy_connector

    assert construct_calls == [{"type": "mock", "connector": {"mock": {}}}]


def test_lifespan_defaults_to_mock_connector_when_control_system_type_unreadable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Task 3.4: fail-SAFE default — no project config context at all (the
    normal unit-test environment; `control_system.type` resolves via the
    caller-supplied default) builds the mock `type_config`, never
    `virtual_accelerator`/`epics`. This is the "unreadable config" case that
    must never silently connect to real Channel Access.
    """
    pytest.importorskip("bluesky")
    pytest.importorskip("ophyd_async")

    from osprey.connectors.factory import ConnectorFactory

    construct_calls: list[dict] = []
    spy_connector = _SpyConnector()

    async def fake_create_control_system_connector(config):
        construct_calls.append(config)
        return spy_connector

    monkeypatch.setattr(
        ConnectorFactory,
        "create_control_system_connector",
        fake_create_control_system_connector,
    )

    # No `_patch_control_system_type` call: `get_config_value` runs for real
    # against this (unconfigured) test environment, and — same as
    # `_assert_limits_readable_if_writable`'s "no project config context"
    # handling — falls through to the caller-supplied default ("mock").
    monkeypatch.setenv(_SUBSTRATE_ENV, "true")
    monkeypatch.setenv(_MOTORS_ENV, "mot1=TEST:MOTOR:01:SP")

    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert app_module.get_connector() is spy_connector

    assert construct_calls == [{"type": "mock", "connector": {"mock": {}}}]


def test_lifespan_does_not_construct_connector_when_flag_unset() -> None:
    """No EPICS substrate, no connector — mirrors the existing FakePlanRunner-default test."""
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert app_module.get_connector() is None

    assert app_module.get_connector() is None


def test_lifespan_does_not_construct_connector_when_extra_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Substrate flag set but the bluesky stack not importable: fall back to
    `FakePlanRunner` with no connector constructed at all (mirrors the existing
    `test_flag_set_but_extra_absent_falls_back_to_fake_scanner` simulation).
    """
    purged = {
        name: mod
        for name, mod in sys.modules.items()
        if name == "osprey.services.bluesky_bridge.devices"
        or name.startswith("osprey.services.bluesky_bridge.devices.")
        or name.startswith("ophyd_async.")
    }
    for name in purged:
        del sys.modules[name]
    monkeypatch.setitem(sys.modules, "ophyd_async", None)
    try:
        monkeypatch.setenv(_SUBSTRATE_ENV, "true")
        monkeypatch.setenv(_MOTORS_ENV, "mot1=TEST:MOTOR:01:SP")

        with TestClient(app) as client:
            resp = client.get("/health")
            assert resp.status_code == 200
            assert app_module.get_connector() is None

        assert app_module._runner_factory is FakePlanRunner
        assert app_module.get_connector() is None
    finally:
        for name, mod in purged.items():
            sys.modules[name] = mod


# =========================================================================
# Real factory path: isinstance(connector, EPICSConnector) for
# virtual_accelerator, no CA server needed (gateway-less config)
# =========================================================================


async def test_real_factory_path_yields_epics_connector_subclass() -> None:
    from osprey.connectors.control_system.epics_connector import EPICSConnector
    from osprey.connectors.factory import ConnectorFactory, register_builtin_connectors

    register_builtin_connectors()  # idempotent; must run before create
    cs_config = {
        "type": "virtual_accelerator",
        "connector": {"virtual_accelerator": {"timeout": 5.0}},
    }
    connector = await ConnectorFactory.create_control_system_connector(cs_config)
    try:
        # VirtualAcceleratorConnector is a subclass of EPICSConnector (FR5) —
        # assert via isinstance, not exact type.
        assert isinstance(connector, EPICSConnector)
    finally:
        await connector.disconnect()


# =========================================================================
# Import-cleanliness: the bridge core app module never leaks ophyd_async/
# pyepics into sys.modules, even after Task 2.1's connector wiring
# =========================================================================


def _run_import_check(module: str) -> subprocess.CompletedProcess[str]:
    code = f"import osprey.services.bluesky_bridge.app, sys; assert {module!r} not in sys.modules"
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("module", ["ophyd_async", "pyepics", "epics"])
def test_importing_app_does_not_leak_epics_modules(module: str) -> None:
    """Mirrors `test_app_import_clean.py`'s subprocess pattern: only a fresh
    interpreter, spawned before anything else has touched `sys.modules`, can
    answer whether a bare `import app` leaks these. `BLUESKY_EPICS_SUBSTRATE`
    is unset in the child process, so `_lifespan` never runs its connector-
    construction branch at all — this only guards module-scope imports.
    """
    result = _run_import_check(module)
    assert result.returncode == 0, (
        "importing osprey.services.bluesky_bridge.app leaked a top-level "
        f"import of {module!r} (child exit {result.returncode}):\n{result.stderr}"
    )
