"""Tests for the bridge's single long-lived OSPREY connector (task 2.1).

`app.py`'s `_lifespan` hook constructs exactly one OSPREY connector — via
`ConnectorFactory.create_control_system_connector` with a gateway-less
``virtual_accelerator`` `type_config` — whenever the EPICS substrate is
enabled (`_is_epics_substrate_enabled()`) and the bluesky-bridge extra is
importable, holds it as the module-level `_connector` singleton for the
process's whole lifetime, and disconnects it exactly once on shutdown. This
task does NOT wire the connector into the scanner factory (`_epics_scanner_factory`
still builds devices straight from `epics.build_devices`) — that is a later
task; here we only prove construct/hold/disconnect.

Exercised here:

- Lifespan constructs exactly one connector and disconnects it exactly once
  on shutdown (spy `ConnectorFactory.create_control_system_connector`).
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
from osprey.services.bluesky_bridge.app import app, set_scanner_factory
from osprey.services.bluesky_bridge.runs import registry
from osprey.services.bluesky_bridge.scanner import FakeScanner

_SUBSTRATE_ENV = "BLUESKY_EPICS_SUBSTRATE"
_DEMO_ENV = "BLUESKY_DEMO_SCANNER"
_MOTORS_ENV = "BLUESKY_EPICS_MOTORS"
_DETECTORS_ENV = "BLUESKY_EPICS_DETECTORS"
_TILED_URI_ENV = "BLUESKY_TILED_URI"
_TILED_API_KEY_ENV = "BLUESKY_TILED_API_KEY"


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch: pytest.MonkeyPatch):
    """Every test gets a clean flag set, registry, scanner factory, and connector global."""
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
    set_scanner_factory(FakeScanner)
    app_module._connector = None
    yield
    registry._runs.clear()
    set_scanner_factory(FakeScanner)
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
    # compose-inherited EPICS_CA_NAME_SERVERS survive untouched.
    assert construct_calls == [
        {
            "type": "virtual_accelerator",
            "connector": {"virtual_accelerator": {"timeout": 5.0}},
        }
    ]


def test_lifespan_does_not_construct_connector_when_flag_unset() -> None:
    """No EPICS substrate, no connector — mirrors the existing FakeScanner-default test."""
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert app_module.get_connector() is None

    assert app_module.get_connector() is None


def test_lifespan_does_not_construct_connector_when_extra_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Substrate flag set but the bluesky-bridge extra missing: fall back to
    `FakeScanner` with no connector constructed at all (mirrors the existing
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

        assert app_module._scanner_factory is FakeScanner
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
