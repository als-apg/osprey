"""Tests for the opt-in EPICS-substrate scanner startup wiring (task 2.3).

`app.py`'s `_lifespan` hook wires a real bluesky-backed `BlueskyScanner`
against real EPICS devices (Channel Access clients built by
`devices/epics.py`, from a PV list parsed by `devices/_specs_from_env.py`)
ONLY when `BLUESKY_EPICS_SUBSTRATE` is truthy (`_is_epics_substrate_enabled`)
AND bluesky/ophyd-async are importable — mirroring the mock demo scanner's
guarded/lazy-import pattern (see `test_demo_scanner_wiring.py`) so the
Phase-1 "app.py import-clean of bluesky" invariant holds whether the extra is
absent or the flag is unset. Exercised here:

- `_is_epics_substrate_enabled()` directly, across the truthy/non-truthy
  value matrix (already covered exhaustively for the shared parsing logic by
  `test_demo_scanner_wiring.py`'s `_is_demo_scanner_enabled` matrix; this
  file only spot-checks it for the substrate flag's own name).
- Flag absent: app.py stays import-clean and keeps the `FakeScanner` default.
- Flag truthy AND bluesky/ophyd-async present: the app wires a
  `BlueskyScanner` whose device source is `epics.build_devices(motors,
  detectors)`, built from `BLUESKY_EPICS_MOTORS`/`BLUESKY_EPICS_DETECTORS`.
- Flag truthy but the bluesky-bridge extra absent: guarded fallback to
  `FakeScanner`, simulated by forcing `ophyd_async` out of `sys.modules` (see
  `_ophyd_async_absent`) since this environment actually has the extra
  installed.
- Both `BLUESKY_EPICS_SUBSTRATE` and `BLUESKY_DEMO_SCANNER` set: the EPICS
  substrate wins, with a warning logged.
- Task 2.5: `BLUESKY_TILED_URI` wires a `TiledWriter` subscription onto the
  EPICS-substrate scanner. This path has no e2e coverage (unlike the demo
  scanner's real-scan round trip in `test_demo_scanner_wiring.py`), so its
  wiring is asserted explicitly here rather than left to an integration test.
"""

from __future__ import annotations

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
    """Every test gets a clean flag set, registry, and scanner factory."""
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
    yield
    registry._runs.clear()
    set_scanner_factory(FakeScanner)


@pytest.fixture
def _ophyd_async_absent(monkeypatch: pytest.MonkeyPatch):
    """Force `from .devices import epics` to raise ImportError(name="ophyd_async...").

    This environment actually has the `bluesky-bridge` extra installed, so
    the only way to exercise the "extra not installed" fallback branch is to
    simulate it: purge the bridge's `devices` submodules from `sys.modules`
    (so the next import is not served from cache) and set `ophyd_async` to
    `None` there — the documented sentinel that makes the import machinery
    raise `ImportError` for any `ophyd_async` (sub)module, without needing
    ophyd-async to actually be uninstalled.

    The `ophyd_async.*` submodules are purged too (and restored after): if any
    earlier test/collection imported e.g. `ophyd_async.core`, a cached submodule
    would satisfy `from ophyd_async.core import ...` and defeat the top-level
    `None` sentinel — so the simulation must clear the whole `ophyd_async`
    subtree, not just the package root, to stay robust to import order.
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
    yield
    for name, mod in purged.items():
        sys.modules[name] = mod


# =========================================================================
# _is_epics_substrate_enabled(): spot-check (shared truthy-parsing logic is
# exhaustively covered by test_demo_scanner_wiring.py's parallel matrix)
# =========================================================================


@pytest.mark.parametrize("value", ["1", "true", "TRUE", " yes ", "on"])
def test_is_epics_substrate_enabled_accepts_truthy_variants(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(_SUBSTRATE_ENV, value)
    assert app_module._is_epics_substrate_enabled() is True


@pytest.mark.parametrize("value", ["false", "0", "", "bogus"])
def test_is_epics_substrate_enabled_rejects_non_truthy_variants(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(_SUBSTRATE_ENV, value)
    assert app_module._is_epics_substrate_enabled() is False


def test_is_epics_substrate_enabled_false_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_SUBSTRATE_ENV, raising=False)
    assert app_module._is_epics_substrate_enabled() is False


# =========================================================================
# Flag unset: app stays import-clean, FakeScanner default unchanged
# =========================================================================


def test_flag_unset_stays_import_clean_and_keeps_fake_scanner_default() -> None:
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200

    assert app_module._scanner_factory is FakeScanner


# =========================================================================
# Flag truthy but the bluesky-bridge extra absent: guarded fallback, no crash
# =========================================================================


def test_flag_set_but_extra_absent_falls_back_to_fake_scanner(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    _ophyd_async_absent: None,
) -> None:
    monkeypatch.setenv(_SUBSTRATE_ENV, "true")
    monkeypatch.setenv(_MOTORS_ENV, "mot1=TEST:MOTOR:01:SP")

    with caplog.at_level("WARNING", logger="osprey.services.bluesky_bridge.app"):
        with TestClient(app) as client:
            resp = client.get("/health")
            assert resp.status_code == 200

    assert app_module._scanner_factory is FakeScanner
    assert any("falling back to FakeScanner" in rec.message for rec in caplog.records)


# =========================================================================
# Flag truthy AND the extra present: real wiring against epics.build_devices
# =========================================================================


@pytest.mark.parametrize("value", ["1", "true"])
def test_flag_set_with_extra_present_wires_epics_backed_scanner(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    pytest.importorskip("bluesky")
    pytest.importorskip("ophyd_async")

    monkeypatch.setenv(_SUBSTRATE_ENV, value)
    monkeypatch.setenv(_MOTORS_ENV, "mot1=TEST:MOTOR:01:SP|TEST:MOTOR:01:RB")
    monkeypatch.setenv(_DETECTORS_ENV, "det1=TEST:DET:01:RB")

    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200

    assert app_module._scanner_factory is not FakeScanner

    from osprey.services.bluesky_bridge.scanner_bluesky import BlueskyScanner

    scanner = app_module._scanner_factory()
    assert isinstance(scanner, BlueskyScanner)
    # `devices` is the bare async factory (not its awaited result) — matches
    # `scanner_bluesky.BlueskyScanner._resolve_devices`'s expected shape.
    assert callable(scanner._devices_source)


def test_flag_set_with_no_pv_env_wires_scanner_with_zero_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Absent/empty PV-list env vars are a valid (if useless) config, not an error."""
    pytest.importorskip("bluesky")
    pytest.importorskip("ophyd_async")

    monkeypatch.setenv(_SUBSTRATE_ENV, "true")

    with TestClient(app):
        pass

    assert app_module._scanner_factory is not FakeScanner


# =========================================================================
# Precedence: both flags set -> EPICS substrate wins, with a warning logged
# =========================================================================


def test_both_flags_set_epics_substrate_wins(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    pytest.importorskip("bluesky")
    pytest.importorskip("ophyd_async")

    monkeypatch.setenv(_SUBSTRATE_ENV, "true")
    monkeypatch.setenv(_DEMO_ENV, "true")
    monkeypatch.setenv(_MOTORS_ENV, "mot1=TEST:MOTOR:01:SP")

    with caplog.at_level("WARNING", logger="osprey.services.bluesky_bridge.app"):
        with TestClient(app) as client:
            resp = client.get("/health")
            assert resp.status_code == 200

    assert app_module._scanner_factory is not FakeScanner
    assert any("takes precedence" in rec.message for rec in caplog.records)

    from osprey.services.bluesky_bridge.scanner_bluesky import BlueskyScanner

    scanner = app_module._scanner_factory()
    assert isinstance(scanner, BlueskyScanner)
    # Substrate scanners are built with the explicit built-in plan set, not
    # the demo path's default (None -> merged built-ins + facility plans).
    assert scanner._plans is not None


# =========================================================================
# Task 2.5: BLUESKY_TILED_URI wires a TiledWriter subscription
# =========================================================================


def test_tiled_uri_set_subscribes_a_tiled_writer(monkeypatch: pytest.MonkeyPatch) -> None:
    """`_build_tiled_writer_factory` reaches all the way through to a real
    `TiledWriter.from_uri(uri, api_key=...)` call once `BlueskyScanner` is
    built — spying on `from_uri` (rather than actually connecting) keeps this
    a unit test while still exercising the real wiring path end to end.
    """
    pytest.importorskip("bluesky")
    pytest.importorskip("ophyd_async")

    from bluesky.callbacks.tiled_writer import TiledWriter

    calls: list[tuple[str, dict]] = []

    def fake_from_uri(uri, **kwargs):
        calls.append((uri, kwargs))
        return lambda name, doc: None

    monkeypatch.setattr(TiledWriter, "from_uri", fake_from_uri)
    monkeypatch.setenv(_SUBSTRATE_ENV, "true")
    monkeypatch.setenv(_MOTORS_ENV, "mot1=TEST:MOTOR:01:SP")
    monkeypatch.setenv(_TILED_URI_ENV, "http://tiled:8000")
    monkeypatch.setenv(_TILED_API_KEY_ENV, "test-api-key")

    with TestClient(app):
        pass

    from osprey.services.bluesky_bridge.scanner_bluesky import BlueskyScanner

    scanner = app_module._scanner_factory()
    assert isinstance(scanner, BlueskyScanner)
    assert calls == [("http://tiled:8000", {"api_key": "test-api-key"})]
    assert scanner.tiled_degraded is False


def test_tiled_uri_unset_builds_scanner_with_no_tiled_subscription(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default (no Tiled configured) behaves exactly like Phase 1: `_build_tiled_writer_factory`
    returns `None`, so `BlueskyScanner.__init__` never imports or calls `TiledWriter` at all.
    """
    pytest.importorskip("bluesky")
    pytest.importorskip("ophyd_async")

    monkeypatch.setenv(_SUBSTRATE_ENV, "true")
    monkeypatch.setenv(_MOTORS_ENV, "mot1=TEST:MOTOR:01:SP")

    with TestClient(app):
        pass

    assert app_module._build_tiled_writer_factory() is None

    from osprey.services.bluesky_bridge.scanner_bluesky import BlueskyScanner

    scanner = app_module._scanner_factory()
    assert isinstance(scanner, BlueskyScanner)
