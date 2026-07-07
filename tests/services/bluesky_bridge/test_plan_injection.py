"""Contract test for the facility plan-injection seam (`plan_loader.py`).

Loads a fake facility plan module from a temp dir OUTSIDE the
`osprey.services.bluesky_bridge` package — exactly the "facility repo, not
this framework" shape the contract promises — and asserts the bridge serves
its plans via `GET /plans` and constructs its devices. The fake module needs
NO bluesky import (only `osprey.services.bluesky_bridge.plan_types.PlanSpec`,
which is itself bluesky-clean), so this test also proves `plan_loader.py`
works in a bluesky-less bridge process.
"""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from osprey.services.bluesky_bridge import plan_loader
from osprey.services.bluesky_bridge.app import app

_ENV_VAR = "BLUESKY_PLAN_MODULE"

_FACILITY_MODULE_SOURCE = '''
"""A fake facility plan module — deliberately outside the osprey package."""

from osprey.services.bluesky_bridge.plan_types import PlanSpec
from pydantic import BaseModel


class WiggleParams(BaseModel):
    amplitude: float = 1.0


def _wiggle_plan(devices, params):
    """No bluesky needed: just prove `devices` and `params` were threaded through."""
    return {"device": devices["wiggler"], "amplitude": params.amplitude}


PLANS = {
    "wiggle": PlanSpec(
        name="wiggle",
        plan=_wiggle_plan,
        schema=WiggleParams,
        description="A facility-specific plan not in the built-in set.",
    ),
}


def get_devices():
    return {"wiggler": "fake-wiggler-device"}
'''


@pytest.fixture(autouse=True)
def _isolated_facility_plans(monkeypatch: pytest.MonkeyPatch):
    """Every test gets a clean plan_loader cache and no leftover env var."""
    monkeypatch.delenv(_ENV_VAR, raising=False)
    plan_loader.reset_facility_plans()
    yield
    plan_loader.reset_facility_plans()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _write_facility_module(tmp_path: Path) -> Path:
    facility_dir = tmp_path / "not_the_osprey_package"
    facility_dir.mkdir()
    module_path = facility_dir / "my_facility_plans.py"
    module_path.write_text(_FACILITY_MODULE_SOURCE)
    return module_path


def test_load_facility_plans_returns_empty_when_unconfigured() -> None:
    result = plan_loader.load_facility_plans()
    assert result.plans == {}
    assert result.devices == {}


def test_load_facility_plans_reads_plans_and_devices_from_a_temp_module(tmp_path: Path) -> None:
    module_path = _write_facility_module(tmp_path)

    result = plan_loader.load_facility_plans(str(module_path))

    assert set(result.plans) == {"wiggle"}
    assert result.plans["wiggle"].name == "wiggle"
    assert result.devices == {"wiggler": "fake-wiggler-device"}


def test_load_facility_plans_resolves_module_path_from_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_path = _write_facility_module(tmp_path)
    monkeypatch.setenv(_ENV_VAR, str(module_path))

    result = plan_loader.load_facility_plans()

    assert set(result.plans) == {"wiggle"}


def test_load_facility_plans_raises_on_missing_module_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        plan_loader.load_facility_plans(str(tmp_path / "does_not_exist.py"))


def test_load_facility_plans_raises_when_plans_attribute_missing(tmp_path: Path) -> None:
    module_path = tmp_path / "no_plans.py"
    module_path.write_text("def get_devices():\n    return {}\n")

    with pytest.raises(AttributeError, match="PLANS"):
        plan_loader.load_facility_plans(str(module_path))


def test_load_facility_plans_raises_when_get_devices_missing(tmp_path: Path) -> None:
    module_path = tmp_path / "no_get_devices.py"
    module_path.write_text("PLANS = {}\n")

    with pytest.raises(AttributeError, match="get_devices"):
        plan_loader.load_facility_plans(str(module_path))


def test_load_facility_plans_propagates_get_devices_errors(tmp_path: Path) -> None:
    """A facility `get_devices()` that raises must surface, not be swallowed —
    a facility that can't build its devices is misconfigured and should fail
    loudly rather than serve plans against an empty device map."""
    module_path = tmp_path / "boom_devices.py"
    module_path.write_text(
        "PLANS = {}\n\ndef get_devices():\n    raise RuntimeError('device backend unavailable')\n"
    )

    with pytest.raises(RuntimeError, match="device backend unavailable"):
        plan_loader.load_facility_plans(str(module_path))


def test_load_facility_plans_registers_module_in_sys_modules(tmp_path: Path) -> None:
    """The loaded facility module is registered in `sys.modules` under its
    synthetic name, so intra-module self-references (dataclasses, pickling)
    resolve during import."""
    module_path = _write_facility_module(tmp_path)
    synthetic_name = f"_bluesky_bridge_facility_plans_{module_path.stem}"
    sys.modules.pop(synthetic_name, None)  # clean precondition (other tests share the stem)

    plan_loader.load_facility_plans(str(module_path))

    assert synthetic_name in sys.modules
    del sys.modules[synthetic_name]


def test_load_facility_plans_does_not_leave_module_in_sys_modules_on_exec_failure(
    tmp_path: Path,
) -> None:
    """A facility module that raises during import leaves no half-initialized
    entry behind in `sys.modules` to shadow a later, fixed load."""
    module_path = tmp_path / "explodes_on_import.py"
    module_path.write_text("raise RuntimeError('boom at import time')\n")
    synthetic_name = f"_bluesky_bridge_facility_plans_{module_path.stem}"

    with pytest.raises(RuntimeError, match="boom at import time"):
        plan_loader.load_facility_plans(str(module_path))

    assert synthetic_name not in sys.modules


def test_get_plans_serves_facility_injected_plan_via_http(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """End-to-end: GET /plans includes the facility's plan, with its schema."""
    module_path = _write_facility_module(tmp_path)
    monkeypatch.setenv(_ENV_VAR, str(module_path))

    resp = client.get("/plans")

    assert resp.status_code == 200
    plans_by_name = {p["name"]: p for p in resp.json()}
    assert "wiggle" in plans_by_name
    wiggle = plans_by_name["wiggle"]
    assert wiggle["description"] == "A facility-specific plan not in the built-in set."
    assert wiggle["schema"]["properties"]["amplitude"]["default"] == 1.0


def test_get_facility_plans_constructs_devices_once_and_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`get_facility_plans()` is a cached singleton — one `get_devices()` call per process."""
    module_path = _write_facility_module(tmp_path)
    monkeypatch.setenv(_ENV_VAR, str(module_path))

    first = plan_loader.get_facility_plans()
    second = plan_loader.get_facility_plans()

    assert first is second
    assert first.devices == {"wiggler": "fake-wiggler-device"}


def test_load_facility_plans_warns_when_a_plan_shadows_a_builtin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A facility plan reusing a built-in's name (e.g. `count`) logs a warning
    at load time — silent shadowing would otherwise be a surprising way to
    lose a built-in without any trace. Injects a fake `plans` module into
    `sys.modules` so this is testable without bluesky installed: relative
    imports resolve against `sys.modules` first, so `plan_loader`'s guarded
    `from .plans import BUILTIN_PLANS` picks up the fake instead of trying
    (and failing) to import the real bluesky-backed module.
    """
    fake_plans_module = types.ModuleType("osprey.services.bluesky_bridge.plans")
    fake_plans_module.BUILTIN_PLANS = {"count": object()}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "osprey.services.bluesky_bridge.plans", fake_plans_module)

    facility_dir = tmp_path / "shadowing_facility"
    facility_dir.mkdir()
    module_path = facility_dir / "shadow_plans.py"
    module_path.write_text(
        "from osprey.services.bluesky_bridge.plan_types import PlanSpec\n"
        "from pydantic import BaseModel\n\n"
        "class P(BaseModel):\n"
        "    pass\n\n"
        "PLANS = {'count': PlanSpec(name='count', plan=lambda d, p: None, schema=P, "
        "description='facility count')}\n\n"
        "def get_devices():\n"
        "    return {}\n"
    )

    with caplog.at_level(logging.WARNING, logger="osprey.services.bluesky_bridge.plan_loader"):
        result = plan_loader.load_facility_plans(str(module_path))

    assert "count" in result.plans
    assert any(
        record.levelno == logging.WARNING and "count" in record.message for record in caplog.records
    )


def test_get_plans_reraises_a_non_bridge_import_error_from_builtins(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """`GET /plans` must only swallow an ImportError caused by a missing
    bridge dependency (bluesky/ophyd/ophyd_async/tiled) — anything else (e.g.
    `plans.py` itself missing an expected attribute) is a genuine bug and
    must propagate, not silently degrade to an empty/facility-only list.

    Injects a fake, attribute-less `plans` module into `sys.modules` so
    `from .plans import BUILTIN_PLANS` raises "cannot import name", whose
    `ImportError.name` is the `plans` module itself — not a bridge dependency
    — so the route's allow-list must re-raise rather than swallow it.
    """
    broken_plans_module = types.ModuleType("osprey.services.bluesky_bridge.plans")
    monkeypatch.setitem(sys.modules, "osprey.services.bluesky_bridge.plans", broken_plans_module)

    with pytest.raises(ImportError):
        client.get("/plans")
