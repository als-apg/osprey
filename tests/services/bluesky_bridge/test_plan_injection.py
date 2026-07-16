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

import sys
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
    """End-to-end: GET /plans includes the facility's plan, with its schema.

    Also asserts every plan carries `metadata`/`provenance` keys (task 1.3):
    the legacy contract's `wiggle` plan never set `PlanSpec.metadata`, so it
    stays `None`, but its `provenance` is loader-normalized to `"facility"`
    regardless of what the module declared (see `load_facility_plans`). A
    shipped plan (`orm`) is the other half of the contract: `provenance ==
    "shipped"` with real authored `PLAN_METADATA` — see
    `test_plan_loader_layered.py` for the directory-layer case in isolation.
    """
    module_path = _write_facility_module(tmp_path)
    monkeypatch.setenv(_ENV_VAR, str(module_path))

    resp = client.get("/plans")

    assert resp.status_code == 200
    plans_by_name = {p["name"]: p for p in resp.json()}
    assert "wiggle" in plans_by_name
    wiggle = plans_by_name["wiggle"]
    assert wiggle["description"] == "A facility-specific plan not in the built-in set."
    assert wiggle["schema"]["properties"]["amplitude"]["default"] == 1.0
    assert wiggle["metadata"] is None
    assert wiggle["provenance"] == "facility"

    orm = plans_by_name["orm"]
    assert orm["metadata"] is not None
    assert orm["provenance"] == "shipped"


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


def test_get_plans_serves_directory_layer_metadata_via_http(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """A directory-layer plan (task 1.2's `BLUESKY_PLAN_DIRS`, `facility` tier)
    authors real `PLAN_METADATA` — unlike the legacy single-module contract
    above, `GET /plans` must surface it verbatim, not just `provenance`."""
    facility_dir = tmp_path / "facility_layer"
    facility_dir.mkdir()
    (facility_dir / "sniff.py").write_text(
        "from pydantic import BaseModel\n\n\n"
        "PLAN_METADATA = {\n"
        '    "name": "sniff",\n'
        '    "description": "A directory-layer test plan.",\n'
        '    "category": "accelerator",\n'
        '    "required_devices": ["sniffer"],\n'
        '    "writes": False,\n'
        "}\n\n\n"
        "def build_plan(devices, params):\n"
        '    return {"plan": "sniff"}\n'
    )
    monkeypatch.setenv("BLUESKY_PLAN_DIRS", str(facility_dir))

    resp = client.get("/plans")

    assert resp.status_code == 200
    plans_by_name = {p["name"]: p for p in resp.json()}
    sniff = plans_by_name["sniff"]
    assert sniff["provenance"] == "facility"
    assert sniff["metadata"] == {
        "name": "sniff",
        "description": "A directory-layer test plan.",
        "category": "accelerator",
        "required_devices": ["sniffer"],
        "writes": False,
    }
