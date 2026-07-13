"""End-to-end verification of the facility plan-injection contract (task 2.4)
via `config.yml`'s `scan.plan_module` — no framework source changes here,
purely a test asserting the seam the framework already exposes.

Unlike `test_plan_injection.py` (task 2.4's own contract test, which mostly
drives `plan_loader.py` directly and via the `BLUESKY_PLAN_MODULE` env var),
this test goes through the full `scan.plan_module` config.yml path — the
"local/dev convenience" resolution tier — mirroring how
`tests/mcp_server/test_scan_context.py` verifies `scan.bridge_url`/
`scan.promote_token` config fallback. A custom plan module living in a temp
dir (PLANS + get_devices, no bluesky import) is loaded purely from config,
its plans show up over real HTTP via `GET /plans`, and its devices construct.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from osprey.services.bluesky_bridge import plan_loader
from osprey.services.bluesky_bridge.app import app

_FACILITY_MODULE_SOURCE = '''
"""A custom facility plan module — deliberately outside the osprey package,
loaded purely via config.yml's `scan.plan_module`, no bluesky import."""

from osprey.services.bluesky_bridge.plan_types import PlanSpec
from pydantic import BaseModel


class SnifflePathParams(BaseModel):
    sniff_time: float = 0.5


def _sniffle_plan(devices, params):
    return {"sniffer": devices["sniffer"], "sniff_time": params.sniff_time}


PLANS = {
    "sniffle_path": PlanSpec(
        name="sniffle_path",
        plan=_sniffle_plan,
        schema=SnifflePathParams,
        description="A custom facility plan loaded entirely via config.yml.",
    ),
}


def get_devices():
    return {"sniffer": "constructed-sniffer-device"}
'''


def _write_facility_module(tmp_path: Path) -> Path:
    facility_dir = tmp_path / "facility_repo" / "plans"
    facility_dir.mkdir(parents=True)
    module_path = facility_dir / "custom_plans.py"
    module_path.write_text(_FACILITY_MODULE_SOURCE)
    return module_path


def _write_config(tmp_path: Path, config_dict: dict) -> Path:
    config_file = tmp_path / "config.yml"
    config_file.write_text(yaml.dump(config_dict))
    return config_file


@pytest.fixture(autouse=True)
def _isolated_facility_plans():
    """Clean plan_loader cache before and after every test in this module.

    `OSPREY_CONFIG`/config caching is already reset by the repo-wide
    `reset_state_between_tests` autouse fixture (tests/conftest.py); this
    fixture only owns the `plan_loader` singleton, which that one doesn't
    know about.
    """
    plan_loader.reset_facility_plans()
    yield
    plan_loader.reset_facility_plans()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_facility_plan_module_loaded_via_config_yml_appears_in_get_plans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """The full config-only path: no env var, just `scan.plan_module` in config.yml."""
    monkeypatch.delenv("BLUESKY_PLAN_MODULE", raising=False)
    module_path = _write_facility_module(tmp_path)
    _write_config(tmp_path, {"scan": {"plan_module": str(module_path)}})
    monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "config.yml"))

    resp = client.get("/plans")

    assert resp.status_code == 200
    plans_by_name = {p["name"]: p for p in resp.json()}
    assert "sniffle_path" in plans_by_name
    sniffle = plans_by_name["sniffle_path"]
    assert sniffle["description"] == "A custom facility plan loaded entirely via config.yml."
    assert sniffle["schema"]["properties"]["sniff_time"]["default"] == 0.5


def test_facility_devices_construct_from_config_loaded_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`get_devices()` from the config-pointed module actually runs and its
    result is what the bridge holds — not just that the plan schema shows up.

    Isolates the aggregate from the shipped/facility directory layers (an
    empty `_SHIPPED_PLANS_DIR`, no `BLUESKY_PLAN_DIRS`) so the exact-set
    assertion below reflects only the config-loaded legacy module, not
    whatever exemplar plans the real `plans_core/` dir happens to ship.
    """
    monkeypatch.delenv("BLUESKY_PLAN_MODULE", raising=False)
    monkeypatch.delenv("BLUESKY_PLAN_DIRS", raising=False)
    monkeypatch.setattr(plan_loader, "_SHIPPED_PLANS_DIR", tmp_path / "no_shipped_plans")
    module_path = _write_facility_module(tmp_path)
    _write_config(tmp_path, {"scan": {"plan_module": str(module_path)}})
    monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "config.yml"))

    facility = plan_loader.get_facility_plans()

    assert facility.devices == {"sniffer": "constructed-sniffer-device"}
    assert set(facility.plans) == {"sniffle_path"}


def test_no_plan_module_configured_means_no_facility_plans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """A config.yml with no `scan.plan_module` key injects nothing — sanity
    check that the previous tests' plan really came from config, not a
    leftover default."""
    monkeypatch.delenv("BLUESKY_PLAN_MODULE", raising=False)
    _write_config(tmp_path, {"scan": {"bridge_url": "http://127.0.0.1:8090"}})
    monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "config.yml"))

    resp = client.get("/plans")

    assert resp.status_code == 200
    assert "sniffle_path" not in {p["name"] for p in resp.json()}


def test_env_var_still_wins_over_config_plan_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """`BLUESKY_PLAN_MODULE` outranks `scan.plan_module`, mirroring
    bridge_url/promote_token's env-wins-over-config precedent."""
    config_module_path = _write_facility_module(tmp_path)
    _write_config(tmp_path, {"scan": {"plan_module": str(config_module_path)}})
    monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "config.yml"))

    env_dir = tmp_path / "env_facility"
    env_dir.mkdir()
    env_module_path = env_dir / "env_plans.py"
    env_module_path.write_text(
        "from osprey.services.bluesky_bridge.plan_types import PlanSpec\n"
        "from pydantic import BaseModel\n\n"
        "class P(BaseModel):\n"
        "    pass\n\n"
        "PLANS = {'env_only_plan': PlanSpec(name='env_only_plan', plan=lambda d, p: None, "
        "schema=P, description='from env')}\n\n"
        "def get_devices():\n"
        "    return {}\n"
    )
    monkeypatch.setenv("BLUESKY_PLAN_MODULE", str(env_module_path))

    resp = client.get("/plans")

    names = {p["name"] for p in resp.json()}
    assert "env_only_plan" in names
    assert "sniffle_path" not in names
