"""The ``topoff`` and ``topoff_lifetime`` seed kinds: their schema, their rules, and reseed.

A ``topoff`` seed states ``ceiling``, ``lifetime_h`` and ``period_s`` on a float
channel and carries no ``nominal``; a ``topoff_lifetime`` seed names, in ``of``,
the channel whose ``topoff`` seed's lifetime it reads. The demo seed generator
carries both through as data, stamping rule motion on a ``topoff`` readback and
none on a ``topoff_lifetime`` one.
"""

from __future__ import annotations

import importlib.util
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any

import pydantic
import pytest
import yaml

from osprey.facility.schema import core
from osprey.facility.validate import StageReport, run_stages

REPO_ROOT = Path(__file__).resolve().parents[2]
RESEED = REPO_ROOT / "scripts/demo_seeds/reseed.py"
EXAMPLE_TREE = REPO_ROOT / "src/osprey/templates/facilities/example"

DCCT = {"id": "SR/DCCT1", "class": "BeamCurrentMonitor"}
CURRENT = {"id": "DCCT:CURRENT", "on": {"device": "SR/DCCT1"}, "unit": "mA"}
LIFETIME = {"id": "DCCT:LIFETIME", "on": {"device": "SR/DCCT1"}, "unit": "h"}
OTHER = {"id": "DCCT:OTHER", "on": {"device": "SR/DCCT1"}, "unit": "mA"}
STATE = {"id": "DCCT:STATE", "value_type": "bool", "options": ["OFF", "ON"]}

TOPOFF = {"topoff": {"ceiling": 500.0, "lifetime_h": 10.0, "period_s": 30.0}}
READS_LIFETIME = {"topoff_lifetime": {"of": "DCCT:CURRENT"}}


def _write(root: Path, files: dict[str, Any]) -> Path:
    for rel, data in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return root


def _files(seeds: dict[str, Any]) -> dict[str, Any]:
    return {
        "records/devices.yaml": [DCCT],
        "records/channels.yaml": [CURRENT, LIFETIME, OTHER, STATE],
        "seeds.yaml": seeds,
    }


def _run(tmp_path: Path, seeds: dict[str, Any]) -> StageReport:
    return run_stages(_write(tmp_path / "facility", _files(seeds)), project_name="topoff")


def _lines(report: StageReport) -> list[str]:
    return [error.format_message() for error in report.errors]


def _one(tmp_path: Path, seeds: dict[str, Any], stage: str) -> Any:
    report = _run(tmp_path, seeds)
    assert report.failed == stage, _lines(report)
    assert len(report.errors) == 1, _lines(report)
    return report.errors[0]


# --- the generated model ---------------------------------------------------------------


def test_the_model_carries_both_kinds() -> None:
    seed = core.Seed.model_validate({**TOPOFF})
    assert seed.topoff is not None
    assert (seed.topoff.ceiling, seed.topoff.lifetime_h, seed.topoff.period_s) == (
        500.0,
        10.0,
        30.0,
    )
    assert core.Seed.model_validate(READS_LIFETIME).topoff_lifetime.of == "DCCT:CURRENT"


@pytest.mark.parametrize("missing", ["ceiling", "lifetime_h", "period_s"])
def test_a_topoff_states_all_three_fields(missing: str) -> None:
    fields = {k: v for k, v in TOPOFF["topoff"].items() if k != missing}
    with pytest.raises(pydantic.ValidationError, match=missing):
        core.Topoff.model_validate(fields)


def test_a_topoff_lifetime_states_of() -> None:
    with pytest.raises(pydantic.ValidationError, match="of"):
        core.TopoffLifetime.model_validate({})


def test_a_topoff_takes_no_other_field() -> None:
    with pytest.raises(pydantic.ValidationError):
        core.Topoff.model_validate({**TOPOFF["topoff"], "depth": 1.0})


def test_a_missing_field_stops_the_schema_stage(tmp_path: Path) -> None:
    seeds = {"DCCT:CURRENT": {"topoff": {"ceiling": 500.0, "lifetime_h": 10.0}}}
    error = _one(tmp_path, seeds, "schema")
    assert error.record_id == "channels.0.simulation.topoff.period_s"


# --- the rules -------------------------------------------------------------------------


def test_a_topoff_and_its_lifetime_build(tmp_path: Path) -> None:
    report = _run(tmp_path, {"DCCT:CURRENT": TOPOFF, "DCCT:LIFETIME": READS_LIFETIME})
    assert report.ok, _lines(report)


@pytest.mark.parametrize(
    ("slot", "value"),
    [("lifetime_h", 0.0), ("lifetime_h", -1.0), ("period_s", 0.0), ("period_s", -30.0)],
)
def test_lifetime_and_period_are_above_zero(tmp_path: Path, slot: str, value: float) -> None:
    seeds = {"DCCT:CURRENT": {"topoff": {**TOPOFF["topoff"], slot: value}}}
    error = _one(tmp_path, seeds, "records")
    assert error.kind == "value-invalid"
    assert f"`topoff.{slot}` is {value}, not above 0" in error.detail
    assert error.sources == ("seeds.yaml",)


def test_a_topoff_carries_no_nominal(tmp_path: Path) -> None:
    seeds = {"DCCT:CURRENT": {"nominal": 400.0, **TOPOFF}}
    error = _one(tmp_path, seeds, "records")
    assert error.kind == "seed-invalid"
    assert error.detail == "carries `nominal` beside `topoff`"


def test_a_topoff_on_a_non_float_channel(tmp_path: Path) -> None:
    error = _one(tmp_path, {"DCCT:STATE": TOPOFF}, "records")
    assert error.kind == "value-invalid"
    assert "a bool channel carries `topoff`" in error.detail


@pytest.mark.parametrize(
    ("of", "seeds", "why"),
    [
        ("DCCT:OTHER", {}, "DCCT:OTHER has no `topoff` seed"),
        ("DCCT:OTHER", {"DCCT:OTHER": {"nominal": 1.0}}, "DCCT:OTHER has no `topoff` seed"),
        ("DCCT:NOWHERE", {}, "DCCT:NOWHERE is no channel"),
    ],
    ids=["unseeded", "no-topoff", "unknown"],
)
def test_of_names_a_topoff_seed(tmp_path: Path, of: str, seeds: dict[str, Any], why: str) -> None:
    seeds = {**seeds, "DCCT:LIFETIME": {"topoff_lifetime": {"of": of}}}
    report = _run(tmp_path, seeds)
    assert report.failed == "records", _lines(report)
    details = [error.detail for error in report.errors if error.record_id == "DCCT:LIFETIME"]
    assert details == [f"`topoff_lifetime.of` {why}"], _lines(report)


def test_the_example_facility_still_validates() -> None:
    report = run_stages(EXAMPLE_TREE, project_name="example")
    assert report.ok, _lines(report)


# --- reseed ----------------------------------------------------------------------------


@cache
def reseed_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("demo_seeds_reseed_topoff", RESEED)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RULES: list[dict[str, Any]] = [
    {
        "class": "BeamCurrentMonitor",
        "signal": "beam_intensity_readback",
        "noise": {"value": 0.01, "unit": "mA"},
        "source": "test",
    },
    {
        "class": "BeamCurrentMonitor",
        "signal": "beam_lifetime_readback",
        "noise": {"value": 0.01, "unit": "h"},
        "source": "test",
    },
]


@pytest.fixture
def reseed_tree(tmp_path: Path) -> Path:
    current = {**CURRENT, "signal": "beam_intensity_readback"}
    lifetime = {**LIFETIME, "signal": "beam_lifetime_readback"}
    seeds = {"DCCT:CURRENT": TOPOFF, "DCCT:LIFETIME": READS_LIFETIME}
    return _write(
        tmp_path,
        {
            "records/devices.yaml": [DCCT],
            "records/channels.yaml": [current, lifetime],
            "seeds.yaml": seeds,
        },
    )


def test_key_order_appends_both_kinds_after_linear() -> None:
    order = reseed_module().KEY_ORDER
    assert order[order.index("linear") + 1 :] == ("topoff", "topoff_lifetime")


def test_moves_excludes_topoff_lifetime_only() -> None:
    moves = reseed_module().moves
    assert moves(CURRENT, TOPOFF)
    assert not moves(LIFETIME, READS_LIFETIME)


def test_reseed_keeps_rule_motion_on_a_topoff_and_none_on_its_lifetime(
    reseed_tree: Path,
) -> None:
    module = reseed_module()
    seeds = module.reseed(reseed_tree, rules=RULES)
    assert seeds["DCCT:CURRENT"] == {**TOPOFF, "noise": {"absolute": 0.01}}
    assert list(seeds["DCCT:CURRENT"]) == ["noise", "topoff"]
    assert seeds["DCCT:LIFETIME"] == READS_LIFETIME


def test_a_moving_topoff_lifetime_is_refused(reseed_tree: Path) -> None:
    module = reseed_module()
    seeds = {"DCCT:CURRENT": TOPOFF, "DCCT:LIFETIME": {**READS_LIFETIME, "noise": {"absolute": 1}}}
    (reseed_tree / "seeds.yaml").write_text(yaml.safe_dump(seeds), encoding="utf-8")
    with pytest.raises(module.ReseedError, match="DCCT:LIFETIME"):
        module.reseed(reseed_tree, rules=RULES)
