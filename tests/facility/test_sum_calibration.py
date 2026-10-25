"""The example facility declares each monitor's charge-to-sum factor in the machine.

Every turn-by-turn sum record of the example's ``models.yaml`` carries its
monitor's factor ``k_b`` as its calibration, ``gain = 1/k_b`` (hardware to
physics), drawn once by ``scripts/demo_seeds/sum_calibration.py`` from a seeded,
order-free draw and committed. These tests hold the committed file to the
script, and the gains to the draw they come from.
"""

from __future__ import annotations

import importlib.util
import math
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import yaml

from osprey_connectors.simulation.series import channel_key_bytes, keyed_normals

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/demo_seeds/sum_calibration.py"
MODELS = REPO_ROOT / "src/osprey/templates/facilities/example/models.yaml"

#: The owner-ruled width of ``log k_b``.
SIGMA = 0.2
#: Monitors in the example's storage lattice.
MONITORS = 72


@cache
def script() -> ModuleType:
    """``scripts/demo_seeds/sum_calibration.py`` as a module."""
    spec = importlib.util.spec_from_file_location("demo_seeds_sum_calibration_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@cache
def sum_records() -> dict[str, dict[str, Any]]:
    """Every wiring record of ``models.yaml`` whose engine reads the sum, by address."""
    models = yaml.safe_load(MODELS.read_text(encoding="utf-8"))
    return {
        record["address"]: record
        for model in models
        for record in model.get("wiring") or ()
        if (record.get("engine") or {}).get("axis") == "sum"
    }


def _linear(record: dict[str, Any]) -> dict[str, float]:
    return record["calibration"]["curve"]["linear"]


def test_committed_models_match_the_script() -> None:
    assert script().main(["--check"]) == 0


def test_stamping_is_idempotent() -> None:
    text = MODELS.read_text(encoding="utf-8")
    assert script().stamp(text) == text


def test_every_sum_record_is_a_monitor_sum_and_all_are_stamped() -> None:
    records = sum_records()
    assert len(records) == MONITORS
    assert all(address.endswith(":TBT:SUM") for address in records)


@pytest.mark.parametrize("address", sorted(sum_records()))
def test_gain_is_the_inverse_of_the_drawn_factor(address: str) -> None:
    linear = _linear(sum_records()[address])
    z = float(keyed_normals(channel_key_bytes(address) + b":sum_calibration", np.asarray([0]))[0])
    expected = 1.0 / math.exp(SIGMA * z)
    assert linear["offset"] == 0.0
    assert linear["gain"] == pytest.approx(expected, rel=1e-5)
    assert float(f"{linear['gain']:.6g}") == linear["gain"]


def test_factors_spread_like_the_ruled_width() -> None:
    logs = np.array([-math.log(_linear(r)["gain"]) for r in sum_records().values()])
    assert len(set(logs.round(6))) == MONITORS
    assert 0.12 < logs.std() < 0.28
    assert abs(logs.mean()) < 0.1


def test_a_sum_record_without_a_gain_line_is_refused() -> None:
    module = script()
    text = "- name: X\n  wiring:\n  - address: A:TBT:SUM\n    element: A\n  - address: A:TBT:X\n"
    with pytest.raises(module.SumCalibrationError, match="A:TBT:SUM"):
        module.stamp(text)


def test_only_gain_lines_of_sum_records_change() -> None:
    module = script()
    text = (
        "  - address: A:TBT:X\n"
        "    calibration:\n"
        "      curve:\n"
        "        linear:\n"
        "          gain: 1.0\n"
        "  - address: A:TBT:SUM\n"
        "    calibration:\n"
        "      curve:\n"
        "        linear:\n"
        "          gain: 1.0\n"
        "          offset: 0.0\n"
    )
    out = module.stamp(text).splitlines()
    assert out[4] == "          gain: 1.0"
    assert out[9] == f"          gain: {module.gain('A:TBT:SUM')!r}"
    assert out[:9] == text.splitlines()[:9]
    assert out[10:] == text.splitlines()[10:]
