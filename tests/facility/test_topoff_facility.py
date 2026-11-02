"""The top-off seed in a built facility: the archive view and the limits band.

The sawtooth itself — shape, held level, motion, envelope — is the connectors
package's and is tested in the package's own suite
(``osprey_connectors_tests/test_topoff_shape.py``), whose seeds and model these
tests share.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from osprey_connectors.simulation.archive import build
from osprey_connectors.simulation.view import SEEDS_FILE
from osprey_connectors_tests.test_topoff_shape import (
    CEILING,
    CURRENT,
    LEVEL,
    LIFETIME,
    LIFETIME_H,
    RECORD,
    SEEDS,
    T0,
    _model,
    _read,
)
from tests._simulator_view import write_texture_view
from tests.facility.test_compute import _ok, _run, _tree

# --- archive --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def topoff_view(tmp_path_factory):
    view = write_texture_view(
        tmp_path_factory.mktemp("topoff"), {CURRENT: {}, LIFETIME: {}, "SR:OTHER": {"nominal": 1.0}}
    )
    seeds = json.loads((view / SEEDS_FILE).read_text())
    seeds["seeds"].update(SEEDS)
    (view / SEEDS_FILE).write_text(json.dumps(seeds))
    return view


def test_live_equals_archive_at_the_same_instant(topoff_view):
    archive = build(topoff_view, [], anchor_s=T0 + 3600.0)
    instants = [T0 + offset for offset in (0.0, 1.5, 14.0, 29.999, 30.0, 47.25, 1800.0)]
    stored = archive.series(CURRENT, instants)
    lifetimes = archive.series(LIFETIME, instants)
    model = _model()
    for instant, sample, lifetime in zip(instants, stored, lifetimes, strict=True):
        assert sample == pytest.approx(_read(model, CURRENT, instant), rel=1e-12)
        assert lifetime == LIFETIME_H
    assert stored[instants.index(T0 + 30.0)] == pytest.approx(CEILING, rel=1e-12)


# --- limits band ----------------------------------------------------------------------


def _limits_tree(low: float, high: float, address: str = "T:DCCT") -> dict[str, Any]:
    files = _tree()
    files["records/channels.yaml"].extend(
        [
            {"id": "T:DCCT", "simulation": {"topoff": RECORD}},
            {"id": "T:LIFE", "simulation": {"topoff_lifetime": {"of": "T:DCCT"}}},
        ]
    )
    files["limits.yaml"] = {"records": [{"address": address, "min_value": low, "max_value": high}]}
    return files


def test_a_limits_band_around_the_held_level_builds(tmp_path):
    _ok(tmp_path, _limits_tree(LEVEL - 0.5, LEVEL + 0.5))


def test_a_limits_band_excluding_the_held_level_names_the_derived_value(tmp_path):
    result = _run(tmp_path, _limits_tree(0.0, 499.0))
    assert [(e.kind, e.record_id) for e in result.errors] == [("seed-invalid", "T:DCCT")]
    detail = result.errors[0].detail
    assert "held level of `topoff`" in detail
    assert repr(LEVEL).removesuffix(".0") in detail
    assert "above `max_value` 499" in detail


def test_a_lifetime_band_judges_the_referenced_lifetime(tmp_path):
    _ok(tmp_path / "in", _limits_tree(1.0, 10.0, address="T:LIFE"))
    result = _run(tmp_path / "out", _limits_tree(10.0, 20.0, address="T:LIFE"))
    assert [(e.kind, e.record_id) for e in result.errors] == [("seed-invalid", "T:LIFE")]
    assert "held lifetime of `topoff_lifetime` 8 lies below `min_value` 10" in (
        result.errors[0].detail
    )
