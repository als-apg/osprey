"""``settle_bound`` reads a setpoint's readback and tolerance from the render's facility file.

Each case stands in a throwaway render: ``build/config.yml`` under the working
directory, with the facility file beside it at ``build/facility.json``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey_connectors.workspace import reset_config_cache
from pyaml_cs_osprey._settle import settle_bound


@pytest.fixture
def render(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A render directory holding ``config.yml``, the working directory's ``build/``."""
    root = tmp_path / "repo"
    (root / "build").mkdir(parents=True)
    (root / "build" / "config.yml").write_text(
        yaml.safe_dump({"control_system": {"type": "epics"}}), encoding="utf-8"
    )
    monkeypatch.chdir(root)
    monkeypatch.delenv("OSPREY_CONFIG", raising=False)
    monkeypatch.delenv("CONFIG_FILE", raising=False)
    reset_config_cache()
    yield root / "build"
    reset_config_cache()


def _plant(render: Path, *channels: dict[str, Any]) -> Path:
    path = render / "facility.json"
    path.write_text(json.dumps({"channels": list(channels)}), encoding="utf-8")
    return path


def test_an_absolute_tolerance_is_the_bound_on_the_pair(render: Path) -> None:
    _plant(render, {"id": "Q1:SP", "pair": "Q1:RB", "tolerance": {"absolute": -0.05}})
    assert settle_bound("Q1:SP") == ("Q1:RB", 0.05)


def test_a_relative_tolerance_scales_the_seed_nominal(render: Path) -> None:
    _plant(
        render,
        {
            "id": "RF:SP",
            "pair": "RF:RB",
            "tolerance": {"relative": 1.0e-3},
            "simulation": {"nominal": -500.0},
        },
    )
    pair, bound = settle_bound("RF:SP") or ("", 0.0)
    assert pair == "RF:RB"
    assert bound == pytest.approx(0.5)


def test_a_setpoint_stating_no_tolerance_has_no_bound(render: Path) -> None:
    _plant(render, {"id": "Q1:SP", "pair": "Q1:RB"})
    assert settle_bound("Q1:SP") is None


def test_a_setpoint_with_no_pair_is_its_own_readback(render: Path) -> None:
    _plant(render, {"id": "Q1:SP", "tolerance": {"absolute": 0.1}})
    assert settle_bound("Q1:SP") == ("Q1:SP", 0.1)


def test_an_address_that_is_not_a_channel_has_no_bound(render: Path) -> None:
    _plant(render, {"id": "Q1:SP", "pair": "Q1:RB", "tolerance": {"absolute": 0.1}})
    assert settle_bound("Q2:SP") is None


def test_a_render_with_no_facility_file_has_no_bound(render: Path) -> None:
    assert not (render / "facility.json").exists()
    assert settle_bound("Q1:SP") is None


def test_no_configuration_at_all_has_no_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OSPREY_CONFIG", raising=False)
    monkeypatch.delenv("CONFIG_FILE", raising=False)
    reset_config_cache()
    try:
        assert settle_bound("Q1:SP") is None
    finally:
        reset_config_cache()


def test_a_corrupt_facility_file_raises(render: Path) -> None:
    (render / "facility.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        settle_bound("Q1:SP")


def test_a_rewritten_facility_file_is_read_again(render: Path) -> None:
    path = _plant(render, {"id": "Q1:SP", "pair": "Q1:RB", "tolerance": {"absolute": 0.1}})
    assert settle_bound("Q1:SP") == ("Q1:RB", 0.1)

    _plant(render, {"id": "Q1:SP", "pair": "Q1:RB", "tolerance": {"absolute": 0.2}})
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))

    assert settle_bound("Q1:SP") == ("Q1:RB", 0.2)
