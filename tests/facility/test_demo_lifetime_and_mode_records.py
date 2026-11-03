"""The demo facility's beam-lifetime and operation-mode records, and its DCCT units.

The storage ring states how long its beam lasts and what the machine is doing:
``SR:DIAG:DCCT:01:LIFETIME:RB`` is the DCCT's lifetime readback in hours and
``SR:OPS:MODE:RB`` is a read-only, ring-level enum of four modes. Both are in
the in_context view. Both DCCT current readbacks read in mA.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.facility.validate import run_stages, signal_roles

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "src/osprey/templates/facilities/example"
CHANNELS = EXAMPLE / "records" / "channels.yaml"

LIFETIME = "SR:DIAG:DCCT:01:LIFETIME:RB"
MODE = "SR:OPS:MODE:RB"
MODES = ["User operation", "Injection", "Machine studies", "Off"]
DCCT_CURRENTS = ("SR:DIAG:DCCT:01:CURRENT:RB", "BR:DIAG:DCCT:01:CURRENT:RB")


@pytest.fixture(scope="module")
def rows() -> list[dict[str, Any]]:
    loaded: list[dict[str, Any]] = yaml.safe_load(CHANNELS.read_text(encoding="utf-8"))
    return loaded


@pytest.fixture(scope="module")
def stated(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in rows}


@pytest.fixture(scope="module")
def built() -> dict[str, dict[str, Any]]:
    report = run_stages(EXAMPLE, project_name="demo")
    assert report.ok, [error.format_message() for error in report.errors]
    return {channel["id"]: channel for channel in report.validated.document["channels"]}


def test_the_lifetime_is_a_float_readback_in_hours_on_the_dcct(
    stated: dict[str, dict[str, Any]],
) -> None:
    row = stated[LIFETIME]
    assert row["on"] == {"device": "SR/DCCT01"}
    assert row["signal"] == "beam_lifetime_readback"
    assert row["unit"] == "h"
    assert row.get("value_type", "float") == "float"
    assert row.get("role", "readback") == "readback"
    assert "in_context" in row["tags"]
    assert row["description"]


def test_the_mode_is_a_read_only_ring_enum_of_four_modes_in_order(
    stated: dict[str, dict[str, Any]],
) -> None:
    row = stated[MODE]
    assert row["on"] == {"place": "SR"}
    assert row["signal"] == "operation_mode_status"
    assert row["value_type"] == "enum"
    assert row["options"] == MODES
    assert row.get("role", "readback") == "readback"
    assert row.get("writable", False) is False
    assert "unit" not in row
    assert "in_context" in row["tags"]
    assert row["description"]


def test_both_signals_are_vocabulary_roles() -> None:
    assert {"beam_lifetime_readback", "operation_mode_status"} <= signal_roles()


@pytest.mark.parametrize("address", DCCT_CURRENTS)
def test_both_dcct_currents_read_in_milliamperes(
    stated: dict[str, dict[str, Any]], address: str
) -> None:
    assert stated[address]["unit"] == "mA"


def test_both_rows_sit_in_address_order_beside_their_siblings(
    rows: list[dict[str, Any]],
) -> None:
    ids = [row["id"] for row in rows]
    assert ids == sorted(ids)
    assert len(ids) == len(set(ids))
    assert ids[ids.index("SR:DIAG:DCCT:01:CURRENT:RB") + 1] == LIFETIME


def test_the_build_keeps_both_records_as_stated(built: dict[str, dict[str, Any]]) -> None:
    lifetime = built[LIFETIME]
    assert lifetime["on"] == {"device": "SR/DCCT01"}
    assert lifetime["unit"] == "h"
    assert lifetime.get("value_type", "float") == "float"
    assert "in_context" in lifetime["tags"]

    mode = built[MODE]
    assert mode["on"] == {"place": "SR"}
    assert mode["value_type"] == "enum"
    assert mode["options"] == MODES
    assert mode.get("writable", False) is False
    assert "in_context" in mode["tags"]

    for address in DCCT_CURRENTS:
        assert built[address]["unit"] == "mA"
