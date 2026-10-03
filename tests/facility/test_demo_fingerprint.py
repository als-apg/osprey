"""The built demo against its frozen fingerprint.

``osprey build`` on the control-assistant preset writes ``build/facility.json``.
Its channels, joined as the fingerprint joins them (address, role, value type,
names, description), hold every frozen row unchanged, and what they hold beyond
the frozen rows is exactly the additions file. The file validates as the
generated ``Facility`` model, and the committed sources validate clean. The
channel roster read off the built render enumerates every frozen address with
the direction its role states.
"""

from __future__ import annotations

import hashlib
import io
import json
from typing import TYPE_CHECKING, Any

import pytest

from tests.facility.test_cf_view_parity import (
    FINGERPRINT_ROW_KEYS,
    FINGERPRINT_SHA256,
    fingerprint_rows,
    load_golden,
)

if TYPE_CHECKING:
    from tests.facility.conftest import BuiltProject

# xdist_group("built_control_assistant"): every module reading the session's one
# control-assistant build shares a worker, so the build runs once per run.
pytestmark = [pytest.mark.slow, pytest.mark.xdist_group("built_control_assistant")]


def _rows(document: dict[str, Any]) -> list[dict[str, Any]]:
    """The facility file's channels as fingerprint rows, sorted by address."""
    rows = [
        {"address": channel["id"], **{key: channel[key] for key in FINGERPRINT_ROW_KEYS[1:]}}
        for channel in document["channels"]
    ]
    return sorted(rows, key=lambda row: row["address"])


def _sha256(rows: list[dict[str, Any]]) -> str:
    compact = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(compact.encode("utf-8")).hexdigest()


def test_the_frozen_rows_are_unchanged() -> None:
    rows = fingerprint_rows()
    assert len(rows) == 2908
    assert _sha256(rows) == FINGERPRINT_SHA256


def test_every_frozen_row_is_served_unchanged(built_control_assistant: BuiltProject) -> None:
    served = {row["address"]: row for row in _rows(built_control_assistant.facility)}
    missing = [row["address"] for row in fingerprint_rows() if row["address"] not in served]
    changed = [row for row in fingerprint_rows() if served.get(row["address"], row) != row]
    assert missing == []
    assert changed == []


def test_the_rows_beyond_the_frozen_ones_are_exactly_the_additions(
    built_control_assistant: BuiltProject,
) -> None:
    frozen = {row["address"] for row in fingerprint_rows()}
    beyond = [
        row for row in _rows(built_control_assistant.facility) if row["address"] not in frozen
    ]
    assert beyond == load_golden("demo_fingerprint_additions.json")["rows"]


def test_the_facility_file_validates_as_the_model(built_control_assistant: BuiltProject) -> None:
    from osprey.facility.schema import Facility

    Facility.model_validate(built_control_assistant.facility)
    assert built_control_assistant.facility["schema"] == "osprey.facility.facility/1"


def test_the_sources_validate_clean(built_control_assistant: BuiltProject) -> None:
    from osprey.facility.validate import validate

    out = io.StringIO()
    code = validate(
        built_control_assistant.facility_dir,
        project_name=built_control_assistant.repo.name,
        file=out,
    )
    assert (code, out.getvalue()) == (0, "")


def test_twelve_sectors_lie_under_sr(
    built_control_assistant: BuiltProject,
) -> None:
    places = built_control_assistant.facility["places"]
    sectors = [place["id"] for place in places if place.get("level") == "sector"]
    assert sectors == sorted(f"SR/SECT{n}" for n in range(1, 13))


def test_the_roster_enumerates_every_frozen_address_with_its_direction(
    built_control_assistant: BuiltProject,
) -> None:
    import yaml

    from osprey.channel_roster import registered_channels

    config = yaml.safe_load((built_control_assistant.build_dir / "config.yml").read_text())
    config["config_dir"] = str(built_control_assistant.build_dir)

    roster = registered_channels(config)

    assert roster.absence is None
    directions = {record.address: record.direction for record in roster.records}
    stated = {"setpoint": "write", "readback": "read", "none": None}
    missing = [row["address"] for row in fingerprint_rows() if row["address"] not in directions]
    changed = [
        row["address"]
        for row in fingerprint_rows()
        if directions.get(row["address"], stated[row["role"]]) != stated[row["role"]]
    ]
    assert missing == []
    assert changed == []
