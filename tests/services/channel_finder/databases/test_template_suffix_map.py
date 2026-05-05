"""Regression tests for `suffix_map` handling in template expansion.

When a template declares ``sub_channels`` whose names differ from their EPICS
address suffixes (e.g. ``"CurrentSetPoint"`` displayed to humans, ``"SP"`` on
the wire), the template must declare a ``suffix_map`` so the renderer can
translate one to the other.

These tests pin the contract:
  * ``channel`` (the human-readable alias) keeps the raw ``sub_channel`` name.
  * ``address`` is built from the *mapped* suffix.
  * Templates without ``suffix_map`` continue to use ``sub_channel`` names
    as the address suffix (backward-compat path).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from osprey.services.channel_finder.databases.template import ChannelDatabase


def _write_db(path: Path, channels: list[dict]) -> ChannelDatabase:
    path.write_text(json.dumps({"channels": channels}, indent=2))
    return ChannelDatabase(str(path))


class TestSuffixMapExpansion:
    def test_address_uses_mapped_suffix(self, tmp_path: Path):
        db = _write_db(
            tmp_path / "db.json",
            [
                {
                    "template": True,
                    "base_name": "DipoleMagnet",
                    "instances": [1, 3],
                    "sub_channels": ["CurrentSetPoint", "CurrentReadBack"],
                    "address_pattern": "SR:MAG:DIPOLE:B{instance:02d}:CURRENT:{suffix}",
                    "suffix_map": {"CurrentSetPoint": "SP", "CurrentReadBack": "RB"},
                }
            ],
        )

        sp = db.channel_map["DipoleMagnet01CurrentSetPoint"]
        rb = db.channel_map["DipoleMagnet01CurrentReadBack"]

        assert sp["address"] == "SR:MAG:DIPOLE:B01:CURRENT:SP"
        assert rb["address"] == "SR:MAG:DIPOLE:B01:CURRENT:RB"

    def test_alias_keeps_raw_sub_channel_name(self, tmp_path: Path):
        db = _write_db(
            tmp_path / "db.json",
            [
                {
                    "template": True,
                    "base_name": "DipoleMagnet",
                    "instances": [5, 5],
                    "sub_channels": ["CurrentSetPoint"],
                    "address_pattern": "SR:MAG:DIPOLE:B{instance:02d}:CURRENT:{suffix}",
                    "suffix_map": {"CurrentSetPoint": "SP"},
                }
            ],
        )

        ch = db.channel_map["DipoleMagnet05CurrentSetPoint"]
        assert ch["channel"] == "DipoleMagnet05CurrentSetPoint"
        assert ch["address"] == "SR:MAG:DIPOLE:B05:CURRENT:SP"

    def test_partial_map_falls_back_to_raw_for_unmapped(self, tmp_path: Path):
        db = _write_db(
            tmp_path / "db.json",
            [
                {
                    "template": True,
                    "base_name": "Thing",
                    "instances": [1, 1],
                    "sub_channels": ["Aliased", "Plain"],
                    "address_pattern": "DEV:{instance:02d}:{suffix}",
                    "suffix_map": {"Aliased": "AL"},
                }
            ],
        )

        assert db.channel_map["Thing01Aliased"]["address"] == "DEV:01:AL"
        assert db.channel_map["Thing01Plain"]["address"] == "DEV:01:Plain"

    def test_no_suffix_map_preserves_legacy_behavior(self, tmp_path: Path):
        db = _write_db(
            tmp_path / "db.json",
            [
                {
                    "template": True,
                    "base_name": "Quad",
                    "instances": [1, 1],
                    "sub_channels": ["SP", "RB"],
                    "address_pattern": "SR:MAG:QF:{instance:02d}:{suffix}",
                }
            ],
        )

        assert db.channel_map["Quad01SP"]["address"] == "SR:MAG:QF:01:SP"
        assert db.channel_map["Quad01RB"]["address"] == "SR:MAG:QF:01:RB"

    def test_multi_axis_with_suffix_map(self, tmp_path: Path):
        db = _write_db(
            tmp_path / "db.json",
            [
                {
                    "template": True,
                    "base_name": "Steerer",
                    "instances": [1, 1],
                    "axes": ["H", "V"],
                    "sub_channels": ["CurrentSetPoint"],
                    "address_pattern": "SR:STR:{axis}{instance:02d}:CURRENT:{suffix}",
                    "suffix_map": {"CurrentSetPoint": "SP"},
                }
            ],
        )

        h = db.channel_map["Steerer01HCurrentSetPoint"]
        v = db.channel_map["Steerer01VCurrentSetPoint"]
        assert h["address"] == "SR:STR:H01:CURRENT:SP"
        assert v["address"] == "SR:STR:V01:CURRENT:SP"


class TestShippedTemplateDatabaseRenders:
    """Pin the addresses produced by the in_context.json template DB shipped
    with the control_assistant app. This DB drives `osprey init
    --channel-finder-mode=in_context` and the InContext e2e benchmark."""

    @pytest.fixture()
    def shipped_db(self) -> ChannelDatabase:
        path = (
            Path(__file__).parents[4]
            / "src/osprey/templates/apps/control_assistant/data/channel_databases/in_context.json"
        )
        return ChannelDatabase(str(path))

    @pytest.mark.parametrize(
        "channel, expected_address",
        [
            ("DipoleMagnet05CurrentSetPoint", "SR:MAG:DIPOLE:B05:CURRENT:SP"),
            ("DipoleMagnet05CurrentReadBack", "SR:MAG:DIPOLE:B05:CURRENT:RB"),
            ("FocusingQuad03CurrentSetPoint", "SR:MAG:QF:QF03:CURRENT:SP"),
            ("HorizontalCorrectorMagnet07CurrentSetPoint", "SR:MAG:HCM:H07:CURRENT:SP"),
            ("VerticalCorrectorMagnet02CurrentReadBack", "SR:MAG:VCM:V02:CURRENT:RB"),
        ],
    )
    def test_canonical_addresses(
        self, shipped_db: ChannelDatabase, channel: str, expected_address: str
    ):
        assert shipped_db.channel_map[channel]["address"] == expected_address
