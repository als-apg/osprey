"""Tests for template-database expansion and prompt-presentation formatting.

The write-path is covered elsewhere; this file pins the untested read path:
``_expand_template`` axis/description handling, the ``explicit`` vs ``template``
presentation modes (``format_chunk_for_prompt`` and its two backends), the
pattern detector, and ``get_statistics``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from osprey.services.channel_finder.databases.template import ChannelDatabase


def _write_db(
    path: Path, channels: list[dict], presentation_mode: str = "explicit"
) -> ChannelDatabase:
    path.write_text(json.dumps({"channels": channels}, indent=2))
    return ChannelDatabase(str(path), presentation_mode=presentation_mode)


# ---------------------------------------------------------------------------
# Template expansion
# ---------------------------------------------------------------------------


class TestExpandTemplate:
    def test_no_sub_channels_yields_bare_instances(self, tmp_path: Path):
        db = _write_db(
            tmp_path / "db.json",
            [{"template": True, "base_name": "Solo", "instances": [1, 2]}],
        )
        assert set(db.channel_map) == {"Solo01", "Solo02"}
        assert db.channel_map["Solo01"]["address"] == "Solo01"

    def test_axis_expansion_and_description_substitution(self, tmp_path: Path):
        db = _write_db(
            tmp_path / "db.json",
            [
                {
                    "template": True,
                    "base_name": "Steer",
                    "instances": [1, 1],
                    "axes": ["X", "Y"],
                    "sub_channels": ["SP"],
                    "description": "Steerer {instance:02d} {axis}",
                }
            ],
        )
        assert set(db.channel_map) == {"Steer01XSP", "Steer01YSP"}
        assert db.channel_map["Steer01XSP"]["description"] == "Steerer 01 X"
        assert db.channel_map["Steer01YSP"]["description"] == "Steerer 01 Y"

    def test_per_sub_channel_descriptions_win_over_generic(self, tmp_path: Path):
        db = _write_db(
            tmp_path / "db.json",
            [
                {
                    "template": True,
                    "base_name": "Corr",
                    "instances": [1, 1],
                    "sub_channels": ["SP", "RB"],
                    "description": "generic {instance:02d}",
                    "channel_descriptions": {
                        "SP": "Corrector {instance:02d} setpoint",
                        "RB": "Corrector {instance:02d} readback",
                    },
                }
            ],
        )
        assert db.channel_map["Corr01SP"]["description"] == "Corrector 01 setpoint"
        assert db.channel_map["Corr01RB"]["description"] == "Corrector 01 readback"


# ---------------------------------------------------------------------------
# _detect_pattern
# ---------------------------------------------------------------------------


class TestDetectPattern:
    @pytest.fixture()
    def db(self, tmp_path: Path) -> ChannelDatabase:
        return _write_db(tmp_path / "db.json", [{"channel": "X", "address": "X"}])

    def test_empty_group(self, db: ChannelDatabase):
        assert db._detect_pattern([]) == {"pattern": ""}

    def test_single_suffix_range(self, db: ChannelDatabase):
        chans = [{"channel": f"Dipole{i:02d}SP"} for i in range(1, 11)]
        assert db._detect_pattern(chans)["pattern"] == "Dipole{01-10}SP"

    def test_multiple_suffixes_alternation(self, db: ChannelDatabase):
        chans = [{"channel": "Q01SP"}, {"channel": "Q01RB"}, {"channel": "Q02SP"}]
        assert db._detect_pattern(chans)["pattern"] == "Q{01-02}{RB|SP}"

    def test_no_suffix_range(self, db: ChannelDatabase):
        chans = [{"channel": "BPM01"}, {"channel": "BPM02"}]
        assert db._detect_pattern(chans)["pattern"] == "BPM{01-02}"

    def test_fallback_when_names_do_not_match(self, db: ChannelDatabase):
        # Leading-digit names don't match the [A-Za-z]+ base pattern.
        chans = [{"channel": "123abc"}, {"channel": "456def"}]
        assert db._detect_pattern(chans)["pattern"] == "123abc ... 456def"


# ---------------------------------------------------------------------------
# Presentation modes
# ---------------------------------------------------------------------------


class TestFormatExplicit:
    def test_uniform_descriptions_render_compact_header(self, tmp_path: Path):
        db = _write_db(
            tmp_path / "db.json",
            [
                {
                    "template": True,
                    "base_name": "Dipole",
                    "instances": [1, 3],
                    "sub_channels": ["SP"],
                    "description": "Dipole {instance:02d} setpoint",
                }
            ],
        )
        out = db.format_chunk_for_prompt(db.channels)
        # Compact: one normalized header, then bare channel names (no per-line desc).
        assert "Dipole devices: Dipole {N} setpoint" in out
        assert "- Dipole01SP" in out
        assert "Dipole01SP:" not in out  # no repeated per-channel description

    def test_differing_descriptions_render_individually(self, tmp_path: Path):
        db = _write_db(
            tmp_path / "db.json",
            [
                {
                    "template": True,
                    "base_name": "Corr",
                    "instances": [1, 2],
                    "sub_channels": ["SP", "RB"],
                    "channel_descriptions": {
                        "SP": "Corrector {instance:02d} setpoint",
                        "RB": "Corrector {instance:02d} readback",
                    },
                }
            ],
        )
        out = db.format_chunk_for_prompt(db.channels)
        assert "Corr devices:" in out
        assert "- Corr01SP: Corrector 01 setpoint" in out
        assert "- Corr01RB: Corrector 01 readback" in out

    def test_compact_group_includes_addresses(self, tmp_path: Path):
        db = _write_db(
            tmp_path / "db.json",
            [
                {
                    "template": True,
                    "base_name": "Dipole",
                    "instances": [1, 2],
                    "sub_channels": ["SP"],
                    "description": "Dipole {instance:02d} setpoint",
                }
            ],
        )
        out = db.format_chunk_for_prompt(db.channels, include_addresses=True)
        assert "- Dipole01SP (Address: Dipole01SP)" in out

    def test_standalone_channel_with_address(self, tmp_path: Path):
        db = _write_db(
            tmp_path / "db.json",
            [{"channel": "SOLO:CH", "address": "SOLO:ADDR", "description": "solo"}],
        )
        out = db.format_chunk_for_prompt(db.channels, include_addresses=True)
        assert out == "- SOLO:CH (Address: SOLO:ADDR): solo"


class TestFormatTemplateMode:
    def test_pattern_and_examples_rendered(self, tmp_path: Path):
        db = _write_db(
            tmp_path / "db.json",
            [
                {
                    "template": True,
                    "base_name": "Dipole",
                    "instances": [1, 3],
                    "sub_channels": ["SP"],
                    "description": "Dipole {instance:02d} setpoint",
                }
            ],
            presentation_mode="template",
        )
        out = db.format_chunk_for_prompt(db.channels)
        assert "Pattern: Dipole{01-03}SP" in out
        assert "Examples: Dipole01SP, Dipole02SP, ... (3 total)" in out

    def test_standalone_rendered_alongside_pattern(self, tmp_path: Path):
        db = _write_db(
            tmp_path / "db.json",
            [
                {
                    "template": True,
                    "base_name": "Dipole",
                    "instances": [1, 3],
                    "sub_channels": ["SP"],
                },
                {"channel": "SOLO:CH", "address": "SOLO:ADDR", "description": "solo"},
            ],
            presentation_mode="template",
        )
        out = db.format_chunk_for_prompt(db.channels, include_addresses=True)
        assert "- SOLO:CH (Address: SOLO:ADDR): solo" in out
        assert "Pattern: Dipole{01-03}SP" in out

    def test_mode_selects_backend(self, tmp_path: Path):
        channels = [
            {
                "template": True,
                "base_name": "Dipole",
                "instances": [1, 3],
                "sub_channels": ["SP"],
                "description": "Dipole {instance:02d} setpoint",
            }
        ]
        explicit = _write_db(tmp_path / "e.json", channels, presentation_mode="explicit")
        template = _write_db(tmp_path / "t.json", channels, presentation_mode="template")
        assert "Pattern:" not in explicit.format_chunk_for_prompt(explicit.channels)
        assert "Pattern:" in template.format_chunk_for_prompt(template.channels)


class TestStatistics:
    def test_statistics_report_template_and_standalone_counts(self, tmp_path: Path):
        db = _write_db(
            tmp_path / "db.json",
            [
                {
                    "template": True,
                    "base_name": "Dipole",
                    "instances": [1, 3],
                    "sub_channels": ["SP"],
                },
                {"channel": "SOLO:CH", "address": "SOLO:CH"},
            ],
            presentation_mode="template",
        )
        stats = db.get_statistics()
        assert stats["format"] == "template"
        assert stats["presentation_mode"] == "template"
        assert stats["template_entries"] == 1
        assert stats["standalone_entries"] == 1
        assert stats["total_channels"] == 4  # 3 expanded + 1 standalone
