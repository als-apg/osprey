"""Direct-call tests for the pure helpers of the channel database builder.

These exercise ``load_csv``, ``group_by_family``, ``find_common_description``
and ``create_template`` against ``tmp_path`` CSV files and in-memory row dicts.
The full ``build_database`` pipeline is covered elsewhere; nothing here touches
the filesystem outside ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.services.channel_finder.tools.build_database import (
    create_template,
    find_common_description,
    group_by_family,
    load_csv,
)

# ---------------------------------------------------------------------------
# load_csv
# ---------------------------------------------------------------------------


def _write_csv(tmp_path: Path, body: str, name: str = "channels.csv") -> Path:
    csv_path = tmp_path / name
    csv_path.write_text(body, encoding="utf-8")
    return csv_path


def test_load_csv_skips_comment_and_empty_address_rows(tmp_path: Path) -> None:
    csv_path = _write_csv(
        tmp_path,
        "address,description,family_name,instances,sub_channel\n"
        "# this is a comment row,ignored,,,\n"
        ",orphan row with no address,,,\n"
        "SR:BPM1:X,Beam position horizontal,,,\n"
        "   ,whitespace-only address,,,\n"
        "SR:SCH1:SP,Corrector setpoint,SCH,12,:SP\n",
    )

    rows = load_csv(csv_path)

    assert [r["address"] for r in rows] == ["SR:BPM1:X", "SR:SCH1:SP"]
    assert rows[0]["description"] == "Beam position horizontal"
    assert rows[1]["family_name"] == "SCH"


def test_load_csv_normalizes_blank_fields_to_none_and_strips(tmp_path: Path) -> None:
    csv_path = _write_csv(
        tmp_path,
        "address,description,family_name,instances,sub_channel\n"
        "  SR:BPM1:X  ,  Beam position horizontal  ,,,   \n",
    )

    (row,) = load_csv(csv_path)

    assert row["address"] == "SR:BPM1:X"
    assert row["description"] == "Beam position horizontal"
    assert row["family_name"] is None
    assert row["instances"] is None
    assert row["sub_channel"] is None


def test_load_csv_honors_custom_delimiter(tmp_path: Path) -> None:
    csv_path = _write_csv(
        tmp_path,
        "address\tdescription\tfamily_name\n"
        "#comment\tignored\t\n"
        "SR:BPM1:X\tBeam position horizontal\t\n",
        name="channels.tsv",
    )

    rows = load_csv(csv_path, delimiter="\t")

    assert [r["address"] for r in rows] == ["SR:BPM1:X"]
    assert rows[0]["description"] == "Beam position horizontal"


# ---------------------------------------------------------------------------
# group_by_family
# ---------------------------------------------------------------------------


def test_group_by_family_splits_families_from_standalone() -> None:
    channels = [
        {"address": "SR:SCH1:SP", "family_name": "SCH"},
        {"address": "SR:SCH1:AM", "family_name": "SCH"},
        {"address": "SR:BPM1:X", "family_name": None},
        {"address": "SR:QF1:SP", "family_name": "QF"},
    ]

    families, standalone = group_by_family(channels)

    assert set(families) == {"SCH", "QF"}
    assert [c["address"] for c in families["SCH"]] == ["SR:SCH1:SP", "SR:SCH1:AM"]
    assert [c["address"] for c in standalone] == ["SR:BPM1:X"]


# ---------------------------------------------------------------------------
# find_common_description
# ---------------------------------------------------------------------------


def test_find_common_description_empty_list_returns_empty() -> None:
    assert find_common_description([]) == ""


def test_find_common_description_single_entry_returns_it_verbatim() -> None:
    # The single-description short circuit bypasses the >=3-word guard.
    assert find_common_description(["Beam"]) == "Beam"
    assert (
        find_common_description(["Horizontal corrector magnet current setpoint"])
        == "Horizontal corrector magnet current setpoint"
    )


def test_find_common_description_no_shared_first_word_returns_empty() -> None:
    descriptions = [
        "Horizontal corrector magnet current setpoint",
        "Quadrupole focusing magnet current readback",
    ]

    assert find_common_description(descriptions) == ""


def test_find_common_description_short_shared_prefix_returns_empty() -> None:
    # Common prefix is only "Beam" -- fewer than three words, so it is rejected.
    descriptions = ["Beam current monitor", "Beam voltage monitor"]

    assert find_common_description(descriptions) == ""


def test_find_common_description_returns_shared_prefix_preserving_case() -> None:
    descriptions = [
        "Horizontal Corrector Magnet current setpoint",
        "horizontal corrector magnet current readback",
        "HORIZONTAL CORRECTOR MAGNET CURRENT status",
    ]

    # Case-insensitive matching, but capitalization comes from the first entry.
    assert find_common_description(descriptions) == "Horizontal Corrector Magnet current"


def test_find_common_description_strips_trailing_punctuation() -> None:
    descriptions = [
        "Corrector magnet current - horizontal",
        "Corrector magnet current - vertical",
    ]

    assert find_common_description(descriptions) == "Corrector magnet current"


# ---------------------------------------------------------------------------
# create_template
# ---------------------------------------------------------------------------


def _family_row(sub_channel: str, description: str, instances: str = "12") -> dict:
    return {
        "address": f"SR:SCH01{sub_channel}",
        "description": description,
        "family_name": "SCH",
        "instances": instances,
        "sub_channel": sub_channel,
    }


def test_create_template_strips_common_prefix_from_sub_channel_descriptions() -> None:
    channels = [
        _family_row(":SP", "Horizontal corrector magnet current setpoint"),
        _family_row(":AM", "Horizontal corrector magnet current readback"),
        _family_row(":ON", "Horizontal corrector magnet current"),
        _family_row(":ST", "HORIZONTAL CORRECTOR MAGNET CURRENT status"),
    ]

    template = create_template("SCH", channels)

    assert template["template"] is True
    assert template["base_name"] == "SCH"
    assert template["instances"] == [1, 12]
    assert template["sub_channels"] == [":SP", ":AM", ":ON", ":ST"]
    assert template["description"] == "Horizontal corrector magnet current"
    assert template["address_pattern"] == "SCH{instance:02d}{suffix}"
    assert template["channel_descriptions"] == {
        # Prefix stripped and the remainder lower-cased at its first character.
        ":SP": "setpoint",
        ":AM": "readback",
        # Description equal to the prefix leaves nothing unique behind.
        ":ON": "",
        # Different capitalization does not match the prefix, so it is kept whole.
        ":ST": "HORIZONTAL CORRECTOR MAGNET CURRENT status",
    }


def test_create_template_lowercases_only_the_first_character_of_the_remainder() -> None:
    channels = [
        _family_row(":SP", "Storage ring corrector magnet Setpoint Value"),
        _family_row(":AM", "Storage ring corrector magnet Readback Value"),
    ]

    template = create_template("SCH", channels)

    assert template["description"] == "Storage ring corrector magnet"
    assert template["channel_descriptions"] == {
        ":SP": "setpoint Value",
        ":AM": "readback Value",
    }


def test_create_template_falls_back_to_generic_description_and_keeps_originals() -> None:
    channels = [
        _family_row(":SP", "Horizontal corrector setpoint", instances="4"),
        _family_row(":AM", "Quadrupole focusing readback", instances="4"),
    ]

    template = create_template("SCH", channels)

    assert template["description"] == "SCH device family"
    # With the generic fallback, sub-channel descriptions are left untouched.
    assert template["channel_descriptions"] == {
        ":SP": "Horizontal corrector setpoint",
        ":AM": "Quadrupole focusing readback",
    }


def test_create_template_deduplicates_repeated_sub_channels() -> None:
    channels = [
        _family_row(":SP", "Horizontal corrector magnet current setpoint"),
        _family_row(":SP", "A later duplicate that must be ignored"),
        _family_row(":AM", "Horizontal corrector magnet current readback"),
    ]

    template = create_template("SCH", channels)

    assert template["sub_channels"] == [":SP", ":AM"]
    assert set(template["channel_descriptions"]) == {":SP", ":AM"}


def test_create_template_ignores_rows_without_a_sub_channel() -> None:
    channels = [
        {
            "address": "SR:SCH01",
            "description": "Horizontal corrector magnet family root",
            "family_name": "SCH",
            "instances": "8",
            "sub_channel": None,
        },
        _family_row(":SP", "Horizontal corrector magnet current setpoint", instances="8"),
        _family_row(":AM", "Horizontal corrector magnet current readback", instances="8"),
    ]

    template = create_template("SCH", channels)

    assert template["instances"] == [1, 8]
    assert template["sub_channels"] == [":SP", ":AM"]
    assert template["description"] == "Horizontal corrector magnet current"


def test_create_template_rejects_non_numeric_instance_count() -> None:
    channels = [_family_row(":SP", "Horizontal corrector setpoint", instances="many")]

    with pytest.raises(ValueError):
        create_template("SCH", channels)
