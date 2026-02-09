"""
Unit tests for build_database tool.

Tests core functions: load_csv, group_by_family, find_common_description,
create_template, and build_database end-to-end.
"""

import json

from osprey.services.channel_finder.tools.build_database import (
    build_database,
    create_template,
    find_common_description,
    group_by_family,
    load_csv,
)


class TestLoadCsv:
    """Test CSV loading with comment skipping, empty row handling, and value cleaning."""

    def test_loads_valid_csv(self, tmp_path):
        """Loads valid CSV with expected columns."""
        csv = tmp_path / "test.csv"
        csv.write_text(
            "address,description,family_name,instances,sub_channel\n"
            "PV:CH1,Channel one,,,\n"
            "PV:CH2,Channel two,BPM,3,X\n"
        )
        result = load_csv(csv)
        assert len(result) == 2
        assert result[0]["address"] == "PV:CH1"
        assert result[1]["family_name"] == "BPM"

    def test_skips_comment_lines(self, tmp_path):
        """Rows where address starts with # are skipped."""
        csv = tmp_path / "test.csv"
        csv.write_text(
            "address,description,family_name,instances,sub_channel\n"
            "#COMMENT,ignore this,,,\n"
            "PV:CH1,Real channel,,,\n"
        )
        result = load_csv(csv)
        assert len(result) == 1
        assert result[0]["address"] == "PV:CH1"

    def test_skips_empty_address_rows(self, tmp_path):
        """Rows with empty address are skipped."""
        csv = tmp_path / "test.csv"
        csv.write_text(
            "address,description,family_name,instances,sub_channel\n"
            ",no address,,,\n"
            "PV:CH1,Valid,,,\n"
        )
        result = load_csv(csv)
        assert len(result) == 1

    def test_cleans_whitespace_from_values(self, tmp_path):
        """Whitespace is stripped from all values."""
        csv = tmp_path / "test.csv"
        csv.write_text(
            "address,description,family_name,instances,sub_channel\n"
            "  PV:CH1  ,  Some description  ,  BPM  ,  3  ,  X  \n"
        )
        result = load_csv(csv)
        assert result[0]["address"] == "PV:CH1"
        assert result[0]["description"] == "Some description"
        assert result[0]["family_name"] == "BPM"

    def test_converts_empty_values_to_none(self, tmp_path):
        """Empty or whitespace-only values become None."""
        csv = tmp_path / "test.csv"
        csv.write_text("address,description,family_name,instances,sub_channel\nPV:CH1,A desc,,, \n")
        result = load_csv(csv)
        assert result[0]["family_name"] is None
        assert result[0]["sub_channel"] is None

    def test_header_only_returns_empty(self, tmp_path):
        """CSV with only a header row returns empty list."""
        csv = tmp_path / "test.csv"
        csv.write_text("address,description,family_name,instances,sub_channel\n")
        result = load_csv(csv)
        assert result == []


class TestGroupByFamily:
    """Test family grouping vs standalone separation."""

    def test_groups_channels_with_family_name(self):
        """Channels with family_name are grouped into families dict."""
        channels = [
            {"address": "BPM01X", "family_name": "BPM", "sub_channel": "X"},
            {"address": "BPM01Y", "family_name": "BPM", "sub_channel": "Y"},
        ]
        families, standalone = group_by_family(channels)
        assert "BPM" in families
        assert len(families["BPM"]) == 2
        assert standalone == []

    def test_standalone_channels_without_family(self):
        """Channels without family_name go to standalone list."""
        channels = [
            {"address": "PV:BEAM", "family_name": None},
            {"address": "PV:VAC", "family_name": None},
        ]
        families, standalone = group_by_family(channels)
        assert families == {}
        assert len(standalone) == 2

    def test_mixed_input_separates_correctly(self):
        """Mixed input correctly separates families and standalone."""
        channels = [
            {"address": "PV:BEAM", "family_name": None},
            {"address": "BPM01X", "family_name": "BPM"},
            {"address": "BPM01Y", "family_name": "BPM"},
            {"address": "PV:VAC", "family_name": None},
            {"address": "COR01X", "family_name": "COR"},
        ]
        families, standalone = group_by_family(channels)
        assert len(families) == 2
        assert len(families["BPM"]) == 2
        assert len(families["COR"]) == 1
        assert len(standalone) == 2

    def test_empty_input(self):
        """Empty input returns empty families and empty standalone."""
        families, standalone = group_by_family([])
        assert families == {}
        assert standalone == []


class TestFindCommonDescription:
    """Test common prefix extraction from descriptions."""

    def test_empty_list_returns_empty(self):
        """Empty list returns empty string."""
        assert find_common_description([]) == ""

    def test_single_description_returns_it(self):
        """Single description returns that description."""
        assert find_common_description(["Beam position monitor readback"]) == (
            "Beam position monitor readback"
        )

    def test_multiple_with_common_prefix(self):
        """Multiple descriptions with common prefix extracts it."""
        descs = [
            "Beam position monitor horizontal readback",
            "Beam position monitor vertical readback",
            "Beam position monitor sum signal",
        ]
        result = find_common_description(descs)
        assert result == "Beam position monitor"

    def test_no_common_prefix_returns_empty(self):
        """No common prefix returns empty string."""
        descs = [
            "Horizontal position readback",
            "Vertical steering current",
        ]
        result = find_common_description(descs)
        assert result == ""

    def test_short_common_prefix_returns_empty(self):
        """Common prefix shorter than 3 words returns empty string."""
        descs = [
            "BPM horizontal readback",
            "BPM vertical readback",
        ]
        result = find_common_description(descs)
        # "BPM" is only 1 word, too short
        assert result == ""


class TestCreateTemplate:
    """Test template creation from family groups."""

    def test_creates_template_with_correct_fields(self):
        """Creates template with correct base_name, instances, sub_channels."""
        channels = [
            {
                "address": "BPM01X",
                "family_name": "BPM",
                "instances": "5",
                "sub_channel": "X",
                "description": "BPM horizontal",
            },
            {
                "address": "BPM01Y",
                "family_name": "BPM",
                "instances": "5",
                "sub_channel": "Y",
                "description": "BPM vertical",
            },
        ]
        template = create_template("BPM", channels)
        assert template["template"] is True
        assert template["base_name"] == "BPM"
        assert template["instances"] == [1, 5]
        assert template["sub_channels"] == ["X", "Y"]

    def test_extracts_channel_descriptions(self):
        """Extracts channel_descriptions per sub-channel."""
        channels = [
            {
                "address": "BPM01X",
                "family_name": "BPM",
                "instances": "3",
                "sub_channel": "X",
                "description": "X position",
            },
            {
                "address": "BPM01Y",
                "family_name": "BPM",
                "instances": "3",
                "sub_channel": "Y",
                "description": "Y position",
            },
        ]
        template = create_template("BPM", channels)
        assert "X" in template["channel_descriptions"]
        assert "Y" in template["channel_descriptions"]

    def test_strips_common_prefix_from_descriptions(self):
        """Strips common prefix from sub-channel descriptions."""
        channels = [
            {
                "address": "BPM01X",
                "family_name": "BPM",
                "instances": "3",
                "sub_channel": "X",
                "description": "Beam position monitor horizontal readback value",
            },
            {
                "address": "BPM01Y",
                "family_name": "BPM",
                "instances": "3",
                "sub_channel": "Y",
                "description": "Beam position monitor vertical readback value",
            },
            {
                "address": "BPM01S",
                "family_name": "BPM",
                "instances": "3",
                "sub_channel": "Sum",
                "description": "Beam position monitor sum signal readback",
            },
        ]
        template = create_template("BPM", channels)
        # Common prefix "Beam position monitor" should be extracted as description
        assert template["description"] == "Beam position monitor"
        # Sub-channel descriptions should have the common prefix stripped
        for desc in template["channel_descriptions"].values():
            assert not desc.startswith("Beam position monitor")

    def test_fallback_description_when_no_common(self):
        """Falls back to '{family} device family' when no common description."""
        channels = [
            {
                "address": "BPM01X",
                "family_name": "BPM",
                "instances": "2",
                "sub_channel": "X",
                "description": "Horizontal readback",
            },
            {
                "address": "BPM01Y",
                "family_name": "BPM",
                "instances": "2",
                "sub_channel": "Y",
                "description": "Vertical readback",
            },
        ]
        template = create_template("BPM", channels)
        assert template["description"] == "BPM device family"


class TestBuildDatabase:
    """Test end-to-end build_database function."""

    def test_csv_to_json_end_to_end(self, tmp_path):
        """End-to-end: CSV in -> JSON out, correct structure with _metadata and channels."""
        csv_file = tmp_path / "input.csv"
        csv_file.write_text(
            "address,description,family_name,instances,sub_channel\n"
            "BEAM:CURRENT,Total beam current,,,\n"
            "BPM01X,BPM horizontal,BPM,3,X\n"
            "BPM01Y,BPM vertical,BPM,3,Y\n"
        )
        output_file = tmp_path / "output" / "db.json"

        db = build_database(csv_file, output_file)

        assert output_file.exists()
        assert "_metadata" in db
        assert "channels" in db
        assert db["_metadata"]["stats"]["total_entries"] == len(db["channels"])

    def test_mixed_produces_standalone_and_templates(self, tmp_path):
        """Mixed input produces both standalone channels and templates."""
        csv_file = tmp_path / "input.csv"
        csv_file.write_text(
            "address,description,family_name,instances,sub_channel\n"
            "BEAM:CURRENT,Total beam current,,,\n"
            "BPM01X,BPM horizontal,BPM,3,X\n"
            "BPM01Y,BPM vertical,BPM,3,Y\n"
        )
        output_file = tmp_path / "db.json"

        db = build_database(csv_file, output_file)

        standalone = [ch for ch in db["channels"] if not ch.get("template")]
        templates = [ch for ch in db["channels"] if ch.get("template")]
        assert len(standalone) == 1
        assert len(templates) == 1
        assert db["_metadata"]["stats"]["standalone_entries"] == 1
        assert db["_metadata"]["stats"]["template_entries"] == 1

    def test_output_creates_parent_dirs(self, tmp_path):
        """Output file is created with parents=True."""
        csv_file = tmp_path / "input.csv"
        csv_file.write_text(
            "address,description,family_name,instances,sub_channel\nPV:CH1,A channel,,,\n"
        )
        output_file = tmp_path / "deep" / "nested" / "dir" / "db.json"

        build_database(csv_file, output_file)

        assert output_file.exists()
        data = json.loads(output_file.read_text())
        assert "channels" in data
