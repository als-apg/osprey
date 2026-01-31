"""
Unit tests for Middle Layer Pipeline Tool Functions

Tests the individual LangChain tools that the React agent uses to query the database.
These tests validate tool behavior without requiring full LLM integration.
"""

import json
from unittest.mock import MagicMock

import pytest
import yaml

from osprey.templates.apps.control_assistant.services.channel_finder.databases.middle_layer import (
    MiddleLayerDatabase,
)
from osprey.templates.apps.control_assistant.services.channel_finder.pipelines.middle_layer import (
    MiddleLayerPipeline,
)


class TestMiddleLayerTools:
    """Test individual tool functions created by the pipeline."""

    def test_list_systems_tool(self, sample_middle_layer_pipeline) -> None:
        """Test that list_systems tool returns correct data."""
        pipeline = sample_middle_layer_pipeline
        tools = pipeline._create_tools()

        # Find the list_systems tool
        list_systems_tool = next(t for t in tools if t.name == "list_systems")

        # Execute the tool
        result = list_systems_tool.func()

        # Should return list of dicts with name and description
        assert isinstance(result, list)
        assert len(result) == 3  # SR, BR, BTS

        system_names = {s["name"] for s in result}
        assert system_names == {"SR", "BR", "BTS"}

        # Check structure
        for system in result:
            assert "name" in system
            assert "description" in system
            assert isinstance(system["description"], str)

    def test_list_families_tool(self, sample_middle_layer_pipeline) -> None:
        """Test that list_families tool returns correct data."""
        pipeline = sample_middle_layer_pipeline
        tools = pipeline._create_tools()

        list_families_tool = next(t for t in tools if t.name == "list_families")

        # Test valid system
        result = list_families_tool.func("SR")
        assert isinstance(result, list)
        assert len(result) > 0

        family_names = {f["name"] for f in result}
        assert "BPM" in family_names
        assert "HCM" in family_names

        # Test invalid system (should return error dict)
        result_invalid = list_families_tool.func("INVALID")
        assert isinstance(result_invalid, dict)
        assert "error" in result_invalid

    def test_inspect_fields_tool(self, sample_middle_layer_pipeline) -> None:
        """Test that inspect_fields tool returns correct structure."""
        pipeline = sample_middle_layer_pipeline
        tools = pipeline._create_tools()

        inspect_fields_tool = next(t for t in tools if t.name == "inspect_fields")

        # Test top-level fields
        result = inspect_fields_tool.func("SR", "BPM")
        assert isinstance(result, dict)
        assert "Monitor" in result
        assert "Setpoint" in result

        # Check field info structure
        assert "type" in result["Monitor"]
        assert "description" in result["Monitor"]

        # Test subfield inspection
        result_subfields = inspect_fields_tool.func("SR", "BPM", "Setpoint")
        assert "X" in result_subfields
        assert "Y" in result_subfields

        # Test invalid family
        result_invalid = inspect_fields_tool.func("SR", "INVALID")
        assert isinstance(result_invalid, dict)
        assert "error" in result_invalid

    def test_list_channel_names_tool(self, sample_middle_layer_pipeline) -> None:
        """Test that list_channel_names tool retrieves correct channels."""
        pipeline = sample_middle_layer_pipeline
        tools = pipeline._create_tools()

        list_channel_names_tool = next(t for t in tools if t.name == "list_channel_names")

        # Test simple field
        channels = list_channel_names_tool.func("SR", "BPM", "Monitor")
        assert isinstance(channels, list)
        assert "SR01C:BPM1:X" in channels
        assert "SR01C:BPM1:Y" in channels

        # Test with subfield
        channels_x = list_channel_names_tool.func("SR", "BPM", "Setpoint", subfield="X")
        assert "SR01C:BPM1:XSet" in channels_x
        assert len(channels_x) == 4  # 2 sectors × 2 devices

        # Test with sector filter
        channels_sector1 = list_channel_names_tool.func(
            "SR", "BPM", "Setpoint", subfield="X", sectors=[1]
        )
        assert "SR01C:BPM1:XSet" in channels_sector1
        assert "SR02C:BPM1:XSet" not in channels_sector1
        assert len(channels_sector1) == 2

        # Test with device filter
        channels_device1 = list_channel_names_tool.func(
            "SR", "BPM", "Setpoint", subfield="X", devices=[1]
        )
        assert "SR01C:BPM1:XSet" in channels_device1
        assert "SR01C:BPM2:XSet" not in channels_device1

        # Test invalid field
        result_invalid = list_channel_names_tool.func("SR", "BPM", "INVALID")
        assert isinstance(result_invalid, dict)
        assert "error" in result_invalid

    def test_get_common_names_tool(self, sample_middle_layer_pipeline) -> None:
        """Test that get_common_names tool returns device names."""
        pipeline = sample_middle_layer_pipeline
        tools = pipeline._create_tools()

        get_common_names_tool = next(t for t in tools if t.name == "get_common_names")

        # Test family with common names
        result = get_common_names_tool.func("SR", "BPM")
        assert isinstance(result, list)
        assert len(result) > 0

        # Test family without common names (should return empty list)
        result_empty = get_common_names_tool.func("SR", "DCCT")
        assert isinstance(result_empty, list)
        # May or may not have common names depending on fixture

    def test_report_results_tool(self, sample_middle_layer_pipeline) -> None:
        """Test that report_results tool validates and returns confirmation."""
        pipeline = sample_middle_layer_pipeline
        tools = pipeline._create_tools()

        report_results_tool = next(t for t in tools if t.name == "report_results")

        # Test with valid results
        result = report_results_tool.func(
            channels=["SR01C:BPM1:X", "SR01C:BPM1:Y"],
            description="Found BPM positions in SR:BPM:Monitor field",
        )
        assert "2 channel(s) found" in result

        # Test with empty results
        result_empty = report_results_tool.func(channels=[], description="No channels found")
        assert "0 channel(s) found" in result_empty


class TestMiddleLayerToolIntegration:
    """Test tool interaction patterns that the agent would use."""

    def test_typical_search_workflow(self, sample_middle_layer_pipeline) -> None:
        """Test a typical workflow: systems → families → fields → channels."""
        pipeline = sample_middle_layer_pipeline
        tools = pipeline._create_tools()

        # Get tool functions
        list_systems = next(t for t in tools if t.name == "list_systems").func
        list_families = next(t for t in tools if t.name == "list_families").func
        inspect_fields = next(t for t in tools if t.name == "inspect_fields").func
        list_channel_names = next(t for t in tools if t.name == "list_channel_names").func

        # Step 1: List systems
        systems = list_systems()
        assert len(systems) > 0

        # Step 2: Pick a system (SR) and list families
        families = list_families("SR")
        assert len(families) > 0

        # Step 3: Pick a family (BPM) and inspect fields
        fields = inspect_fields("SR", "BPM")
        assert "Monitor" in fields

        # Step 4: Get channels for Monitor field
        channels = list_channel_names("SR", "BPM", "Monitor")
        assert len(channels) > 0
        assert all(isinstance(ch, str) for ch in channels)

    def test_subfield_navigation_workflow(self, sample_middle_layer_pipeline) -> None:
        """Test workflow with subfield navigation."""
        pipeline = sample_middle_layer_pipeline
        tools = pipeline._create_tools()

        inspect_fields = next(t for t in tools if t.name == "inspect_fields").func
        list_channel_names = next(t for t in tools if t.name == "list_channel_names").func

        # Inspect top-level fields
        fields = inspect_fields("SR", "BPM")

        # Check if Setpoint has subfields (type should indicate it)
        setpoint_info = fields.get("Setpoint")
        assert setpoint_info is not None

        # Inspect Setpoint subfields
        subfields = inspect_fields("SR", "BPM", "Setpoint")
        assert "X" in subfields
        assert "Y" in subfields

        # Get channels for X subfield
        channels_x = list_channel_names("SR", "BPM", "Setpoint", subfield="X")
        assert len(channels_x) > 0
        assert all("XSet" in ch for ch in channels_x)

    def test_filtering_workflow(self, sample_middle_layer_pipeline) -> None:
        """Test workflow with sector and device filtering."""
        pipeline = sample_middle_layer_pipeline
        tools = pipeline._create_tools()

        list_channel_names = next(t for t in tools if t.name == "list_channel_names").func

        # Get all channels
        all_channels = list_channel_names("SR", "BPM", "Setpoint", subfield="X")

        # Filter by sector
        sector1_channels = list_channel_names("SR", "BPM", "Setpoint", subfield="X", sectors=[1])
        assert len(sector1_channels) < len(all_channels)
        assert all("SR01C" in ch for ch in sector1_channels)

        # Filter by device
        device1_channels = list_channel_names("SR", "BPM", "Setpoint", subfield="X", devices=[1])
        assert all("BPM1" in ch for ch in device1_channels)

        # Combined filtering
        specific_channels = list_channel_names(
            "SR", "BPM", "Setpoint", subfield="X", sectors=[1], devices=[1]
        )
        assert len(specific_channels) == 1
        assert specific_channels[0] == "SR01C:BPM1:XSet"


class TestStringChannelNamesNormalization:
    """Test that string ChannelNames are correctly normalized to lists.

    MATLAB Middle Layer exports may produce bare strings for single-channel
    entries instead of single-element arrays. The code must handle both formats.
    """

    def test_string_channel_names_in_list_channel_names(
        self, string_channel_names_pipeline
    ) -> None:
        """Test list_channel_names handles string ChannelNames correctly."""
        pipeline = string_channel_names_pipeline
        tools = pipeline._create_tools()

        list_channel_names_tool = next(t for t in tools if t.name == "list_channel_names")

        # DCCT has string ChannelNames "SR:DCCT" (not a list)
        channels = list_channel_names_tool.func("SR", "DCCT", "Monitor")

        # Should return ["SR:DCCT"], NOT ['S', 'R', ':', 'D', 'C', 'C', 'T']
        assert isinstance(channels, list)
        assert len(channels) == 1
        assert channels[0] == "SR:DCCT"

    def test_string_channel_names_in_channel_map(self, string_channel_names_pipeline) -> None:
        """Test that string ChannelNames are correctly added to channel_map."""
        pipeline = string_channel_names_pipeline
        db = pipeline.database

        # The channel should be in the map with correct metadata
        channel_info = db.get_channel("SR:DCCT")
        assert channel_info is not None
        assert channel_info["channel"] == "SR:DCCT"
        assert channel_info["system"] == "SR"
        assert channel_info["family"] == "DCCT"
        assert channel_info["field"] == "Monitor"

    def test_string_channel_names_validation(self, string_channel_names_pipeline) -> None:
        """Test that string ChannelNames pass validation."""
        pipeline = string_channel_names_pipeline
        db = pipeline.database

        # Should validate successfully
        assert db.validate_channel("SR:DCCT") is True
        # Characters from string iteration should NOT be valid
        assert db.validate_channel("S") is False
        assert db.validate_channel("R") is False
        assert db.validate_channel(":") is False

    def test_mixed_string_and_list_channel_names(self, string_channel_names_pipeline) -> None:
        """Test database with both string and list ChannelNames."""
        pipeline = string_channel_names_pipeline
        tools = pipeline._create_tools()

        list_channel_names_tool = next(t for t in tools if t.name == "list_channel_names")

        # BPM has list ChannelNames (normal case)
        bpm_channels = list_channel_names_tool.func("SR", "BPM", "Monitor")
        assert len(bpm_channels) == 2
        assert "SR01C:BPM1:X" in bpm_channels
        assert "SR01C:BPM1:Y" in bpm_channels

        # DCCT has string ChannelNames (edge case)
        dcct_channels = list_channel_names_tool.func("SR", "DCCT", "Monitor")
        assert len(dcct_channels) == 1
        assert dcct_channels[0] == "SR:DCCT"


class TestMiddleLayerErrorHandling:
    """Test error handling in tool functions."""

    def test_invalid_system_error(self, sample_middle_layer_pipeline) -> None:
        """Test that invalid system name returns error."""
        pipeline = sample_middle_layer_pipeline
        tools = pipeline._create_tools()

        list_families = next(t for t in tools if t.name == "list_families").func

        result = list_families("NONEXISTENT")
        assert isinstance(result, dict)
        assert "error" in result
        assert "not found" in result["error"].lower()

    def test_invalid_family_error(self, sample_middle_layer_pipeline) -> None:
        """Test that invalid family name returns error."""
        pipeline = sample_middle_layer_pipeline
        tools = pipeline._create_tools()

        inspect_fields = next(t for t in tools if t.name == "inspect_fields").func

        result = inspect_fields("SR", "NONEXISTENT")
        assert isinstance(result, dict)
        assert "error" in result

    def test_invalid_field_error(self, sample_middle_layer_pipeline) -> None:
        """Test that invalid field name returns error."""
        pipeline = sample_middle_layer_pipeline
        tools = pipeline._create_tools()

        list_channel_names = next(t for t in tools if t.name == "list_channel_names").func

        result = list_channel_names("SR", "BPM", "NONEXISTENT")
        assert isinstance(result, dict)
        assert "error" in result

    def test_invalid_subfield_error(self, sample_middle_layer_pipeline) -> None:
        """Test that invalid subfield name returns error."""
        pipeline = sample_middle_layer_pipeline
        tools = pipeline._create_tools()

        list_channel_names = next(t for t in tools if t.name == "list_channel_names").func

        result = list_channel_names("SR", "BPM", "Setpoint", subfield="NONEXISTENT")
        assert isinstance(result, dict)
        assert "error" in result


# === Fixtures ===


@pytest.fixture
def sample_middle_layer_db_path(tmp_path) -> str:
    """Create a sample middle layer database for testing."""
    db_file = tmp_path / "test_middle_layer.json"

    # Create a minimal test database
    test_data = {
        "SR": {
            "_description": "Storage Ring - main synchrotron light source",
            "BPM": {
                "_description": "Beam Position Monitors - measure beam X/Y position",
                "Monitor": {
                    "_description": "Position readback values",
                    "ChannelNames": [
                        "SR01C:BPM1:X",
                        "SR01C:BPM1:Y",
                        "SR01C:BPM2:X",
                        "SR01C:BPM2:Y",
                        "SR02C:BPM1:X",
                        "SR02C:BPM1:Y",
                        "SR02C:BPM2:X",
                        "SR02C:BPM2:Y",
                    ],
                },
                "Setpoint": {
                    "X": {
                        "ChannelNames": [
                            "SR01C:BPM1:XSet",
                            "SR01C:BPM2:XSet",
                            "SR02C:BPM1:XSet",
                            "SR02C:BPM2:XSet",
                        ]
                    },
                    "Y": {
                        "ChannelNames": [
                            "SR01C:BPM1:YSet",
                            "SR01C:BPM2:YSet",
                            "SR02C:BPM1:YSet",
                            "SR02C:BPM2:YSet",
                        ]
                    },
                },
                "setup": {
                    "CommonNames": ["BPM 1-1", "BPM 1-2", "BPM 2-1", "BPM 2-2"],
                    "DeviceList": [[1, 1], [1, 2], [2, 1], [2, 2]],
                },
            },
            "HCM": {
                "_description": "Horizontal Corrector Magnets",
                "Monitor": {
                    "_description": "Current readback in Amperes",
                    "ChannelNames": [
                        "SR01C:HCM1:Current",
                        "SR01C:HCM2:Current",
                    ],
                },
                "Setpoint": {
                    "ChannelNames": [
                        "SR01C:HCM1:SetCurrent",
                        "SR01C:HCM2:SetCurrent",
                    ]
                },
                "setup": {
                    "CommonNames": ["H Corrector 1", "H Corrector 2"],
                    "DeviceList": [[1, 1], [1, 2]],
                },
            },
            "VCM": {
                "Monitor": {
                    "ChannelNames": [
                        "SR01C:VCM1:Current",
                        "SR01C:VCM2:Current",
                    ]
                },
                "Setpoint": {
                    "ChannelNames": [
                        "SR01C:VCM1:SetCurrent",
                        "SR01C:VCM2:SetCurrent",
                    ]
                },
                "setup": {
                    "CommonNames": ["V Corrector 1", "V Corrector 2"],
                },
            },
            "DCCT": {
                "Monitor": {"ChannelNames": ["SR:DCCT:Current"]},
                "setup": {"CommonNames": ["Beam Current Monitor"]},
            },
        },
        "BR": {
            "_description": "Booster Ring",
            "BPM": {
                "Monitor": {"ChannelNames": ["BR:BPM1:X", "BR:BPM1:Y"]},
                "setup": {"CommonNames": ["BR BPM 1"]},
            },
            "DCCT": {
                "Monitor": {"ChannelNames": ["BR:DCCT:Current"]},
                "setup": {"CommonNames": ["BR Beam Current"]},
            },
        },
        "BTS": {
            "BPM": {
                "Monitor": {"ChannelNames": ["BTS:BPM1:X", "BTS:BPM1:Y"]},
                "setup": {"CommonNames": ["BTS BPM 1"]},
            }
        },
    }

    with open(db_file, "w") as f:
        json.dump(test_data, f, indent=2)

    return str(db_file)


@pytest.fixture
def sample_middle_layer_pipeline(
    sample_middle_layer_db_path, monkeypatch, tmp_path
) -> "MiddleLayerPipeline":
    """Create a sample middle layer pipeline for testing."""
    # Create a minimal config file for testing
    config_data = {
        "project_root": str(tmp_path),
        "channel_finder": {"prompts": {"path": "prompts/middle_layer"}},
    }

    config_file = tmp_path / "config.yml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    # Set CONFIG_FILE environment variable
    monkeypatch.setenv("CONFIG_FILE", str(config_file))

    # Create a mock prompts module
    mock_prompts = MagicMock()
    mock_prompts.query_splitter = MagicMock()
    mock_prompts.query_splitter.get_prompt = MagicMock(return_value="Mock query splitter prompt")

    # Mock load_prompts to return our mock
    def mock_load_prompts(config, require_query_splitter=True):
        return mock_prompts

    monkeypatch.setattr(
        "osprey.templates.apps.control_assistant.services.channel_finder.pipelines.middle_layer.pipeline.load_prompts",
        mock_load_prompts,
    )

    db = MiddleLayerDatabase(sample_middle_layer_db_path)

    # Create pipeline with minimal config (no LLM needed for tool tests)
    model_config = {
        "provider": "anthropic",
        "model_id": "claude-haiku-4-5-20251001",
        "api_key": "test-key",  # Not used in tool tests
        "max_tokens": 4096,
    }

    pipeline = MiddleLayerPipeline(
        database=db,
        model_config=model_config,
        facility_name="Test Facility",
        facility_description="Test accelerator facility",
    )

    return pipeline


@pytest.fixture
def string_channel_names_db_path(tmp_path) -> str:
    """Create a database with string ChannelNames (not arrays) for testing.

    This simulates MATLAB Middle Layer exports that produce bare strings
    for single-channel entries instead of single-element arrays.
    """
    db_file = tmp_path / "test_string_channel_names.json"

    # Database with string ChannelNames (the bug case)
    test_data = {
        "SR": {
            "_description": "Storage Ring",
            "BPM": {
                "Monitor": {
                    "ChannelNames": ["SR01C:BPM1:X", "SR01C:BPM1:Y"],  # Normal: list
                },
                "setup": {"CommonNames": ["BPM 1"]},
            },
            "DCCT": {
                "_description": "DC Current Transformer",
                "Monitor": {
                    "_description": "Beam current readback",
                    "ChannelNames": "SR:DCCT",  # Bug case: STRING, not list
                },
                "setup": {"CommonNames": ["Beam Current Monitor"]},
            },
            "RF": {
                "Frequency": {
                    "ChannelNames": "SR:RF:Freq",  # Another string case
                },
                "Power": {
                    "ChannelNames": ["SR:RF:Pwr1", "SR:RF:Pwr2"],  # Normal list
                },
            },
        },
    }

    with open(db_file, "w") as f:
        json.dump(test_data, f, indent=2)

    return str(db_file)


@pytest.fixture
def string_channel_names_pipeline(
    string_channel_names_db_path, monkeypatch, tmp_path
) -> "MiddleLayerPipeline":
    """Create a pipeline with string ChannelNames for testing."""
    config_data = {
        "project_root": str(tmp_path),
        "channel_finder": {"prompts": {"path": "prompts/middle_layer"}},
    }

    config_file = tmp_path / "config.yml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    monkeypatch.setenv("CONFIG_FILE", str(config_file))

    mock_prompts = MagicMock()
    mock_prompts.query_splitter = MagicMock()
    mock_prompts.query_splitter.get_prompt = MagicMock(return_value="Mock query splitter prompt")

    def mock_load_prompts(config, require_query_splitter=True):
        return mock_prompts

    monkeypatch.setattr(
        "osprey.templates.apps.control_assistant.services.channel_finder.pipelines.middle_layer.pipeline.load_prompts",
        mock_load_prompts,
    )

    db = MiddleLayerDatabase(string_channel_names_db_path)

    model_config = {
        "provider": "anthropic",
        "model_id": "claude-haiku-4-5-20251001",
        "api_key": "test-key",
        "max_tokens": 4096,
    }

    pipeline = MiddleLayerPipeline(
        database=db,
        model_config=model_config,
        facility_name="Test Facility",
        facility_description="Test accelerator facility",
    )

    return pipeline
