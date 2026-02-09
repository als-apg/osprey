"""
Unit tests for llm_channel_namer tool.

Tests non-API methods: _is_valid_channel_name, _create_prompt_for_batch,
_create_duplicate_resolution_prompt, and resolve_duplicates (with mocked LLM).
"""

from unittest.mock import MagicMock, patch

import pytest

from osprey.services.channel_finder.tools.llm_channel_namer import LLMChannelNamer


@pytest.fixture
def namer():
    """Create an LLMChannelNamer instance for testing."""
    return LLMChannelNamer(provider="test", model_id="test/model")


class TestIsValidChannelName:
    """Test channel name validation rules."""

    def test_valid_pascal_case(self, namer):
        """Valid PascalCase name returns True."""
        assert namer._is_valid_channel_name("BeamPositionMonitor") is True

    def test_name_with_underscore(self, namer):
        """Name with underscore is allowed (replace('_', '').isalnum())."""
        assert namer._is_valid_channel_name("Beam_Position") is True

    def test_empty_string(self, namer):
        """Empty string returns False."""
        assert namer._is_valid_channel_name("") is False

    def test_too_short_one_char(self, namer):
        """Single character returns False."""
        assert namer._is_valid_channel_name("A") is False

    def test_too_short_two_chars(self, namer):
        """Two characters returns False."""
        assert namer._is_valid_channel_name("Ab") is False

    def test_starts_with_lowercase(self, namer):
        """Name starting with lowercase returns False."""
        assert namer._is_valid_channel_name("beamPosition") is False

    def test_contains_special_characters(self, namer):
        """Name with special characters returns False."""
        assert namer._is_valid_channel_name("Beam:Position") is False
        assert namer._is_valid_channel_name("Beam@Monitor") is False

    def test_exceeds_max_length(self, namer):
        """Name exceeding 80 chars returns False."""
        long_name = "A" * 81
        assert namer._is_valid_channel_name(long_name) is False

    def test_exactly_80_chars(self, namer):
        """Name of exactly 80 chars returns True."""
        name = "A" * 80
        assert namer._is_valid_channel_name(name) is True

    def test_exactly_3_chars(self, namer):
        """Name of exactly 3 chars returns True."""
        assert namer._is_valid_channel_name("Abc") is True


class TestCreatePromptForBatch:
    """Test prompt construction for batch naming."""

    def test_prompt_contains_channel_info(self, namer):
        """Prompt contains all channel short_names and descriptions."""
        channels = [
            {"short_name": "BPM01X", "description": "Horizontal position"},
            {"short_name": "BPM01Y", "description": "Vertical position"},
        ]
        prompt = namer._create_prompt_for_batch(channels)
        assert "BPM01X" in prompt
        assert "Horizontal position" in prompt
        assert "BPM01Y" in prompt
        assert "Vertical position" in prompt

    def test_prompt_includes_count_instruction(self, namer):
        """Prompt includes correct count instruction."""
        channels = [
            {"short_name": "CH1", "description": "Desc 1"},
            {"short_name": "CH2", "description": "Desc 2"},
            {"short_name": "CH3", "description": "Desc 3"},
        ]
        prompt = namer._create_prompt_for_batch(channels)
        assert "EXACTLY 3" in prompt

    def test_prompt_for_single_channel(self, namer):
        """Prompt works for single channel."""
        channels = [{"short_name": "BPM01", "description": "BPM readback"}]
        prompt = namer._create_prompt_for_batch(channels)
        assert "EXACTLY 1" in prompt
        assert "BPM01" in prompt


class TestCreateDuplicateResolutionPrompt:
    """Test duplicate resolution prompt construction."""

    def test_prompt_contains_duplicate_info(self, namer):
        """Prompt contains duplicate group information."""
        groups = {
            "BeamMonitor": [
                {
                    "short_name": "BPM01X",
                    "description": "Horizontal BPM",
                    "original_name": "BeamMonitor",
                },
                {
                    "short_name": "BPM01Y",
                    "description": "Vertical BPM",
                    "original_name": "BeamMonitor",
                },
            ]
        }
        prompt = namer._create_duplicate_resolution_prompt(groups)
        assert "BeamMonitor" in prompt
        assert "BPM01X" in prompt
        assert "BPM01Y" in prompt
        assert "EXACTLY 2" in prompt

    def test_prompt_counts_total_channels(self, namer):
        """Prompt includes total count across all groups."""
        groups = {
            "NameA": [
                {"short_name": "CH1", "description": "D1", "original_name": "NameA"},
                {"short_name": "CH2", "description": "D2", "original_name": "NameA"},
            ],
            "NameB": [
                {"short_name": "CH3", "description": "D3", "original_name": "NameB"},
            ],
        }
        prompt = namer._create_duplicate_resolution_prompt(groups)
        assert "EXACTLY 3" in prompt


class TestResolveDuplicates:
    """Test duplicate detection and resolution with mocked LLM."""

    def test_no_duplicates_returns_unchanged(self, namer):
        """No duplicates returns names unchanged."""
        channels = [
            {"short_name": "CH1", "description": "D1"},
            {"short_name": "CH2", "description": "D2"},
        ]
        names = ["UniqueNameA", "UniqueNameB"]
        result = namer.resolve_duplicates(channels, names)
        assert result == names

    @patch("osprey.services.channel_finder.tools.llm_channel_namer.get_chat_completion")
    def test_duplicates_resolved_via_llm(self, mock_llm, namer):
        """With duplicates, calls LLM and replaces duplicated names."""
        channels = [
            {"short_name": "BPM01X", "description": "Horizontal"},
            {"short_name": "BPM01Y", "description": "Vertical"},
            {"short_name": "CH3", "description": "Unique"},
        ]
        names = ["BeamMonitor", "BeamMonitor", "UniqueChannel"]

        mock_response = MagicMock()
        mock_response.names = ["HorizontalBeamMonitor", "VerticalBeamMonitor"]
        mock_llm.return_value = mock_response

        result = namer.resolve_duplicates(channels, names)

        assert result[0] == "HorizontalBeamMonitor"
        assert result[1] == "VerticalBeamMonitor"
        assert result[2] == "UniqueChannel"
        mock_llm.assert_called_once()

    @patch("osprey.services.channel_finder.tools.llm_channel_namer.get_chat_completion")
    def test_llm_failure_returns_originals(self, mock_llm, namer):
        """LLM failure during dedup returns original names with duplicates intact."""
        channels = [
            {"short_name": "CH1", "description": "D1"},
            {"short_name": "CH2", "description": "D2"},
        ]
        names = ["SameName", "SameName"]

        mock_llm.side_effect = Exception("API error")

        result = namer.resolve_duplicates(channels, names)
        assert result == ["SameName", "SameName"]
