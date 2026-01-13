"""Unit Tests for State Identification Agent and Tools.

This module tests the state identification subsystem including:
- Reference file tools (mock and real modes)
- Channel access tools
- State identification agent
"""

import pytest

from osprey.services.xopt_optimizer.models import MachineState
from osprey.services.xopt_optimizer.state_identification.tools.reference_files import (
    MOCK_REFERENCE_FILES,
    create_list_files_tool,
    create_read_file_tool,
    create_reference_file_tools,
)


class TestReferenceFileTools:
    """Test reference file tools."""

    def test_mock_list_files_returns_files(self):
        """Mock list files tool should return mock file names."""
        list_files = create_list_files_tool(mock_mode=True)
        result = list_files.invoke({})

        assert "machine_ready_criteria.md" in result
        assert "optimization_channels.md" in result
        assert "safety_procedures.md" in result

    def test_mock_read_file_returns_content(self):
        """Mock read file tool should return mock content."""
        read_file = create_read_file_tool(mock_mode=True)
        result = read_file.invoke({"filename": "machine_ready_criteria.md"})

        assert "Machine Ready Criteria" in result
        assert "BEAM:CURRENT" in result
        assert "READY" in result
        assert "NOT_READY" in result

    def test_mock_read_file_not_found(self):
        """Mock read file tool should handle missing files."""
        read_file = create_read_file_tool(mock_mode=True)
        result = read_file.invoke({"filename": "nonexistent.md"})

        assert "not found" in result.lower()
        assert "machine_ready_criteria.md" in result  # Shows available files

    def test_real_mode_no_path_returns_message(self):
        """Real mode without path should return informative message."""
        list_files = create_list_files_tool(reference_path=None, mock_mode=False)
        result = list_files.invoke({})

        assert "not configured" in result.lower() or "not specified" in result.lower()

    def test_real_mode_missing_path_returns_message(self):
        """Real mode with non-existent path should return error message."""
        list_files = create_list_files_tool(
            reference_path="/nonexistent/path", mock_mode=False
        )
        result = list_files.invoke({})

        assert "does not exist" in result.lower()

    def test_create_reference_file_tools_returns_two_tools(self):
        """create_reference_file_tools should return list and read tools."""
        tools = create_reference_file_tools(mock_mode=True)

        assert len(tools) == 2
        tool_names = [t.name for t in tools]
        assert "list_reference_files" in tool_names
        assert "read_reference_file" in tool_names

    def test_mock_files_contain_expected_content(self):
        """Mock files should contain machine state assessment content."""
        # Check that mock files have the essential content for the agent
        criteria_file = MOCK_REFERENCE_FILES["machine_ready_criteria.md"]
        assert "BEAM:CURRENT" in criteria_file
        assert "SAFETY:INTERLOCK" in criteria_file
        assert "READY" in criteria_file
        assert "NOT_READY" in criteria_file

        channels_file = MOCK_REFERENCE_FILES["optimization_channels.md"]
        assert "Channel Name" in channels_file
        assert "VACUUM:PRESSURE" in channels_file


class TestReferenceFileToolsWithRealPath:
    """Test reference file tools with real file paths."""

    def test_real_mode_with_temp_dir(self, tmp_path):
        """Real mode should read files from provided path."""
        # Create test file
        test_file = tmp_path / "test_criteria.md"
        test_file.write_text("# Test Criteria\nThis is a test file.")

        list_files = create_list_files_tool(
            reference_path=str(tmp_path), mock_mode=False
        )
        result = list_files.invoke({})

        assert "test_criteria.md" in result

    def test_real_mode_read_file(self, tmp_path):
        """Real mode should read file contents correctly."""
        # Create test file
        test_file = tmp_path / "test_criteria.md"
        test_content = "# Test Criteria\nBeam current must be > 10 mA"
        test_file.write_text(test_content)

        read_file = create_read_file_tool(
            reference_path=str(tmp_path), mock_mode=False
        )
        result = read_file.invoke({"filename": "test_criteria.md"})

        assert "Test Criteria" in result
        assert "Beam current" in result

    def test_real_mode_prevents_path_traversal(self, tmp_path):
        """Real mode should prevent reading files outside reference path."""
        read_file = create_read_file_tool(
            reference_path=str(tmp_path), mock_mode=False
        )

        # Try to read parent directory file
        result = read_file.invoke({"filename": "../../../etc/passwd"})

        assert "denied" in result.lower() or "not found" in result.lower()


class TestChannelAccessTools:
    """Test channel access tools."""

    @pytest.mark.asyncio
    async def test_read_channels_tool_creation(self):
        """read_channels tool should be creatable."""
        from osprey.services.xopt_optimizer.state_identification.tools.channel_access import (
            create_read_channels_tool,
        )

        tool = create_read_channels_tool()
        assert tool.name == "read_channel_values"

    @pytest.mark.asyncio
    async def test_create_channel_access_tools_returns_list(self):
        """create_channel_access_tools should return tool list."""
        from osprey.services.xopt_optimizer.state_identification.tools.channel_access import (
            create_channel_access_tools,
        )

        tools = create_channel_access_tools()
        assert len(tools) == 1
        assert tools[0].name == "read_channel_values"


class TestMockConnectorStatusChannel:
    """Test that MockConnector returns expected values for status channels."""

    @pytest.mark.asyncio
    async def test_machine_status_returns_one(self):
        """MockConnector should return 1 for MACHINE:STATUS channel."""
        from osprey.connectors.control_system.mock_connector import MockConnector

        connector = MockConnector()
        await connector.connect({})

        result = await connector.read_channel("MACHINE:STATUS")

        # Value should be approximately 1 (with small noise)
        assert 0.9 < result.value < 1.1, f"Expected ~1, got {result.value}"

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_status_channels_return_one(self):
        """MockConnector should return 1 for various status-like channels."""
        from osprey.connectors.control_system.mock_connector import MockConnector

        connector = MockConnector()
        await connector.connect({"noise_level": 0})  # No noise for exact test

        for channel in ["MACHINE:STATUS", "BEAM:READY", "SYSTEM:ENABLE"]:
            result = await connector.read_channel(channel)
            assert result.value == 1.0, f"Expected 1 for {channel}, got {result.value}"

        await connector.disconnect()


class TestStateIdentificationAgent:
    """Test the state identification agent."""

    def test_agent_creation(self):
        """Agent should be creatable with mock mode."""
        from osprey.services.xopt_optimizer.state_identification import (
            create_state_identification_agent,
        )

        # This should not raise - agent is created but not invoked
        agent = create_state_identification_agent(
            mock_files=True,
            model_config={"provider": "openai", "model_id": "gpt-4"},
        )

        assert agent is not None
        assert agent.mock_files is True

    def test_agent_creation_with_reference_path(self, tmp_path):
        """Agent should accept reference path."""
        from osprey.services.xopt_optimizer.state_identification import (
            create_state_identification_agent,
        )

        agent = create_state_identification_agent(
            reference_path=str(tmp_path),
            mock_files=False,
            model_config={"provider": "openai", "model_id": "gpt-4"},
        )

        assert agent.reference_path == str(tmp_path)
        assert agent.mock_files is False

    def test_agent_tools_include_file_and_channel_tools(self):
        """Agent should have both file and channel tools."""
        from osprey.services.xopt_optimizer.state_identification import (
            create_state_identification_agent,
        )

        agent = create_state_identification_agent(
            mock_files=True,
            model_config={"provider": "openai", "model_id": "gpt-4"},
        )

        tools = agent._get_tools()

        tool_names = [t.name for t in tools]
        assert "list_reference_files" in tool_names
        assert "read_reference_file" in tool_names
        assert "read_channel_values" in tool_names


class TestMachineStateAssessmentModel:
    """Test the MachineStateAssessment Pydantic model."""

    def test_assessment_model_creation(self):
        """MachineStateAssessment should be creatable."""
        from osprey.services.xopt_optimizer.state_identification.agent import (
            MachineStateAssessment,
        )

        assessment = MachineStateAssessment(
            state=MachineState.READY,
            reasoning="All criteria met",
            channels_checked=["BEAM:CURRENT", "VACUUM:PRESSURE"],
            key_observations={"beam_current": 100.0},
        )

        assert assessment.state == MachineState.READY
        assert "All criteria" in assessment.reasoning
        assert len(assessment.channels_checked) == 2

    def test_assessment_model_defaults(self):
        """MachineStateAssessment should have sensible defaults."""
        from osprey.services.xopt_optimizer.state_identification.agent import (
            MachineStateAssessment,
        )

        assessment = MachineStateAssessment(
            state=MachineState.NOT_READY,
            reasoning="Interlock active",
        )

        assert assessment.channels_checked == []
        assert assessment.key_observations == {}
