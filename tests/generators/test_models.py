"""Tests for generators/models.py - Pydantic models for capability generation."""

import pytest
from pydantic import ValidationError

from osprey.generators.models import (
    CapabilityMetadata,
    ClassifierAnalysis,
    ClassifierExampleRaw,
    ExampleStepRaw,
    OrchestratorAnalysis,
    ToolPattern,
)


class TestClassifierExampleRaw:
    """Tests for ClassifierExampleRaw model."""

    def test_create_classifier_example(self):
        """Test creating a valid classifier example."""
        example = ClassifierExampleRaw(
            query="Show me temperature data", reason="Mentions temperature which is data retrieval"
        )
        assert example.query == "Show me temperature data"
        assert example.reason == "Mentions temperature which is data retrieval"

    def test_missing_required_fields(self):
        """Test that required fields are enforced."""
        with pytest.raises(ValidationError):
            ClassifierExampleRaw(query="test query")  # missing reason

    def test_serialization(self):
        """Test model serialization."""
        example = ClassifierExampleRaw(query="test query", reason="test reason")
        data = example.model_dump()
        assert data == {"query": "test query", "reason": "test reason"}

    def test_deserialization(self):
        """Test model deserialization."""
        data = {"query": "test query", "reason": "test reason"}
        example = ClassifierExampleRaw(**data)
        assert example.query == "test query"
        assert example.reason == "test reason"


class TestClassifierAnalysis:
    """Tests for ClassifierAnalysis model."""

    def test_create_classifier_analysis(self):
        """Test creating a valid classifier analysis."""
        analysis = ClassifierAnalysis(
            activation_criteria="When user asks about weather",
            keywords=["weather", "temperature", "forecast"],
            positive_examples=[
                ClassifierExampleRaw(query="What's the weather?", reason="Direct weather question")
            ],
            negative_examples=[
                ClassifierExampleRaw(query="What time is it?", reason="Not related to weather")
            ],
            edge_cases=["User asks about climate vs weather"],
        )
        assert analysis.activation_criteria == "When user asks about weather"
        assert len(analysis.keywords) == 3
        assert len(analysis.positive_examples) == 1
        assert len(analysis.negative_examples) == 1
        assert len(analysis.edge_cases) == 1

    def test_empty_lists(self):
        """Test that lists can be empty."""
        analysis = ClassifierAnalysis(
            activation_criteria="criteria",
            keywords=[],
            positive_examples=[],
            negative_examples=[],
            edge_cases=[],
        )
        assert analysis.keywords == []
        assert analysis.positive_examples == []

    def test_multiple_examples(self):
        """Test with multiple positive and negative examples."""
        analysis = ClassifierAnalysis(
            activation_criteria="test",
            keywords=["a", "b"],
            positive_examples=[
                ClassifierExampleRaw(query="q1", reason="r1"),
                ClassifierExampleRaw(query="q2", reason="r2"),
            ],
            negative_examples=[
                ClassifierExampleRaw(query="q3", reason="r3"),
                ClassifierExampleRaw(query="q4", reason="r4"),
            ],
            edge_cases=["case1", "case2"],
        )
        assert len(analysis.positive_examples) == 2
        assert len(analysis.negative_examples) == 2


class TestExampleStepRaw:
    """Tests for ExampleStepRaw model."""

    def test_create_example_step(self):
        """Test creating a valid example step."""
        step = ExampleStepRaw(
            context_key="current_weather_sf",
            task_objective="Get current weather for San Francisco",
            scenario="User wants to know if they need an umbrella",
            tool_name="get_weather",
        )
        assert step.context_key == "current_weather_sf"
        assert step.task_objective == "Get current weather for San Francisco"
        assert step.scenario == "User wants to know if they need an umbrella"
        assert step.tool_name == "get_weather"

    def test_optional_tool_name(self):
        """Test that tool_name is optional with default empty string."""
        step = ExampleStepRaw(
            context_key="analysis_results",
            task_objective="Analyze data",
            scenario="High-level planning step",
        )
        assert step.tool_name == ""

    def test_explicit_empty_tool_name(self):
        """Test explicit empty tool_name for high-level planning."""
        step = ExampleStepRaw(
            context_key="planning_step",
            task_objective="Plan next actions",
            scenario="Strategic planning",
            tool_name="",
        )
        assert step.tool_name == ""


class TestToolPattern:
    """Tests for ToolPattern model."""

    def test_create_tool_pattern(self):
        """Test creating a valid tool pattern."""
        pattern = ToolPattern(
            tool_name="fetch_data", typical_scenario="When user requests historical data"
        )
        assert pattern.tool_name == "fetch_data"
        assert pattern.typical_scenario == "When user requests historical data"

    def test_required_fields(self):
        """Test that both fields are required."""
        with pytest.raises(ValidationError):
            ToolPattern(tool_name="test")  # missing typical_scenario


class TestOrchestratorAnalysis:
    """Tests for OrchestratorAnalysis model."""

    def test_create_orchestrator_analysis(self):
        """Test creating a valid orchestrator analysis."""
        analysis = OrchestratorAnalysis(
            when_to_use="Use when multi-step workflow needed",
            example_steps=[
                ExampleStepRaw(
                    context_key="step1",
                    task_objective="First step",
                    scenario="Initial scenario",
                    tool_name="tool1",
                )
            ],
            common_sequences=["Pattern A", "Pattern B"],
            important_notes=["Note 1", "Note 2"],
            tool_usage_patterns=[ToolPattern(tool_name="tool1", typical_scenario="scenario1")],
        )
        assert analysis.when_to_use == "Use when multi-step workflow needed"
        assert len(analysis.example_steps) == 1
        assert len(analysis.common_sequences) == 2
        assert len(analysis.important_notes) == 2
        assert len(analysis.tool_usage_patterns) == 1

    def test_optional_tool_usage_patterns(self):
        """Test that tool_usage_patterns defaults to empty list."""
        analysis = OrchestratorAnalysis(
            when_to_use="guidance",
            example_steps=[],
            common_sequences=[],
            important_notes=[],
        )
        assert analysis.tool_usage_patterns == []

    def test_complex_orchestrator_analysis(self):
        """Test orchestrator analysis with multiple steps and patterns."""
        analysis = OrchestratorAnalysis(
            when_to_use="Complex multi-tool workflow",
            example_steps=[
                ExampleStepRaw(
                    context_key="fetch_data",
                    task_objective="Retrieve sensor data",
                    scenario="User wants analysis",
                    tool_name="archiver_tool",
                ),
                ExampleStepRaw(
                    context_key="analyze_data",
                    task_objective="Process retrieved data",
                    scenario="Second step after data fetch",
                    tool_name="python_executor",
                ),
            ],
            common_sequences=[
                "Fetch then analyze pattern",
                "Analyze then visualize pattern",
            ],
            important_notes=[
                "Always validate data before analysis",
                "Handle missing data gracefully",
            ],
            tool_usage_patterns=[
                ToolPattern(tool_name="archiver_tool", typical_scenario="Data retrieval"),
                ToolPattern(tool_name="python_executor", typical_scenario="Data processing"),
            ],
        )
        assert len(analysis.example_steps) == 2
        assert len(analysis.tool_usage_patterns) == 2
        assert analysis.example_steps[0].tool_name == "archiver_tool"
        assert analysis.example_steps[1].tool_name == "python_executor"


class TestCapabilityMetadata:
    """Tests for CapabilityMetadata model."""

    def test_create_capability_metadata(self):
        """Test creating valid capability metadata."""
        metadata = CapabilityMetadata(
            capability_name_suggestion="weather_retrieval",
            description="Retrieves weather data from API",
            context_type_suggestion="WEATHER_DATA",
        )
        assert metadata.capability_name_suggestion == "weather_retrieval"
        assert metadata.description == "Retrieves weather data from API"
        assert metadata.context_type_suggestion == "WEATHER_DATA"

    def test_naming_conventions(self):
        """Test that naming follows conventions (snake_case and UPPER_CASE)."""
        metadata = CapabilityMetadata(
            capability_name_suggestion="my_new_capability",
            description="Test capability",
            context_type_suggestion="MY_CONTEXT_TYPE",
        )
        # Test naming patterns (though validation is not enforced in the model itself)
        assert "_" in metadata.capability_name_suggestion
        assert metadata.context_type_suggestion.isupper()

    def test_all_fields_required(self):
        """Test that all metadata fields are required."""
        with pytest.raises(ValidationError):
            CapabilityMetadata(
                capability_name_suggestion="test",
                description="test description",
                # missing context_type_suggestion
            )


class TestIntegration:
    """Integration tests combining multiple models."""

    def test_full_classifier_workflow(self):
        """Test creating a complete classifier analysis structure."""
        positive_ex = ClassifierExampleRaw(
            query="Get weather forecast", reason="Direct weather request"
        )
        negative_ex = ClassifierExampleRaw(
            query="What time is dinner?", reason="Not weather related"
        )

        analysis = ClassifierAnalysis(
            activation_criteria="Weather-related queries",
            keywords=["weather", "forecast", "temperature"],
            positive_examples=[positive_ex],
            negative_examples=[negative_ex],
            edge_cases=["Climate vs weather distinction"],
        )

        # Verify structure
        assert len(analysis.positive_examples) == 1
        assert analysis.positive_examples[0].query == "Get weather forecast"
        assert len(analysis.keywords) == 3

    def test_full_orchestrator_workflow(self):
        """Test creating a complete orchestrator analysis structure."""
        step1 = ExampleStepRaw(
            context_key="data_fetch",
            task_objective="Fetch sensor data",
            scenario="User wants to analyze trends",
            tool_name="data_fetcher",
        )

        step2 = ExampleStepRaw(
            context_key="data_analysis",
            task_objective="Analyze fetched data",
            scenario="Process the retrieved data",
            tool_name="python_executor",
        )

        pattern = ToolPattern(
            tool_name="data_fetcher", typical_scenario="Retrieving time-series data"
        )

        analysis = OrchestratorAnalysis(
            when_to_use="Multi-step data analysis workflows",
            example_steps=[step1, step2],
            common_sequences=["Fetch → Analyze → Visualize"],
            important_notes=["Validate data between steps"],
            tool_usage_patterns=[pattern],
        )

        # Verify complete structure
        assert len(analysis.example_steps) == 2
        assert analysis.example_steps[0].context_key == "data_fetch"
        assert analysis.example_steps[1].tool_name == "python_executor"
        assert len(analysis.tool_usage_patterns) == 1

    def test_json_round_trip(self):
        """Test JSON serialization and deserialization."""
        original = CapabilityMetadata(
            capability_name_suggestion="test_capability",
            description="Test description",
            context_type_suggestion="TEST_CONTEXT",
        )

        # Serialize to JSON
        json_data = original.model_dump_json()

        # Deserialize back
        restored = CapabilityMetadata.model_validate_json(json_data)

        # Verify equality
        assert restored.capability_name_suggestion == original.capability_name_suggestion
        assert restored.description == original.description
        assert restored.context_type_suggestion == original.context_type_suggestion
