"""Shared Pydantic models for capability generators.

These models define the structure for LLM-generated analysis used across
different generator pipelines (MCP, prompt-based, OpenAPI, etc.).
"""

from pydantic import BaseModel, Field

# =============================================================================
# Classifier Models
# =============================================================================


class ClassifierExampleRaw(BaseModel):
    """Raw classifier example from LLM."""

    query: str = Field(description="User query example")
    reason: str = Field(description="Why this should/shouldn't activate")


class ClassifierAnalysis(BaseModel):
    """LLM analysis for classifier guide generation."""

    activation_criteria: str = Field(description="When to activate")
    keywords: list[str] = Field(description="Key indicators")
    positive_examples: list[ClassifierExampleRaw] = Field(description="Should activate")
    negative_examples: list[ClassifierExampleRaw] = Field(description="Should not activate")
    edge_cases: list[str] = Field(description="Tricky scenarios")


# =============================================================================
# Orchestrator Models
# =============================================================================


class ExampleStepRaw(BaseModel):
    """Raw example step from LLM."""

    context_key: str = Field(
        description="Descriptive identifier for this step (e.g., 'current_weather_sf', 'alerts_boston')"
    )
    task_objective: str = Field(description="What user wants to accomplish")
    scenario: str = Field(description="Real-world scenario description")
    tool_name: str = Field(
        default="", description="Tool to invoke (optional, can be empty for high-level planning)"
    )


class ToolPattern(BaseModel):
    """Tool usage pattern from LLM (for MCP-specific orchestration)."""

    tool_name: str = Field(description="Tool name")
    typical_scenario: str = Field(description="When to use this tool")


class OrchestratorAnalysis(BaseModel):
    """LLM analysis for orchestrator guide generation."""

    when_to_use: str = Field(description="General guidance")
    example_steps: list[ExampleStepRaw] = Field(description="Example steps")
    common_sequences: list[str] = Field(description="Common patterns")
    important_notes: list[str] = Field(description="Important reminders")
    tool_usage_patterns: list[ToolPattern] = Field(
        default_factory=list, description="Tool-specific patterns (optional)"
    )


# =============================================================================
# Metadata Models
# =============================================================================


class CapabilityMetadata(BaseModel):
    """Metadata about the capability from LLM."""

    capability_name_suggestion: str = Field(description="Suggested snake_case name")
    description: str = Field(description="Brief description of what this capability does")
    context_type_suggestion: str = Field(description="Suggested UPPER_CASE context type")
