"""Few-shot example types and guide models for orchestration and classification.

Provides BaseExample (abstract), OrchestratorExample, ClassifierExample,
and the guide Pydantic models (OrchestratorGuide, TaskClassifierGuide)
used in prompt construction.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from .planning import PlannedStep


@dataclass
class BaseExample(ABC):
    """Abstract base for few-shot examples.

    Subclasses must implement ``format_for_prompt()`` to produce a string
    suitable for LLM prompt inclusion.
    """

    @abstractmethod
    def format_for_prompt(self) -> str:
        """Return a formatted string representation for LLM prompt inclusion."""
        pass

    @staticmethod
    def join(
        examples: list["BaseExample"],
        separator: str = "\n",
        max_examples: int | None = None,
        randomize: bool = False,
        add_numbering: bool = False,
    ) -> str:
        """Join multiple examples into a formatted string for prompt inclusion.

        This method combines a list of examples into a single formatted string
        suitable for LLM consumption. It provides flexible formatting options
        while maintaining consistency across all example types.

        Args:
            examples: List of example objects to format
            separator: String to join examples (default: "\n")
            max_examples: Optional limit on number of examples to include
            randomize: Whether to randomize order (prevents positional bias)
            add_numbering: Whether to add numbered headers to each example

        Returns:
            Formatted string ready for prompt inclusion, empty string if no examples

        Examples:
            Basic usage::

                examples = [ex1, ex2, ex3]
                formatted = BaseExample.join(examples)
                # Returns: "ex1_content\nex2_content\nex3_content"

            With numbering and spacing::

                formatted = BaseExample.join(examples, separator="\n\n", add_numbering=True)
                # Returns: "**Example 1:**\nex1_content\n\n**Example 2:**\nex2_content..."

            With randomization (for bias prevention)::

                formatted = BaseExample.join(examples, randomize=True)
                # Returns examples in random order

        .. note::
           This method provides a unified interface for formatting example collections.
           All customization is handled through parameters.

        .. seealso::
           :meth:`format_for_prompt` : Individual example formatting method
        """
        if not examples:
            return ""

        examples_to_use = examples[:max_examples] if max_examples else examples

        if randomize:
            import random

            examples_to_use = examples_to_use.copy()
            random.shuffle(examples_to_use)

        formatted = []
        for i, ex in enumerate(examples_to_use):
            content = ex.format_for_prompt()

            if add_numbering:
                content = f"**Example {i + 1}:**\n{content}"

            formatted.append(content)

        return separator.join(formatted)


@dataclass
class OrchestratorExample(BaseExample):
    """Structured example for orchestrator prompt showing how to plan steps with this capability.

    This class provides rich examples that demonstrate how to plan execution steps
    with specific capabilities. Each example includes the planned step, scenario
    context, requirements, and optional notes to guide the orchestrator in
    creating effective execution plans.

    :param step: The planned execution step demonstrating capability usage
    :type step: PlannedStep
    :param scenario_description: Human-readable description of when/why to use this capability
    :type scenario_description: str
    :param context_requirements: What data needs to be available in execution context
    :type context_requirements: Optional[Dict[str, str]]
    :param notes: Additional guidance, caveats, or usage tips
    :type notes: Optional[str]
    """

    step: PlannedStep
    scenario_description: str  # Human-readable description of when/why to use this
    context_requirements: dict[str, str] | None = None  # What needs to be in context
    notes: str | None = None  # Additional guidance or caveats

    def format_for_prompt(self) -> str:
        """Format this example as a scenario with step specification for the orchestrator."""
        formatted_text = f"**{self.scenario_description}**\n"

        if self.context_requirements:
            formatted_text += "   - Context requirements:\n"
            for key, desc in self.context_requirements.items():
                formatted_text += f"     * {key}: {desc}\n"

        formatted_text += "   PlannedStep(\n"

        step_fields = PlannedStep.__annotations__.keys()

        for field_name in step_fields:
            field_value = self.step.get(field_name, None)

            if field_value is None or (isinstance(field_value, (list, dict)) and not field_value):
                continue

            formatted_value = self._format_field_value(field_name, field_value)
            formatted_text += f"       {field_name}={formatted_value},\n"

        formatted_text = formatted_text.rstrip(",\n") + "\n"
        formatted_text += "   )\n"

        if self.notes:
            formatted_text += f"   - Note: {self.notes}\n"

        return formatted_text

    def _format_field_value(self, field_name: str, value: Any) -> str:
        """Format a field value as a Python-like string for prompt display."""
        if value is None:
            return "None"
        elif isinstance(value, str):
            return f'"{value}"'
        elif isinstance(value, dict):
            return json.dumps(value) if value else "{}"
        elif isinstance(value, (list, set)):
            return (
                json.dumps(list(value)) if value else ("[]" if isinstance(value, list) else "set()")
            )
        else:
            return repr(value)


@dataclass
class ClassifierExample(BaseExample):
    """Example for few-shot learning in classifiers.

    This class represents training examples used for few-shot learning in
    classification tasks. Each example contains a query, expected result,
    and reasoning to help the classifier learn decision patterns.

    :param query: Input query text to be classified
    :type query: str
    :param result: Expected boolean classification result
    :type result: bool
    :param reason: Explanation of why this classification is correct
    :type reason: str
    """

    query: str
    result: bool
    reason: str

    def format_for_prompt(self) -> str:
        """Format as ``User Query: "..." -> Expected Output: ... -> Reason: ...``."""
        return (
            f'User Query: "{self.query}" -> Expected Output: {self.result} -> Reason: {self.reason}'
        )


class ClassifierActions(BaseModel):
    """Placeholder for classifier match actions."""

    pass


class TaskClassifierGuide(BaseModel):
    """Classification guide with instructions and few-shot examples for a capability.

    Examples are automatically randomized during prompt formatting to prevent
    positional bias.
    """

    instructions: str
    examples: list[ClassifierExample] = Field(default_factory=list)
    actions_if_true: ClassifierActions = Field(default_factory=ClassifierActions)


class OrchestratorGuide(BaseModel):
    """Orchestration guide with instructions, examples, and priority ordering.

    Lower priority values appear first when multiple guides are concatenated.
    """

    instructions: str
    examples: list[OrchestratorExample] = Field(default_factory=list)
    priority: int = 0
