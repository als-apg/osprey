"""Base capability generator with shared functionality.

Provides common infrastructure for all capability generator pipelines.
"""

import asyncio
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from osprey.models.completion import get_chat_completion
from osprey.utils.config import get_model_config

from .models import ClassifierAnalysis, OrchestratorAnalysis


class BaseCapabilityGenerator:
    """Base class for capability generators with common functionality."""

    def __init__(
        self,
        capability_name: str,
        verbose: bool = False,
        provider: str | None = None,
        model_id: str | None = None,
    ):
        """Initialize base generator.

        Args:
            capability_name: Name for the generated capability
            verbose: Whether to print progress messages
            provider: Optional LLM provider override
            model_id: Optional model ID override
        """
        self.capability_name = capability_name
        self.verbose = verbose
        self.provider = provider
        self.model_id = model_id

    def _get_model_kwargs(self) -> dict[str, Any]:
        """Get model kwargs for LLM calls.

        Returns:
            Dictionary with either explicit provider/model or model_config
        """
        model_config = get_model_config("orchestrator")

        # Allow explicit provider/model override
        if self.provider and self.model_id:
            if self.verbose:
                print(f"   Using explicit model: {self.provider}/{self.model_id}")
            return {
                "provider": self.provider,
                "model_id": self.model_id,
                "max_tokens": model_config.get("max_tokens", 4096),
            }
        else:
            # Use orchestrator config from registry
            if self.verbose:
                provider = model_config.get("provider", "unknown")
                model_id = model_config.get("model_id", "unknown")
                print(f"   Using orchestrator model: {provider}/{model_id}")
            return {"model_config": model_config}

    async def _call_llm(
        self, prompt: str, output_model: type[BaseModel], max_attempts: int = 2
    ) -> BaseModel:
        """Call LLM with retry logic.

        Args:
            prompt: The prompt to send
            output_model: Pydantic model for structured output
            max_attempts: Number of retry attempts

        Returns:
            Parsed response as output_model instance

        Raises:
            RuntimeError: If all attempts fail
        """
        model_kwargs = self._get_model_kwargs()
        last_error = None

        for attempt in range(1, max_attempts + 1):
            try:
                if self.verbose and attempt > 1:
                    print(f"   Retry attempt {attempt}/{max_attempts}...")

                # Set caller context for API call logging (propagates through asyncio.to_thread)
                from osprey.models import set_api_call_context

                set_api_call_context(
                    function="_call_llm",
                    module="base_generator",
                    class_name=self.__class__.__name__,
                    extra={"attempt": attempt, "max_attempts": max_attempts},
                )

                response = await asyncio.to_thread(
                    get_chat_completion, message=prompt, **model_kwargs, output_model=output_model
                )

                return response

            except Exception as e:
                last_error = e
                if self.verbose:
                    print(f"   Attempt {attempt} failed: {str(e)[:100]}")

                if attempt == max_attempts:
                    break

        # All attempts failed
        error_msg = (
            f"\n❌ Failed to generate after {max_attempts} attempts.\n\n"
            f"Last error: {str(last_error)}\n\n"
            f"Suggestions:\n"
            f"  1. Verify your LLM provider is accessible\n"
            f"  2. Try a more capable model (e.g., Claude Sonnet)\n"
            f"  3. Use --provider and --model flags to override model\n\n"
        )
        raise RuntimeError(error_msg) from last_error

    @staticmethod
    def _to_class_name(name: str, suffix: str = "Capability") -> str:
        """Convert snake_case name to CapitalizedClassName.

        Args:
            name: Snake case name (e.g., 'weather_mcp')
            suffix: Suffix to append (default: 'Capability')

        Returns:
            CamelCase class name (e.g., 'WeatherMcpCapability')
        """
        return "".join(word.title() for word in name.split("_")) + suffix

    @staticmethod
    def _get_timestamp() -> str:
        """Get ISO format timestamp for generation metadata.

        Returns:
            ISO format timestamp string
        """
        return datetime.now().isoformat()

    def _build_classifier_examples_code(
        self, classifier_analysis: ClassifierAnalysis, indent: str = "            "
    ) -> str:
        """Build classifier examples code block.

        Args:
            classifier_analysis: Classifier analysis from LLM
            indent: Indentation string

        Returns:
            Formatted Python code string for classifier examples
        """
        examples = []

        for ex in classifier_analysis.positive_examples:
            examples.append(
                f"{indent}ClassifierExample(\n"
                f'{indent}    query="{ex.query}",\n'
                f"{indent}    result=True,\n"
                f'{indent}    reason="{ex.reason}"\n'
                f"{indent})"
            )

        for ex in classifier_analysis.negative_examples:
            examples.append(
                f"{indent}ClassifierExample(\n"
                f'{indent}    query="{ex.query}",\n'
                f"{indent}    result=False,\n"
                f'{indent}    reason="{ex.reason}"\n'
                f"{indent})"
            )

        return ",\n".join(examples)

    def _build_orchestrator_examples_code(
        self,
        orchestrator_analysis: OrchestratorAnalysis,
        context_type: str,
        indent: str = "            ",
    ) -> str:
        """Build orchestrator examples code block.

        Args:
            orchestrator_analysis: Orchestrator analysis from LLM
            context_type: Context type string
            indent: Indentation string

        Returns:
            Formatted Python code string for orchestrator examples
        """
        examples = []

        for i, ex in enumerate(orchestrator_analysis.example_steps):
            # Use LLM-generated context_key (descriptive), fallback to generic if missing
            context_key = (
                ex.context_key if ex.context_key else f"{self.capability_name}_result_{i+1}"
            )
            examples.append(
                f"{indent}OrchestratorExample(\n"
                f"{indent}    step=PlannedStep(\n"
                f'{indent}        context_key="{context_key}",\n'
                f'{indent}        capability="{self.capability_name}",\n'
                f'{indent}        task_objective="{ex.task_objective}",\n'
                f'{indent}        expected_output="{context_type}",\n'
                f'{indent}        success_criteria="Successfully completed task",\n'
                f"{indent}        inputs=[]\n"
                f"{indent}    ),\n"
                f'{indent}    scenario_description="{ex.scenario}",\n'
                f"{indent}    notes=\"{ex.tool_name if ex.tool_name else 'Implementation-specific notes'}\"\n"
                f"{indent})"
            )

        return ",\n".join(examples)
