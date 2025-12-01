"""Basic LLM-based code generator.

This module implements a straightforward LLM-based code generation strategy using
a simple prompt-response pattern with the configured language model.

The basic generator provides:
- Single-pass code generation with minimal overhead
- Clean prompt building with structured error feedback from previous attempts
- Automatic handling of common LLM formatting issues
- Simple, proven reliability and stability

This generator is ideal for self-hosted models, simple setups, and users who
want straightforward code generation without additional dependencies like the
Claude SDK.

.. seealso::
   :class:`osprey.services.python_executor.generation.interface.CodeGenerator`
   :class:`osprey.services.python_executor.generation.factory.create_code_generator`
   :func:`osprey.models.get_chat_completion`

Examples:
    Using the basic generator directly::

        >>> generator = BasicLLMCodeGenerator()
        >>> request = PythonExecutionRequest(
        ...     user_query="Calculate statistics",
        ...     task_objective="Compute mean and std",
        ...     execution_folder_name="stats"
        ... )
        >>> code = await generator.generate_code(request, [])
        >>> print(code)
        import numpy as np
        results = {"mean": np.mean(data), "std": np.std(data)}

    Using with custom model configuration::

        >>> model_config = {"model": "gpt-4", "temperature": 0.3}
        >>> generator = BasicLLMCodeGenerator(model_config=model_config)
        >>> code = await generator.generate_code(request, [])
"""

import re
import textwrap
from pathlib import Path
from typing import Any

from osprey.models import get_chat_completion
from osprey.utils.config import get_model_config
from osprey.utils.logger import get_logger

from ..exceptions import CodeGenerationError
from ..models import ExecutionError, PythonExecutionRequest

logger = get_logger("basic_generator")


class BasicLLMCodeGenerator:
    """Basic LLM-based code generator.

    This class implements a straightforward LLM-based code generation strategy using
    a simple prompt-response pattern. It builds comprehensive prompts from the
    execution request, includes error feedback for iterative improvement, and
    automatically cleans up common LLM formatting issues.

    The generator uses the framework's standard LLM interface (get_chat_completion)
    with configurable model settings. It handles all aspects of prompt construction,
    code generation, and post-processing to produce clean, executable Python code.

    :param model_config: Optional model configuration dict. If None, uses the
                        default "python_code_generator" configuration from the
                        framework config.
    :type model_config: Dict[str, Any], optional

    .. note::
       The generator automatically cleans markdown code blocks and other common
       LLM formatting artifacts, even when explicitly instructed not to use them.

    .. seealso::
       :class:`CodeGenerator` : Protocol interface this class implements
       :func:`osprey.models.get_chat_completion` : LLM interface used for generation
    """

    def __init__(self, model_config=None):
        """Initialize basic generator with optional model configuration.

        Args:
            model_config: Optional model configuration dict. If not provided,
                         uses the default "python_code_generator" config.
        """
        self._provided_model_config = model_config
        self._model_config = None

        # Save prompts: save all prompts and responses for transparency
        self._save_prompts = (model_config or {}).get("save_prompts", False)
        self._prompt_data: dict[str, Any] = {}  # Stores prompts/responses for inspection
        self._execution_folder: Path | None = None  # Set during generation

    @property
    def model_config(self):
        """Lazy-load model configuration when needed."""
        if self._model_config is None:
            if self._provided_model_config is not None:
                # Check if it's a reference to a model config
                if isinstance(self._provided_model_config, dict):
                    model_config_name = self._provided_model_config.get('model_config_name')
                    if model_config_name:
                        # Reference to models section
                        logger.debug(f"Loading model config from: {model_config_name}")
                        self._model_config = get_model_config(model_config_name)
                    else:
                        # Inline config provided
                        self._model_config = self._provided_model_config
                else:
                    self._model_config = self._provided_model_config
            else:
                # No config provided - use default
                logger.info("No model config provided, using default 'python_code_generator'")
                self._model_config = get_model_config("python_code_generator")
        return self._model_config

    async def generate_code(
        self,
        request: PythonExecutionRequest,
        error_chain: list[ExecutionError]
    ) -> str:
        """Generate code using simple LLM chat completion with structured error feedback.

        Builds a comprehensive prompt from the request details and structured error
        feedback, sends it to the configured LLM, and returns cleaned, executable
        Python code.

        Args:
            request: Execution request with task details
            error_chain: Previous ExecutionError objects for iterative improvement

        Returns:
            Generated Python code as string

        Raises:
            CodeGenerationError: If generation fails or returns empty code

        .. note::
           The method uses structured ExecutionError objects to provide rich
           context about previous failures including code, tracebacks, and
           error analysis to improve generation quality.
        """
        # Set execution folder for saving prompts
        if self._save_prompts and hasattr(request, 'execution_folder_path') and request.execution_folder_path:
            self._execution_folder = Path(request.execution_folder_path)
            # Initialize prompt data structure
            self._prompt_data = {
                "generation_prompt": None,
                "raw_response": None,
                "cleaned_code": None,
            }
            logger.info(f"📝 Will save prompts to: {self._execution_folder / 'prompts'}")

        try:
            # Build prompt
            prompt = self._build_code_generation_prompt(request, error_chain)

            logger.info(f"Generating code with prompt length: {len(prompt)} characters")

            # Save prompt if enabled
            if self._save_prompts:
                self._prompt_data["generation_prompt"] = prompt

            # Generate code (model_config loaded lazily)
            generated_code = get_chat_completion(
                model_config=self.model_config,
                message=prompt
            )

            if not generated_code or not generated_code.strip():
                raise CodeGenerationError(
                    "LLM returned empty code",
                    generation_attempt=1,
                    error_chain=error_chain
                )

            # Save raw response if enabled
            if self._save_prompts:
                self._prompt_data["raw_response"] = generated_code

            # Clean up formatting
            cleaned_code = self._clean_generated_code(generated_code)

            # Save cleaned code if enabled
            if self._save_prompts:
                self._prompt_data["cleaned_code"] = cleaned_code

            logger.success(f"Generated {len(cleaned_code)} characters of code")
            return cleaned_code

        except Exception as e:
            if isinstance(e, CodeGenerationError):
                raise

            raise CodeGenerationError(
                f"LLM code generation failed: {str(e)}",
                generation_attempt=1,
                error_chain=error_chain,
                technical_details={"original_error": str(e)}
            ) from e
        finally:
            # Save prompts if enabled
            if self._save_prompts:
                self._save_prompt_data()

    def _build_code_generation_prompt(
        self,
        request: PythonExecutionRequest,
        error_chain: list[ExecutionError]
    ) -> str:
        """Build prompt for code generation with structured error feedback.

        Constructs a comprehensive prompt that includes:
        - System instructions for code generation
        - Task objective (step-specific goal from orchestrator) and user query (original request)
        - Expected results structure template if provided
        - Capability-specific prompts (domain guidance, output formats, context access)
        - Structured error feedback from previous attempts (last 2)

        The capability_prompts field enables sophisticated prompt engineering by
        allowing capabilities to inject domain-specific guidance, execution plans,
        required output structures, and context access patterns.

        The task_objective comes from the orchestrator's execution plan and is a
        self-sufficient description of what THIS specific step should accomplish.
        The user_query provides the original user request for broader context.

        Args:
            request: Execution request with task details and capability_prompts
                    - task_objective: Step-specific goal from orchestrator's plan
                    - user_query: Original user request for context
            error_chain: Previous ExecutionError objects with rich context

        Returns:
            Complete prompt string for LLM with formatted error details

        .. seealso::
           See documentation for detailed examples of leveraging capability_prompts
           for sophisticated prompt engineering in domain-specific scenarios.
        """
        prompt_parts = []

        # === MINIMAL SYSTEM ROLE ===
        prompt_parts.append("You are an expert Python code generator.")

        # === CRITICAL OUTPUT CONSTRAINTS (safety net) ===
        # These are essential constraints that MUST always be enforced
        prompt_parts.append(textwrap.dedent("""
            CRITICAL REQUIREMENTS:
            1. Generate ONLY executable Python code (no markdown, no explanations)
            2. Store computed results in a dictionary variable named 'results'
            3. Include all necessary imports at the top
            """).strip())

        # Task details
        # Note: task_objective is the STEP-SPECIFIC objective from the orchestrator's execution plan
        #       user_query is the ORIGINAL user request for broader context
        prompt_parts.append(f"**User Query:** {request.user_query}")
        prompt_parts.append(f"\n**Your Task:** {request.task_objective}")

        if request.expected_results:
            prompt_parts.append(f"**Expected Results:** {request.expected_results}")

        # === CAPABILITY-SPECIFIC PROMPTS ===
        # These include detailed instructions from prompt builder system
        if request.capability_prompts:
            for prompt in request.capability_prompts:
                if prompt:
                    prompt_parts.append(prompt)
        else:
            # Fallback if no capability prompts provided
            prompt_parts.append(textwrap.dedent("""
                GUIDANCE:
                - Write clean, working Python code for the task
                - Use clear variable names and add comments
                - Handle common edge cases appropriately
                """).strip())

        # Structured error feedback
        if error_chain:
            prompt_parts.append("\n=== PREVIOUS ATTEMPT(S) FAILED - LEARN FROM THESE ERRORS ===")
            prompt_parts.append("Analyze what went wrong and fix the root cause, not just symptoms.")

            # Show last 2 attempts with full structured context
            for error in error_chain[-2:]:
                prompt_parts.append(f"\n{'='*60}")
                prompt_parts.append(error.to_prompt_text())

            prompt_parts.append(f"\n{'='*60}")
            prompt_parts.append("Generate IMPROVED code that fixes these issues.")

        prompt_parts.append("\nGenerate ONLY the Python code.")

        return "\n".join(prompt_parts)

    def _clean_generated_code(self, raw_code: str) -> str:
        """Clean up LLM formatting issues.

        Removes common formatting artifacts that LLMs add despite being instructed
        not to, including markdown code blocks and other wrapper formatting.

        Args:
            raw_code: Raw code string from LLM

        Returns:
            Cleaned Python code without formatting artifacts

        .. note::
           This handles multiple formatting patterns that different LLMs use,
           including both ```python and plain ``` code blocks.
        """
        cleaned = raw_code.strip()

        # Pattern 1: Standard markdown code blocks ```python ... ```
        # This handles both ```python and ``` python (with space)
        markdown_pattern = r'^```\s*python\s*\n(.*?)\n```$'
        match = re.match(markdown_pattern, cleaned, re.DOTALL | re.IGNORECASE)
        if match:
            logger.info("Detected and removed markdown code block formatting")
            cleaned = match.group(1).strip()

        # Pattern 2: Plain code blocks ``` ... ``` (without python specifier)
        elif cleaned.startswith('```') and cleaned.endswith('```'):
            logger.info("Detected and removed plain markdown code block")
            lines = cleaned.split('\n')
            # Remove first and last lines if they're just ```
            if lines[0].strip() == '```' and lines[-1].strip() == '```':
                cleaned = '\n'.join(lines[1:-1]).strip()

        # Pattern 3: Inline code blocks with backticks (less common but possible)
        elif cleaned.count('`') >= 2:
            # Only clean if the entire content is wrapped in backticks
            if cleaned.startswith('`') and cleaned.endswith('`'):
                logger.info("Detected and removed inline code formatting")
                cleaned = cleaned.strip('`').strip()

        return cleaned

    def _save_prompt_data(self) -> None:
        """Save all prompts and responses to execution folder for transparency.

        Creates a prompts/ subdirectory containing:
        - generation_prompt.txt: The complete prompt sent to the LLM
        - raw_response.txt: The raw LLM response before cleaning
        - cleaned_code.py: The final cleaned code
        - metadata.json: Generation metadata (config, error chain info)
        """
        if not self._save_prompts or not self._execution_folder:
            return

        try:
            import json
            prompts_dir = self._execution_folder / "prompts"
            prompts_dir.mkdir(exist_ok=True)

            # Save generation prompt
            if self._prompt_data.get("generation_prompt"):
                (prompts_dir / "generation_prompt.txt").write_text(
                    self._prompt_data["generation_prompt"], encoding="utf-8"
                )

            # Save raw LLM response
            if self._prompt_data.get("raw_response"):
                (prompts_dir / "raw_response.txt").write_text(
                    self._prompt_data["raw_response"], encoding="utf-8"
                )

            # Save cleaned code
            if self._prompt_data.get("cleaned_code"):
                (prompts_dir / "cleaned_code.py").write_text(
                    self._prompt_data["cleaned_code"], encoding="utf-8"
                )

            # Save metadata (model config, generation info)
            metadata = {
                "generator": "BasicLLMCodeGenerator",
                "model_config": {
                    "provider": self.model_config.get("provider"),
                    "model_id": self.model_config.get("model_id"),
                    "temperature": self.model_config.get("temperature"),
                    "max_tokens": self.model_config.get("max_tokens"),
                },
                "generation_stats": {
                    "prompt_length": len(self._prompt_data.get("generation_prompt", "")),
                    "raw_response_length": len(self._prompt_data.get("raw_response", "")),
                    "cleaned_code_length": len(self._prompt_data.get("cleaned_code", "")),
                },
            }
            (prompts_dir / "metadata.json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )

            logger.info(f"💾 Prompts saved to: {prompts_dir}")

        except Exception as e:
            logger.warning(f"Failed to save prompts: {e}")

    def get_generation_metadata(self) -> dict[str, Any]:
        """Get metadata from the last code generation.

        Returns:
            Empty dictionary. Basic generator doesn't provide metadata.
        """
        return {}

