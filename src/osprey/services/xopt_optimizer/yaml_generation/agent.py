"""ReAct Agent for XOpt YAML Configuration Generation.

This module provides a ReAct agent that generates XOpt YAML configurations.
The agent dynamically adapts based on whether example files are available:

- **With examples**: Agent gets file reading tools and is instructed to
  learn from historical configurations before generating new ones.
- **Without examples**: Agent generates YAML from its built-in knowledge
  of XOpt configuration patterns.

This design avoids requiring pre-created example files while still
benefiting from them when available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from osprey.models.langchain import get_langchain_model
from osprey.utils.logger import get_logger

logger = get_logger("xopt_optimizer")


# =============================================================================
# DYNAMIC PROMPTS
# =============================================================================

# Prompt when example files ARE available
PROMPT_WITH_EXAMPLES = """You are an expert XOpt configuration generator for accelerator optimization.

You have access to example XOpt YAML configurations that you should read and learn from.

## Your Workflow

1. **READ EXAMPLES FIRST**: Use the `list_yaml_files` tool to see what examples are available,
   then use `read_yaml_file` to read them. Study the structure carefully:
   - Variable definitions (types, bounds)
   - Objective specifications
   - Generator selection patterns
   - Comments explaining configuration choices

2. **Understand the Objective**: Parse the user's optimization request to understand:
   - What they want to optimize
   - What strategy is appropriate (exploration vs optimization)
   - Any constraints mentioned

3. **Generate Configuration**: Create a valid XOpt YAML based on:
   - The patterns you learned from examples
   - The specific optimization objective
   - Best practices for XOpt

## Output Format

Your final output MUST be a complete, valid YAML configuration wrapped in ```yaml``` code blocks.
Include comments explaining key configuration choices.

## Important Notes

- Always read the examples first - they show the expected structure
- Use placeholder names (param_1, param_2, objective_1) unless the user provides specific names
- Do NOT invent specific accelerator channel names or parameters
- Always include: generator, evaluator, vocs sections
"""

# Prompt when NO example files are available
PROMPT_WITHOUT_EXAMPLES = """You are an expert XOpt configuration generator for accelerator optimization.

No example configurations are available, so you must generate YAML from your knowledge of XOpt.

## XOpt Configuration Structure

A valid XOpt YAML configuration includes:

```yaml
# Generator - how to sample new points
generator:
  name: random  # Options: random, latin_hypercube, sobol, bayesian

# Evaluator - how to assess points
evaluator:
  function: objective_function_name

# VOCS - Variables, Objectives, Constraints, Statics
vocs:
  variables:
    param_name:
      type: continuous  # or discrete, ordinal
      lower: 0.0
      upper: 10.0
  objectives:
    objective_name:
      type: minimize  # or maximize
  constraints: {}
  statics: {}

# Runtime settings
n_initial: 5
max_evaluations: 20
```

## Your Workflow

1. **Understand the Objective**: Parse the user's optimization request
2. **Select Generator**: Based on strategy (exploration → random/latin_hypercube, optimization → bayesian)
3. **Define Variables**: Create placeholder variables with reasonable defaults
4. **Define Objectives**: Based on what user wants to optimize
5. **Generate YAML**: Complete, valid configuration

## Output Format

Your final output MUST be a complete, valid YAML configuration wrapped in ```yaml``` code blocks.
Include comments explaining key configuration choices.

## Important Notes

- Use placeholder names (param_1, param_2, objective_1) unless the user provides specific names
- Do NOT invent specific accelerator channel names or parameters
- Include reasonable default bounds (e.g., 0.0 to 10.0 for continuous variables)
- Always include: generator, evaluator, vocs sections
"""


# =============================================================================
# FILE TOOLS (only created when examples exist)
# =============================================================================


def _create_file_tools(examples_path: Path) -> list[Any]:
    """Create file reading tools for the agent.

    These tools are only created when example files exist.

    Args:
        examples_path: Path to directory containing example YAML files

    Returns:
        List of LangChain tools for file operations
    """

    @tool
    def list_yaml_files() -> str:
        """List available YAML configuration files in the examples directory.

        Use this tool first to see what examples are available before reading them.

        Returns:
            List of available YAML files with brief descriptions from their first comment.
        """
        yaml_files = list(examples_path.glob("**/*.yaml")) + list(examples_path.glob("**/*.yml"))

        if not yaml_files:
            return "No YAML files found in examples directory."

        results = ["Available YAML configurations:"]
        for yaml_file in yaml_files:
            rel_path = yaml_file.relative_to(examples_path)
            # Try to extract description from first comment line
            try:
                first_lines = yaml_file.read_text(encoding="utf-8").split("\n")[:5]
                description = ""
                for line in first_lines:
                    if line.startswith("#") and not line.startswith("# ="):
                        description = line.lstrip("# ").strip()
                        break
                if description:
                    results.append(f"  - {rel_path}: {description}")
                else:
                    results.append(f"  - {rel_path}")
            except Exception:
                results.append(f"  - {rel_path}")

        return "\n".join(results)

    @tool
    def read_yaml_file(filename: str) -> str:
        """Read the contents of a YAML configuration file.

        Use this after listing files to read specific examples and learn their structure.

        Args:
            filename: Name of the YAML file to read (e.g., 'exploration_basic.yaml')

        Returns:
            Contents of the YAML file, or error message if not found.
        """
        # Security: only allow reading from examples directory
        file_path = examples_path / filename

        # Check for path traversal attacks
        try:
            file_path = file_path.resolve()
            examples_resolved = examples_path.resolve()
            if not str(file_path).startswith(str(examples_resolved)):
                return f"Error: Cannot read files outside examples directory."
        except Exception:
            return f"Error: Invalid file path."

        if not file_path.exists():
            # Try searching subdirectories
            matches = list(examples_path.glob(f"**/{filename}"))
            if matches:
                file_path = matches[0]
            else:
                return f"Error: File '{filename}' not found. Use list_yaml_files to see available files."

        try:
            content = file_path.read_text(encoding="utf-8")
            return f"=== {filename} ===\n{content}"
        except Exception as e:
            return f"Error reading file: {e}"

    return [list_yaml_files, read_yaml_file]


# =============================================================================
# AGENT CLASS
# =============================================================================


class YamlGenerationAgent:
    """ReAct agent for generating XOpt YAML configurations.

    This agent dynamically adapts based on whether example files are available:
    - With examples: Gets file tools and prompt to read examples first
    - Without examples: Generates from knowledge with appropriate prompt

    Attributes:
        examples_path: Path to directory containing example YAML files (optional)
        model_config: Configuration for the LLM model to use
    """

    def __init__(
        self,
        examples_path: str | Path | None = None,
        model_config: dict[str, Any] | None = None,
    ):
        """Initialize the YAML generation agent.

        Args:
            examples_path: Path to directory containing example YAML files.
                If None or directory doesn't exist/is empty, agent generates from knowledge.
            model_config: Optional model configuration. If not provided,
                uses the 'fast' model from osprey config.
        """
        self.examples_path = Path(examples_path) if examples_path else None
        self.model_config = model_config
        self._agent = None
        self._has_examples = False

    def _check_examples_exist(self) -> bool:
        """Check if example YAML files exist.

        Returns:
            True if examples directory exists and contains YAML files
        """
        if not self.examples_path:
            return False

        if not self.examples_path.exists():
            return False

        yaml_files = list(self.examples_path.glob("**/*.yaml")) + list(
            self.examples_path.glob("**/*.yml")
        )
        return len(yaml_files) > 0

    def _get_tools(self) -> list[Any]:
        """Get tools for the agent based on file availability.

        Returns:
            List of tools (file tools if examples exist, empty otherwise)
        """
        if self._has_examples and self.examples_path:
            return _create_file_tools(self.examples_path)
        else:
            return []

    def _get_prompt(self) -> str:
        """Get the appropriate system prompt based on file availability.

        Returns:
            System prompt string
        """
        if self._has_examples:
            return PROMPT_WITH_EXAMPLES
        else:
            return PROMPT_WITHOUT_EXAMPLES

    def _get_model(self):
        """Get the LangChain model for the agent.

        Uses model_config provided during initialization.
        The node.py handles fallback to orchestrator model if xopt-specific
        model is not configured.

        Returns:
            LangChain BaseChatModel instance

        Raises:
            ValueError: If no model_config is available
        """
        if self.model_config:
            return get_langchain_model(model_config=self.model_config)

        # This shouldn't happen if node.py fallback is working
        raise ValueError(
            "No model_config provided to YamlGenerationAgent. "
            "Ensure xopt_optimizer.yaml_generation.model_config_name is set in config.yml "
            "or that 'orchestrator' model is configured as fallback."
        )

    def _get_agent(self):
        """Get or create the ReAct agent with dynamic configuration.

        Returns:
            Compiled ReAct agent graph
        """
        if self._agent is None:
            # Check for examples at agent creation time
            self._has_examples = self._check_examples_exist()

            model = self._get_model()
            tools = self._get_tools()

            # Create agent with or without tools
            self._agent = create_react_agent(
                model=model,
                tools=tools,
            )

        return self._agent

    async def generate_yaml(
        self,
        objective: str,
        strategy: str,
        additional_context: dict[str, Any] | None = None,
    ) -> str:
        """Generate XOpt YAML configuration using the ReAct agent.

        Args:
            objective: The optimization objective (e.g., "maximize injection efficiency")
            strategy: The selected strategy ("exploration" or "optimization")
            additional_context: Optional additional context to include in the prompt

        Returns:
            Generated YAML configuration as a string

        Raises:
            ValueError: If YAML generation fails or produces invalid output
        """
        agent = self._get_agent()

        # Build the user message
        user_message = f"""Generate an XOpt YAML configuration for the following:

**Optimization Objective:** {objective}
**Strategy:** {strategy}

{"First, use the tools to read available example configurations. " if self._has_examples else ""}Generate a complete, valid YAML configuration based on {
    "what you learn from the examples" if self._has_examples else "your knowledge of XOpt configuration patterns"
}.

Remember:
- Use generic parameter names unless specific names are provided
- Include comments explaining your configuration choices
- Output the final YAML in ```yaml``` code blocks
"""

        if additional_context:
            user_message += f"\n**Additional Context:** {additional_context}"

        # Run the agent
        logger.info("Starting YAML generation agent...")

        try:
            result = await agent.ainvoke(
                {
                    "messages": [
                        {"role": "system", "content": self._get_prompt()},
                        {"role": "user", "content": user_message},
                    ]
                }
            )

            # Extract the final response
            messages = result.get("messages", [])
            if not messages:
                raise ValueError("Agent did not produce any output")

            # Get the last message content
            last_message = messages[-1]
            content = (
                last_message.content
                if hasattr(last_message, "content")
                else str(last_message)
            )

            # Extract YAML from response
            yaml_content = self._extract_yaml(content)

            if not yaml_content:
                logger.warning(f"Could not extract YAML from response. Response: {content[:500]}")
                raise ValueError("Agent did not produce valid YAML output")

            logger.info(f"YAML generation complete: {len(yaml_content)} characters")
            return yaml_content

        except Exception as e:
            logger.error(f"YAML generation agent failed: {e}")
            raise ValueError(f"YAML generation failed: {e}") from e

    def _extract_yaml(self, content: str) -> str | None:
        """Extract YAML content from agent response.

        Args:
            content: The agent's response text

        Returns:
            Extracted YAML content or None if not found
        """
        import re

        # Try to find YAML code blocks
        yaml_pattern = r"```yaml\n(.*?)```"
        matches = re.findall(yaml_pattern, content, re.DOTALL)

        if matches:
            return matches[-1].strip()

        # Try generic code blocks
        code_pattern = r"```\n(.*?)```"
        matches = re.findall(code_pattern, content, re.DOTALL)

        for match in matches:
            # Check if it looks like YAML
            if "generator:" in match or "vocs:" in match or "evaluator:" in match:
                return match.strip()

        # If no code blocks, check if the whole response is YAML-like
        if "generator:" in content and "vocs:" in content:
            # Try to extract just the YAML part
            lines = content.split("\n")
            yaml_lines = []
            in_yaml = False

            for line in lines:
                if line.strip().startswith(("#", "generator:", "evaluator:", "vocs:")):
                    in_yaml = True
                if in_yaml:
                    yaml_lines.append(line)

            if yaml_lines:
                return "\n".join(yaml_lines).strip()

        return None


def create_yaml_generation_agent(
    examples_path: str | Path | None = None,
    model_config: dict[str, Any] | None = None,
) -> YamlGenerationAgent:
    """Factory function to create a YAML generation agent.

    The agent dynamically adapts based on whether example files exist:
    - If examples_path has YAML files: Agent gets tools to read them
    - If no examples: Agent generates from its built-in knowledge

    Args:
        examples_path: Path to directory containing example YAML files (optional)
        model_config: Optional model configuration

    Returns:
        Configured YamlGenerationAgent instance
    """
    return YamlGenerationAgent(
        examples_path=examples_path,
        model_config=model_config,
    )
