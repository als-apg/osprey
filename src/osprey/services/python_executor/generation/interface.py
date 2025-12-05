"""Code Generator Protocol for Python Executor.

This module defines the Protocol interface that all code generators must implement
to work with the Python Executor service. The Protocol-based design enables
pluggable code generation strategies while maintaining a clean, type-safe interface.

The Protocol approach provides several advantages:
- Duck typing friendly - no inheritance required
- Runtime type checking with @runtime_checkable
- Clean interface separation between different generator implementations
- Easy to add new generators without modifying existing code

All code generators must implement the generate_code() method with the specified
signature to ensure compatibility with the executor service's workflow.

Generators may optionally implement get_generation_metadata() to provide additional
context about the generation process (thinking blocks, tool usage, cost, etc).

.. note::
   This is a Protocol, not an abstract base class. Generators don't need to
   explicitly inherit from CodeGenerator - they just need to implement the
   required method signature.

.. seealso::
   :class:`osprey.services.python_executor.generation.basic_generator.BasicLLMCodeGenerator`
   :class:`osprey.services.python_executor.generation.claude_code_generator.ClaudeCodeGenerator`
   :class:`osprey.services.python_executor.generation.factory.create_code_generator`
   :class:`osprey.services.python_executor.models.PythonExecutionRequest`

Examples:
    Implementing a custom code generator::

        >>> class MyCustomGenerator:
        ...     async def generate_code(
        ...         self,
        ...         request: PythonExecutionRequest,
        ...         error_chain: list[ExecutionError]
        ...     ) -> str:
        ...         # Build prompt with structured error feedback
        ...         prompt = f"Task: {request.task_objective}"
        ...         if error_chain:
        ...             prompt += "\\n\\nPrevious errors:\\n"
        ...             for error in error_chain:
        ...                 prompt += error.to_prompt_text()
        ...         return "print('Hello, World!')"
        ...
        ...     def get_generation_metadata(self) -> dict[str, Any]:
        ...         # Optional: provide metadata about the generation
        ...         return {"model": "my-model-v1", "tokens": 150}
        ...
        >>> # No need to inherit from CodeGenerator - Protocol checks duck typing
        >>> generator = MyCustomGenerator()
        >>> assert isinstance(generator, CodeGenerator)  # True!

    Using the protocol for type hints::

        >>> def process_with_generator(generator: CodeGenerator, request):
        ...     code = await generator.generate_code(request, [])
        ...     # Check if generator provides metadata
        ...     if hasattr(generator, 'get_generation_metadata'):
        ...         metadata = generator.get_generation_metadata()
        ...     return code
"""

from typing import Any, Protocol, runtime_checkable

from ..models import ExecutionError, PythonExecutionRequest


@runtime_checkable
class CodeGenerator(Protocol):
    """Protocol for code generators.

    All code generators must implement this interface to work with
    the Python Executor service. The Protocol uses duck typing, so
    generators don't need to explicitly inherit from this class - they
    just need to implement the generate_code() method.

    The @runtime_checkable decorator enables isinstance() checks at runtime,
    allowing the factory and service to verify that a generator implements
    the correct interface.

    .. note::
       This is a Protocol, not a base class. Don't inherit from it - just
       implement the generate_code() method with the correct signature.

    .. seealso::
       :class:`osprey.services.python_executor.models.PythonExecutionRequest`
       :class:`osprey.services.python_executor.exceptions.CodeGenerationError`
    """

    async def generate_code(
        self, request: PythonExecutionRequest, error_chain: list[ExecutionError]
    ) -> str:
        """Generate Python code based on request and structured error feedback.

        This method is called by the executor service to generate Python code
        for a given execution request. The generator should use the request
        details to create appropriate code and incorporate any error feedback
        from previous generation attempts.

        Args:
            request: Execution request with task details, user query, context data,
                    and generation guidance. Contains all information needed to
                    understand what code needs to be generated.
            error_chain: Previous ExecutionError objects from failed generation,
                        analysis, or execution attempts. Empty list for first attempt.
                        Each error contains the failed code, error message, traceback,
                        and other context. Generators should use this rich feedback
                        to improve code quality in subsequent attempts.

        Returns:
            Generated Python code as a string. The code should be complete,
            executable Python with all necessary imports and proper structure.

        Raises:
            CodeGenerationError: If generation fails or produces invalid output.
                               Should include details about why generation failed
                               and any relevant context for debugging.

        .. note::
           The generated code should follow Python best practices and include
           any necessary imports. Most generators should store results in a
           'results' dictionary for automatic collection by the executor.

        .. warning::
           Generators must handle errors gracefully and raise CodeGenerationError
           with descriptive messages rather than letting raw exceptions propagate.

        Examples:
            Simple generator implementation::

                >>> async def generate_code(self, request, error_chain):
                ...     # Build prompt from request
                ...     prompt = f"Task: {request.task_objective}"
                ...
                ...     # Add structured error feedback if retrying
                ...     if error_chain:
                ...         prompt += "\\n\\nPrevious Errors:\\n"
                ...         for error in error_chain:
                ...             prompt += error.to_prompt_text()  # Rich formatting
                ...
                ...     # Generate code
                ...     code = await my_llm.generate(prompt)
                ...
                ...     if not code:
                ...         raise CodeGenerationError("Failed to generate code")
                ...
                ...     return code
        """
        ...

    def get_generation_metadata(self) -> dict[str, Any]:
        """Get metadata from the last code generation.

        This is an optional method that generators can implement to provide
        additional context about the generation process. The metadata structure
        is generator-specific and can include information like:

        - Cost information (API costs, token usage)
        - Performance metrics (duration, number of retries)
        - Reasoning artifacts (thinking blocks, planning steps)
        - Tool usage (files read, commands executed)
        - Model information (which model was used, temperature settings)

        The executor service will store this metadata in the LangGraph state
        if available, making it accessible to downstream nodes and persisted
        in checkpoints.

        Returns:
            Dictionary containing generator-specific metadata. Structure varies
            by implementation. Returns empty dict if no metadata available.

        .. note::
           This method is optional. Generators that don't provide metadata
           don't need to implement it. The executor checks for its presence
           using hasattr() before calling.

        Examples:
            Claude Code generator metadata::

                >>> metadata = generator.get_generation_metadata()
                >>> # Returns:
                >>> {
                ...     "thinking_blocks": [
                ...         {"content": "First I'll...", "signature": "abc123"}
                ...     ],
                ...     "tool_uses": [
                ...         {"name": "Read", "input": {"file_path": "example.py"}}
                ...     ],
                ...     "cost_usd": 0.0023,
                ...     "duration_ms": 1542,
                ...     "turns": 3
                ... }

            Basic LLM generator metadata::

                >>> metadata = generator.get_generation_metadata()
                >>> # Returns:
                >>> {}  # Basic generator doesn't provide metadata
        """
        ...
