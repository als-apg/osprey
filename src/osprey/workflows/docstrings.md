---
workflow: docstring-guidelines
category: documentation
applies_when: [writing_code, code_review, new_functions, api_design]
estimated_time: reference as needed
ai_ready: true
related: [comments, update-documentation, pre-merge-cleanup]
---

# DocString Guidelines for Professional Python Development

Comprehensive guidelines for writing clear, consistent Sphinx-format docstrings that provide meaningful documentation for any Python codebase.

## Docstring Structure and Components

Every docstring follows this structure with specific purpose for each section:

```python
def function_name(param1, param2=None):
    """Brief one-line summary of what the function does.

    More detailed explanation of the function's purpose, behavior, and context.
    This section explains the "why" and "how" when the function is complex or
    when the brief summary isn't sufficient. Include implementation details
    that affect usage, performance considerations, or important behavioral notes.

    :param param1: Clear description of what this parameter represents and how it's used
    :type param1: str
    :param param2: Description including what None means and default behavior
    :type param2: int, optional
    :raises ValueError: Specific conditions that trigger this exception
    :raises ConnectionError: When this error occurs and what it means for the caller
    :return: Detailed description of what is returned and its structure
    :rtype: bool

    .. note::
       Important usage notes, warnings about thread safety, performance considerations,
       or other crucial information for users.

    .. warning::
       Critical warnings about dangerous operations, state mutations, or breaking changes.

    Examples:
        Basic usage with realistic parameters::

            >>> processor = DataProcessor(config={'timeout': 30})
            >>> result = function_name("input_data", 42)
            >>> print(f"Success: {result}")
            True

        Error handling pattern::

            >>> try:
            ...     result = function_name("", -1)
            ... except ValueError as e:
            ...     print(f"Invalid input: {e}")

    .. seealso::
       :func:`related_function` : Related functionality
       :class:`RelatedClass` : Associated class documentation
    """
```

## Writing Effective Descriptions

### Brief Summary (First Line)
The first line is crucial - it appears in API documentation summaries and IDE tooltips.

**Good Examples:**
```python
"""Execute a capability node and return processing results."""
"""Load configuration from YAML file with validation and defaults."""
"""Initialize database connection with retry logic and connection pooling."""
```

**Poor Examples:**
```python
"""This function does processing."""  # Too vague
"""Execute step"""  # Too brief, no context
"""Get the data from the database using the provided parameters."""  # Too long, redundant
```

**Guidelines:**
- Start with an action verb (Execute, Load, Initialize, Calculate, Process)
- Be specific about what the function accomplishes
- Avoid redundant phrases like "This function..." or "This method..."
- End with a period
- Keep under 79 characters
- Don't repeat the function name or parameter names

### Detailed Description (Optional Second Paragraph)
Include this when the function is complex, has important behavioral details, or when the brief summary needs context.

**When to include:**
- Complex algorithms or multi-step processes
- Important performance characteristics
- State mutations or side effects
- Integration with external systems
- Non-obvious behavior or edge cases

**Example:**
```python
async def process_data_batch(processor, data_batch, config=None):
    """Process a batch of data through the processing pipeline.

    This function handles the complete lifecycle of batch processing including
    validation, transformation, error handling, and result collection.
    The processing is asynchronous and integrates with logging and monitoring
    systems for progress tracking. State is managed through clean separation
    of concerns between validation, processing, and result handling.

    Processing follows this pattern:
    1. Validate input data batch
    2. Apply configured transformations
    3. Measure processing time and capture results
    4. Handle errors with appropriate classification
    5. Return processed results with metadata
    """
```

## Documenting Parameters

### Parameter Descriptions
Be specific about what each parameter represents and how it's used within the function.

**Good Examples:**
```python
:param config: Configuration dictionary containing processing settings and feature flags
:param timeout: Maximum time in seconds to wait for operation completion
:param data: Input DataFrame with required columns 'timestamp' and 'value'
:param observer: Optional observer for status updates and execution tracing
```

**Poor Examples:**
```python
:param config: The configuration  # Too vague
:param data: Data to process      # Doesn't specify format or requirements
:param flag: A boolean flag       # Doesn't explain what it controls
```

### Parameter Types
Use precise type information, especially for complex types common in your application domain.

**Modern Python Type Hints vs. Explicit Documentation:**
- Use function signatures with type hints for simple, self-explanatory types
- Add explicit `:type:` documentation for complex types that need explanation
- Always document when types are optional, have special meaning, or have constraints

```python
:param config: Configuration dictionary containing processing settings and feature flags
:type config: dict
:param processors: List of available data processors for pipeline execution
:type processors: list[DataProcessor]
:param settings: Application-specific configuration object
:type settings: AppConfig
:param logger: Logger instance for debugging and monitoring
:type logger: logging.Logger, optional
```

## Return Values and Exceptions

### Return Documentation
Clearly explain what the function returns, including structure for complex return types.

```python
def analyze_processing_results(processing_history):
    """Analyze processing history and generate performance metrics.

    :param processing_history: List of completed processing records
    :type processing_history: list[ProcessingRecord]
    :return: Dictionary containing performance metrics with keys:
        - 'total_time': Total processing time in seconds
        - 'batch_count': Number of processed batches
        - 'error_rate': Percentage of failed operations
        - 'processor_usage': Dictionary mapping processor names to usage counts
    :rtype: dict[str, Union[float, int, dict]]
    """
```

### Exception Documentation
Document exceptions that callers should handle, with specific conditions that trigger them.

```python
:raises FileNotFoundError: If the configuration file doesn't exist at the specified path
:raises ValidationError: If configuration format is invalid or required fields are missing
:raises ConnectionError: If unable to connect to external services after retry attempts
:raises TimeoutError: If operation exceeds the specified timeout duration
```

## Common Patterns

### Async Functions
Document async behavior and integration with the event loop.

```python
async def process_data_async(processor, data, config):
    """Process data asynchronously with full error handling support.

    This function integrates with asyncio patterns, supporting concurrent
    execution where possible and proper exception propagation through
    the async call stack.

    :param processor: Processor instance to execute
    :type processor: DataProcessor
    :param data: Input data to process
    :type data: dict
    :param config: Processing configuration and parameters
    :type config: ProcessingConfig
    :raises asyncio.TimeoutError: If processing exceeds configured timeout
    :return: Processing result with success status and data
    :rtype: ProcessingResult

    .. note::
       This function should be awaited and integrates with asyncio's
       cancellation and timeout mechanisms.
    """
```

### State Management Functions
Document state mutations and thread safety considerations.

```python
def update_application_state(state, category, key, value):
    """Update application state with new data.

    Safely updates the application state with proper validation and
    type checking. This function modifies shared state and should be
    called with appropriate synchronization in multi-threaded contexts.

    :param state: Application state object to modify
    :type state: ApplicationState
    :param category: Category identifier for the state data
    :type category: str
    :param key: Unique key for storing the state data
    :type key: str
    :param value: State data to store
    :type value: Any
    :raises KeyError: If category is not registered in the state system
    :raises ValueError: If value doesn't match expected type for category

    .. warning::
       This function mutates shared state. Consider thread safety
       implications in concurrent environments.
    """
```

### Configuration Classes
Document configuration options and their effects.

```python
@dataclass
class ProcessingConfig:
    """Configuration for data processing behavior and limits.

    This configuration controls core processing parameters including timeouts,
    retry behavior, and resource limits. Settings can be loaded from JSON/YAML
    configuration files or set programmatically.

    :param max_batch_size: Maximum number of items to process in a single batch
    :type max_batch_size: int
    :param timeout: Timeout in seconds for individual processing operations
    :type timeout: float
    :param enable_parallel_processing: Whether to allow concurrent processing
    :type enable_parallel_processing: bool
    :param retry_failed_items: Whether to retry items that fail with retriable errors
    :type retry_failed_items: bool

    .. note::
       Default values are suitable for most use cases. Adjust timeouts based
       on your application's performance characteristics.

    Examples:
        Default configuration::

            >>> config = ProcessingConfig()
            >>> print(f"Max batch size: {config.max_batch_size}")

        Custom configuration for high-throughput processing::

            >>> config = ProcessingConfig(
            ...     max_batch_size=1000,
            ...     timeout=120.0,
            ...     enable_parallel_processing=True
            ... )
    """

    max_batch_size: int = 100
    timeout: float = 60.0
    enable_parallel_processing: bool = False
    retry_failed_items: bool = True
```

## Examples and Usage Patterns

### Writing Effective Examples
Examples should demonstrate realistic usage patterns, not just syntax.

**Good Example:**
```python
Examples:
    Basic data processing operation::

        >>> from myapp.processors import DataAnalysisProcessor
        >>> from myapp.state import AppState
        >>>
        >>> processor = DataAnalysisProcessor()
        >>> state = AppState.create_new()
        >>> config = ProcessingConfig(
        ...     task_type="analyze_trends",
        ...     data_source="user_metrics",
        ...     time_window="1h"
        ... )
        >>> result = await processor.execute(config, state)
        >>> print(f"Analysis complete: {result.success}")

    Error handling in production code::

        >>> try:
        ...     result = await processor.execute(config, state)
        ... except ValidationError as e:
        ...     logger.error(f"Invalid configuration: {e}")
        ... except TimeoutError:
        ...     logger.warning("Processing timed out, will retry")
```

### Domain-Specific Examples
Include examples that show integration with your specific domain systems when relevant.

```python
Examples:
    Financial data processing::

        >>> trading_symbols = ["AAPL", "GOOGL", "MSFT"]
        >>> result = await get_market_data(trading_symbols)
        >>> print(f"Retrieved data for {len(result)} symbols")

    Scientific instrument integration::

        >>> instrument_config = InstrumentConfig(
        ...     device_address="192.168.1.100",
        ...     port=8080,
        ...     timeout=5.0
        ... )
        >>> async with InstrumentConnection(instrument_config) as conn:
        ...     measurements = await conn.read_sensors(sensor_ids)

    Web API integration::

        >>> api_client = APIClient(base_url="https://api.example.com")
        >>> response = await api_client.fetch_user_data(user_id=12345)
        >>> print(f"User: {response['name']}, Status: {response['status']}")
```

## Quality Guidelines

### What Makes Good Documentation

**Clarity:**
- Use precise, unambiguous language
- Define technical terms specific to your domain
- Explain abbreviations and acronyms
- Provide context for complex operations

**Completeness:**
- Document all public parameters and return values
- Include important exceptions and error conditions
- Show realistic usage examples
- Cross-reference related functionality

**Consistency:**
- Use the same terminology throughout your codebase
- Follow the same structure for similar functions
- Maintain consistent style in examples
- Use standard Sphinx directives consistently

### What to Avoid

**Over-documentation:**
- Don't document every private method in detail
- Skip obvious things like `filename: str` for a parameter clearly named `filename`
- Don't repeat information that's already in the function signature

**Under-documentation:**
- Don't skip complex functions just because they're hard to explain
- Don't omit error conditions that callers need to handle
- Don't skip examples for non-obvious usage patterns

## Checklist for Every Docstring

- [ ] **Brief description** clearly explains the function's purpose
- [ ] **Detailed description** included for complex functions
- [ ] **All parameters** documented with clear descriptions and types
- [ ] **Return value** documented with structure details for complex types
- [ ] **Important exceptions** documented with trigger conditions
- [ ] **Examples** show realistic usage patterns
- [ ] **Cross-references** link to related functionality
- [ ] **Warnings and notes** highlight important usage information
- [ ] **Examples actually work** when copied and executed

This approach ensures your documentation is thorough enough to be genuinely useful while remaining focused and practical for day-to-day development.

## See Also

- [ai-code-review.md](ai-code-review.md) — Review API consistency before writing docstrings
- [pre-merge-cleanup.md](pre-merge-cleanup.md) — Ensure all new functions have docstrings
- [comments.md](comments.md) — When to use inline comments vs docstrings