---
workflow: create-capability
category: development
applies_when: [adding_capability, extending_framework, building_features, new_integration]
estimated_time: 30-60 minutes
ai_ready: true
related: [testing-workflow, docstrings, ai-code-review]
skill_description: >-
  Guides users through creating new capabilities in Osprey applications.
  Use when the user wants to add new functionality, create a capability,
  extend the framework, integrate an external API, or build a new feature
  that requires a capability class. Helps with context classes, capability
  implementation, registry configuration, and testing.
---

# Creating a New Capability

Step-by-step guide for building production-ready capabilities in the Osprey Framework.

**Target Audience**: Developers who need to add new functionality to an Osprey application by creating custom capabilities.

**Important**: This workflow is interactive. DO NOT start writing code immediately. First gather requirements, understand the domain, and design the data structures before implementation.

---

## AI Quick Start

**Paste this prompt to your AI assistant:**

```
Following @src/osprey/assist/tasks/create-capability/instructions.md, help me create a new capability.

Your approach:
1. DO NOT immediately write code
2. DO ask for concrete information about what I'm building
3. DO understand my domain and data before designing
4. DO read the template files to understand patterns
5. DO guide me through each phase systematically
6. DO validate my requirements before implementation

Start by asking:
1. What functionality should this capability provide? (Be specific)
2. What external services/APIs does it need to call? (If any)
3. What data does it produce? (Output structure)
4. What data does it consume? (Input requirements)
5. Where does this capability fit in my workflow? (Dependencies)

Before writing any code:
- Read the minimal template: src/osprey/templates/apps/minimal/
- Read example capability: src/osprey/templates/apps/minimal/capabilities/example_capability.py.j2
- Read context class pattern: src/osprey/templates/apps/minimal/context_classes.py.j2
- Understand my application's existing registry structure

Remember:
- Ask clarifying questions when requirements are vague
- Design context classes before implementing capability
- Keep capabilities focused (single responsibility)
- Reference actual code patterns, not hypothetical examples
```

**Related workflows**: [testing-workflow.md](../testing-workflow/instructions.md), [ai-code-review.md](../ai-code-review/instructions.md)

---

## Quick Reference: Capability Components

### What is a Capability?

A capability is a self-contained business logic component that:
- Extends `BaseCapability` and uses `@capability_node` decorator
- Executes specific domain functionality (API calls, data processing, etc.)
- Stores results in typed Pydantic context objects
- Integrates with LangGraph for orchestration

### Required Components

| Component | Purpose | Location |
|-----------|---------|----------|
| **Context Class** | Define output data structure | `context_classes.py` |
| **Capability Class** | Business logic implementation | `capabilities/your_capability.py` |
| **Registry Entry** | Register for framework discovery | `registry.py` |

### Component Relationships

```
User Query
    │
    ▼
┌─────────────────────┐
│  Task Classifier    │ ← Decides which capabilities to invoke
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Your Capability    │ ← @capability_node decorator
│  - name             │
│  - description      │
│  - provides = [...]  │ ← Output context types
│  - requires = [...]  │ ← Input context types (optional)
│  - execute()        │ ← Business logic here
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Context Object     │ ← CapabilityContext subclass
│  - CONTEXT_TYPE     │
│  - data fields      │
│  - get_access_details()
│  - get_summary()    │
└─────────────────────┘
    │
    ▼
  Stored in State → Available to other capabilities / response generation
```

---

## Phase 1: Requirements Gathering

**Goal**: Understand exactly what the capability should do before writing any code.

### Essential Questions

Before designing anything, the AI assistant must ask these questions:

**1. Functionality**
```
"What specific task should this capability perform?"
- What problem does it solve?
- What's the user's goal when invoking this capability?
- Can you give me a concrete example of a user request that should trigger this?
```

**2. External Dependencies**
```
"Does this capability need to call external services?"
- APIs to call? (REST, gRPC, etc.)
- Databases to query?
- Control systems to interact with? (EPICS, Tango, etc.)
- Local services? (Python executor, memory storage, etc.)
```

**3. Input Requirements**
```
"What information does this capability need to do its job?"
- User query text only?
- Output from another capability? (specify context type)
- Configuration parameters?
- Session/user context?
```

**4. Output Structure**
```
"What data does this capability produce?"
- What fields/values?
- How will this data be used? (displayed to user, consumed by other capabilities, etc.)
- Any relationships to other data types?
```

**5. Workflow Position**
```
"Where does this capability fit in your application's workflow?"
- Standalone capability?
- Part of a multi-step workflow? (depends on / provides for other capabilities)
- What capabilities already exist?
```

### Information Checklist

Before proceeding to design, verify you have:

- [ ] **Clear functionality description** with example user queries
- [ ] **External service details** (endpoints, authentication, response format)
- [ ] **Input specification** (what data is needed, where it comes from)
- [ ] **Output specification** (what data is produced, its structure)
- [ ] **Dependency mapping** (other capabilities it requires/provides for)
- [ ] **Error scenarios** (what can go wrong, how to handle it)

### Red Flags: Stop and Clarify

**User is too vague:**
```
User: "I want to add a capability"
❌ Don't start designing
✅ Ask: "What specific functionality should this capability provide? Give me a concrete example."
```

**Scope is too broad:**
```
User: "I want a capability that handles all weather data"
❌ Don't design one huge capability
✅ Ask: "Let's break this down. What's the most important operation? Current weather? Forecasts? Historical data?"
```

**Unclear output:**
```
User: "It should return the results"
❌ Don't assume what results means
✅ Ask: "What fields should be in the results? How will they be used?"
```

---

## Phase 2: Context Class Design

**Goal**: Define the data structure for what your capability produces.

### When to Create a Context Class

| Situation | Action |
|-----------|--------|
| Capability produces structured data | Create context class |
| Data will be used by other capabilities | Create context class |
| Data needs to be displayed/formatted | Create context class |
| Capability only performs side effects | May not need context class |

### Context Class Template

Read the actual template first:
```
src/osprey/templates/apps/minimal/context_classes.py.j2
```

**Structure:**
```python
from typing import ClassVar, Dict, Any, List, Optional
from pydantic import Field
from osprey.context.base import CapabilityContext

class YourDataContext(CapabilityContext):
    """Description of what this context represents.

    This context is created by the [capability_name] capability when [scenario].
    """

    # Required: Unique identifier (UPPERCASE_WITH_UNDERSCORES)
    CONTEXT_TYPE: ClassVar[str] = "YOUR_DATA"

    # Required: Category for organization
    CONTEXT_CATEGORY: ClassVar[str] = "COMPUTATIONAL_DATA"  # or KNOWLEDGE_DATA, LIVE_DATA, etc.

    # Data fields with descriptions (descriptions shown to LLM)
    primary_field: str = Field(..., description="Main data value")
    secondary_field: int = Field(default=0, description="Optional count")
    nested_data: Dict[str, Any] = Field(default_factory=dict, description="Additional details")

    def get_access_details(self, key: str) -> Dict[str, Any]:
        """Guide LLM on how to access this data in generated code.

        Returns dict with:
        - access_pattern: Python code pattern to access data
        - available_fields: List of fields that can be accessed
        - example_usage: Concrete usage example
        """
        return {
            "context_type": self.CONTEXT_TYPE,
            "key": key,
            "access_pattern": f"context.{self.CONTEXT_TYPE}.{key}.primary_field",
            "available_fields": ["primary_field", "secondary_field", "nested_data"],
            "example_usage": f"value = context.{self.CONTEXT_TYPE}.{key}.primary_field"
        }

    def get_summary(self) -> Dict[str, Any]:
        """Generate human-readable summary for logs and UI.

        Returns structured dict (not pre-formatted string).
        """
        return {
            "type": "Your Data",
            "primary": self.primary_field,
            "secondary": self.secondary_field,
            "has_nested": bool(self.nested_data)
        }
```

### Context Type Naming Conventions

| Pattern | Example | Use Case |
|---------|---------|----------|
| `{DOMAIN}_DATA` | `WEATHER_DATA` | General domain data |
| `{DOMAIN}_RESULTS` | `ANALYSIS_RESULTS` | Computed/processed results |
| `{DOMAIN}_CONTEXT` | `TIME_RANGE_CONTEXT` | Contextual information |
| `{ACTION}_OUTPUT` | `QUERY_OUTPUT` | Action-specific output |

### Field Design Guidelines

**DO:**
- Use descriptive field names (LLM reads these)
- Add `description` to every field (shown to LLM for code generation)
- Use simple types when possible (str, int, float, bool, List, Dict)
- Mark optional fields with `Optional[T]` and `default=None`

**DON'T:**
- Use `datetime` directly (not JSON serializable) - use ISO string instead
- Create deeply nested structures (hard for LLM to navigate)
- Use abbreviated field names (be explicit)
- Skip descriptions (LLM relies on them)

### Example: Weather Context

```python
class CurrentWeatherContext(CapabilityContext):
    """Current weather conditions for a location."""

    CONTEXT_TYPE: ClassVar[str] = "CURRENT_WEATHER"
    CONTEXT_CATEGORY: ClassVar[str] = "LIVE_DATA"

    location: str = Field(..., description="Location name (city, region)")
    temperature_celsius: float = Field(..., description="Temperature in Celsius")
    conditions: str = Field(..., description="Weather conditions (sunny, cloudy, etc.)")
    humidity_percent: int = Field(..., description="Relative humidity percentage")
    timestamp: str = Field(..., description="ISO format timestamp of observation")

    def get_access_details(self, key: str) -> Dict[str, Any]:
        return {
            "context_type": self.CONTEXT_TYPE,
            "key": key,
            "location": self.location,
            "temperature": f"{self.temperature_celsius}°C",
            "access_pattern": f"context.CURRENT_WEATHER.{key}.temperature_celsius",
            "available_fields": ["location", "temperature_celsius", "conditions", "humidity_percent", "timestamp"]
        }

    def get_summary(self) -> Dict[str, Any]:
        return {
            "type": "Current Weather",
            "location": self.location,
            "temperature": f"{self.temperature_celsius}°C",
            "conditions": self.conditions
        }
```

---

## Phase 3: Capability Implementation

**Goal**: Implement the business logic in a capability class.

### Read the Template First

**Always read before implementing:**
```
src/osprey/templates/apps/minimal/capabilities/example_capability.py.j2
```

This 400+ line template contains comprehensive patterns and documentation.

### Capability Structure

```python
from typing import Dict, Any, Optional
from osprey.base import BaseCapability, capability_node
from osprey.base.errors import ErrorClassification, ErrorSeverity
from your_app.context_classes import YourDataContext

@capability_node
class YourCapability(BaseCapability):
    """One-line description of what this capability does.

    Longer description explaining:
    - When this capability is invoked
    - What external services it uses
    - What data it produces
    """

    # Required: Unique identifier (lowercase_with_underscores)
    name = "your_capability"

    # Required: Description (shown to orchestrator/classifier)
    description = "Process data and return structured results"

    # Required: What context types this capability creates
    provides = ["YOUR_DATA"]

    # Optional: What context types this capability needs as input
    requires = []  # e.g., ["TIME_RANGE", "INPUT_DATA"]

    async def execute(self) -> Dict[str, Any]:
        """Execute the capability's core business logic.

        Returns:
            Dict with state updates (use self.store_output_context())
        """
        # 1. Get unified logger
        logger = self.get_logger()

        try:
            # 2. Get task objective (what user/orchestrator wants)
            task_objective = self.get_task_objective()
            logger.status("Processing request...")

            # 3. Get required contexts if needed
            # contexts = self.get_required_contexts()
            # input_data = contexts["INPUT_DATA"]

            # 4. Get parameters if passed by orchestrator
            # params = self.get_parameters()
            # timeout = params.get("timeout", 30)

            # 5. YOUR BUSINESS LOGIC HERE
            # - Call external APIs
            # - Process data
            # - Transform results
            result = await self._do_work(task_objective)

            # 6. Create output context
            context = YourDataContext(
                primary_field=result["value"],
                secondary_field=result.get("count", 0)
            )

            # 7. Store and return
            logger.success("Processing complete!")
            return self.store_output_context(context)

        except Exception as e:
            logger.error(f"Error: {e}")
            raise

    async def _do_work(self, task: str) -> Dict[str, Any]:
        """Internal helper method for business logic."""
        # Implementation here
        pass

    @staticmethod
    def classify_error(exc: Exception, context: dict) -> ErrorClassification:
        """Classify errors for framework recovery strategies.

        Severity options:
        - RETRIABLE: Temporary issue, retry may succeed
        - CRITICAL: Cannot recover, end execution
        - FATAL: System failure, terminate immediately
        """
        if isinstance(exc, (ConnectionError, TimeoutError)):
            return ErrorClassification(
                severity=ErrorSeverity.RETRIABLE,
                user_message="Service temporarily unavailable, retrying...",
                metadata={"technical_details": str(exc)}
            )

        # Default: critical error
        return ErrorClassification(
            severity=ErrorSeverity.CRITICAL,
            user_message=f"Error processing request: {exc}",
            metadata={"technical_details": str(exc)}
        )
```

### Helper Methods Reference

| Method | Purpose | Example |
|--------|---------|---------|
| `self.get_logger()` | Get unified logger with streaming | `logger.status("Working...")` |
| `self.get_task_objective()` | Get current task from state | `task = self.get_task_objective()` |
| `self.get_parameters()` | Get step parameters | `params = self.get_parameters()` |
| `self.get_required_contexts()` | Auto-extract contexts by requires field | `contexts = self.get_required_contexts()` |
| `self.store_output_context(ctx)` | Store single output context | `return self.store_output_context(ctx)` |
| `self.store_output_contexts(*ctxs)` | Store multiple contexts | `return self.store_output_contexts(a, b)` |

### Logger Usage

```python
logger = self.get_logger()

# Status updates (CLI + Web UI)
logger.status("Phase 1: Fetching data...")

# Detailed logging (CLI only)
logger.info(f"Processing {count} items")
logger.debug(f"Raw response: {response}")

# Completion messages (CLI + Web UI)
logger.success("Operation completed!")
logger.error("Operation failed")
logger.warning("Partial results returned")
```

### Capabilities with Dependencies

When your capability depends on output from another:

```python
@capability_node
class ProcessingCapability(BaseCapability):
    name = "processor"
    description = "Process fetched data"
    provides = ["PROCESSED_DATA"]
    requires = ["RAW_DATA"]  # Must have RAW_DATA from another capability

    async def execute(self) -> Dict[str, Any]:
        logger = self.get_logger()

        # Auto-extract required contexts
        contexts = self.get_required_contexts()
        raw_data = contexts["RAW_DATA"]  # Dict access

        # Or tuple unpacking (order matches requires)
        # raw_data, = self.get_required_contexts()

        # Process the input
        processed = transform(raw_data)

        context = ProcessedDataContext(data=processed)
        return self.store_output_context(context)
```

### Cardinality Constraints

Specify how many context instances are needed:

```python
requires = [
    "REQUIRED_DATA",                # Any number of instances
    ("TIME_RANGE", "single"),       # Exactly one instance
    ("CHANNEL", "multiple")         # One or more instances
]
```

---

## Phase 4: Registry Configuration

**Goal**: Register your capability and context class for framework discovery.

### Read the Template

```
src/osprey/templates/apps/minimal/registry.py.j2
```

### Registration Structure

```python
from osprey.registry import (
    RegistryConfigProvider,
    RegistryConfig,
    extend_framework_registry,  # Use this to include framework capabilities
    CapabilityRegistration,
    ContextClassRegistration
)

class YourAppRegistryProvider(RegistryConfigProvider):
    """Registry provider for your application."""

    def get_registry_config(self) -> RegistryConfig:
        return extend_framework_registry(
            capabilities=[
                CapabilityRegistration(
                    name="your_capability",  # Must match capability class name attribute
                    module_path="your_app.capabilities.your_capability",
                    class_name="YourCapability",
                    description="Process data and return structured results",
                    provides=["YOUR_DATA"],  # Must match capability class
                    requires=[]  # Must match capability class
                ),
            ],

            context_classes=[
                ContextClassRegistration(
                    context_type="YOUR_DATA",  # Must match CONTEXT_TYPE
                    module_path="your_app.context_classes",
                    class_name="YourDataContext",
                    description="Structured output from your capability"
                ),
            ],

            # Optional: exclude framework capabilities you don't need
            # exclude_capabilities=["python"],

            # Optional: override framework capabilities
            # override_capabilities=[...],
        )
```

### Registration Checklist

- [ ] `name` in CapabilityRegistration matches capability class `name` attribute
- [ ] `module_path` is correct import path to capability file
- [ ] `class_name` matches capability class name exactly
- [ ] `provides` matches capability class `provides` list
- [ ] `requires` matches capability class `requires` list
- [ ] `context_type` in ContextClassRegistration matches `CONTEXT_TYPE`
- [ ] Context class `module_path` and `class_name` are correct

### Framework Capabilities

Using `extend_framework_registry()` automatically includes these framework capabilities:

| Capability | Purpose |
|------------|---------|
| `routing` | Route tasks to appropriate capabilities |
| `memory` | Short and long-term memory storage |
| `python` | Execute Python code for analysis |
| `time_range_parser` | Parse time expressions ("last week") |
| `classifier` | Classify user tasks |
| `user_approval` | Request human approval for sensitive operations |

---

## Phase 5: Testing and Validation

**Goal**: Verify your capability works correctly.

### Unit Test Structure

```python
"""tests/capabilities/test_your_capability.py"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from your_app.capabilities.your_capability import YourCapability
from your_app.context_classes import YourDataContext

class TestYourCapability:
    """Unit tests for YourCapability."""

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful execution."""
        # Arrange
        capability = YourCapability()
        capability._state = create_test_state("process this data")
        capability._step = {
            'context_key': 'test_key',
            'task_objective': 'process this data',
            'parameters': {}
        }

        # Mock external service
        with patch.object(capability, '_do_work', new_callable=AsyncMock) as mock_work:
            mock_work.return_value = {"value": "result", "count": 5}

            # Act
            result = await capability.execute()

            # Assert
            assert "capability_context_data" in result
            assert "YOUR_DATA" in str(result)

    def test_context_class_structure(self):
        """Test context class creates valid structure."""
        context = YourDataContext(
            primary_field="test",
            secondary_field=10
        )

        assert context.CONTEXT_TYPE == "YOUR_DATA"

        summary = context.get_summary()
        assert "primary" in summary

        details = context.get_access_details("test_key")
        assert "access_pattern" in details

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error classification."""
        capability = YourCapability()

        # Test retriable error
        error = ConnectionError("timeout")
        classification = capability.classify_error(error, {})

        assert classification.severity == ErrorSeverity.RETRIABLE
```

### Integration Testing

```python
@pytest.mark.asyncio
async def test_capability_with_registry(mock_registry):
    """Test capability is discoverable through registry."""
    from osprey.registry import get_registry

    registry = await get_registry()

    assert "your_capability" in registry.capabilities
    assert "YOUR_DATA" in registry.context_classes
```

### Manual Testing with Chat

```bash
# Start chat interface
osprey chat

# Test your capability with relevant queries
> [query that should trigger your capability]
```

### Testing Checklist

- [ ] Unit tests for execute() success path
- [ ] Unit tests for execute() error paths
- [ ] Context class structure validation
- [ ] Error classification tests
- [ ] Registry integration test
- [ ] Manual chat testing with real queries
- [ ] Verify correct capability is triggered

---

## Common Patterns

### Pattern 1: API Integration Capability

```python
@capability_node
class ExternalAPICapability(BaseCapability):
    name = "external_api"
    description = "Fetch data from external API"
    provides = ["API_RESPONSE"]
    requires = []

    async def execute(self) -> Dict[str, Any]:
        logger = self.get_logger()

        task = self.get_task_objective()
        logger.status("Calling external API...")

        # Call API
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{API_URL}?q={task}") as response:
                data = await response.json()

        context = APIResponseContext(
            results=data["results"],
            count=len(data["results"])
        )

        return self.store_output_context(context)

    @staticmethod
    def classify_error(exc, context):
        if isinstance(exc, aiohttp.ClientError):
            return ErrorClassification(
                severity=ErrorSeverity.RETRIABLE,
                user_message="API temporarily unavailable"
            )
        return ErrorClassification(
            severity=ErrorSeverity.CRITICAL,
            user_message=f"API error: {exc}"
        )
```

### Pattern 2: Data Processing Capability

```python
@capability_node
class DataProcessorCapability(BaseCapability):
    name = "data_processor"
    description = "Process and analyze data"
    provides = ["PROCESSED_DATA"]
    requires = ["RAW_DATA"]

    async def execute(self) -> Dict[str, Any]:
        logger = self.get_logger()

        # Get input context
        contexts = self.get_required_contexts()
        raw_data = contexts["RAW_DATA"]

        logger.status("Processing data...")

        # Transform data
        processed = self._transform(raw_data)

        context = ProcessedDataContext(
            summary=processed["summary"],
            statistics=processed["stats"]
        )

        return self.store_output_context(context)
```

### Pattern 3: LLM-Assisted Capability

```python
@capability_node
class LLMAssistedCapability(BaseCapability):
    name = "llm_assisted"
    description = "Use LLM for complex reasoning"
    provides = ["REASONING_OUTPUT"]
    requires = []

    async def execute(self) -> Dict[str, Any]:
        logger = self.get_logger()

        task = self.get_task_objective()
        logger.status("Analyzing with LLM...")

        # Use framework LLM utilities
        from osprey.models import get_chat_completion

        response = await get_chat_completion(
            message=f"Analyze: {task}",
            output_model=AnalysisOutput  # Pydantic model for structured output
        )

        context = ReasoningContext(
            analysis=response.analysis,
            confidence=response.confidence
        )

        return self.store_output_context(context)
```

### Pattern 4: Multi-Output Capability

```python
@capability_node
class MultiOutputCapability(BaseCapability):
    name = "multi_output"
    description = "Produce multiple context types"
    provides = ["PRIMARY_DATA", "METADATA"]
    requires = []

    async def execute(self) -> Dict[str, Any]:
        logger = self.get_logger()

        # ... do work ...

        primary = PrimaryDataContext(data=results)
        metadata = MetadataContext(info=meta)

        # Store multiple contexts
        return self.store_output_contexts(primary, metadata)
```

---

## Anti-Patterns: Common Mistakes

### ❌ Don't: Create God Capabilities

```python
# BAD: One capability that does everything
@capability_node
class DoEverythingCapability(BaseCapability):
    name = "do_everything"
    provides = ["WEATHER", "NEWS", "STOCKS", "CALENDAR"]  # Too much!
```

```python
# GOOD: Focused single-responsibility capabilities
@capability_node
class WeatherCapability(BaseCapability):
    name = "weather"
    provides = ["WEATHER_DATA"]
```

### ❌ Don't: Skip Context Classes

```python
# BAD: Return raw dict without context class
async def execute(self):
    return {"raw": "data"}  # No type safety, LLM can't understand structure
```

```python
# GOOD: Use typed context class
async def execute(self):
    context = WeatherContext(temperature=72, conditions="sunny")
    return self.store_output_context(context)
```

### ❌ Don't: Hardcode Configuration

```python
# BAD: Hardcoded values
API_URL = "https://api.example.com"
API_KEY = "abc123"  # Never do this!
```

```python
# GOOD: Use configuration
from osprey.config import get_full_configuration

config = get_full_configuration()
api_url = config.get("external_api", {}).get("url")
```

### ❌ Don't: Ignore Error Handling

```python
# BAD: No error classification
@capability_node
class UnsafeCapability(BaseCapability):
    async def execute(self):
        response = requests.get(url)  # Could fail in many ways
        return process(response)
```

```python
# GOOD: Comprehensive error handling
@capability_node
class SafeCapability(BaseCapability):
    async def execute(self):
        try:
            response = await fetch(url)
            return self.store_output_context(process(response))
        except Exception as e:
            self.get_logger().error(f"Error: {e}")
            raise

    @staticmethod
    def classify_error(exc, context):
        # Map exceptions to recovery strategies
        ...
```

### ❌ Don't: Misalign Names

```python
# BAD: Names don't match
class WeatherCapability(BaseCapability):
    name = "get_weather"  # Different from class name pattern
    provides = ["WEATHER"]  # Different from context CONTEXT_TYPE

class WeatherDataContext(CapabilityContext):
    CONTEXT_TYPE = "WEATHER_DATA"  # Doesn't match provides!
```

```python
# GOOD: Consistent naming
class WeatherCapability(BaseCapability):
    name = "weather"
    provides = ["WEATHER_DATA"]

class WeatherDataContext(CapabilityContext):
    CONTEXT_TYPE = "WEATHER_DATA"  # Matches provides!
```

---

## Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| "Capability not found" | Name mismatch in registry | Ensure `name` matches exactly |
| "Context type not found" | Missing context registration | Add ContextClassRegistration |
| "Module not found" | Wrong module_path | Check import path is correct |
| "LLM doesn't trigger capability" | Poor description | Improve description, add classifier guide |
| "Context serialization error" | Non-JSON types | Use ISO strings for datetime, simple types |
| "Circular dependency" | provides/requires loop | Check dependency graph |

### Debug Commands

```bash
# Check registry loaded correctly
osprey health

# View registered capabilities
osprey config show

# Test with verbose logging
osprey chat --verbose
```

---

## Code Reference Guide

**Templates to Read:**
```
src/osprey/templates/apps/minimal/capabilities/example_capability.py.j2  (435 lines - comprehensive)
src/osprey/templates/apps/minimal/context_classes.py.j2
src/osprey/templates/apps/minimal/registry.py.j2
```

**Working Examples:**
```
src/osprey/templates/apps/hello_world_weather/  (complete weather example)
src/osprey/templates/apps/control_assistant/    (advanced control system)
```

**Framework Capabilities (for reference):**
```
src/osprey/capabilities/memory/memory_capability.py
src/osprey/capabilities/python/python_capability.py
src/osprey/capabilities/time_range_parsing/time_range_parsing_capability.py
```

**Documentation:**
```
docs/source/developer-guides/02_quick-start-patterns/01_building-your-first-capability.rst
docs/source/developer-guides/02_quick-start-patterns/02_state-and-context-essentials.rst
docs/source/developer-guides/03_core-framework-systems/02_context-management-system.rst
```

---

## Decision Checklist

### Before Implementation

- [ ] I have clear requirements documented
- [ ] I know what external services/APIs are involved
- [ ] I understand the input/output data structures
- [ ] I've identified dependencies on other capabilities
- [ ] I've read the template files

### After Implementation

- [ ] Context class has CONTEXT_TYPE, get_access_details(), get_summary()
- [ ] Capability has @capability_node decorator
- [ ] name, description, provides, requires are defined
- [ ] execute() uses helper methods properly
- [ ] Error handling with classify_error() is implemented
- [ ] Registry entries are correct and names match
- [ ] Unit tests cover success and error paths
- [ ] Manual chat testing works

---

## Summary

Creating a capability involves:

1. **Gather Requirements** - Understand what you're building before coding
2. **Design Context Class** - Define output data structure with proper methods
3. **Implement Capability** - Write business logic using framework patterns
4. **Configure Registry** - Register for framework discovery
5. **Test Thoroughly** - Unit tests, integration tests, manual testing

**Remember**: Capabilities should be focused, well-documented, and follow the framework conventions. When in doubt, read the templates and existing examples.

---

## Additional Resources

- **Testing Workflow**: `testing-workflow.md` - Comprehensive testing guidance
- **AI Code Review**: `ai-code-review.md` - Review AI-generated capability code
- **Docstrings**: `docstrings.md` - Documentation standards
