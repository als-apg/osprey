# Python Executor Service

A comprehensive, LangGraph-based service for Python code generation, static analysis, approval workflows, and secure execution with flexible deployment options.

## Directory Structure

```
python_executor/
├── __init__.py                    # Main exports and public API
├── service.py                     # PythonExecutorService orchestrator
├── models.py                      # Shared data models and state
├── exceptions.py                  # Exception hierarchy
├── config.py                      # Configuration classes
├── services.py                    # File/notebook management utilities
│
├── generation/                    # Code generation subsystem
│   ├── README.md                 # Subsystem documentation
│   ├── __init__.py              # Subsystem exports
│   ├── interface.py             # CodeGenerator protocol
│   ├── factory.py               # Registry-based factory
│   ├── basic_generator.py       # Simple LLM-based generator
│   ├── claude_code_generator.py # Claude Code SDK generator
│   ├── mock_generator.py        # Mock for testing
│   └── node.py                  # LangGraph generator node
│
├── analysis/                      # Code analysis subsystem
│   ├── README.md
│   ├── __init__.py
│   ├── node.py                  # LangGraph analyzer node
│   ├── pattern_detection.py     # Security pattern detection
│   └── policy_analyzer.py       # Execution policy analysis
│
├── execution/                     # Code execution subsystem
│   ├── README.md
│   ├── __init__.py
│   ├── node.py                  # LangGraph executor node
│   ├── container_engine.py      # Container execution
│   ├── wrapper.py               # Execution wrapper utilities
│   └── control.py               # Execution mode configuration
│
└── approval/                      # Human approval subsystem
    ├── README.md
    ├── __init__.py
    └── node.py                  # LangGraph approval node
```

## Architecture

### Service Pipeline

```
User Request
     ↓
┌────────────────────┐
│   GENERATION       │  ← generation/
│  - Select strategy │
│  - Generate code   │
│  - Handle retries  │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│    ANALYSIS        │  ← analysis/
│  - Syntax check    │
│  - Security scan   │
│  - Policy decision │
└─────────┬──────────┘
          ↓
     ┌────┴────┐
     │ Approval │
     │ Required?│
     └────┬────┘
    Yes   │   No
     ↓    │    ↓
┌─────────────────┐
│   APPROVAL      │  ← approval/
│  - Human review │
│  - Interrupt    │
└─────────┬───────┘
          ↓
┌────────────────────┐
│   EXECUTION        │  ← execution/
│  - Container/Local │
│  - Run code        │
│  - Collect results │
└─────────┬──────────┘
          ↓
    Results + Notebook
```

## Quick Start

### Basic Usage
```python
from osprey.services.python_executor import PythonExecutorService

# Create service
service = PythonExecutorService()

# Create request
from osprey.services.python_executor import PythonExecutionRequest
request = PythonExecutionRequest(
    user_query="Calculate mean of sensor data",
    task_objective="Statistical analysis",
    execution_folder_name="sensor_stats"
)

# Execute
result = await service.ainvoke(request, config=config)

# Access results
print(result.execution_result.results)
print(result.execution_result.notebook_link)
```

### Subsystem Access
```python
# Code generation
from osprey.services.python_executor.generation import create_code_generator
generator = create_code_generator()
code = await generator.generate_code(request, [])

# Analysis
from osprey.services.python_executor.analysis import create_analyzer_node
analyzer = create_analyzer_node()

# Execution
from osprey.services.python_executor.execution import create_executor_node
executor = create_executor_node()

# Approval
from osprey.services.python_executor.approval import create_approval_node
approval = create_approval_node()
```

## Configuration

### Minimal Configuration
```yaml
osprey:
  execution:
    # Execution environment
    execution_method: "container"  # or "local"

    # Code generator
    code_generator: "legacy"  # or "claude_code"

    # Container endpoint (if using container execution)
    container_endpoints:
      default:
        host: "localhost"
        port: 8888
```

### Advanced Configuration
```yaml
osprey:
  execution:
    # Generator selection and config
    code_generator: "claude_code"
    generators:
      claude_code:
        profile: "fast"
        max_budget_usd: 0.50
      legacy:
        model_config_name: "python_code_generator"

    # Execution settings
    execution_method: "container"
    max_retries: 3
    timeout_seconds: 600

    # Approval settings
    approval:
      modes_requiring_approval:
        - "WRITE_ENABLED"
        - "EPICS_CONTROL"
```

## Subsystem Documentation

Each subsystem has its own README with detailed information:

- [**Generation**](generation/README.md) - Code generation strategies
- [**Analysis**](analysis/README.md) - Security and policy analysis
- [**Execution**](execution/README.md) - Container and local execution
- [**Approval**](approval/README.md) - Human-in-the-loop workflows

## Testing

### Run All Tests
```bash
pytest tests/services/python_executor/
```

### Test by Subsystem
```bash
# Code generation tests
pytest tests/services/python_executor/generation/

# Analysis tests
pytest tests/services/python_executor/analysis/

# Execution tests
pytest tests/services/python_executor/execution/

# Approval tests
pytest tests/services/python_executor/approval/
```

### Integration Tests
```bash
# Full pipeline test
pytest tests/services/python_executor/test_service.py

# End-to-end with real execution
pytest tests/services/python_executor/test_integration.py
```

## Extensibility

### Custom Code Generator
```python
# 1. Implement protocol in myapp/generators/custom.py
class CustomGenerator:
    async def generate_code(self, request, error_chain) -> str:
        # Your implementation
        return code

# 2. Register in myapp/registry.py
CodeGeneratorRegistration(
    name="custom",
    module_path="myapp.generators.custom",
    class_name="CustomGenerator",
    description="Custom generator"
)

# 3. Configure in config.yml
code_generator: "custom"
```

### Custom Policy Analyzer
```python
# Similar pattern - register via registry system
ExecutionPolicyAnalyzerRegistration(...)
```

## Migration from Old Structure

If you have code importing from the old flat structure:

```python
# Old imports (DEPRECATED - update your code)
# from osprey.services.python_executor.legacy_generator import LegacyLLMCodeGenerator  # DEPRECATED
# from osprey.services.python_executor.analyzer_node import create_analyzer_node  # DEPRECATED

# New imports (recommended)
from osprey.services.python_executor.generation import BasicLLMCodeGenerator
from osprey.services.python_executor.analysis import create_analyzer_node

# Best practice (use main package exports)
from osprey.services.python_executor import (
    BasicLLMCodeGenerator,  # Re-exported from generation
    create_code_generator    # Factory recommended
)
```

## Key Concepts

### Subsystem Independence
- Each subsystem has minimal coupling
- Clear interfaces between subsystems
- Independent testing possible
- Easy to understand and modify

### Registry Integration
- Code generators discovered via registry
- Extensible by applications
- No hardcoded generator lists
- Consistent with framework patterns

### Safety by Design
- Multiple validation layers
- Read-only by default
- Human approval for risky operations
- Comprehensive audit trails

## Performance

### Code Generation
- Legacy: ~2-5 seconds
- Claude Code (fast): ~5 seconds
- Claude Code (fast): ~20 seconds
- Claude Code (robust): ~30 seconds

### Execution
- Container: ~1-3 seconds (+ code runtime)
- Local: ~0.5-1 second (+ code runtime)

## See Also

- [Service Documentation](../../docs/source/developer-guides/python-executor-architecture.rst)
- [Claude Code Integration](../../docs/source/developer-guides/claude-code-integration.rst)
- [Main Package Exports](__init__.py)
- [LangGraph Service](service.py)
