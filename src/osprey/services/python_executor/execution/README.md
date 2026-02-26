# Code Execution Subsystem

This subsystem handles secure execution of Python code in isolated environments, supporting both container-based and local execution with unified interfaces.

## Components

### LangGraph Node
- **`node.py`**: Executor LangGraph node
  - Orchestrates execution process
  - Selects execution engine (container vs local)
  - Handles execution retries
  - Creates execution notebooks
  - Collects and validates results

### Execution Engines
- **`container_engine.py`**: Container-based execution
  - Jupyter container integration
  - WebSocket communication
  - Session management
  - Isolated environment execution
  - Full dependency support

### Execution Infrastructure
- **`wrapper.py`**: Unified execution wrapper
  - Provides 'context' object for generated code
  - Handles result collection
  - Manages execution environment
  - Works with both container and local execution

- **`control.py`**: Execution mode configuration
  - `ExecutionMode` enum (READ_ONLY, WRITE_ENABLED, EPICS_CONTROL)
  - `ExecutionControlConfig`: Configuration management
  - Mode validation and selection

## Execution Flow

```
Code + Analysis Result
         ↓
┌─────────────────────────┐
│  Select Engine          │
│  - container (default)  │
│  - local (development)  │
└───────────┬─────────────┘
            ↓
    ┌───────┴───────┐
    │               │
┌───▼────────┐  ┌──▼──────────┐
│ Container  │  │   Local     │
│ Executor   │  │  Executor   │
└─────┬──────┘  └──┬──────────┘
      │            │
      └────────┬───┘
               ↓
     ┌─────────────────┐
     │ Execution       │
     │ Wrapper         │
     │ - context obj   │
     │ - result collect│
     └────────┬────────┘
              ↓
     Execution Result
     (results dict + metadata)
```

## Usage

### Via LangGraph Node
```python
from osprey.services.python_executor.execution import create_executor_node

# Create node
executor_node = create_executor_node()

# Add to graph
graph.add_node("executor", executor_node)
```

### Container Execution (Direct)
```python
from osprey.services.python_executor.execution.container_engine import ContainerExecutor

# Create executor
executor = ContainerExecutor(endpoint_config, configurable)

# Execute code
result = await executor.execute_code(
    code=python_code,
    execution_mode=ExecutionMode.READ_ONLY,
    execution_folder=folder_path,
    context_data={"data": some_data}
)

# Access results
print(result.results)  # User's results dict
print(result.stdout)   # Standard output
print(result.notebook_path)  # Generated notebook
```

### Local Execution
```python
# In config.yml
osprey:
  execution:
    execution_method: "local"  # Instead of "container"

# Code automatically executes locally
# Same interface, different isolation level
```

## Execution Modes

### READ_ONLY (Default)
- No file writes
- No subprocess calls
- Safe for untrusted code
- Fast execution

### WRITE_ENABLED
- File writes allowed
- Data persistence enabled
- Requires approval (typically)
- Use for data analysis

### EPICS_CONTROL
- EPICS control operations allowed
- Can modify accelerator setpoints
- Requires approval (always)
- Use for control operations

## Configuration

### Container Execution
```yaml
osprey:
  execution:
    execution_method: "container"
    container_endpoints:
      default:
        host: "localhost"
        port: 8888
        token: "your-token"
```

### Local Execution
```yaml
osprey:
  execution:
    execution_method: "local"
    # No container configuration needed
```

## Testing

```bash
# Test execution subsystem
pytest tests/services/python_executor/execution/

# Test container engine
pytest tests/services/python_executor/execution/test_container_engine.py

# Test with local execution
EXECUTION_METHOD=local pytest tests/services/python_executor/execution/
```

## See Also

- [Execution Node](node.py) - Main executor implementation
- [Container Engine](container_engine.py) - Container execution
- [Execution Wrapper](wrapper.py) - Unified execution interface
- [Execution Control](control.py) - Mode configuration
- [Parent Module](../__init__.py) - Main Python executor exports
