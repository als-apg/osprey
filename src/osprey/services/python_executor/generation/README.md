# Code Generation Subsystem

This subsystem handles all Python code generation for the executor, providing a clean Protocol-based interface with multiple generator implementations and registry-based extensibility.

## Components

### Core Interface
- **`interface.py`**: `CodeGenerator` Protocol that all generators must implement
  - Single method: `async def generate_code(request, error_chain) -> str`
  - Runtime checkable with `@runtime_checkable`
  - Duck typing friendly - no inheritance required

### Factory Pattern
- **`factory.py`**: Registry-integrated factory for generator creation
  - Discovers generators from Osprey registry
  - Handles optional dependencies gracefully
  - Supports fallback to basic generator
  - Configuration-driven selection

### Generator Implementations
- **`basic_generator.py`**: Simple LLM-based generator
  - Uses direct API calls (OpenAI, Anthropic, etc.)
  - Fast and simple
  - Always available (no optional dependencies)

- **`claude_code_generator.py`**: Claude Code SDK-based generator
  - Multi-turn agentic reasoning
  - Codebase-aware (reads successful examples)
  - Two quality profiles: fast (single-phase, DEFAULT) / robust (multi-phase)
  - Included as core dependency (claude-agent-sdk)

- **`mock_generator.py`**: Mock generator for testing
  - Configurable behavior (success/syntax_error/runtime_error)
  - No external dependencies
  - Useful for unit tests

### LangGraph Integration
- **`node.py`**: LangGraph node function
  - Creates generator via factory
  - Handles streaming updates
  - Integrates with executor state

## Usage

### Basic Usage
```python
from osprey.services.python_executor.generation import create_code_generator

# Create generator (uses config from config.yml)
generator = create_code_generator()

# Generate code
code = await generator.generate_code(request, error_chain=[])
```

### Testing
```python
from osprey.services.python_executor.generation import MockCodeGenerator

# Mock successful generation
generator = MockCodeGenerator(behavior="success")
code = await generator.generate_code(request, [])

# Mock syntax error
generator = MockCodeGenerator(behavior="syntax_error")
try:
    code = await generator.generate_code(request, [])
except CodeGenerationError:
    # Handle error
    pass
```

### Custom Generator via Registry
```python
# In your application's registry.py
from osprey.registry import CodeGeneratorRegistration, RegistryConfig

class MyAppRegistryProvider(RegistryConfigProvider):
    def get_registry_config(self) -> RegistryConfig:
        return RegistryConfig(
            code_generators=[
                CodeGeneratorRegistration(
                    name="domain_specific",
                    module_path="myapp.generators.domain",
                    class_name="DomainGenerator",
                    description="Domain-specific code generator"
                )
            ]
        )

# Then in config.yml:
# code_generator: "domain_specific"
```

## Architecture

```
┌─────────────────────────────────────────┐
│       CodeGenerator Protocol            │
│  (interface.py)                         │
│  - async generate_code(request, errors) │
└────────────┬────────────────────────────┘
             │
    ┌────────┴──────────┬──────────────────┐
    │                   │                  │
┌───▼──────────┐  ┌────▼────────────┐  ┌──▼─────────┐
│  Legacy      │  │  Claude Code    │  │   Mock     │
│  Generator   │  │  Generator      │  │ Generator  │
│              │  │                 │  │            │
│ (Direct API) │  │ (SDK/Agentic)   │  │ (Testing)  │
└──────────────┘  └─────────────────┘  └────────────┘
         │                 │                  │
         └─────────────────┴──────────────────┘
                           │
                  ┌────────▼─────────┐
                  │  Generator Node  │
                  │  (LangGraph)     │
                  └──────────────────┘
```

## Configuration

### Quick Start (Legacy)
```yaml
osprey:
  execution:
    code_generator: "legacy"
    generators:
      legacy:
        model_config_name: "python_code_generator"
```

### Claude Code
```yaml
osprey:
  execution:
    code_generator: "claude_code"
    generators:
      claude_code:
        profile: "fast"  # fast (DEFAULT, single-phase) | robust (multi-phase)
        max_budget_usd: 0.50
```

### Custom Generator
```yaml
osprey:
  execution:
    code_generator: "my_custom_gen"
    generators:
      my_custom_gen:
        # Your generator's configuration
        custom_param: value
```

## Testing

```bash
# Test all generators
pytest tests/services/python_executor/generation/

# Test specific generator
pytest tests/services/python_executor/generation/test_basic_generator.py

# Test factory
pytest tests/services/python_executor/generation/test_factory.py
```

## Adding a New Generator

1. **Implement the Protocol**
   ```python
   class MyGenerator:
       async def generate_code(self, request, error_chain) -> str:
           # Your implementation
           return generated_code
   ```

2. **Register via Registry**
   ```python
   # In your registry.py
   CodeGeneratorRegistration(
       name="my_gen",
       module_path="myapp.generators",
       class_name="MyGenerator",
       description="My custom generator"
   )
   ```

3. **Configure**
   ```yaml
   # In config.yml
   code_generator: "my_gen"
   ```

That's it! The factory will discover and use your generator automatically.

## See Also

- [Code Generator Interface](interface.py) - Protocol definition
- [Factory Implementation](factory.py) - Generator creation logic
- [Basic Generator](basic_generator.py) - Simple reference implementation
- [Claude Code Generator](claude_code_generator.py) - Advanced implementation
- [Parent Module](../__init__.py) - Main Python executor exports
