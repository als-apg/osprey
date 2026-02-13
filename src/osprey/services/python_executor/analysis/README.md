# Code Analysis Subsystem

This subsystem performs static analysis of generated Python code to detect security issues, determine execution policies, and ensure code safety before execution.

## Components

### LangGraph Node
- **`node.py`**: Analyzer LangGraph node
  - Performs syntax validation
  - Runs security pattern detection
  - Applies execution policy analysis
  - Creates analysis notebooks
  - Determines approval requirements

### Analysis Infrastructure
- **`policy_analyzer.py`**: Execution policy analysis
  - `ExecutionPolicyAnalyzer`: Determines execution mode (read-only vs write-enabled)
  - `BasicAnalysisResult`: Analysis result structure
  - `DomainAnalysisManager`: Domain-specific pattern detection
  - `ExecutionPolicyManager`: Combines domain and policy analysis

- **`pattern_detection.py`**: Security pattern detection
  - Detects file I/O operations
  - Identifies network operations
  - Finds subprocess calls
  - Detects EPICS control operations
  - AST-based analysis

## Analysis Flow

```
Generated Code
     ↓
┌────────────────────────┐
│  Syntax Validation     │ ← ast.parse()
└──────────┬─────────────┘
           ↓
┌────────────────────────┐
│  Pattern Detection     │ ← PatternDetector
│  - File operations     │
│  - Network calls       │
│  - Subprocess usage    │
│  - EPICS operations    │
└──────────┬─────────────┘
           ↓
┌────────────────────────┐
│  Policy Analysis       │ ← ExecutionPolicyAnalyzer
│  - Domain analysis     │
│  - Execution mode      │
│  - Approval decision   │
└──────────┬─────────────┘
           ↓
     Analysis Result
     (pass to approval or executor)
```

## Usage

### Via LangGraph Node
```python
from osprey.services.python_executor.analysis import create_analyzer_node

# Create node
analyzer_node = create_analyzer_node()

# Add to graph
graph.add_node("analyzer", analyzer_node)
```

### Direct Analysis
```python
from osprey.services.python_executor.analysis.policy_analyzer import get_execution_policy_analyzer

# Get analyzer
analyzer = get_execution_policy_analyzer(configurable)

# Analyze code
result = await analyzer.analyze_code(code, state)

# Check results
if result.requires_approval:
    # Request human approval
    pass
```

### Custom Policy Analyzer
```python
from osprey.services.python_executor.analysis.policy_analyzer import ExecutionPolicyAnalyzer

class MyPolicyAnalyzer(ExecutionPolicyAnalyzer):
    async def analyze_code(self, code, state):
        # Custom analysis logic
        return BasicAnalysisResult(
            recommended_execution_mode=ExecutionMode.READ_ONLY,
            requires_approval=False,
            reasoning="Safe code"
        )

# Register via registry system
```

## Security Patterns Detected

- **File I/O**: `open()`, `Path.write_text()`, etc.
- **Network**: `requests`, `urllib`, `socket`, etc.
- **Subprocess**: `subprocess`, `os.system()`, etc.
- **EPICS Control**: `caget()`, `caput()`, etc.
- **Database**: `pymongo`, `psycopg2`, etc.

## Execution Modes

- **READ_ONLY**: Safe, no write operations allowed
- **WRITE_ENABLED**: File writes allowed
- **EPICS_CONTROL**: EPICS control operations allowed

## See Also

- [Analysis Node](node.py) - Main analyzer implementation
- [Policy Analyzer](policy_analyzer.py) - Execution policy logic
- [Pattern Detection](pattern_detection.py) - Security pattern detection
- [Parent Module](../__init__.py) - Main Python executor exports
