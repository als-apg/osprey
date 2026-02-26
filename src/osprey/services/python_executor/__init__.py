"""Python Executor Service - Python Code Generation and Execution Framework.

This package provides a service for Python code generation, static analysis,
approval workflows, and secure execution with flexible deployment options.

## **Key Features**

### **1. Flexible Execution Environments**
Switch between containerized and local execution with a single configuration line:
```yaml
osprey:
  execution:
    execution_method: "container"  # or "local" - that's it!
```
- **Container Execution**: Secure, isolated Jupyter environments with full dependency management
- **Local Execution**: Direct host execution with automatic Python environment detection
- **Seamless Switching**: Same code, same results, different isolation levels

### **2. Comprehensive Jupyter Notebook Generation**
Automatic creation of rich, interactive notebooks for human evaluation and review:
- **Multi-Stage Notebooks**: Generated at code creation, analysis, and execution stages
- **Rich Metadata**: Complete execution context, analysis results, and error information
- **Direct Jupyter Access**: Click-to-open URLs for immediate notebook review
- **Audit Trails**: Complete history of execution attempts with detailed context
- **Figure Integration**: Automatic embedding of generated plots and visualizations

### **3. Human-in-the-Loop Approval System**
Production-ready approval workflows for high-stakes scientific and industrial environments:
- **Security Analysis Integration**: Automatic detection of potentially dangerous operations
- **Rich Approval Context**: Detailed safety assessments, code analysis, and execution plans
- **Configurable Policies**: Domain-specific approval rules for different operation types

## Architecture Overview

The service implements a multi-stage pipeline with clean exception-based
architecture:

1. **Code Generation**: LLM-based Python code generation with context awareness
2. **Static Analysis**: Security and policy analysis with configurable domain-specific rules
3. **Approval Workflows**: Human oversight system with rich context and safety assessments
4. **Flexible Execution**: Container or local execution with unified result collection
5. **Notebook Generation**: Jupyter notebook creation for human evaluation
6. **Result Processing**: Structured result handling with artifact management and audit trails

Core Components:
    - :class:`PythonExecutionRequest`: Type-safe execution request with context data
    - :class:`FileManager`: File operations and execution folder management
    - :class:`NotebookManager`: Jupyter notebook creation and management
    - :class:`ContainerExecutor`: Secure container-based Python execution engine

Exception Hierarchy:
    The package provides a comprehensive exception system organized by error category:

    - **Infrastructure Errors**: Container connectivity and configuration issues
        - :exc:`ContainerConnectivityError`: Container unreachable or connection failed
        - :exc:`ContainerConfigurationError`: Invalid container configuration

    - **Code-Related Errors**: Issues requiring code regeneration
        - :exc:`CodeGenerationError`: LLM failed to generate valid code
        - :exc:`CodeSyntaxError`: Generated code has syntax errors
        - :exc:`CodeRuntimeError`: Code execution failed with runtime errors

    - **Workflow Errors**: Service workflow and control issues
        - :exc:`ExecutionTimeoutError`: Code execution exceeded timeout limits
        - :exc:`MaxAttemptsExceededError`: Exceeded maximum retry attempts
        - :exc:`WorkflowError`: General workflow control errors

Configuration System:
    The service integrates with the framework's configuration system and
    supports multiple execution environments with configurable security policies.
    Execution modes range from read-only safe environments to write-enabled
    environments for EPICS control operations.

Security Features:
    - Static code analysis with security pattern detection
    - Configurable execution policies with domain-specific rules
    - Container-based execution isolation
    - Approval workflows for sensitive operations
    - Comprehensive audit logging and execution tracking

.. note::
   This service requires Docker containers for secure code execution. Container
   endpoints must be configured in the application configuration.

.. warning::
   Python code execution can perform system operations depending on the configured
   execution mode. Always review approval policies before enabling write access.

## Configuration Examples

### **Execution Environment Configuration**
```yaml
# Container execution (default) - maximum security and isolation
osprey:
  execution:
    execution_method: "container"
    modes:
      read_only:
        kernel_name: "python3-epics-readonly"
        allows_writes: false
      write_access:
        kernel_name: "python3-epics-write"
        allows_writes: true
        requires_approval: true

# Local execution - direct host execution for development
osprey:
  execution:
    execution_method: "local"
    python_env_path: "${LOCAL_PYTHON_VENV}"  # Optional: specific Python environment
```

### **Approval Workflow Configuration**
```yaml
# High-stakes scientific environment with strict approvals
agent_control_defaults:
  epics_writes_enabled: true  # Enable write operations with approval

osprey:
  execution:
    modes:
      write_access:
        requires_approval: true  # Force human approval for write operations
```
"""

# Import from restructured subsystems
from .analysis import (
    detect_control_system_operations,
    get_framework_standard_patterns,
)
from .exceptions import (  # Code errors (retry code generation); Infrastructure errors (retry execution); Workflow errors (special handling); Base
    ChannelLimitsViolationError,
    CodeGenerationError,
    CodeRuntimeError,
    CodeSyntaxError,
    ContainerConfigurationError,
    ContainerConnectivityError,
    ErrorCategory,
    ExecutionTimeoutError,
    MaxAttemptsExceededError,
    PythonExecutorException,
    WorkflowError,
)
from .execution.control import ExecutionControlConfig, ExecutionMode, get_execution_control_config
from .generation import (
    CLAUDE_SDK_AVAILABLE,
    BasicLLMCodeGenerator,
    ClaudeCodeGenerator,
    CodeGenerator,
    MockCodeGenerator,
    create_code_generator,
)
from .models import (
    ContainerEndpointConfig,
    ExecutionModeConfig,
    NotebookAttempt,
    NotebookType,
    PythonExecutionContext,
    PythonExecutionRequest,
    PythonExecutionState,
    PythonExecutionSuccess,
    PythonServiceResult,
)
from .services import (
    FileManager,
    NotebookManager,
    make_json_serializable,
    serialize_results_to_file,
)

__all__ = [
    # Core types
    "PythonExecutionRequest",
    "PythonExecutionSuccess",
    "PythonExecutionState",
    "PythonServiceResult",
    # Code generator interfaces
    "CodeGenerator",
    "BasicLLMCodeGenerator",
    "ClaudeCodeGenerator",  # Optional - requires claude-agent-sdk
    "MockCodeGenerator",  # For testing - no external dependencies
    "CLAUDE_SDK_AVAILABLE",
    "create_code_generator",
    # Note: Generator registration now via registry system (see osprey.registry.base.CodeGeneratorRegistration)
    # Analysis utilities
    "detect_control_system_operations",
    "get_framework_standard_patterns",
    # Execution context and notebook management
    "NotebookAttempt",
    "NotebookType",
    "PythonExecutionContext",
    "FileManager",
    "NotebookManager",
    # Configuration utilities
    "ExecutionModeConfig",
    "ContainerEndpointConfig",
    "ExecutionMode",
    "ExecutionControlConfig",
    "get_execution_control_config",
    # Exception hierarchy
    "PythonExecutorException",
    "ErrorCategory",
    "ContainerConnectivityError",
    "ContainerConfigurationError",
    "CodeGenerationError",
    "CodeSyntaxError",
    "CodeRuntimeError",
    "ChannelLimitsViolationError",
    "ExecutionTimeoutError",
    "MaxAttemptsExceededError",
    "WorkflowError",
    # Serialization utilities
    "make_json_serializable",
    "serialize_results_to_file",
]
