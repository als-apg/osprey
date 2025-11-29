"""Mock Code Generator for Testing.

Provides a deterministic, fast mock implementation of the CodeGenerator protocol for
testing the Python executor service without external dependencies or API calls.

Key Features:
    - Instant code generation (no API calls)
    - Deterministic output (same input → same output)
    - Configurable behaviors (success, errors, EPICS operations, etc.)
    - Protocol compliant (drop-in replacement for real generators)

Usage:
    >>> generator = MockCodeGenerator(behavior="success")
    >>> code = await generator.generate_code(request, [])

    # Custom code
    >>> generator.set_code("results = {'value': 42}")

    # Code sequence for retry testing
    >>> generator.set_code_sequence([
    ...     "results = 1 / 0",           # Fails first time
    ...     "results = {'value': 42}"    # Succeeds on retry
    ... ])

Behaviors:
    - "success": Valid code that executes successfully
    - "syntax_error": Code with syntax error
    - "runtime_error": Code that fails at runtime
    - "missing_results": Code without results dictionary
    - "epics_write": Code with EPICS write operations
    - "epics_read": Code with EPICS read operations
    - "security_risk": Code with security concerns
    - "error_aware": Adapts code based on error feedback

.. note::
   For testing only. Do not use in production.
"""

from typing import Any

from ..models import PythonExecutionRequest, ExecutionError


class MockCodeGenerator:
    """Mock code generator for testing.

    Provides configurable mock code generation for testing the Python executor
    service without API calls or external dependencies.

    Operation Modes:
        - **Static**: Returns same code every time
        - **Sequence**: Returns different code on successive calls
        - **Behavior**: Uses predefined scenarios (success, errors, EPICS, etc.)
        - **Error-Aware**: Adapts code based on error feedback

    Args:
        model_config: Optional config dict (for protocol compatibility)
        behavior: Predefined behavior ("success", "syntax_error", etc.)

    Attributes:
        call_count: Number of generate_code calls
        last_request: Last PythonExecutionRequest received
        last_error_chain: Last error_chain received

    Examples:
        >>> generator = MockCodeGenerator(behavior="success")
        >>> code = await generator.generate_code(request, [])

        >>> generator.set_code_sequence([
        ...     "results = 1 / 0",           # First fails
        ...     "results = {'value': 42}"    # Retry succeeds
        ... ])
    """

    def __init__(self, model_config: dict[str, Any] | None = None, behavior: str | None = None):
        """Initialize mock generator.

        Args:
            model_config: Optional config dict (protocol compatibility)
            behavior: Predefined behavior ("success", "syntax_error", etc.)
        """
        self.model_config = model_config or {}
        self.behavior = behavior

        # Code generation state
        self.static_code: str | None = None
        self.code_sequence: list[str] = []
        self.call_count: int = 0

        # Call tracking for test assertions
        self.last_request: PythonExecutionRequest | None = None
        self.last_error_chain: list[ExecutionError] = []

        # Initialize with behavior if specified
        if behavior:
            self._apply_behavior(behavior)

    async def generate_code(
        self,
        request: PythonExecutionRequest,
        error_chain: list[ExecutionError]
    ) -> str:
        """Generate Python code based on configured behavior.

        Args:
            request: Execution request with task details
            error_chain: Previous errors from failed attempts

        Returns:
            Generated Python code string

        Raises:
            ValueError: If generator not configured with code or behavior

        Note:
            Tracks all calls in call_count, last_request, and last_error_chain
            for test assertions.
        """
        # Track call for test assertions
        self.call_count += 1
        self.last_request = request
        self.last_error_chain = error_chain  # Store as ExecutionError objects

        # Sequence mode: return next code in sequence
        if self.code_sequence:
            # Use call_count - 1 as index (0-based)
            idx = min(self.call_count - 1, len(self.code_sequence) - 1)
            return self.code_sequence[idx]

        # Static mode: return configured code
        if self.static_code is not None:
            return self.static_code

        # Error-aware mode: generate code based on errors
        if self.behavior == "error_aware":
            return self._generate_error_aware_code(request, error_chain)

        # No code configured
        raise ValueError(
            "MockCodeGenerator not configured. Use set_code(), set_code_sequence(), "
            "or specify a behavior in constructor."
        )

    def set_code(self, code: str) -> None:
        """Set static code to return on all calls.

        Args:
            code: Python code string to return
        """
        self.static_code = code
        self.code_sequence = []  # Clear sequence

    def set_code_sequence(self, code_list: list[str]) -> None:
        """Set sequence of codes to return on successive calls.

        Args:
            code_list: List of Python code strings
        """
        self.code_sequence = code_list
        self.static_code = None  # Clear static

    def reset(self) -> None:
        """Reset call tracking state (preserves code configuration)."""
        self.call_count = 0
        self.last_request = None
        self.last_error_chain = []

    def get_generation_metadata(self) -> dict[str, Any]:
        """Get metadata from the last code generation.

        Returns:
            Empty dict (mock generator provides no metadata)
        """
        return {}

    def _apply_behavior(self, behavior: str) -> None:
        """Apply a predefined behavior pattern."""
        behaviors = {
            "success": self._behavior_success,
            "syntax_error": self._behavior_syntax_error,
            "runtime_error": self._behavior_runtime_error,
            "missing_results": self._behavior_missing_results,
            "epics_write": self._behavior_epics_write,
            "epics_read": self._behavior_epics_read,
            "error_aware": None,  # Handled in generate_code
            "security_risk": self._behavior_security_risk,
        }

        if behavior not in behaviors:
            raise ValueError(
                f"Unknown behavior: {behavior}. "
                f"Available: {', '.join(behaviors.keys())}"
            )

        # Apply behavior if it has a generator function
        behavior_func = behaviors[behavior]
        if behavior_func:
            behavior_func()

    def _behavior_success(self) -> None:
        """Generate successful, valid Python code."""
        self.set_code("""
# Mock generated code - success path
import json
from datetime import datetime

result_value = 42
timestamp = datetime.now().isoformat()

results = {
    'value': result_value,
    'timestamp': timestamp,
    'status': 'success'
}
""".strip())

    def _behavior_syntax_error(self) -> None:
        """Generate code with syntax error."""
        self.set_code("""
# Mock generated code - syntax error
def broken_function(
    # Missing closing parenthesis causes syntax error

results = {'value': 42}
""".strip())

    def _behavior_runtime_error(self) -> None:
        """Generate code that fails at runtime."""
        self.set_code("""
# Mock generated code - runtime error
import json

# ZeroDivisionError
value = 100 / 0

results = {'value': value}
""".strip())

    def _behavior_missing_results(self) -> None:
        """Generate valid code without results dictionary."""
        self.set_code("""
# Mock generated code - missing results
import json

value = 42 * 2

# Forgot to create results dictionary
print(f"Value is {value}")
""".strip())

    def _behavior_epics_write(self) -> None:
        """Generate code with EPICS write operations."""
        self.set_code("""
# Mock generated code - EPICS write operation
from epics import caget, caput

current_value = caget('DEVICE:PV:SETPOINT')

# Write new value (requires approval!)
new_value = current_value * 1.1
caput('DEVICE:PV:SETPOINT', new_value)

results = {
    'old_value': current_value,
    'new_value': new_value,
    'pv': 'DEVICE:PV:SETPOINT'
}
""".strip())

    def _behavior_epics_read(self) -> None:
        """Generate code with EPICS read operations only."""
        self.set_code("""
# Mock generated code - EPICS read operation
from epics import caget

pv_value = caget('DEVICE:PV:READBACK')
pv_status = caget('DEVICE:STATUS')

results = {
    'value': pv_value,
    'status': pv_status,
    'operation': 'read_only'
}
""".strip())

    def _behavior_security_risk(self) -> None:
        """Generate code with security concerns."""
        self.set_code("""
# Mock generated code - security risk
import subprocess
import os

# Subprocess call (security risk!)
output = subprocess.run(['ls', '-la'], capture_output=True)

# File system access
files = os.listdir('/')

results = {
    'output': output.stdout.decode(),
    'files': files
}
""".strip())

    def _generate_error_aware_code(
        self,
        request: PythonExecutionRequest,
        error_chain: list[ExecutionError]
    ) -> str:
        """Generate code that adapts based on error feedback."""
        # First attempt
        if not error_chain:
            return """
results = {'values': [1, 2, 3]}
mean_value = sum(results['values']) / len(results['values'])
results['mean'] = mean_value
""".strip()

        # Adapt based on error type - extract error messages from ExecutionError objects
        errors_str = ' '.join(err.error_message for err in error_chain).lower()

        if 'nameerror' in errors_str or 'import' in errors_str:
            return """
import json
from statistics import mean

values = [1, 2, 3]
results = {
    'values': values,
    'mean': mean(values),
    'count': len(values)
}
""".strip()

        if 'zerodivisionerror' in errors_str:
            return """
import json

values = [1, 2, 3]
count = len(values)

if count > 0:
    mean_value = sum(values) / count
else:
    mean_value = 0

results = {
    'values': values,
    'mean': mean_value
}
""".strip()

        if 'syntax' in errors_str:
            return """
import json

def calculate_mean(values):
    if not values:
        return 0
    return sum(values) / len(values)

results = {
    'values': [1, 2, 3],
    'mean': calculate_mean([1, 2, 3])
}
""".strip()

        # Generic fix
        return """
results = {
    'status': 'success',
    'value': 42,
    'message': 'Fixed after error'
}
""".strip()

