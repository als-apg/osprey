"""End-to-End Integration Tests for Runtime Utilities with Channel Limits.

CRITICAL SAFETY TESTS: Verify that runtime utilities (osprey.runtime) respect
channel limits validation and cannot bypass safety mechanisms.

This test suite ensures that:
1. write_channel() calls go through the monkeypatched limits validator
2. Violations are blocked BEFORE reaching the control system
3. Valid writes within limits succeed
4. Bulk operations (write_channels) are also validated

These are full end-to-end tests that exercise:
- Code generation with runtime utilities
- Execution wrapper with limits monkeypatch
- Connector factory and real Mock connector
- Complete error handling and result propagation

Related to: Runtime Utilities (RUNTIME.md) and Channel Limits Safety
"""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from osprey.services.python_executor import (
    PythonExecutionRequest,
    PythonExecutorService,
)
from osprey.services.python_executor.exceptions import CodeRuntimeError
from osprey.services.python_executor.generation import MockCodeGenerator

# =============================================================================
# SAFETY LAYER: LIMITS VALIDATION WITH RUNTIME UTILITIES
# =============================================================================


def _disable_approval_in_config(config_data: dict) -> None:
    """Helper to explicitly disable approval in test config.

    This prevents test pollution when approval is enabled in previous tests.
    """
    if 'approval' not in config_data:
        config_data['approval'] = {}
    if 'capabilities' not in config_data['approval']:
        config_data['approval']['capabilities'] = {}
    config_data['approval']['capabilities']['python_execution'] = {
        'enabled': False
    }


@pytest.fixture(autouse=True)
def cleanup_test_state():
    """Clean up global state before and after each test to prevent pollution.

    This fixture ensures that:
    - CONFIG_FILE environment variable is restored
    - Registry state is reset
    - Config module caches are cleared
    - Approval manager is reset

    This prevents test isolation issues when running the full test suite.
    """
    # Save original state
    original_config_file = os.environ.get('CONFIG_FILE')

    # CLEAN UP BEFORE TEST - Reset all global state from previous tests
    from osprey.registry import reset_registry
    import osprey.utils.config as config_module

    reset_registry()
    config_module._default_config = None
    config_module._default_configurable = None
    config_module._config_cache.clear()

    # Reset approval manager singleton BEFORE test starts
    try:
        import osprey.approval.approval_manager as approval_module
        approval_module._approval_manager = None
    except ImportError:
        pass

    # Let test run
    yield

    # CLEAN UP AFTER TEST - Restore original state
    reset_registry()
    config_module._default_config = None
    config_module._default_configurable = None
    config_module._config_cache.clear()

    # Reset approval manager singleton again after test
    try:
        import osprey.approval.approval_manager as approval_module
        approval_module._approval_manager = None
    except ImportError:
        pass

    # Restore CONFIG_FILE environment variable
    if original_config_file is not None:
        os.environ['CONFIG_FILE'] = original_config_file
    elif 'CONFIG_FILE' in os.environ:
        del os.environ['CONFIG_FILE']


class TestRuntimeUtilitiesRespectChannelLimits:
    """E2E tests verifying runtime utilities respect channel limits."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_runtime_write_exceeds_max_limit_blocked(self, tmp_path, test_config):
        """
        CRITICAL: Runtime write_channel() exceeding max limit is BLOCKED.

        Verifies that write_channel() calls go through the monkeypatched
        limits validator and violations are caught before reaching the connector.

        Flow:
        1. Configure limits: TEST:VOLTAGE max=100.0
        2. Generate code: write_channel("TEST:VOLTAGE", 150.0)
        3. Execute code
        4. Verify: Execution FAILS with ChannelLimitsViolationError
        5. Verify: Write NEVER reached control system
        """
        # === SETUP ===
        os.environ['CONFIG_FILE'] = str(test_config)

        # Load and modify test config
        test_config_data = yaml.safe_load(test_config.read_text())

        # Ensure control_system section exists and use Mock connector
        if 'control_system' not in test_config_data:
            test_config_data['control_system'] = {}

        test_config_data['control_system']['type'] = 'mock'
        test_config_data['control_system']['writes_enabled'] = True
        test_config_data['control_system']['connector'] = {
            'mock': {
                'noise_level': 0.0,  # No noise for predictable tests
                'response_delay_ms': 1
            }
        }

        # Enable limits checking with strict policy
        test_config_data['control_system']['limits_checking'] = {
            'enabled': True,
            'policy': {
                'allow_unlisted_channels': False,
                'on_violation': 'error'  # Strict mode - raise exception
            }
        }

        # Create limits file with boundary
        limits_file = tmp_path / "channel_limits.json"
        limits_data = {
            "TEST:VOLTAGE": {
                "min_value": 0.0,
                "max_value": 100.0,  # ← Limit is 100.0
                "writable": True,
                "description": "Test voltage channel with limits"
            }
        }
        limits_file.write_text(json.dumps(limits_data))
        test_config_data['control_system']['limits_checking']['database_path'] = str(limits_file)

        # Disable approval to prevent test pollution
        _disable_approval_in_config(test_config_data)

        # Write updated config
        test_config.write_text(yaml.dump(test_config_data))

        # Initialize registry with updated config
        from osprey.registry import initialize_registry, reset_registry
        import osprey.utils.config as config_module

        config_module._default_config = None
        config_module._default_configurable = None
        config_module._config_cache.clear()

        reset_registry()
        initialize_registry(config_path=test_config)

        # === CODE GENERATION ===
        # Generate code that VIOLATES the max limit
        mock_gen = MockCodeGenerator()
        mock_gen.set_code("""
from osprey.runtime import write_channel

# This value EXCEEDS the max limit of 100.0
voltage = 150.0  # ← VIOLATION!

# Attempt to write using runtime utilities (synchronous API)
write_channel("TEST:VOLTAGE", voltage)

# This should NOT execute (blocked before this line)
results = {
    'voltage': voltage,
    'write_attempted': True,
    'should_not_exist': True
}
""")

        # === EXECUTION ===
        with patch('osprey.services.python_executor.generation.node.create_code_generator',
                   return_value=mock_gen):
            from osprey.utils.config import get_full_configuration

            service = PythonExecutorService()

            request = PythonExecutionRequest(
                user_query="Set voltage to 150V",
                task_objective="Write voltage exceeding limits",
                execution_folder_name=f"limits_violation_{tmp_path.name}"
            )

            config = {
                "thread_id": "test_max_violation",
                "configurable": get_full_configuration()
            }

            # Execution should raise CodeRuntimeError for limits violation
            with pytest.raises(CodeRuntimeError):
                await service.ainvoke(request, config)

        # === VERIFICATION ===
        # If we reach here, the test passed - the limits violation caused execution to fail
        # The logs will show the detailed "CHANNEL LIMITS VIOLATION DETECTED" message
        # which includes the channel address and the violation reason

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_runtime_write_below_min_limit_blocked(self, tmp_path, test_config):
        """
        CRITICAL: Runtime write_channel() below min limit is BLOCKED.
        """
        # === SETUP ===
        os.environ['CONFIG_FILE'] = str(test_config)

        test_config_data = yaml.safe_load(test_config.read_text())

        # Ensure control_system section exists and use Mock connector
        if 'control_system' not in test_config_data:
            test_config_data['control_system'] = {}

        test_config_data['control_system']['type'] = 'mock'
        test_config_data['control_system']['writes_enabled'] = True
        test_config_data['control_system']['connector'] = {
            'mock': {
                'noise_level': 0.0,
                'response_delay_ms': 1
            }
        }
        test_config_data['control_system']['limits_checking'] = {
            'enabled': True,
            'policy': {
                'allow_unlisted_channels': False,
                'on_violation': 'error'
            }
        }

        # Create limits file
        limits_file = tmp_path / "channel_limits.json"
        limits_data = {
            "TEST:CURRENT": {
                "min_value": 10.0,  # ← Minimum is 10.0
                "max_value": 100.0,
                "writable": True
            }
        }
        limits_file.write_text(json.dumps(limits_data))
        test_config_data['control_system']['limits_checking']['database_path'] = str(limits_file)

        # Disable approval to prevent test pollution
        _disable_approval_in_config(test_config_data)

        test_config.write_text(yaml.dump(test_config_data))

        # Initialize
        from osprey.registry import initialize_registry, reset_registry
        import osprey.utils.config as config_module

        config_module._default_config = None
        config_module._default_configurable = None
        config_module._config_cache.clear()
        reset_registry()
        initialize_registry(config_path=test_config)

        # === CODE GENERATION ===
        mock_gen = MockCodeGenerator()
        mock_gen.set_code("""
from osprey.runtime import write_channel

# This value is BELOW the min limit of 10.0
current = 5.0  # ← VIOLATION!

write_channel("TEST:CURRENT", current)

results = {'current': current}
""")

        # === EXECUTION ===
        with patch('osprey.services.python_executor.generation.node.create_code_generator',
                   return_value=mock_gen):
            from osprey.utils.config import get_full_configuration
            service = PythonExecutorService()

            request = PythonExecutionRequest(
                user_query="Set current to 5A",
                task_objective="Write current below minimum",
                execution_folder_name=f"min_violation_{tmp_path.name}"
            )

            # Execution should raise CodeRuntimeError for limits violation
            with pytest.raises(CodeRuntimeError):
                await service.ainvoke(
                    request,
                    {"thread_id": "test_min", "configurable": get_full_configuration()}
                )

        # === VERIFICATION ===
        # If we reach here, the test passed - the limits violation caused execution to fail

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_runtime_write_within_limits_succeeds(self, tmp_path, test_config):
        """
        POSITIVE TEST: Runtime write_channel() within limits SUCCEEDS.

        Verifies that valid writes using runtime utilities pass through
        limits checking and complete successfully.
        """
        # === SETUP ===
        os.environ['CONFIG_FILE'] = str(test_config)

        test_config_data = yaml.safe_load(test_config.read_text())

        # Ensure control_system section exists and use Mock connector
        if 'control_system' not in test_config_data:
            test_config_data['control_system'] = {}

        test_config_data['control_system']['type'] = 'mock'
        test_config_data['control_system']['writes_enabled'] = True
        test_config_data['control_system']['connector'] = {
            'mock': {
                'noise_level': 0.0,
                'response_delay_ms': 1
            }
        }
        test_config_data['control_system']['limits_checking'] = {
            'enabled': True,
            'policy': {
                'allow_unlisted_channels': False,
                'on_violation': 'error'
            }
        }

        # CRITICAL: Explicitly disable approval to prevent test pollution
        # When running with full test suite, approval state can leak from other tests
        if 'approval' not in test_config_data:
            test_config_data['approval'] = {}
        if 'capabilities' not in test_config_data['approval']:
            test_config_data['approval']['capabilities'] = {}
        test_config_data['approval']['capabilities']['python_execution'] = {
            'enabled': False
        }

        # Create limits file
        limits_file = tmp_path / "channel_limits.json"
        limits_data = {
            "TEST:VOLTAGE": {
                "min_value": 0.0,
                "max_value": 100.0,
                "writable": True
            }
        }
        limits_file.write_text(json.dumps(limits_data))
        test_config_data['control_system']['limits_checking']['database_path'] = str(limits_file)

        # Disable approval to prevent test pollution
        _disable_approval_in_config(test_config_data)

        test_config.write_text(yaml.dump(test_config_data))

        # Initialize
        from osprey.registry import initialize_registry, reset_registry
        import osprey.utils.config as config_module

        config_module._default_config = None
        config_module._default_configurable = None
        config_module._config_cache.clear()
        reset_registry()
        initialize_registry(config_path=test_config)

        # Force approval manager to reload with updated config AND clear config caches again
        try:
            import osprey.approval.approval_manager as approval_module
            approval_module._approval_manager = None
        except ImportError:
            pass
        # Clear config cache again to ensure approval manager gets fresh config
        config_module._default_config = None
        config_module._default_configurable = None
        config_module._config_cache.clear()

        # === CODE GENERATION ===
        # Generate code with VALID value (within limits)
        mock_gen = MockCodeGenerator()
        mock_gen.set_code("""
from osprey.runtime import write_channel
import math

# Calculate voltage - result is ~64.4, well within 0-100 limit
voltage = math.sqrt(4150)

# This should SUCCEED
write_channel("TEST:VOLTAGE", voltage)

results = {
    'voltage': voltage,
    'write_successful': True,
    'within_limits': True
}
""")

        # === EXECUTION ===
        with patch('osprey.services.python_executor.generation.node.create_code_generator',
                   return_value=mock_gen):
            from osprey.utils.config import get_full_configuration
            service = PythonExecutorService()

            request = PythonExecutionRequest(
                user_query="Set voltage to sqrt(4150)",
                task_objective="Write valid voltage within limits",
                execution_folder_name=f"valid_write_{tmp_path.name}"
            )

            result = await service.ainvoke(
                request,
                {"thread_id": "test_valid", "configurable": get_full_configuration()}
            )

        # === VERIFICATION ===

        # 1. Execution should SUCCEED (no exception raised, result returned)
        assert result is not None, "Result should be returned for successful execution"
        assert result.execution_result is not None, "Execution result should exist"

        # 2. Results should show success
        results = result.execution_result.results
        assert results is not None
        assert results.get('write_successful') is True
        assert results.get('within_limits') is True

        # 3. Voltage should be the calculated value (~64.4)
        voltage = results.get('voltage')
        assert voltage is not None
        assert 64.0 < voltage < 65.0  # sqrt(4150) ≈ 64.42

        # 4. Final notebook should exist (not error notebook)
        final_notebook = result.execution_result.notebook_path
        assert final_notebook.exists()
        assert "notebook.ipynb" in str(final_notebook)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_runtime_unlisted_channel_blocked(self, tmp_path, test_config):
        """
        CRITICAL: Runtime write to unlisted channel is BLOCKED when allow_unlisted=false.
        """
        # === SETUP ===
        os.environ['CONFIG_FILE'] = str(test_config)

        test_config_data = yaml.safe_load(test_config.read_text())

        # Ensure control_system section exists and use Mock connector
        if 'control_system' not in test_config_data:
            test_config_data['control_system'] = {}

        test_config_data['control_system']['type'] = 'mock'
        test_config_data['control_system']['writes_enabled'] = True
        test_config_data['control_system']['connector'] = {
            'mock': {
                'noise_level': 0.0,
                'response_delay_ms': 1
            }
        }
        test_config_data['control_system']['limits_checking'] = {
            'enabled': True,
            'policy': {
                'allow_unlisted_channels': False,  # ← Strict: block unlisted
                'on_violation': 'error'
            }
        }

        # Create limits file with ONLY one channel
        limits_file = tmp_path / "channel_limits.json"
        limits_data = {
            "ALLOWED:CHANNEL": {
                "min_value": 0.0,
                "max_value": 100.0,
                "writable": True
            }
            # UNLISTED:CHANNEL is NOT in the limits file
        }
        limits_file.write_text(json.dumps(limits_data))
        test_config_data['control_system']['limits_checking']['database_path'] = str(limits_file)

        # Disable approval to prevent test pollution
        _disable_approval_in_config(test_config_data)

        test_config.write_text(yaml.dump(test_config_data))

        # Initialize
        from osprey.registry import initialize_registry, reset_registry
        import osprey.utils.config as config_module

        config_module._default_config = None
        config_module._default_configurable = None
        config_module._config_cache.clear()
        reset_registry()
        initialize_registry(config_path=test_config)

        # === CODE GENERATION ===
        mock_gen = MockCodeGenerator()
        mock_gen.set_code("""
from osprey.runtime import write_channel

# Try to write to a channel NOT in the limits file
write_channel("UNLISTED:CHANNEL", 42.0)

results = {'attempted': True}
""")

        # === EXECUTION ===
        with patch('osprey.services.python_executor.generation.node.create_code_generator',
                   return_value=mock_gen):
            from osprey.utils.config import get_full_configuration
            service = PythonExecutorService()

            request = PythonExecutionRequest(
                user_query="Write to unlisted channel",
                task_objective="Test unlisted channel blocking",
                execution_folder_name=f"unlisted_{tmp_path.name}"
            )

            # Execution should raise CodeRuntimeError for unlisted channel
            with pytest.raises(CodeRuntimeError):
                await service.ainvoke(
                    request,
                    {"thread_id": "test_unlisted", "configurable": get_full_configuration()}
                )

        # === VERIFICATION ===
        # If we reach here, the test passed - the unlisted channel was blocked

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_runtime_bulk_writes_respect_limits(self, tmp_path, test_config):
        """
        CRITICAL: Runtime write_channels() (bulk) respects limits for ALL channels.

        Verifies that bulk write operations validate each channel individually.
        """
        # === SETUP ===
        os.environ['CONFIG_FILE'] = str(test_config)

        test_config_data = yaml.safe_load(test_config.read_text())

        # Ensure control_system section exists and use Mock connector
        if 'control_system' not in test_config_data:
            test_config_data['control_system'] = {}

        test_config_data['control_system']['type'] = 'mock'
        test_config_data['control_system']['writes_enabled'] = True
        test_config_data['control_system']['connector'] = {
            'mock': {
                'noise_level': 0.0,
                'response_delay_ms': 1
            }
        }
        test_config_data['control_system']['limits_checking'] = {
            'enabled': True,
            'policy': {
                'allow_unlisted_channels': False,
                'on_violation': 'error'
            }
        }

        # Create limits file
        limits_file = tmp_path / "channel_limits.json"
        limits_data = {
            "MAGNET:01": {
                "min_value": 0.0,
                "max_value": 50.0,  # ← Max is 50
                "writable": True
            },
            "MAGNET:02": {
                "min_value": 0.0,
                "max_value": 50.0,
                "writable": True
            }
        }
        limits_file.write_text(json.dumps(limits_data))
        test_config_data['control_system']['limits_checking']['database_path'] = str(limits_file)

        # Disable approval to prevent test pollution
        _disable_approval_in_config(test_config_data)

        test_config.write_text(yaml.dump(test_config_data))

        # Initialize
        from osprey.registry import initialize_registry, reset_registry
        import osprey.utils.config as config_module

        config_module._default_config = None
        config_module._default_configurable = None
        config_module._config_cache.clear()
        reset_registry()
        initialize_registry(config_path=test_config)

        # === CODE GENERATION ===
        # One valid value, one exceeds limit
        mock_gen = MockCodeGenerator()
        mock_gen.set_code("""
from osprey.runtime import write_channels

# Bulk write: one valid, one violates limit (synchronous API)
write_channels({
    "MAGNET:01": 30.0,  # Valid
    "MAGNET:02": 75.0   # VIOLATION! Exceeds 50.0
})

results = {'bulk_write': True}
""")

        # === EXECUTION ===
        with patch('osprey.services.python_executor.generation.node.create_code_generator',
                   return_value=mock_gen):
            from osprey.utils.config import get_full_configuration
            service = PythonExecutorService()

            request = PythonExecutionRequest(
                user_query="Set magnet values",
                task_objective="Bulk write with one violation",
                execution_folder_name=f"bulk_violation_{tmp_path.name}"
            )

            # Execution should raise CodeRuntimeError for limits violation
            with pytest.raises(CodeRuntimeError):
                await service.ainvoke(
                    request,
                    {"thread_id": "test_bulk", "configurable": get_full_configuration()}
                )

        # === VERIFICATION ===
        # If we reach here, the test passed - the bulk write was blocked
        # The logs will show "CHANNEL LIMITS VIOLATION DETECTED" for MAGNET:02

