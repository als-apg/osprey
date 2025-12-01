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

import nbformat
import pytest
import yaml

from osprey.services.python_executor import (
    PythonExecutionRequest,
    PythonExecutorService,
)
from osprey.services.python_executor.generation import MockCodeGenerator


# =============================================================================
# SAFETY LAYER: LIMITS VALIDATION WITH RUNTIME UTILITIES
# =============================================================================

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

        # Enable control system writes
        test_config_data['control_system']['writes_enabled'] = True

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
        test_config_data['control_system']['limits_checking']['limits_file'] = str(limits_file)

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

# Attempt to write using runtime utilities
await write_channel("TEST:VOLTAGE", voltage)

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

            result = await service.ainvoke(request, config)

        # === VERIFICATION ===

        # 1. Execution should FAIL
        assert result.is_failed is True, "Execution should have failed due to limits violation"
        assert result.execution_result is None, "No execution result should exist"

        # 2. Failure reason should mention ChannelLimitsViolationError
        assert result.failure_reason is not None
        assert "ChannelLimitsViolationError" in result.failure_reason or "above maximum" in result.failure_reason.lower()

        # 3. Error should mention the channel and violation details
        assert "TEST:VOLTAGE" in result.failure_reason

        # 4. Error notebook should exist with violation details
        attempts_dir = result.execution_folder.folder_path / "attempts"
        assert attempts_dir.exists(), "Attempts directory should exist"

        error_notebooks = list(attempts_dir.glob("*execution_failed*.ipynb"))
        assert len(error_notebooks) > 0, "Error notebook should exist"

        # Read error notebook and verify it contains violation info
        error_notebook = nbformat.read(error_notebooks[0], as_version=4)
        notebook_text = '\n'.join(cell.source for cell in error_notebook.cells)

        # Should contain error details
        assert "TEST:VOLTAGE" in notebook_text
        assert "150" in notebook_text or "above" in notebook_text.lower()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_runtime_write_below_min_limit_blocked(self, tmp_path, test_config):
        """
        CRITICAL: Runtime write_channel() below min limit is BLOCKED.
        """
        # === SETUP ===
        os.environ['CONFIG_FILE'] = str(test_config)

        test_config_data = yaml.safe_load(test_config.read_text())
        test_config_data['control_system']['writes_enabled'] = True
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
        test_config_data['control_system']['limits_checking']['limits_file'] = str(limits_file)
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

await write_channel("TEST:CURRENT", current)

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

            result = await service.ainvoke(
                request,
                {"thread_id": "test_min", "configurable": get_full_configuration()}
            )

        # === VERIFICATION ===
        assert result.is_failed is True
        assert "below minimum" in result.failure_reason.lower() or "ChannelLimitsViolationError" in result.failure_reason
        assert "TEST:CURRENT" in result.failure_reason

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
        test_config_data['control_system']['writes_enabled'] = True
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
            "TEST:VOLTAGE": {
                "min_value": 0.0,
                "max_value": 100.0,
                "writable": True
            }
        }
        limits_file.write_text(json.dumps(limits_data))
        test_config_data['control_system']['limits_checking']['limits_file'] = str(limits_file)
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
        # Generate code with VALID value (within limits)
        mock_gen = MockCodeGenerator()
        mock_gen.set_code("""
from osprey.runtime import write_channel
import math

# Calculate voltage - result is ~64.4, well within 0-100 limit
voltage = math.sqrt(4150)

# This should SUCCEED
await write_channel("TEST:VOLTAGE", voltage)

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

        # 1. Execution should SUCCEED
        assert result.is_failed is False, f"Execution should succeed, but failed: {result.failure_reason}"
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

        # 5. Context snapshot should exist with control system config
        context_file = result.execution_result.folder_path / 'context.json'
        assert context_file.exists()

        context_data = json.loads(context_file.read_text())
        assert '_execution_config' in context_data
        assert 'control_system' in context_data['_execution_config']

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_runtime_unlisted_channel_blocked(self, tmp_path, test_config):
        """
        CRITICAL: Runtime write to unlisted channel is BLOCKED when allow_unlisted=false.
        """
        # === SETUP ===
        os.environ['CONFIG_FILE'] = str(test_config)

        test_config_data = yaml.safe_load(test_config.read_text())
        test_config_data['control_system']['writes_enabled'] = True
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
        test_config_data['control_system']['limits_checking']['limits_file'] = str(limits_file)
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
await write_channel("UNLISTED:CHANNEL", 42.0)

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

            result = await service.ainvoke(
                request,
                {"thread_id": "test_unlisted", "configurable": get_full_configuration()}
            )

        # === VERIFICATION ===
        assert result.is_failed is True
        assert "UNLISTED" in result.failure_reason or "not found" in result.failure_reason.lower()

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
        test_config_data['control_system']['writes_enabled'] = True
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
        test_config_data['control_system']['limits_checking']['limits_file'] = str(limits_file)
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

# Bulk write: one valid, one violates limit
await write_channels({
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

            result = await service.ainvoke(
                request,
                {"thread_id": "test_bulk", "configurable": get_full_configuration()}
            )

        # === VERIFICATION ===
        # Should fail on the second write (MAGNET:02 exceeds limit)
        assert result.is_failed is True
        assert "MAGNET:02" in result.failure_reason or "above maximum" in result.failure_reason.lower()

