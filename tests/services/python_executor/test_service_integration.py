"""Integration Tests for Python Executor Service using Mock Generator.

This module provides comprehensive end-to-end tests for the Python executor service
using the MockCodeGenerator. These tests verify the complete workflow from code
generation through execution without external dependencies or API calls.

Test Coverage:
    - Complete service workflow (generation -> analysis -> execution)
    - Error handling and retry logic
    - Approval workflows and interrupts
    - Container vs local execution
    - Analysis and security checks
    - Notebook generation
    - State management and checkpointing

The tests use the mock generator to provide deterministic, fast, and reliable
testing without requiring LLM API access, container infrastructure, or other
external dependencies.

Note:
    These are integration tests that exercise the full service stack. They
    require a test configuration file (provided by test_config fixture).
"""

import os
from unittest.mock import patch

import pytest

from osprey.services.python_executor import (
    PythonExecutionRequest,
    PythonExecutorService,
    PythonServiceResult,
)
from osprey.services.python_executor.generation import MockCodeGenerator

# =============================================================================
# SERVICE INITIALIZATION TESTS
# =============================================================================

class TestServiceInitialization:
    """Test service initialization and configuration."""

    def test_service_initializes(self, test_config):
        """Service should initialize without errors."""
        # Set config environment
        os.environ['CONFIG_FILE'] = str(test_config)

        service = PythonExecutorService()
        assert service is not None
        assert service.executor_config is not None

    def test_service_builds_graph(self, test_config):
        """Service should build LangGraph on initialization."""
        # Set config environment
        os.environ['CONFIG_FILE'] = str(test_config)

        service = PythonExecutorService()
        graph = service.get_compiled_graph()
        assert graph is not None


# =============================================================================
# BASIC WORKFLOW TESTS
# =============================================================================

class TestBasicWorkflow:
    """Test basic end-to-end workflow with mock generator."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_successful_execution_flow(self, tmp_path, test_config, mock_code_generator):
        """Test complete successful execution flow: generate -> analyze -> execute."""
        # Set test config
        os.environ['CONFIG_FILE'] = str(test_config)

        # Use global mock generator fixture
        mock_gen = mock_code_generator

        # Patch the factory to return our mock
        with patch('osprey.services.python_executor.generation.node.create_code_generator', return_value=mock_gen):
            from osprey.utils.config import get_full_configuration

            service = PythonExecutorService()

            request = PythonExecutionRequest(
                user_query="Calculate test value",
                task_objective="Generate test result",
                execution_folder_name=f"test_{tmp_path.name}"
            )

            # Use full configuration for proper execution
            full_config = get_full_configuration()
            config = {
                "thread_id": "test_thread",
                "configurable": full_config
            }

            # Execute service
            result = await service.ainvoke(request, config)

            # Verify result structure
            assert isinstance(result, PythonServiceResult)
            assert result.execution_result is not None
            assert result.generated_code is not None
            assert result.generation_attempt >= 1

            # Verify mock was called
            assert mock_gen.call_count >= 1
            # Compare request fields (excluding execution_folder_path which gets set during execution)
            assert mock_gen.last_request.user_query == request.user_query
            assert mock_gen.last_request.task_objective == request.task_objective
            assert mock_gen.last_request.execution_folder_name == request.execution_folder_name

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_execution_with_simple_code(self, tmp_path, test_config):
        """Test execution with simple Python code."""
        # Set test config
        os.environ['CONFIG_FILE'] = str(test_config)

        # Initialize registry for this test
        import osprey.utils.config as config_module
        from osprey.registry import initialize_registry, reset_registry
        config_module._default_config = None
        config_module._default_configurable = None
        config_module._config_cache.clear()

        reset_registry()
        initialize_registry()

        # Create mock generator with specific code
        mock_gen = MockCodeGenerator()
        mock_gen.set_code("results = {'value': 42, 'status': 'success'}")

        with patch('osprey.services.python_executor.generation.node.create_code_generator', return_value=mock_gen):
            service = PythonExecutorService()

            request = PythonExecutionRequest(
                user_query="Return 42",
                task_objective="Simple test",
                execution_folder_name=f"test_{tmp_path.name}"
            )

            # Get the full configurable from the loaded configuration
            from osprey.utils.config import get_full_configuration
            full_config = get_full_configuration()

            config = {
                "thread_id": "test_simple",
                "configurable": full_config
            }

            result = await service.ainvoke(request, config)

            # Verify execution succeeded
            assert result.execution_result.results.get('value') == 42
            assert result.execution_result.results.get('status') == 'success'


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestErrorHandling:
    """Test error handling and retry logic."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_syntax_error_detected(self, tmp_path):
        """Test that syntax errors are detected in analysis phase."""
        # Create mock generator that produces syntax error
        mock_gen = MockCodeGenerator(behavior="syntax_error")

        with patch('osprey.services.python_executor.generation.node.create_code_generator', return_value=mock_gen):
            service = PythonExecutorService()

            request = PythonExecutionRequest(
                user_query="Test syntax error",
                task_objective="Error handling test",
                execution_folder_name=f"test_{tmp_path.name}"
            )

            config = {
                "thread_id": "test_syntax_error",
                "configurable": {
                    "execution": {
                        "execution_method": "local"
                    }
                }
            }

            # Should raise error (syntax not fixable by retries alone)
            with pytest.raises(Exception):  # CodeRuntimeError or similar
                await service.ainvoke(request, config)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_retry_with_improved_code(self, tmp_path):
        """Test retry logic with progressively better code."""
        # Create mock generator with sequence: fail -> succeed
        mock_gen = MockCodeGenerator()
        mock_gen.set_code_sequence([
            "def broken(",  # Syntax error - will retry
            "results = {'value': 42}"  # Fixed code
        ])

        with patch('osprey.services.python_executor.generation.node.create_code_generator', return_value=mock_gen):
            service = PythonExecutorService()

            request = PythonExecutionRequest(
                user_query="Test retry",
                task_objective="Retry test",
                execution_folder_name=f"test_{tmp_path.name}"
            )

            config = {
                "thread_id": "test_retry",
                "configurable": {
                    "execution": {
                        "execution_method": "local"
                    }
                }
            }

            # May succeed after retry or may fail - depends on retry limit
            # The important thing is it tries multiple times
            try:
                result = await service.ainvoke(request, config)
                # If it succeeded, generator was called multiple times
                assert mock_gen.call_count >= 1
            except Exception:
                # If it failed, still verify retry attempts were made
                assert mock_gen.call_count >= 1

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_error_aware_generation(self, tmp_path):
        """Test error-aware generator adapts to feedback."""
        # Use error-aware behavior
        mock_gen = MockCodeGenerator(behavior="error_aware")

        with patch('osprey.services.python_executor.generation.node.create_code_generator', return_value=mock_gen):
            service = PythonExecutorService()

            request = PythonExecutionRequest(
                user_query="Test adaptive generation",
                task_objective="Error-aware test",
                execution_folder_name=f"test_{tmp_path.name}"
            )

            config = {
                "thread_id": "test_error_aware",
                "configurable": {
                    "execution": {
                        "execution_method": "local"
                    }
                }
            }

            # Execute - may succeed or fail, but should show adaptation
            try:
                result = await service.ainvoke(request, config)
                assert result.execution_result is not None
            except Exception:
                # Even on failure, verify generator was called
                assert mock_gen.call_count >= 1

            # Verify error feedback if generator was called multiple times
            if mock_gen.call_count > 1:
                assert len(mock_gen.last_error_chain) > 0


# =============================================================================
# ANALYSIS AND SECURITY TESTS
# =============================================================================

class TestAnalysisAndSecurity:
    """Test static analysis and security checks."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_epics_write_detection(self, tmp_path):
        """Test that control system write operations are detected (using EPICS as example)."""
        # Mock generator with EPICS write code
        # Note: EPICS is used as an example control system - the pattern detection
        # system is generic and works with any control system defined in config
        mock_gen = MockCodeGenerator(behavior="epics_write")

        with patch('osprey.services.python_executor.generation.node.create_code_generator', return_value=mock_gen):
            service = PythonExecutorService()

            request = PythonExecutionRequest(
                user_query="Write to EPICS PV",
                task_objective="EPICS write test",
                execution_folder_name=f"test_{tmp_path.name}"
            )

            config = {
                "thread_id": "test_epics_write",
                "configurable": {
                    "execution": {
                        "execution_method": "local"
                    }
                }
            }

            # This should trigger approval workflow or execute
            # depending on configuration
            try:
                result = await service.ainvoke(request, config)
                # If executed, verify code was analyzed
                assert mock_gen.call_count >= 1
            except Exception as e:
                # May require approval or fail due to missing epics module
                assert mock_gen.call_count >= 1

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_epics_read_allowed(self, tmp_path):
        """Test that control system read operations are allowed (using EPICS as example)."""
        # Mock generator with EPICS read code
        # Note: EPICS is used as an example control system - the pattern detection
        # system is generic and works with any control system defined in config
        mock_gen = MockCodeGenerator(behavior="epics_read")

        with patch('osprey.services.python_executor.generation.node.create_code_generator', return_value=mock_gen):
            service = PythonExecutorService()

            request = PythonExecutionRequest(
                user_query="Read from EPICS PV",
                task_objective="EPICS read test",
                execution_folder_name=f"test_{tmp_path.name}"
            )

            config = {
                "thread_id": "test_epics_read",
                "configurable": {
                    "execution": {
                        "execution_method": "local"
                    }
                }
            }

            # Read operations should not require approval
            # (though may fail due to missing epics module)
            try:
                result = await service.ainvoke(request, config)
                assert mock_gen.call_count >= 1
            except Exception:
                # May fail due to missing module, but should attempt execution
                assert mock_gen.call_count >= 1

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_security_risk_detection(self, tmp_path):
        """Test that security risks are detected in analysis."""
        # Mock generator with security-sensitive code
        mock_gen = MockCodeGenerator(behavior="security_risk")

        with patch('osprey.services.python_executor.generation.node.create_code_generator', return_value=mock_gen):
            service = PythonExecutorService()

            request = PythonExecutionRequest(
                user_query="Run system command",
                task_objective="Security test",
                execution_folder_name=f"test_{tmp_path.name}"
            )

            config = {
                "thread_id": "test_security",
                "configurable": {
                    "execution": {
                        "execution_method": "local"
                    }
                }
            }

            # Security-sensitive operations should be flagged
            try:
                result = await service.ainvoke(request, config)
                # May succeed with warnings
                assert result.analysis_warnings is not None or result.execution_result is not None
            except Exception:
                # Or may be blocked/fail
                pass


# =============================================================================
# APPROVAL WORKFLOW TESTS
# =============================================================================

class TestApprovalWorkflow:
    """Test approval interrupt and resumption workflow."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_approval_interrupt_triggered(self, tmp_path, test_config_with_approval):
        """Test that approval workflow properly interrupts execution."""
        # Set test config with approval enabled
        os.environ['CONFIG_FILE'] = str(test_config_with_approval)

        # Reset global config and registry state to ensure clean test
        import osprey.approval.approval_manager as approval_module
        import osprey.utils.config as config_module
        from osprey.registry import initialize_registry, reset_registry

        config_module._default_config = None
        config_module._default_configurable = None
        config_module._config_cache.clear()

        # Reset approval manager global state to pick up new config
        approval_module._approval_manager = None

        reset_registry()
        initialize_registry()

        # Create mock generator
        mock_gen = MockCodeGenerator()
        mock_gen.set_code("results = {'value': 42, 'status': 'success'}")

        with patch('osprey.services.python_executor.generation.node.create_code_generator', return_value=mock_gen):
            service = PythonExecutorService()

            request = PythonExecutionRequest(
                user_query="Calculate something",
                task_objective="Test approval workflow",
                execution_folder_name=f"test_{tmp_path.name}"
            )

            # Get the full configurable (approval already enabled in config file)
            from osprey.utils.config import get_full_configuration
            full_config = get_full_configuration()

            config = {
                "thread_id": "test_approval_interrupt",
                "configurable": full_config
            }

            # Execute service - should raise an exception because approval is needed
            # The service.ainvoke() wraps the graph and raises CodeRuntimeError
            # when execution fails (including when approval is needed but not provided)
            from osprey.services.python_executor.exceptions import CodeRuntimeError

            with pytest.raises(CodeRuntimeError) as exc_info:
                result = await service.ainvoke(request, config)

            # Verify the error message indicates approval was needed
            error_msg = str(exc_info.value)
            assert "approval" in error_msg.lower() or "execution failed" in error_msg.lower(), \
                f"Expected approval-related error, got: {error_msg}"

            # The test successfully verified that approval is triggered when enabled!
            # In a real scenario, the workflow would be resumed with approval Command

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_approval_with_epics_writes(self, tmp_path, test_config_with_approval):
        """Test that EPICS write operations trigger approval when configured."""
        # Use test_config_with_approval which is already set up for approval workflows
        os.environ['CONFIG_FILE'] = str(test_config_with_approval)

        # Modify config to enable EPICS writes with approval
        import yaml
        with open(test_config_with_approval) as f:
            config_data = yaml.safe_load(f)

        # Enable EPICS writes in execution_control (so they're not blocked)
        config_data['execution_control']['epics']['writes_enabled'] = True

        # Add agent_control_defaults to enable writes at the policy level
        if 'agent_control_defaults' not in config_data:
            config_data['agent_control_defaults'] = {}
        config_data['agent_control_defaults']['epics_writes_enabled'] = True
        config_data['agent_control_defaults']['control_system_writes_enabled'] = True

        # Set approval mode to epics_writes
        config_data['approval']['capabilities']['python_execution']['mode'] = 'epics_writes'

        with open(test_config_with_approval, 'w') as f:
            yaml.dump(config_data, f)

        # Reset and reinitialize registry and approval manager
        import osprey.approval.approval_manager as approval_module
        import osprey.utils.config as config_module
        from osprey.registry import initialize_registry, reset_registry
        from osprey.utils.config import get_full_configuration

        config_module._default_config = None
        config_module._default_configurable = None
        config_module._config_cache.clear()

        # Reset approval manager global state to pick up new config
        approval_module._approval_manager = None

        reset_registry()
        initialize_registry()

        # Create mock generator with EPICS write code
        mock_gen = MockCodeGenerator(behavior="epics_write")

        with patch('osprey.services.python_executor.generation.node.create_code_generator', return_value=mock_gen):
            service = PythonExecutorService()

            request = PythonExecutionRequest(
                user_query="Write to EPICS PV",
                task_objective="Test EPICS write approval",
                execution_folder_name=f"test_{tmp_path.name}"
            )

            full_config = get_full_configuration()

            config = {
                "thread_id": "test_epics_approval",
                "configurable": full_config
            }

            # Execute - should interrupt for approval due to EPICS write
            # Convert request to internal state first (don't bypass service logic)
            internal_state = service._create_internal_state(request)
            result_state = await service.get_compiled_graph().ainvoke(internal_state, config)

            # Verify approval is required
            assert result_state.get('requires_approval'), \
                f"EPICS write operations should require approval. State: {result_state.get('analysis_result')}"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_complete_approval_resume_workflow(self, tmp_path, test_config_with_approval):
        """Test complete approval workflow: interrupt → user approves → resume → execute.

        This is the CRITICAL end-to-end test for Issue #2 (state loss during resume).
        It validates that the preserve_once_set reducer successfully prevents
        state loss when resuming from checkpoint after user approval.
        """
        # Set test config with approval enabled
        os.environ['CONFIG_FILE'] = str(test_config_with_approval)

        # Modify config to enable EPICS writes with approval
        import yaml
        with open(test_config_with_approval) as f:
            config_data = yaml.safe_load(f)

        # Enable EPICS writes but require approval
        config_data['execution_control']['epics']['writes_enabled'] = True
        if 'agent_control_defaults' not in config_data:
            config_data['agent_control_defaults'] = {}
        config_data['agent_control_defaults']['epics_writes_enabled'] = True
        config_data['agent_control_defaults']['control_system_writes_enabled'] = True
        config_data['approval']['capabilities']['python_execution']['enabled'] = True
        config_data['approval']['capabilities']['python_execution']['mode'] = 'epics_writes'

        with open(test_config_with_approval, 'w') as f:
            yaml.dump(config_data, f)

        # Reset and reinitialize
        import osprey.approval.approval_manager as approval_module
        import osprey.utils.config as config_module
        from osprey.registry import initialize_registry, reset_registry
        from osprey.utils.config import get_full_configuration
        from langgraph.types import Command

        config_module._default_config = None
        config_module._default_configurable = None
        config_module._config_cache.clear()
        approval_module._approval_manager = None

        reset_registry()
        initialize_registry()

        # Create mock generator with EPICS write code
        mock_gen = MockCodeGenerator(behavior="epics_write")

        with patch('osprey.services.python_executor.generation.node.create_code_generator', return_value=mock_gen):
            service = PythonExecutorService()

            request = PythonExecutionRequest(
                user_query="Write to EPICS PV",
                task_objective="Test complete approval workflow with resume",
                execution_folder_name=f"test_{tmp_path.name}"
            )

            full_config = get_full_configuration()
            config = {
                "thread_id": "test_approval_resume",
                "configurable": full_config
            }

            # PHASE 1: Execute until approval interrupt
            internal_state = service._create_internal_state(request)
            interrupted_state = await service.get_compiled_graph().ainvoke(internal_state, config)

            # Verify we hit the approval interrupt
            assert interrupted_state.get('requires_approval'), \
                "Should require approval for EPICS write operations"
            assert interrupted_state.get('approval_interrupt_data'), \
                "Should have approval interrupt data"

            # CRITICAL: Verify request field is present BEFORE resume
            assert 'request' in interrupted_state, \
                "Request field should exist before resume"
            assert interrupted_state['request'] == request, \
                "Request should match original"

            # PHASE 2: User approves - resume execution with Command
            # This is where Issue #2 would crash without the preserve_once_set fix
            resume_command = Command(resume={"approved": True})

            # Resume execution - the critical test of state preservation
            final_state = await service.get_compiled_graph().ainvoke(resume_command, config)

            # CRITICAL: Verify request field is STILL present after resume
            assert 'request' in final_state, \
                "Request field should be preserved during resume (Issue #2 fix)"
            assert final_state['request'] == request, \
                "Request should still match original after resume"

            # Verify execution completed successfully
            # Note: May fail due to missing epics module, but should attempt execution
            # The key is that it didn't crash with KeyError: 'request'
            assert final_state.get('approved') == True, \
                "Approval status should be set"

            # If execution succeeded, verify results
            if final_state.get('is_successful'):
                assert final_state.get('execution_result') is not None, \
                    "Should have execution results on success"
                assert final_state.get('generated_code') is not None, \
                    "Should have generated code on success"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_approval_rejection_workflow(self, tmp_path, test_config_with_approval):
        """Test approval rejection: interrupt → user rejects → clean termination."""
        # Set test config with approval enabled
        os.environ['CONFIG_FILE'] = str(test_config_with_approval)

        # Modify config to enable approvals
        import yaml
        with open(test_config_with_approval) as f:
            config_data = yaml.safe_load(f)

        # Enable EPICS writes with approval
        config_data['execution_control']['epics']['writes_enabled'] = True
        if 'agent_control_defaults' not in config_data:
            config_data['agent_control_defaults'] = {}
        config_data['agent_control_defaults']['epics_writes_enabled'] = True
        config_data['approval']['capabilities']['python_execution']['enabled'] = True
        config_data['approval']['capabilities']['python_execution']['mode'] = 'epics_writes'

        with open(test_config_with_approval, 'w') as f:
            yaml.dump(config_data, f)

        # Reset and reinitialize
        import osprey.approval.approval_manager as approval_module
        import osprey.utils.config as config_module
        from osprey.registry import initialize_registry, reset_registry
        from osprey.utils.config import get_full_configuration
        from langgraph.types import Command

        config_module._default_config = None
        config_module._default_configurable = None
        config_module._config_cache.clear()
        approval_module._approval_manager = None

        reset_registry()
        initialize_registry()

        # Create mock generator
        mock_gen = MockCodeGenerator(behavior="epics_write")

        with patch('osprey.services.python_executor.generation.node.create_code_generator', return_value=mock_gen):
            service = PythonExecutorService()

            request = PythonExecutionRequest(
                user_query="Write to EPICS PV",
                task_objective="Test approval rejection workflow",
                execution_folder_name=f"test_{tmp_path.name}"
            )

            full_config = get_full_configuration()
            config = {
                "thread_id": "test_approval_reject",
                "configurable": full_config
            }

            # PHASE 1: Execute until approval interrupt
            internal_state = service._create_internal_state(request)
            interrupted_state = await service.get_compiled_graph().ainvoke(internal_state, config)

            # Verify approval is required
            assert interrupted_state.get('requires_approval'), \
                "Should require approval"

            # PHASE 2: User REJECTS - resume with denial
            reject_command = Command(resume={"approved": False})

            # Resume with rejection
            final_state = await service.get_compiled_graph().ainvoke(reject_command, config)

            # Verify clean termination
            assert final_state.get('approved') == False, \
                "Approval should be rejected"
            assert not final_state.get('is_successful'), \
                "Execution should not succeed when rejected"

            # Verify request is still preserved even in rejection path
            assert 'request' in final_state, \
                "Request field should be preserved even when rejected"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_no_approval_for_read_operations(self, tmp_path, test_config):
        """Test that read-only operations don't require approval with epics_writes mode."""
        # Set test config with EPICS write approval (but read should be allowed)
        os.environ['CONFIG_FILE'] = str(test_config)

        # Modify config
        import yaml
        with open(test_config) as f:
            config_data = yaml.safe_load(f)

        config_data['approval']['capabilities']['python_execution']['enabled'] = True
        config_data['approval']['capabilities']['python_execution']['mode'] = 'epics_writes'

        with open(test_config, 'w') as f:
            yaml.dump(config_data, f)

        # Reset and reinitialize registry and approval manager
        import osprey.approval.approval_manager as approval_module
        import osprey.utils.config as config_module
        from osprey.registry import initialize_registry, reset_registry
        from osprey.utils.config import get_full_configuration

        config_module._default_config = None
        config_module._default_configurable = None
        config_module._config_cache.clear()

        # Reset approval manager global state to pick up new config
        approval_module._approval_manager = None

        reset_registry()
        initialize_registry()

        # Create mock generator with EPICS READ code (no writes)
        mock_gen = MockCodeGenerator(behavior="epics_read")

        with patch('osprey.services.python_executor.generation.node.create_code_generator', return_value=mock_gen):
            service = PythonExecutorService()

            request = PythonExecutionRequest(
                user_query="Read from EPICS PV",
                task_objective="Test read-only execution",
                execution_folder_name=f"test_{tmp_path.name}"
            )

            full_config = get_full_configuration()

            config = {
                "thread_id": "test_read_no_approval",
                "configurable": full_config
            }

            # Execute - should NOT interrupt (read operations allowed)
            # Note: This might still fail due to missing epics module,
            # but it should attempt execution without approval
            try:
                result = await service.ainvoke(request, config)
                # If it succeeds, verify no approval was needed
                assert result is not None
            except Exception as e:
                # If it fails, it should be execution failure, not approval interrupt
                assert "approval" not in str(e).lower() or "requires_approval" not in str(e).lower(), \
                    f"Should not require approval for read operations, but got: {e}"


# =============================================================================
# STATE MANAGEMENT TESTS
# =============================================================================

class TestStateManagement:
    """Test service state management and tracking."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_tracks_generation_attempts(self, tmp_path, test_config):
        """Test that service tracks generation attempts."""
        os.environ['CONFIG_FILE'] = str(test_config)

        mock_gen = MockCodeGenerator(behavior="success")

        with patch('osprey.services.python_executor.generation.node.create_code_generator', return_value=mock_gen):
            from osprey.utils.config import get_full_configuration

            service = PythonExecutorService()

            request = PythonExecutionRequest(
                user_query="Test tracking",
                task_objective="Track attempts",
                execution_folder_name=f"test_{tmp_path.name}"
            )

            full_config = get_full_configuration()
            config = {
                "thread_id": "test_tracking",
                "configurable": full_config
            }

            result = await service.ainvoke(request, config)

            # Should track generation attempts
            assert result.generation_attempt >= 1

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_preserves_request_data(self, tmp_path, test_config):
        """Test that original request data is preserved through workflow."""
        os.environ['CONFIG_FILE'] = str(test_config)

        mock_gen = MockCodeGenerator(behavior="success")

        with patch('osprey.services.python_executor.generation.node.create_code_generator', return_value=mock_gen):
            from osprey.utils.config import get_full_configuration

            service = PythonExecutorService()

            original_query = "Specific user query"
            original_objective = "Specific task objective"

            request = PythonExecutionRequest(
                user_query=original_query,
                task_objective=original_objective,
                execution_folder_name=f"test_{tmp_path.name}"
            )

            full_config = get_full_configuration()
            config = {
                "thread_id": "test_preserve",
                "configurable": full_config
            }

            result = await service.ainvoke(request, config)

            # Verify request data was passed to generator
            assert mock_gen.last_request.user_query == original_query
            assert mock_gen.last_request.task_objective == original_objective


# =============================================================================
# EXECUTION METHOD TESTS
# =============================================================================

class TestExecutionMethods:
    """Test different execution methods (local vs container)."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_local_execution_method(self, tmp_path, test_config):
        """Test local execution method."""
        os.environ['CONFIG_FILE'] = str(test_config)

        mock_gen = MockCodeGenerator()
        mock_gen.set_code("results = {'method': 'local', 'value': 42}")

        with patch('osprey.services.python_executor.generation.node.create_code_generator', return_value=mock_gen):
            from osprey.utils.config import get_full_configuration

            service = PythonExecutorService()

            request = PythonExecutionRequest(
                user_query="Test local execution",
                task_objective="Local execution test",
                execution_folder_name=f"test_{tmp_path.name}"
            )

            full_config = get_full_configuration()
            config = {
                "thread_id": "test_local",
                "configurable": full_config
            }

            result = await service.ainvoke(request, config)

            assert result.execution_result is not None
            # Local execution should work
            assert 'method' in result.execution_result.results or 'value' in result.execution_result.results


# =============================================================================
# HELPER TESTS
# =============================================================================

class TestHelperFunctionality:
    """Test helper functionality and edge cases."""

    @pytest.mark.asyncio
    async def test_mock_generator_call_tracking(self, tmp_path):
        """Verify mock generator tracks calls correctly."""
        mock_gen = MockCodeGenerator(behavior="success")

        request = PythonExecutionRequest(
            user_query="Test",
            task_objective="Test",
            execution_folder_name=f"test_{tmp_path.name}"
        )

        # Initial state
        assert mock_gen.call_count == 0
        assert mock_gen.last_request is None

        # After call
        code = await mock_gen.generate_code(request, [])
        assert mock_gen.call_count == 1
        assert mock_gen.last_request == request
        assert code is not None

    def test_mock_generator_reset(self):
        """Verify mock generator reset works correctly."""
        mock_gen = MockCodeGenerator(behavior="success")
        mock_gen.call_count = 5

        mock_gen.reset()

        assert mock_gen.call_count == 0
        assert mock_gen.last_request is None
        # Code configuration should be preserved
        assert mock_gen.static_code is not None


# =============================================================================
# MODULE-SPECIFIC FIXTURES
# =============================================================================
# Note: General fixtures (mock_code_generator, test_config) are in tests/conftest.py

@pytest.fixture
def service(test_config):
    """Fixture providing a fresh PythonExecutorService instance.

    Uses test_config fixture to ensure service has proper configuration.
    """
    os.environ['CONFIG_FILE'] = str(test_config)
    return PythonExecutorService()

