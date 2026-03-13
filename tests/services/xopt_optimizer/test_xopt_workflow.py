"""Integration test for XOpt optimizer service workflow.

This test runs the full service workflow without approval to verify
the placeholder implementation works end-to-end, and also tests
the workflow with a mocked TuningScriptsClient for the real API path.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from osprey.services.xopt_optimizer.execution.api_client import TuningScriptsAPIError


class TestXOptWorkflow:
    """Test complete XOpt workflow execution."""

    @pytest.mark.asyncio
    async def test_full_workflow_without_approval(self, test_config):
        """Test complete workflow execution without approval.

        This runs the service through all nodes:
        state_id -> decision -> config_gen -> execution -> analysis

        The execution node falls back to placeholder when the API is unreachable.
        """
        os.environ["CONFIG_FILE"] = str(test_config)

        # Clear config cache so test fixture config is picked up
        from osprey.utils import config as config_module

        config_module._default_config = None
        config_module._default_configurable = None
        config_module._config_cache.clear()

        from osprey.services.xopt_optimizer import (
            XOptExecutionRequest,
            XOptOptimizerService,
            XOptServiceResult,
            XOptStrategy,
        )

        # Mock the client to simulate API-unreachable (connection error → placeholder fallback)
        mock_client = MagicMock()
        mock_client.health_check = AsyncMock(
            side_effect=TuningScriptsAPIError("Connection refused")
        )

        service = XOptOptimizerService()

        # Create request with approval disabled
        request = XOptExecutionRequest(
            user_query="Optimize injection efficiency",
            optimization_objective="Maximize injection efficiency",
            max_iterations=2,
            require_approval=False,  # Skip approval for this test
        )

        # Configure for execution
        config = {
            "configurable": {
                "thread_id": "test_workflow",
                "checkpoint_ns": "xopt_test",
            }
        }

        # Run the service with mocked client to avoid depending on external API
        with patch(
            "osprey.services.xopt_optimizer.execution.node.TuningScriptsClient",
            return_value=mock_client,
        ):
            result = await service.ainvoke(request, config)

        # Verify result structure
        assert isinstance(result, XOptServiceResult)
        assert result.strategy == XOptStrategy.EXPLORATION
        assert result.total_iterations == 2  # We set max_iterations=2
        assert isinstance(result.optimization_config, dict)
        assert "algorithm" in result.optimization_config
        assert len(result.recommendations) > 0

    @pytest.mark.asyncio
    async def test_single_iteration_workflow(self, test_config):
        """Test workflow with single iteration."""
        os.environ["CONFIG_FILE"] = str(test_config)

        from osprey.services.xopt_optimizer import (
            XOptExecutionRequest,
            XOptOptimizerService,
            XOptServiceResult,
        )

        service = XOptOptimizerService()

        request = XOptExecutionRequest(
            user_query="Quick test",
            optimization_objective="Test objective",
            max_iterations=1,
            require_approval=False,
        )

        config = {
            "configurable": {
                "thread_id": "test_single",
                "checkpoint_ns": "xopt_test",
            }
        }

        result = await service.ainvoke(request, config)

        assert isinstance(result, XOptServiceResult)
        assert result.total_iterations == 1

    @pytest.mark.asyncio
    async def test_workflow_with_mocked_api_client(self, test_config):
        """Test full workflow with a mocked TuningScriptsClient.

        This simulates the real API path where the execution node talks
        to the tuning_scripts API and receives actual optimization data.
        """
        os.environ["CONFIG_FILE"] = str(test_config)

        from osprey.services.xopt_optimizer import (
            XOptExecutionRequest,
            XOptOptimizerService,
            XOptServiceResult,
        )

        # Mock API responses with real-looking data
        full_state = {
            "job_id": "mock-job-001",
            "status": "completed",
            "data": [
                {"quad1_k1": 0.5, "quad2_k1": -1.0, "injection_efficiency": 0.85},
                {"quad1_k1": 1.0, "quad2_k1": -0.5, "injection_efficiency": 0.92},
                {"quad1_k1": 0.8, "quad2_k1": -0.7, "injection_efficiency": 0.95},
            ],
            "environment_name": "als_injector",
            "objective_name": "injection_efficiency",
            "variable_names": ["quad1_k1", "quad2_k1"],
            "results_path": "2024/mock-job-001",
            "logs": "Optimization completed in 3 evaluations",
        }

        mock_client = MagicMock()
        mock_client.health_check = AsyncMock(return_value={"status": "ok"})
        mock_client.submit_config = AsyncMock(return_value="mock-job-001")
        mock_client.poll_until_complete = AsyncMock(return_value=full_state)

        service = XOptOptimizerService()

        request = XOptExecutionRequest(
            user_query="Optimize injection efficiency",
            optimization_objective="Maximize injection efficiency",
            max_iterations=1,
            require_approval=False,
        )

        config = {
            "configurable": {
                "thread_id": "test_mocked_api",
                "checkpoint_ns": "xopt_test",
            }
        }

        with patch(
            "osprey.services.xopt_optimizer.execution.node.TuningScriptsClient",
            return_value=mock_client,
        ):
            result = await service.ainvoke(request, config)

        assert isinstance(result, XOptServiceResult)
        assert result.total_iterations == 1
        assert result.run_artifact["job_id"] == "mock-job-001"
        assert result.run_artifact["data"] is not None
        assert len(result.run_artifact["data"]) == 3
        # Analysis should have found best point
        assert len(result.recommendations) > 0
        assert any("injection_efficiency" in r or "0.95" in r for r in result.recommendations)
