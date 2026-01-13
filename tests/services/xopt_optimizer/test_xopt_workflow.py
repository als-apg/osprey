"""Integration test for XOpt optimizer service workflow.

This test runs the full service workflow without approval to verify
the placeholder implementation works end-to-end.
"""

import os

import pytest


class TestXOptWorkflow:
    """Test complete XOpt workflow execution."""

    @pytest.mark.asyncio
    async def test_full_workflow_without_approval(self, test_config):
        """Test complete workflow execution without approval.

        This runs the service through all nodes:
        state_id -> decision -> yaml_gen -> execution -> analysis
        """
        os.environ["CONFIG_FILE"] = str(test_config)

        from osprey.services.xopt_optimizer import (
            XOptExecutionRequest,
            XOptOptimizerService,
            XOptServiceResult,
            XOptStrategy,
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

        # Run the service
        result = await service.ainvoke(request, config)

        # Verify result structure
        assert isinstance(result, XOptServiceResult)
        assert result.strategy == XOptStrategy.EXPLORATION
        assert result.total_iterations == 2  # We set max_iterations=2
        assert result.generated_yaml is not None
        # Check for valid XOpt YAML structure (generator and vocs are required)
        yaml_lower = result.generated_yaml.lower()
        assert "generator" in yaml_lower, "Generated YAML should contain generator config"
        assert "vocs" in yaml_lower, "Generated YAML should contain vocs config"
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
