"""End-to-end integration test: optimization capability + xopt_optimizer service.

Validates that the optimization capability can be loaded from a registry
and that the xopt_optimizer service runs the full mock-mode workflow.
This simulates what would happen when a user says "Optimize injection efficiency"
through the weather-agent project.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from osprey.services.xopt_optimizer.execution.api_client import TuningScriptsAPIError


class TestEndToEndOptimization:
    """Test optimization from capability registration through service execution."""

    @pytest.mark.asyncio
    async def test_capability_loads_and_service_runs(self, test_config):
        """Verify OptimizationCapability can be imported and xopt_optimizer service runs."""
        os.environ["CONFIG_FILE"] = str(test_config)

        # Clear config cache
        from osprey.utils import config as config_module

        config_module._default_config = None
        config_module._default_configurable = None
        config_module._config_cache.clear()

        # 1. Verify capability can be imported and instantiated
        from osprey.capabilities.optimization import (
            OptimizationCapability,
            OptimizationResultContext,
        )

        assert OptimizationCapability.name == "optimization"
        assert "OPTIMIZATION_RESULT" in OptimizationCapability.provides

        # 2. Verify context class works
        ctx = OptimizationResultContext(
            run_artifact={"status": "test"},
            strategy="exploration",
            total_iterations=1,
            optimization_config={"algorithm": "random"},
        )
        assert ctx.context_type == "OPTIMIZATION_RESULT"
        summary = ctx.get_summary()
        assert summary["strategy"] == "exploration"
        assert summary["iterations"] == 1

        # 3. Run the service directly (mock mode, no LLM calls)
        from osprey.services.xopt_optimizer import (
            XOptExecutionRequest,
            XOptOptimizerService,
            XOptServiceResult,
        )

        mock_client = MagicMock()
        mock_client.health_check = AsyncMock(
            side_effect=TuningScriptsAPIError("Connection refused")
        )

        service = XOptOptimizerService()
        request = XOptExecutionRequest(
            user_query="Optimize injection efficiency",
            optimization_objective="Maximize injection efficiency",
            max_iterations=1,
            require_approval=False,
        )

        config = {
            "configurable": {
                "thread_id": "test_e2e",
                "checkpoint_ns": "xopt_test",
            }
        }

        with patch(
            "osprey.services.xopt_optimizer.execution.node.TuningScriptsClient",
            return_value=mock_client,
        ):
            result = await service.ainvoke(request, config)

        assert isinstance(result, XOptServiceResult)
        assert isinstance(result.optimization_config, dict)
        assert "algorithm" in result.optimization_config
        assert result.total_iterations == 1

        # 4. Verify context can be created from service result
        from osprey.capabilities.optimization import _create_optimization_context

        ctx = _create_optimization_context(result)
        assert ctx.context_type == "OPTIMIZATION_RESULT"
        assert ctx.strategy == result.strategy.value
        assert ctx.optimization_config == result.optimization_config

    @pytest.mark.asyncio
    async def test_service_with_real_api_data_path(self, test_config):
        """Test full path with mocked API returning real data."""
        os.environ["CONFIG_FILE"] = str(test_config)

        from osprey.capabilities.optimization import _create_optimization_context
        from osprey.services.xopt_optimizer import (
            XOptExecutionRequest,
            XOptOptimizerService,
            XOptServiceResult,
        )

        full_state = {
            "job_id": "e2e-test-001",
            "status": "completed",
            "data": [
                {"x1": 0.5, "x2": -1.0, "objective": 0.85},
                {"x1": 1.0, "x2": -0.5, "objective": 0.92},
                {"x1": 0.8, "x2": -0.7, "objective": 0.95},
            ],
            "environment_name": "test_env",
            "objective_name": "objective",
            "variable_names": ["x1", "x2"],
            "results_path": "2024/e2e-test",
            "logs": "Completed",
        }

        mock_client = MagicMock()
        mock_client.health_check = AsyncMock(return_value={"status": "ok"})
        mock_client.submit_config = AsyncMock(return_value="e2e-test-001")
        mock_client.poll_until_complete = AsyncMock(return_value=full_state)

        service = XOptOptimizerService()
        request = XOptExecutionRequest(
            user_query="Optimize the objective function",
            optimization_objective="Maximize objective",
            max_iterations=1,
            require_approval=False,
        )

        config = {
            "configurable": {
                "thread_id": "test_e2e_api",
                "checkpoint_ns": "xopt_test",
            }
        }

        with patch(
            "osprey.services.xopt_optimizer.execution.node.TuningScriptsClient",
            return_value=mock_client,
        ):
            result = await service.ainvoke(request, config)

        assert isinstance(result, XOptServiceResult)
        assert result.run_artifact["job_id"] == "e2e-test-001"
        assert result.run_artifact["data"] is not None
        assert len(result.run_artifact["data"]) == 3

        # Analysis should identify best point
        assert any("0.95" in r for r in result.recommendations)

        # Context creation should work
        ctx = _create_optimization_context(result)
        assert ctx.optimization_config == result.optimization_config
        assert ctx.total_iterations == 1

    @pytest.mark.asyncio
    async def test_prompt_builder_loads(self):
        """Verify optimization prompt builder is accessible."""
        from osprey.prompts.defaults.optimization import DefaultOptimizationPromptBuilder

        builder = DefaultOptimizationPromptBuilder()

        # Classifier guide should have examples
        classifier = builder.get_classifier_guide()
        assert classifier is not None
        assert len(classifier.examples) > 0

        # Orchestrator guide should exist
        # Note: requires registry to be initialized, so we test the builder directly
        assert hasattr(builder, "get_orchestrator_guide")
        assert hasattr(builder, "get_config_generation_guidance")
        assert hasattr(builder, "get_strategy_selection_guidance")
