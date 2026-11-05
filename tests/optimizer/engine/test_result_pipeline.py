"""Tests for result writers and the process_results function."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import polars as pl
import pytest

from osprey_optimizer.models.config import OptimizationConfig
from osprey_optimizer.services.orchestration import (
    OptimizationMetadata,
    OptimizationResult,
)
from osprey_optimizer.services.result_pipeline import (
    ParquetResultWriter,
    process_results,
)


def _make_result() -> OptimizationResult:
    """Create a dummy OptimizationResult for testing."""
    config = OptimizationConfig(
        n_iterations=3,
        objective_name="charge_ltb",
        environment_config={},
    )
    metadata = OptimizationMetadata(config=config)
    data = pl.DataFrame({"charge_ltb": [1.0, 1.2, 1.3, 1.4], "x": [0.1, 0.2, 0.15, 0.25]})
    best_point = {"charge_ltb": 1.4, "x": 0.25}
    return OptimizationResult(data=data, best_point=best_point, metadata=metadata)


class DummyWriter:
    """Mock result writer for testing process_results."""

    def __init__(self, target: Path) -> None:
        self.target = target
        self.called = False

    def write(self, result: OptimizationResult, *, output_dir: Path) -> Path | None:
        del result, output_dir
        self.called = True
        self.target.mkdir(parents=True, exist_ok=True)
        return self.target


class NullWriter:
    """Result writer that persists nothing and reports no run directory."""

    def __init__(self, target: Path) -> None:
        del target

    def write(self, result: OptimizationResult, *, output_dir: Path) -> None:
        del result, output_dir


class TestParquetResultWriter:
    def test_delegates_to_save(self, tmp_path: Path) -> None:
        fake = _make_result()
        sentinel = tmp_path / "run-2"
        output_dir = tmp_path / "output"
        with patch(
            "osprey_optimizer.services.result_pipeline.save_optimization_results",
            return_value=sentinel,
        ) as mock_save:
            result = ParquetResultWriter().write(fake, output_dir=output_dir)
        assert result is sentinel
        mock_save.assert_called_once()
        # The result's own data/best_point/config are handed to save verbatim,
        # under the caller's output_dir.
        args = mock_save.call_args.args
        assert args[0] is fake.data
        assert args[1] is fake.best_point
        assert args[2] is fake.metadata.config
        assert mock_save.call_args.kwargs.get("output_dir") == output_dir


class TestApplyMatrix:
    """Cover the save / early-return matrix of process_results.

    Source logic in process_results:
        control_checkpoint()
        run_dir = writer.write(result, output_dir=output_dir)
        control_checkpoint()
        if apply_best_point and apply_fn is not None:
            apply_fn(result)
        return run_dir
    """

    @pytest.mark.parametrize(
        "writer_cls, apply_best_point, provide_apply_fn, expect_apply_called, expect_run_dir",
        [
            pytest.param(DummyWriter, False, True, False, True, id="apply-disabled"),
            pytest.param(DummyWriter, True, False, False, True, id="apply-fn-none"),
            pytest.param(DummyWriter, True, True, True, True, id="apply-enabled"),
            pytest.param(NullWriter, False, False, False, False, id="writer-returns-none"),
        ],
    )
    def test_apply_matrix(
        self,
        tmp_path: Path,
        writer_cls: type,
        apply_best_point: bool,
        provide_apply_fn: bool,
        expect_apply_called: bool,
        expect_run_dir: bool,
    ) -> None:
        result = _make_result()
        apply_fn = MagicMock(return_value=result.best_point) if provide_apply_fn else None
        control = MagicMock()
        run_dir = tmp_path / "run"

        returned = process_results(
            result,
            writer=writer_cls(run_dir),
            output_dir=tmp_path,
            check_cancellation=control.check_cancellation,
            wait_if_paused=control.wait_if_paused,
            apply_fn=apply_fn,
            apply_best_point=apply_best_point,
        )

        assert returned == (run_dir if expect_run_dir else None)
        if apply_fn is not None:
            # The apply step receives the result verbatim, and only when enabled.
            assert apply_fn.call_args_list == ([call(result)] if expect_apply_called else [])
        # Both control hooks fire at each checkpoint: before the save and before the apply step.
        assert control.check_cancellation.call_count == 2
        assert control.wait_if_paused.call_count == 2
