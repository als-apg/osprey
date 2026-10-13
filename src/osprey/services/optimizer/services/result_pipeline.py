"""Result writers and the persist-then-apply post-processing function."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from osprey.services.optimizer.models.types import BestPoint

from ..results import ResultMetadata, save_optimization_results
from .orchestration import OptimizationResult


class ResultWriter(Protocol):
    """Protocol for persisting optimization results."""

    def write(self, result: OptimizationResult, *, output_dir: Path) -> Path | None:
        """Persist the given result and return the run directory if available."""


class ParquetResultWriter:
    """Result writer that uses the existing parquet flow."""

    def write(self, result: OptimizationResult, *, output_dir: Path) -> Path:
        return save_optimization_results(
            result.data,
            result.best_point,
            result.metadata.config,
            output_dir=output_dir,
            metadata=ResultMetadata(
                initial_point=result.initial_point,
                job_id=result.metadata.job_id,
                model=result.model,
                model_variables=result.model_variables,
            ),
        )


def process_results(
    result: OptimizationResult,
    *,
    writer: ResultWriter,
    output_dir: Path,
    check_cancellation: Callable[[], None],
    wait_if_paused: Callable[[], None],
    apply_fn: Callable[[OptimizationResult], BestPoint | None] | None = None,
    apply_best_point: bool = True,
) -> Path | None:
    """Persist optimization results, then optionally apply the best point.

    Returns the run directory if the writer produced one, else None.
    """

    def control_checkpoint() -> None:
        check_cancellation()
        wait_if_paused()

    control_checkpoint()

    run_dir = writer.write(result, output_dir=output_dir)

    control_checkpoint()

    if apply_best_point and apply_fn is not None:
        apply_fn(result)

    return run_dir
