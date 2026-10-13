"""Optimization engine result types.

These runtime types carry a polars DataFrame and are used only by the app's
optimization engine, so they live here rather than in the shared
``osprey.services.optimizer.models`` contract layer (which stays polars-free).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from osprey.services.optimizer.models.config import OptimizationConfig
from osprey.services.optimizer.models.types import BestPoint


@dataclass
class OptimizationMetadata:
    """Metadata describing an optimization run."""

    config: OptimizationConfig
    job_id: str | None = None


@dataclass
class OptimizationResult:
    """Structured result returned by the optimization engine."""

    data: pl.DataFrame
    best_point: BestPoint
    metadata: OptimizationMetadata
    initial_point: dict[str, float] | None = None
    model: Any | None = None
    model_variables: list[str] | None = None
