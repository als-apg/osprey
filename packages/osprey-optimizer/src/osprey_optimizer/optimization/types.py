"""Optimization engine result types.

These runtime types carry a polars DataFrame and are used only by the app's
optimization engine, so they live here rather than in the shared
``osprey_optimizer.models`` contract layer (which stays polars-free).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from osprey_optimizer.models.config import OptimizationConfig
from osprey_optimizer.models.types import BestPoint


@dataclass
class OptimizationResult:
    """Structured result returned by the optimization engine."""

    data: pl.DataFrame
    best_point: BestPoint
    config: OptimizationConfig
    job_id: str | None = None
    initial_point: dict[str, float] | None = None
    model: Any | None = None
    model_variables: list[str] | None = None
