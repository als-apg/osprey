"""Type definitions for the simulator package.

This module defines local types to avoid dependency on tuning_scripts_models.
"""

from pydantic import BaseModel, model_validator


class VariableMetadata(BaseModel):
    """Metadata for optimization variables."""

    min_value: float
    max_value: float

    @model_validator(mode="after")
    def validate_bounds(self) -> "VariableMetadata":
        if self.min_value > self.max_value:
            raise ValueError(
                f"Min value ({self.min_value}) cannot be greater than max value ({self.max_value})"
            )
        return self
