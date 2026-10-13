"""Each request-body model must reject unknown top-level fields."""

import pytest
from pydantic import ValidationError

from osprey.services.optimizer.models.api import (
    EvaluateRequest,
    ResolveBoundsRequest,
    SetVariablesRequest,
)
from osprey.services.optimizer.models.config import OptimizationConfig

# (model, minimal-valid-kwargs)
CASES = [
    pytest.param(
        OptimizationConfig,
        {"environment_name": "als_injector_sim", "objective_name": "charge_gtl"},
        id="OptimizationConfig",
    ),
    pytest.param(
        SetVariablesRequest,
        {"variables": {"v1": 1.0}},
        id="SetVariablesRequest",
    ),
    pytest.param(
        EvaluateRequest,
        {"variables": {"v1": 1.0}, "observables": ["obs1"]},
        id="EvaluateRequest",
    ),
    pytest.param(
        ResolveBoundsRequest,
        {"current_values": {"v1": 1.0}},
        id="ResolveBoundsRequest",
    ),
]


@pytest.mark.parametrize(("model", "valid_kwargs"), CASES)
def test_unknown_field_rejected(model, valid_kwargs) -> None:
    model(**valid_kwargs)
    with pytest.raises(ValidationError) as exc_info:
        model(**valid_kwargs, totally_bogus_field=123)
    assert any(err["type"] == "extra_forbidden" for err in exc_info.value.errors())
