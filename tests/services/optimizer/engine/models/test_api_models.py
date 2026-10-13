from osprey.services.optimizer.models.api import (
    EvaluateRequest,
    SetVariablesRequest,
)


def test_write_requests_carry_override_unhealthy_defaulting_false():
    assert SetVariablesRequest(variables={"a": 1.0}).override_unhealthy is False
    assert (
        EvaluateRequest(
            variables={"a": 1.0}, observables=["x"], override_unhealthy=True
        ).override_unhealthy
        is True
    )
