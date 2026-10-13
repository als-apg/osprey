from osprey.services.optimizer.models.api import (
    CurrentEnvironmentRequest,
    CurrentEnvironmentResponse,
    EvaluateRequest,
    SetVariablesRequest,
)


def test_current_environment_response_defaults_to_unset():
    resp = CurrentEnvironmentResponse()
    assert resp.name is None
    assert resp.details is None
    assert resp.status is None


def test_current_environment_request_requires_name():
    assert CurrentEnvironmentRequest(name="als_injector").name == "als_injector"


def test_write_requests_carry_override_unhealthy_defaulting_false():
    assert SetVariablesRequest(variables={"a": 1.0}).override_unhealthy is False
    assert (
        EvaluateRequest(
            variables={"a": 1.0}, observables=["x"], override_unhealthy=True
        ).override_unhealthy
        is True
    )
