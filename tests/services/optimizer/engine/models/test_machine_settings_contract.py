"""Contract added by issue #2: machine-settings gate + resolved-draft predicate."""

import pytest

from osprey.services.optimizer.models.config import OptimizationConfig, VariableOverride
from osprey.services.optimizer.models.exceptions import ALSTuningError, MachineSettingsError
from osprey.services.optimizer.models.protocols import EnvironmentProtocol


def test_machine_settings_error_carries_failures() -> None:
    err = MachineSettingsError("machine not ready", failures=["PV1: got 0, expected 1"])
    assert isinstance(err, ALSTuningError)
    assert err.failures == ["PV1: got 0, expected 1"]


def test_protocol_default_check_machine_settings_is_noop() -> None:
    class _Env(EnvironmentProtocol):
        pass

    # Explicit subclasses inherit the default body; no override required.
    assert _Env.check_machine_settings(object.__new__(_Env)) is None  # type: ignore[func-returns-value,type-abstract]


def _resolved_config() -> OptimizationConfig:
    return OptimizationConfig(
        environment_name="e",
        control_target="va",
        objective_name="obj",
        variables=["a"],
        variable_overrides={"a": VariableOverride(min=0.0, max=1.0, step_size=0.1)},
    )


def test_require_resolved_passes_on_materialized_config() -> None:
    _resolved_config().require_resolved()


def test_require_resolved_rejects_unset_variables() -> None:
    cfg = OptimizationConfig(environment_name="e", control_target="va", objective_name="obj")
    with pytest.raises(ValueError, match="variables"):
        cfg.require_resolved()


@pytest.mark.parametrize(
    "override",
    [
        None,
        VariableOverride(max=1.0, step_size=0.1),
        VariableOverride(min=0.0, step_size=0.1),
        VariableOverride(min=0.0, max=1.0),
    ],
)
def test_require_resolved_rejects_incomplete_override(override: VariableOverride | None) -> None:
    cfg = _resolved_config()
    cfg.variable_overrides = {"a": override} if override is not None else {}
    with pytest.raises(ValueError, match="a"):
        cfg.require_resolved()
