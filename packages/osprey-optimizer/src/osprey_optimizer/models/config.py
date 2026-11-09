"""Configuration models for optimization runs."""

from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .enums import AlgorithmType, ConstraintType, SurrogateModelType
from .types import Direction, require_finite, require_finite_positive, validate_min_le_max

ControlTarget: TypeAlias = Literal["va", "standin", "live"]
"""Which machine a call reaches: the virtual accelerator, the stand-in, or the real one."""

ExecutionMode: TypeAlias = Literal["readonly"]
"""``readonly`` takes writes away from a target; it never grants them."""

OnComplete: TypeAlias = Literal["apply_best", "restore", "hold"]
"""What a finished run leaves: its best point, its starting setpoints, or the last point it tried.

A run that stops early, cancelled or failed, always leaves the machine where it is;
restoring its starting setpoints is the operator's call, through the results restore
endpoint."""


def machine_config(
    target: ControlTarget, execution_mode: ExecutionMode | None = None
) -> dict[str, Any]:
    """The environment-config keys that point an environment at one machine."""
    return {"control_target": target, "execution_mode": execution_mode}


class VariableOverride(BaseModel):
    """Configuration override for a single variable's bounds and optimization parameters.

    Bounds may be specified either absolutely (``min``/``max``) or as a symmetric
    window centered on the variable's current readback. The two modes are mutually
    exclusive; specifying both raises ``ValidationError``.

    In an *authored* config, ``variable_overrides`` is sparse — only variables
    that deliberately deviate from the environment defaults appear. In a
    *persisted* (post-run) config it is fully materialized: every active variable
    carries its effective ``min``/``max``/``step_size``, so the config is
    self-describing without consulting the environment.
    """

    model_config = ConfigDict(extra="forbid")

    min: float | None = None
    max: float | None = None
    step_size: float | None = None
    window_fraction: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Total window width as a fraction of the variable's safety range, centered on the current readback: "
            "bounds = current ± window_fraction * (safety_max - safety_min) / 2."
        ),
    )
    window_abs: float | None = Field(
        default=None,
        gt=0,
        description="Symmetric absolute window in the variable's units: bounds = current ± window_abs.",
    )

    @model_validator(mode="after")
    def validate_bounds_exclusivity(self) -> "VariableOverride":
        if (self.min is not None or self.max is not None) and (
            self.window_fraction is not None or self.window_abs is not None
        ):
            raise ValueError("min/max and window_fraction/window_abs are mutually exclusive")
        if self.window_fraction is not None and self.window_abs is not None:
            raise ValueError("window_fraction and window_abs are mutually exclusive")
        # NaN slips past the min > max check below (all NaN comparisons are False) and
        # reaches the resolved bounds and xopt's VOCS; inf is nonsensical as a bound.
        for name, value in (("min", self.min), ("max", self.max)):
            if value is not None:
                require_finite(value, label=name)
        # step_size constrains the optimizer to a grid (bounds + candidate snapping), so a
        # non-finite or non-positive step is meaningless — reject it at the boundary.
        if self.step_size is not None:
            require_finite_positive(self.step_size, label="step_size")
        if self.min is not None and self.max is not None:
            validate_min_le_max(self.min, self.max)
        return self


class ConstraintSpec(BaseModel):
    """An outcome constraint on an observable: ``observable {type} value``.

    Maps directly onto an xopt VOCS constraint entry ``[type, value]``. A row is
    feasible when the measured observable satisfies the strict inequality against
    ``value``. ``value`` must be finite — a NaN/inf threshold is meaningless and
    would silently propagate into the VOCS, so it is rejected at the boundary.
    """

    model_config = ConfigDict(extra="forbid")

    type: ConstraintType
    value: float

    @field_validator("value")
    @classmethod
    def validate_value_finite(cls, v: float) -> float:
        require_finite(v, label="constraint value")
        return v


class Preset(BaseModel):
    """A named run setup offered by an environment: phase tuning, corrector tuning.

    It fills a draft's unset fields when the draft names it (``preset``): which
    variables to tune, how far each may move, and the algorithm, length and
    objective. Anything the draft sets itself wins.
    """

    model_config = ConfigDict(extra="forbid")

    display_name: str = ""
    description: str = ""
    variables: list[str] | None = Field(
        default=None, min_length=1, description="Variables to tune; omitted means all of them."
    )
    variable_overrides: dict[str, VariableOverride] = Field(
        default_factory=dict, description="Per-variable bounds, windows or step sizes."
    )
    algorithm: AlgorithmType | None = None
    n_iterations: int | None = Field(default=None, ge=1)
    objective_name: str | None = None

    @model_validator(mode="after")
    def _overrides_within_variables(self) -> "Preset":
        if self.variables is not None:
            stray = sorted(set(self.variable_overrides) - set(self.variables))
            if stray:
                raise ValueError(
                    f"variable_overrides name variables the preset leaves out: {stray}"
                )
        return self


class OptimizationConfig(BaseModel):
    """Configuration schema for defining and executing optimization runs."""

    model_config = ConfigDict(extra="forbid")

    environment_name: str | None = None
    environment_config: dict[str, Any] | None = None
    preset: str | None = Field(
        default=None,
        description="One of the environment's presets; it fills the fields this config leaves unset.",
    )

    control_target: ControlTarget | None = Field(
        default=None,
        description=(
            "The machine the run drives: va, standin or live. Required to start a run; "
            "there is no default, so a run never lands on a machine nobody named."
        ),
    )
    execution_mode: ExecutionMode | None = Field(
        default=None, description="readonly: the run may read but every write is refused."
    )
    on_complete: OnComplete = Field(
        default="apply_best",
        description=(
            "On completion: 'apply_best' writes the best point found; 'restore' writes back "
            "the starting setpoints; 'hold' leaves the last point tried. A cancelled or failed "
            "run always holds."
        ),
    )

    objective_name: str | None = Field(
        default=None,
        description=(
            "Observable to optimize. Required to start a run; the objective must be explicit — "
            "there is no implicit fallback to the environment's default_objective."
        ),
    )
    objective_direction: Direction | None = None
    algorithm: AlgorithmType = AlgorithmType.RANDOM
    # 0 is allowed (initial-sampling-only run); negative is meaningless.
    n_iterations: int = Field(default=20, ge=0)
    n_initial_samples: int | None = Field(default=None, ge=1)
    variables: list[str] | None = None
    variable_overrides: dict[str, VariableOverride] | None = None
    surrogate_model: SurrogateModelType = SurrogateModelType.DEFAULT
    model_file: str | None = None
    algorithm_params: dict[str, Any] | None = Field(
        default=None,
        description="Algorithm-specific parameters passed as kwargs to the generator constructor.",
    )
    constraints: dict[str, ConstraintSpec] = Field(
        default_factory=dict,
        description=(
            "Outcome constraints keyed by observable name. Each restricts the search to points "
            "whose measured observable satisfies the constraint; the applied best point is chosen "
            "only from feasible evaluations. Not every algorithm supports constraints."
        ),
    )

    @field_validator("variables")
    @classmethod
    def validate_variables_non_empty(cls, v: list[str] | None) -> list[str] | None:
        # None means "use all environment variables"; an explicit empty list is a
        # mistake — without it we silently optimize all variables instead of none.
        if v is not None and len(v) == 0:
            raise ValueError(
                "variables must be non-empty when provided (omit it to use all variables)"
            )
        return v

    @model_validator(mode="after")
    def validate_constraints(self) -> "OptimizationConfig":
        # Constraining the objective is unsupported in v1 (direction already drives it,
        # and feasibility/selection would need bespoke handling); reject rather than ignore.
        if self.objective_name is not None and self.objective_name in self.constraints:
            raise ValueError(
                f"Constraint key '{self.objective_name}' equals the objective; "
                "constraining the objective is not supported. Constrain a different observable."
            )
        return self

    @model_validator(mode="after")
    def ensure_environment_config(self) -> "OptimizationConfig":
        """Initialize environment config to an empty dict when unset.

        ``variable_overrides`` lives only at the top level; the optimization engine
        reads it from there. It is deliberately not copied into ``environment_config``
        — no environment consumer reads the nested copy.
        """
        if self.environment_config is None:
            self.environment_config = {}

        return self

    def require_resolved(self) -> None:
        """Raise ``ValueError`` unless this config is a fully resolved draft.

        The two-phase job contract (issue #2): a persisted draft carries an
        explicit active-variable list and materialized ``min``/``max``/
        ``step_size`` per variable, so the worker consumes it verbatim.
        """
        if not self.variables:
            raise ValueError("config is not resolved: variables is unset")
        overrides = self.variable_overrides or {}
        unresolved = [
            name
            for name in self.variables
            if name not in overrides
            or overrides[name].min is None
            or overrides[name].max is None
            or overrides[name].step_size is None
        ]
        if unresolved:
            raise ValueError(f"config is not resolved: incomplete overrides for {unresolved}")
