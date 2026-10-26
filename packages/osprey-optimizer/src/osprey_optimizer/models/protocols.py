"""Protocol definitions for the optimizer service.

This module defines abstract interfaces (protocols) for key components,
enabling dependency inversion and better testability.
"""

from collections.abc import Callable
from typing import Annotated, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field, model_validator

from .enums import EnvironmentType, JobStatus
from .types import Direction, PVValues, VariableInputs, validate_min_le_max


@runtime_checkable
class JobStateReader(Protocol):
    """Protocol for reading common job state (status, logs, errors).

    This is the shared surface the job state readers implement. Use this type
    at boundaries that handle any job type without needing domain-specific
    methods.
    """

    def get_status(self) -> JobStatus: ...
    def get_logs(self) -> str: ...
    def get_error_message(self) -> str | None: ...


@runtime_checkable
class JobStateWriter(JobStateReader, Protocol):
    """Protocol for writing common job state (extends reader).

    Shared write surface the job state writers implement.
    """

    def set_status(self, status: JobStatus) -> None: ...
    def set_error_message(self, message: str | None) -> None: ...
    def save_logs(self, logs: str) -> None: ...
    def set_status_cas(self, expected: JobStatus, new: JobStatus) -> bool:
        """Atomically set status to ``new`` only if it currently equals ``expected``.

        Returns True on swap, False if the stored value differed (e.g. a
        worker drove the job terminal in the gap).
        """
        ...


class VariableMetadata(BaseModel):
    """Metadata for optimization variables.

    Defines the contract for variable properties exposed by plugins.
    """

    min_value: float
    """Minimum allowed value for the variable."""

    max_value: float
    """Maximum allowed value for the variable."""

    units: str = ""
    """Physical units of the variable."""

    description: str = ""
    """Human-readable description."""

    readback_pv: str
    """EPICS PV or identifier for reading value."""

    setpoint_pv: str
    """EPICS PV or identifier for setting value."""

    step_size: float = 0.1
    """Smallest discrete step size for changes."""

    max_step: float | None = None
    """Largest change one write may make, from the setpoint's channel limits.

    ``None`` when the channel has no step limit. The engine moves a variable in
    one write, so a search window wider than this is refused when the run is
    resolved: a move across it would be refused by the connector mid-run.
    """

    @model_validator(mode="after")
    def validate_bounds(self) -> "VariableMetadata":
        """Validate min_value and max_value consistency."""
        validate_min_le_max(self.min_value, self.max_value)
        return self


class _ObservableBase(BaseModel):
    """Shared fields for all observable variants."""

    model_config = {"extra": "forbid"}

    display_name: str = ""
    """Short, human-readable label for UI (e.g. "GTL Charge"). Falls back to the observable key when empty."""

    description: str = ""
    """Plain-language explanation of what this observable measures, readable for non-experts."""

    units: str = ""
    """Physical units of the observable."""

    min_value: float | None = None
    """Optional minimum expected value (for plotting/validation)."""

    max_value: float | None = None
    """Optional maximum expected value."""

    direction: Direction = Direction.MAXIMIZE
    """Optimization direction when used as an objective."""

    selectable_as_objective: bool = True
    """Whether this observable may be chosen as an optimization objective. Set
    ``False`` for monitoring-only signals (e.g. BPM positions) so they stay
    readable but don't clutter the objective list."""

    target_value: float | None = None
    """Optional target value for this observable (e.g., desired charge in nC)."""

    @model_validator(mode="after")
    def _validate_bounds(self) -> "_ObservableBase":
        if self.min_value is not None and self.max_value is not None:
            validate_min_le_max(self.min_value, self.max_value)
        return self


class PVObservable(_ObservableBase):
    """Observable whose value is read directly from a single EPICS PV."""

    kind: Literal["pv"] = "pv"
    readback_pv: str
    """EPICS PV or identifier for reading value."""


class CompositeObservable(_ObservableBase):
    """Observable computed from other observables via a registered function."""

    kind: Literal["composite"] = "composite"
    source_observables: list[str]
    """Names of observables this computed observable depends on."""


class ReducedObservable(_ObservableBase):
    """Observable reduced to one number from many channels (e.g. orbit RMS over BPMs).

    Computed in the service from one batched read; the reading is unconnected
    (value ``None``) when any channel fails to read, so a partial read never
    becomes an objective value.
    """

    kind: Literal["reduce"] = "reduce"
    reduce: Literal["rms", "mean", "max"]
    """Reduction over the channel values: root mean square, mean, or maximum."""
    channels: list[str] = Field(min_length=1)
    """Channels read for every evaluation of this observable."""


ObservableMetadata = Annotated[
    PVObservable | CompositeObservable | ReducedObservable, Field(discriminator="kind")
]
"""Discriminated-union alias for observable metadata."""


class PVReading(BaseModel):
    """A single PV read: its native value plus whatever metadata EPICS reports.

    ``value`` is the channel's value in JSON-safe native form — a number or
    string for scalars, a list for waveforms, ``None`` when the read failed
    (``connected=False``) or the underlying value was NaN. ``connected`` lets
    callers tell a failed read apart from a genuine null/NaN value.

    Metadata fields are best-effort: the batch :meth:`InterfaceProtocol.read_many`
    fills control metadata (``units`` / ``precision`` / ``enum_strings``), while
    the lightweight :meth:`InterfaceProtocol.read_one` may leave them ``None`` to
    avoid an extra round-trip on hot polling paths.
    """

    value: Any = None
    connected: bool = True
    units: str | None = None
    precision: int | None = None
    severity: int | None = None
    status: int | None = None
    timestamp: float | None = None
    enum_strings: list[str] | None = None
    count: int | None = None


@runtime_checkable
class EnvironmentProtocol(Protocol):
    """Protocol for environment plugins.

    Defines the contract that plugins must satisfy to integrate with the
    optimization framework. A plugin class may also offer named run setups as
    ``PRESETS``, a mapping of preset name to
    :class:`~osprey_optimizer.models.config.Preset` or its dict form.

    The service loads a plugin only when ``$OPTIMIZER_PLUGINS`` names it. A
    plugin must reach the machine through
    :class:`~osprey_optimizer.environments.interface.ConnectorInterface`, so
    its writes are checked like an environment file's; this is not yet
    enforced for plugin code.
    """

    def __init__(self, config: dict[str, Any]) -> None: ...

    @property
    def environment_type(self) -> EnvironmentType:
        """Get the type of environment."""
        ...

    @property
    def variables(self) -> dict[str, VariableMetadata]:
        """Get variable metadata."""
        ...

    @property
    def observables(self) -> dict[str, ObservableMetadata]:
        """Get observable metadata.

        Returns:
            Dictionary mapping observable names to their metadata.
        """
        ...

    def get_variables(self, variable_names: list[str]) -> dict[str, PVReading]:
        """Get current readings for variables via their readback PVs."""
        ...

    def get_setpoints(self, variable_names: list[str]) -> dict[str, PVReading]:
        """Get the values currently written to the variables' setpoint PVs.

        These are the optimizer's coordinates: a run's starting point, the
        centre of each search window and the values a restore writes back.
        A readback can differ from its setpoint (calibration offset, noise),
        so the two reads are not interchangeable.
        """
        ...

    def set_variables(
        self,
        variable_inputs: dict[str, float],
        *,
        check_cancellation: Callable[[], None] | None = None,
    ) -> None:
        """Set values for variables and wait until hardware has fully settled."""
        ...

    def get_observables(
        self,
        observable_names: list[str],
        *,
        check_cancellation: Callable[[], None] | None = None,
    ) -> dict[str, PVReading]:
        """Get current readings for observables (composites resolved)."""
        ...

    def evaluate(
        self,
        variable_inputs: dict[str, float],
        observable_names: list[str],
        *,
        check_cancellation: Callable[[], None] | None = None,
    ) -> dict[str, PVReading]:
        """Set variables and read observable readings in one transaction."""
        ...

    def read_pvs(self, names: list[str]) -> dict[str, PVReading]:
        """Raw caget-style read of arbitrary PV names through this env's CA context.

        Unlike :meth:`get_variables` and :meth:`get_observables`, this does
        not validate names against the env's metadata — callers can pass any
        PV name reachable from this env's interface (e.g. facility-wide
        readings such as beam current). Returns one :class:`PVReading` per
        requested name, preserving native types (waveforms come back as lists)
        and EPICS metadata. PVs that fail to read come back with
        ``connected=False`` rather than being omitted.
        """
        ...

    @property
    def available_objectives(self) -> list[str]:
        """Get list of available objective names supported by this environment."""
        ...

    @property
    def default_objective(self) -> str:
        """Get the default objective name for this environment."""
        ...

    def check_connection(self) -> bool | None:
        """Check if the environment is reachable."""
        ...

    def check_machine_settings(self) -> None:
        """Verify the live machine is fit to start an optimization run.

        Default: no checks, passes trivially. Implementations raise
        ``MachineSettingsError`` aggregating every unmet condition.
        """
        return None

    def __enter__(self) -> "EnvironmentProtocol":
        """Context manager entry."""
        ...

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: Any
    ) -> None:
        """Context manager exit for resource cleanup."""
        ...


EnvironmentFactory = Callable[[str, dict[str, Any]], EnvironmentProtocol]


@runtime_checkable
class OptimizerProtocol(Protocol):
    """Backend-agnostic ask/tell optimizer seam.

    A minimal, pure suggest/ingest surface, deliberately
    compatible with blop 1.0's ``blop.protocols.Optimizer`` so any future
    backend (xopt today, blop later) is a drop-in implementation. The
    search space, objective, and direction are constructor concerns of each
    implementation, not part of this protocol.

    Error contract (part of the interface):

    - The optimizer is pure ask/tell. It never raises or handles *evaluation*
      errors. Hard control-system faults (``ControlSystemUnreachableError``,
      ``EvaluationError``) propagate from the caller's evaluation call and are
      handled there — never through the optimizer.
    - An unreadable objective arrives at :meth:`ingest` as a ``NaN`` value in
      the point dict; implementations must tolerate NaN outcomes rather than
      raising or dropping the point.
    - A failure *inside* the backend (construction, suggest, ingest) surfaces as
      ``OptimizerError``. Backend-specific exception types are translated at the
      implementation's own seam and never cross this protocol.
    """

    def suggest(self, num_points: int | None = None) -> list[dict[str, Any]]:
        """Propose the next point(s) to evaluate.

        Returns a list of point dicts mapping variable names to values. When
        *num_points* is ``None`` the optimizer chooses (typically one). Backend
        bookkeeping keys (e.g. xopt's ``xopt_*``) may be present and must
        round-trip untouched back through :meth:`ingest`.
        """
        ...

    def ingest(self, points: list[dict[str, Any]]) -> None:
        """Tell the optimizer the outcome of evaluated point(s).

        Accepts both previously suggested points and externally injected points
        (e.g. the current machine state used as a seed). Each dict maps
        variable/objective names to values; any backend bookkeeping keys carried
        over from :meth:`suggest` must be accepted untouched. An unreadable
        objective is passed as ``NaN`` and must be tolerated.
        """
        ...


class InterfaceProtocol(Protocol):
    """Protocol for hardware/simulator interface implementations.

    This protocol defines the interface for communicating with either
    real EPICS hardware or mock simulators.
    """

    def read_one(self, channel: str) -> PVReading:
        """Read a single channel, returning its value plus alarm metadata.

        Preserves the native type (scalars as number/str, waveforms as a list).
        **Raises** ``TimeoutError`` / ``ConnectionError`` on failure rather than
        reporting it in the reading — callers that poll (e.g. shot-gating) want
        the exception so they can retry. Lightweight: skips the control-variable
        round-trip, so ``units`` / ``precision`` / ``enum_strings`` may be
        ``None``. Use :meth:`read_many` when you need those.
        """
        ...

    def read_many(self, channel_names: list[str]) -> dict[str, PVReading]:
        """Batch-read channels with retry, enriched with control metadata.

        Lenient per channel: an unreachable PV comes back as a ``PVReading``
        with ``connected=False`` rather than poisoning the batch or raising, so
        one bad name can't sink the rest. Always returns one entry per requested
        name.
        """
        ...

    def set_values(self, channel_inputs: VariableInputs) -> PVValues:
        """Set values to multiple channels.

        Args:
            channel_inputs: Dictionary mapping channel names to values

        Returns:
            Dictionary mapping channel names to actual set values

        Note:
            Errors are raised as exceptions, not returned as None.
        """
        ...

    def check_connection(self, pv_name: str | None = None) -> bool | None:
        """Check if the interface is connected to the underlying hardware/simulator.

        Args:
            pv_name: Optional PV name to check connectivity against.

        Returns:
            True if connected, False if disconnected, None if no PV name provided.
        """
        ...
