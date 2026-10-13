"""Custom exceptions for ALS injector optimization."""

import functools
from collections.abc import Callable


class ALSTuningError(Exception):
    """Base exception for ALS tuning errors."""

    pass


class CancellationError(ALSTuningError):
    """Raised when optimization is cancelled by user."""

    pass


class InterfaceError(ALSTuningError):
    """Hardware interface communication error."""

    pass


class ConnectionError(InterfaceError):
    """Failed to connect to hardware."""

    pass


class TimeoutError(InterfaceError):
    """Operation timed out."""

    pass


class WriteRefusedError(InterfaceError):
    """Write refused by control-system policy before being attempted (blocked).

    Distinct from a write that was attempted but is unverified (failed): a
    refusal never retries — policy does not become true by retrying — and the
    run loop aborts the phase loudly with the operator-facing ``reason``.
    """

    def __init__(self, message: str, *, reason: str) -> None:
        super().__init__(message)
        self.reason = reason

    def __reduce__(self) -> tuple[Callable[..., "WriteRefusedError"], tuple[str]]:
        # The default reduce replays args positionally, dropping the keyword-only reason.
        return functools.partial(type(self), reason=self.reason), (str(self),)


class ControlSystemUnreachableError(InterfaceError):
    """Raised when the control system is persistently unreachable.

    Distinct from a transient ConnectionError/TimeoutError: this signals that
    retries have been exhausted and the optimization run should abort.
    """

    pass


class EvaluationError(ALSTuningError):
    """Error during optimization evaluation."""

    pass


class InvalidSuggestionError(EvaluationError):
    """Optimizer suggested a variable value that is not a finite real number.

    Raised at the suggest seam (before the value can reach hardware) when a
    generator produces NaN/inf, a non-numeric value, or omits a variable.
    Subclasses ``EvaluationError`` so the run loop's existing phase-failure
    handling aborts the phase loudly.
    """

    pass


class OptimizerError(ALSTuningError):
    """The optimizer backend failed internally.

    Raised by an :class:`~osprey.services.optimizer.models.protocols.OptimizerProtocol`
    implementation when its backend fails during construction, ``suggest``, or
    ``ingest``. Backend-specific exception types are translated to this at the
    adapter so they never cross the protocol seam.
    """

    pass


class VariableOutOfBoundsError(ALSTuningError):
    """Variable value is outside defined bounds."""

    pass


class ObservableNotFoundError(ALSTuningError):
    """Requested observable does not exist."""

    pass


class MachineSettingsError(ALSTuningError):
    """The live machine's settings refuse an optimization start.

    Raised by ``EnvironmentProtocol.check_machine_settings`` with every unmet
    condition aggregated in ``failures``; surfaced as a start-time precondition
    failure, never during a running optimization.
    """

    def __init__(self, message: str, *, failures: list[str]) -> None:
        super().__init__(message)
        self.failures = failures


class RestoreFailedError(ALSTuningError):
    """A run ended (cancelled, failed, or finished with ``on_complete: restore``)
    and writing its starting values back failed.

    Classified as ERROR, never CANCELLED: the machine is not where the operator
    was promised it would be, and the job's outcome must say so.
    """
