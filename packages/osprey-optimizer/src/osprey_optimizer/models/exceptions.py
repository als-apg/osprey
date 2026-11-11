"""Custom exceptions for the optimizer service."""

import functools
from collections.abc import Callable


class OptimizerServiceError(Exception):
    """Base exception for optimizer errors."""


class CancellationError(OptimizerServiceError):
    """Raised when optimization is cancelled by user."""


class InterfaceError(OptimizerServiceError):
    """Hardware interface communication error."""


class ConnectionError(InterfaceError):
    """Failed to connect to hardware."""


class TimeoutError(InterfaceError):
    """Operation timed out."""


class WriteRefusedError(InterfaceError):
    """Write refused by control-system policy before being attempted (blocked).

    Distinct from a write that was attempted but is unverified (failed): a
    refusal never retries — policy does not become true by retrying — and the
    run loop aborts the phase loudly with the operator-facing ``reason``.
    """

    def __init__(
        self, message: str, *, reason: str | None, verdict_reason: str | None = None
    ) -> None:
        super().__init__(message)
        self.reason = reason
        #: The connector's write-verdict reason (``switch_off``, ``auth_range`` ...), or ``None``.
        self.verdict_reason = verdict_reason

    def __reduce__(self) -> tuple[Callable[..., "WriteRefusedError"], tuple[str]]:
        # The default reduce replays args positionally, dropping the keyword-only reasons.
        return (
            functools.partial(type(self), reason=self.reason, verdict_reason=self.verdict_reason),
            (str(self),),
        )


class AuthorizationEndedError(WriteRefusedError):
    """The run's write authorization ended, or was never had: no further write is made.

    ``reason`` is the write-verdict reason that ended it (``auth_exhausted``,
    ``auth_revoked``, ``auth_range``, ``auth_lease_lapsed`` ...), the reason an
    authorization could not be had at all (``owner_unknown``, ``tier_never`` ...),
    or one of the optimizer's own words (:data:`AUTHORIZATION_TIMEOUT`,
    :data:`AUTHORIZATION_TOO_MANY_STEPS`).
    Nothing asks again: the run ends here, holding the machine where its last
    write left it.
    """

    def __init__(self, reason: str, detail: str | None = None) -> None:
        self.detail = detail
        super().__init__(_ended_message(reason, detail), reason=reason, verdict_reason=reason)

    def __reduce__(self) -> tuple[Callable[..., "WriteRefusedError"], tuple[str]]:
        # The message is rebuilt from the reason; replay the reason and the detail.
        return functools.partial(type(self), detail=self.detail), (str(self.reason),)


#: The reason a run ends when nobody approved its request in time.
AUTHORIZATION_TIMEOUT = "authorization_timeout"
#: The reason a run ends when it needs more write calls than one authorization may allow.
AUTHORIZATION_TOO_MANY_STEPS = "authorization_too_many_steps"


def _ended_message(reason: str, detail: str | None) -> str:
    """The operator's words for an authorization that ended for *reason*."""
    from osprey_connectors.refusals import refusal_phrase

    try:
        phrase: str | None = refusal_phrase(reason)
    except ValueError:  # one of the optimizer's own words, not a verdict reason
        phrase = None
    words = " ".join(part for part in (phrase, detail) if part)
    return f"The run's write authorization ended [{reason}]" + (f": {words}" if words else ".")


class ControlSystemUnreachableError(InterfaceError):
    """Raised when the control system is persistently unreachable.

    Distinct from a transient ConnectionError/TimeoutError: this signals that
    retries have been exhausted and the optimization run should abort.
    """


class EvaluationError(OptimizerServiceError):
    """Error during optimization evaluation."""


class InvalidSuggestionError(EvaluationError):
    """Optimizer suggested a variable value that is not a finite real number.

    Raised at the suggest seam (before the value can reach hardware) when a
    generator produces NaN/inf, a non-numeric value, or omits a variable.
    Subclasses ``EvaluationError`` so the run loop's existing phase-failure
    handling aborts the phase loudly.
    """


class OptimizerError(OptimizerServiceError):
    """The optimizer backend failed internally.

    Raised by an :class:`~osprey_optimizer.models.protocols.OptimizerProtocol`
    implementation when its backend fails during construction, ``suggest``, or
    ``ingest``. Backend-specific exception types are translated to this at the
    adapter so they never cross the protocol seam.
    """


class VariableOutOfBoundsError(OptimizerServiceError):
    """Variable value is outside defined bounds."""


class ObservableNotFoundError(OptimizerServiceError):
    """Requested observable does not exist."""


class MachineSettingsError(OptimizerServiceError):
    """The live machine's settings refuse an optimization start.

    Raised by ``EnvironmentProtocol.check_machine_settings`` with every unmet
    condition aggregated in ``failures``; surfaced as a start-time precondition
    failure, never during a running optimization.
    """

    def __init__(self, message: str, *, failures: list[str]) -> None:
        super().__init__(message)
        self.failures = failures


class RestoreFailedError(OptimizerServiceError):
    """A run ended (cancelled, failed, or finished with ``on_complete: restore``)
    and writing its starting values back failed.

    Classified as ERROR, never CANCELLED: the machine is not where the operator
    was promised it would be, and the job's outcome must say so.
    """
