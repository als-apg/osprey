"""Deployment-subsystem errors.

Recurring domain errors for container deployment. Framework-level, cross-cutting
errors live in :mod:`osprey.errors`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


class DeploymentError(Exception):
    """Base class for deployment failures."""


class ComposeInterpolationError(DeploymentError):
    """A secret bound for a compose ``env_file:`` contains ``$``.

    Compose interpolates env_file *values*, so such a secret reaches the
    container truncated (or, when the text after ``$`` names a variable set on
    the deploy host, with the host's value spliced in) while the file on disk
    reads correctly. Nothing downstream can tell the difference between a
    truncated secret and a wrong one, so the symptom is an authentication
    failure pointing nowhere near the cause.

    That invisibility is why this refuses the deploy rather than warning. A
    warning would scroll past and leave a stack running with a credential no
    service accepts; the honest outcome is to stop before compose is invoked,
    while the operator still has the context to fix it.

    Carries the offending variable *names* only. The values are secrets and are
    never rendered — the same discipline as
    ``service_tokens._raise_invalid_var``.
    """

    def __init__(self, variables: Sequence[str], path: str | Path) -> None:
        self.variables = list(variables)
        self.path = str(path)
        named = ", ".join(self.variables)
        super().__init__(
            f"{named} in {self.path} contain(s) '$'. Docker Compose interpolates "
            f"env_file values, so the container would receive a truncated secret "
            f"while {self.path} still reads correctly. Refusing to deploy. "
            f"(Values not shown.) Re-issue or rotate the listed secret(s) to a "
            f"'$'-free value — '$' cannot be escaped portably here, because '$$' "
            f"means a literal '$' to Docker Compose but two characters to a "
            f"runtime that does not interpolate."
        )


class DevModeUnavailableError(DeploymentError):
    """``--dev`` was requested but the local osprey wheel cannot be produced.

    ``--dev`` is a statement about *which code runs in the containers*: build a
    wheel from the local checkout so the containers run it instead of the pinned
    PyPI release. When that cannot be done, continuing would deploy a different
    codebase than the one asked for — the containers would come up healthy,
    running released osprey, with no signal that the local checkout was never
    involved. That is worse than not deploying: it silently invalidates whatever
    the deployment was meant to test.

    So this is an error rather than a warning. It carries a ``remedy`` describing
    how to satisfy the precondition, which the CLI renders beneath the reason.
    """

    def __init__(self, reason: str, remedy: str) -> None:
        self.reason = reason
        self.remedy = remedy
        super().__init__(f"--dev cannot be honored: {reason}")


class UnreleasedVersionPinError(DeploymentError):
    """An ``osprey-framework==`` pin was needed, but this build is not a release.

    The same reasoning as :class:`DevModeUnavailableError`, from the other
    direction. A development checkout sits some number of commits past the last
    tag, and no distribution on PyPI corresponds to it. Pinning to the nearest
    release would produce containers running code the operator never wrote, with
    nothing in the logs saying the two differ; emitting no pin at all would leave
    the image tracking whatever PyPI resolves to that day.

    So this refuses, and carries a ``remedy`` the CLI renders beneath the reason.
    """

    def __init__(self, reason: str, remedy: str) -> None:
        self.reason = reason
        self.remedy = remedy
        super().__init__(f"Cannot pin osprey-framework: {reason}")
