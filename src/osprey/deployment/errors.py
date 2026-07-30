"""Deployment-subsystem errors.

Recurring domain errors for container deployment. Framework-level, cross-cutting
errors live in :mod:`osprey.errors`.
"""

from __future__ import annotations


class DeploymentError(Exception):
    """Base class for deployment failures."""


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
