"""Shared control/cancellation helpers for optimization pipelines."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from osprey.services.optimizer.utils.cancellation import CancellationToken


class Controllable(Protocol):
    """Any context that supports cancellation and pause control."""

    check_cancellation: Callable[[], None] | None
    wait_if_paused: Callable[[], None] | None


def handle_control_callbacks(context: Controllable) -> None:
    """Invoke cancellation and pause callbacks if provided."""
    if context.check_cancellation:
        context.check_cancellation()
    if context.wait_if_paused:
        context.wait_if_paused()


def check_cancellation_token(token: CancellationToken | None) -> None:
    """Check cancellation and wait-if-paused on a :class:`CancellationToken`.

    Convenience wrapper used by multiple service classes that hold a
    ``CancellationToken | None`` rather than a full :class:`Controllable`.
    """
    if token:
        token.check_cancellation()
        token.wait_if_paused()
