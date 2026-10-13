"""Tests for utils/control.py — handle_control_callbacks and Controllable protocol."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest

from osprey.services.optimizer.utils.control import (
    check_cancellation_token,
    handle_control_callbacks,
)


@dataclass
class FakeControllable:
    """Test double satisfying the Controllable protocol."""

    check_cancellation: Callable[[], None] | None = field(default=None)
    wait_if_paused: Callable[[], None] | None = field(default=None)


class TestHandleControlCallbacks:
    @pytest.mark.parametrize(
        ("set_cancel", "set_pause", "expected_order"),
        [
            pytest.param(False, False, [], id="both_none_is_noop"),
            pytest.param(True, False, ["cancel"], id="only_check_cancellation"),
            pytest.param(False, True, ["pause"], id="only_wait_if_paused"),
            pytest.param(True, True, ["cancel", "pause"], id="both_called_in_order"),
        ],
    )
    def test_calls_each_configured_callback_once_in_order(
        self, set_cancel: bool, set_pause: bool, expected_order: list[str]
    ) -> None:
        """Only the callbacks that are set are invoked, each exactly once, cancel first."""
        call_order: list[str] = []
        cancel = MagicMock(side_effect=lambda: call_order.append("cancel"))
        pause = MagicMock(side_effect=lambda: call_order.append("pause"))
        ctx = FakeControllable(
            check_cancellation=cancel if set_cancel else None,
            wait_if_paused=pause if set_pause else None,
        )

        handle_control_callbacks(ctx)

        assert call_order == expected_order


def test_check_cancellation_token_none_is_noop() -> None:
    """token None -> 33->exit (no calls)."""
    check_cancellation_token(None)  # must not raise


def test_check_cancellation_token_invokes_both() -> None:
    token = MagicMock()
    check_cancellation_token(token)
    token.check_cancellation.assert_called_once()
    token.wait_if_paused.assert_called_once()
