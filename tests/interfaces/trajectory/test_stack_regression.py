"""Replay of the frozen manual steps against a deployed trajectory stack.

``test_manual_steps.py`` proves each counted step of the manual on the
in-process stack's fake machine, on every pull request. This module replays
the same steps, keyed by the same :data:`MANUAL_STEPS` tuple, in a real
browser against a stack the e2e deploy fixture builds and starts, so the
deployed images are held to what the frozen steps froze.

It needs Docker and an image build, so every per-PR run deselects the
``stack_regression`` marker; the module runs only in the on-demand
``trajectory-stack-regression`` job of a ``run_benchmarks`` dispatch.
"""

from __future__ import annotations

import pytest

from tests.interfaces.trajectory.test_manual_steps import MANUAL_STEPS

pytestmark = [pytest.mark.stack_regression]


def test_the_replay_is_keyed_by_every_counted_step() -> None:
    """The replay walks the manual's eighteen counted steps, each once."""
    assert [step.number for step in MANUAL_STEPS] == list(range(1, 19))
