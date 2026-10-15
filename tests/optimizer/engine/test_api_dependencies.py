"""Direct-call tests for api/dependencies.py.

These bypass ``dependency_overrides`` so the function bodies actually run.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

from osprey_optimizer.api.dependencies import (
    get_environment_registry,
    get_job_tracker,
)


def _fake_request(**state: object) -> SimpleNamespace:
    """A minimal stand-in for a FastAPI Request with an app.state namespace."""
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(**state)))


@pytest.mark.parametrize(
    "getter,state_key",
    [
        (get_job_tracker, "job_tracker"),
        (get_environment_registry, "environment_registry"),
    ],
    ids=["job_tracker", "environment_registry"],
)
def test_getter_reads_app_state(getter: Callable[[Any], Any], state_key: str) -> None:
    """Each getter returns the object stored on app.state."""
    sentinel = object()
    req = _fake_request(**{state_key: sentinel})
    assert getter(req) is sentinel
