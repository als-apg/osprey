"""The bridge's startup posture for the plan queue: ``bluesky.queue_autostart``.

On, the bridge arms the manager (autostart on) once the worker environment is
open, so a plan added to the queue runs without a separate Start. Off, the
manager is explicitly disarmed — the flag persists in Redis across manager
restarts, so a deployment reconfigured from "runs by default" to "starts on
request" must not carry the old posture forward.
"""

from __future__ import annotations

from typing import Any

import pytest

from osprey.services.bluesky_bridge import app as app_module
from osprey.services.bluesky_bridge.queue_backend import QueueBackend

from .test_queue_routes import FakeManager


@pytest.fixture(autouse=True)
def _isolated_backend():
    app_module.set_queue_backend(None)
    yield
    app_module.set_queue_backend(None)


def _configure(monkeypatch: pytest.MonkeyPatch, config: Any) -> None:
    """Stand in for the bridge's config read (`get_config_value`), the same
    seam the route tests' `connector` fixture patches."""

    def fake_get_config_value(key: str, default: Any = None) -> Any:
        if isinstance(config, Exception):
            raise config
        return config.get(key, default)

    monkeypatch.setattr("osprey.utils.config.get_config_value", fake_get_config_value)


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"bluesky": {"queue_autostart": True}}, True),
        ({"bluesky": {"queue_autostart": False}}, False),
        ({"bluesky": {}}, False),
        ({}, False),
        # Only a literal true arms: a string or a number is not a posture.
        ({"bluesky": {"queue_autostart": "yes"}}, False),
        ({"bluesky": {"queue_autostart": 1}}, False),
        (RuntimeError("no config"), False),
    ],
)
def test_the_posture_is_read_off_config_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch, config: Any, expected: bool
) -> None:
    _configure(monkeypatch, config)
    assert app_module.queue_autostart_configured() is expected


async def test_an_on_posture_arms_the_manager_once_the_environment_is_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure(monkeypatch, {"bluesky": {"queue_autostart": True}})
    manager = FakeManager()
    app_module.set_queue_backend(QueueBackend(manager))

    await app_module._apply_queue_autostart_posture(environment_open=True)

    assert manager.kwargs_for("queue_autostart") == [{"enable": True}]


async def test_an_on_posture_does_not_arm_over_a_closed_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Autostart drains only with an open environment; an armed flag over a
    # closed one would read as "ready" while nothing could run.
    _configure(monkeypatch, {"bluesky": {"queue_autostart": True}})
    manager = FakeManager()
    app_module.set_queue_backend(QueueBackend(manager))

    await app_module._apply_queue_autostart_posture(environment_open=False)

    assert manager.calls == []


async def test_an_off_posture_disarms_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    # The manager remembers the flag across its own restarts, so "off" is a
    # write, not an absence of one.
    _configure(monkeypatch, {"bluesky": {}})
    manager = FakeManager()
    app_module.set_queue_backend(QueueBackend(manager))

    await app_module._apply_queue_autostart_posture(environment_open=True)

    assert manager.kwargs_for("queue_autostart") == [{"enable": False}]
