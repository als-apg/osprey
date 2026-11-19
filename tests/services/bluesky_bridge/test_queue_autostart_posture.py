"""The bridge's startup posture for the plan queue: ``bluesky.queue_autostart``.

On, the bridge arms the manager (autostart on) once the worker environment is
open, so a plan added to the queue runs without a separate Start. Off, the
manager is explicitly disarmed — the flag persists in Redis across manager
restarts, so a deployment reconfigured from "runs by default" to "starts on
request" must not carry the old posture forward.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from osprey.services.bluesky_bridge import app as app_module
from osprey.services.bluesky_bridge.queue_backend import QueueBackend
from osprey_connectors.testing import config_of
from osprey_connectors_tests._config_trees import serve_config
from tests.services.bluesky_bridge._queue_fakes import FakeManager

#: Stands for a deployment whose config cannot be read at all.
_UNREADABLE = "unreadable"


def _configure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, config: Any) -> None:
    """Serve *config* as the process config the bridge reads; :data:`_UNREADABLE`
    points the process at a config file that is not there."""
    if config == _UNREADABLE:
        monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "missing.yml"))
        monkeypatch.chdir(tmp_path)
        return
    serve_config(config_of(config))


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
        (_UNREADABLE, False),
    ],
)
def test_the_posture_is_read_off_config_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, config: Any, expected: bool
) -> None:
    _configure(monkeypatch, tmp_path, config)
    assert app_module.queue_autostart_configured() is expected


async def test_an_on_posture_arms_the_manager_once_the_environment_is_open() -> None:
    serve_config(config_of({"bluesky": {"queue_autostart": True}}))
    manager = FakeManager()
    app_module.set_queue_backend(QueueBackend(manager))

    await app_module._apply_queue_autostart_posture(environment_open=True)

    assert manager.kwargs_for("queue_autostart") == [{"enable": True}]


async def test_an_on_posture_does_not_arm_over_a_closed_environment() -> None:
    # Autostart drains only with an open environment; an armed flag over a
    # closed one would read as "ready" while nothing could run.
    serve_config(config_of({"bluesky": {"queue_autostart": True}}))
    manager = FakeManager()
    app_module.set_queue_backend(QueueBackend(manager))

    await app_module._apply_queue_autostart_posture(environment_open=False)

    assert manager.calls == []


async def test_an_off_posture_disarms_explicitly() -> None:
    # The manager remembers the flag across its own restarts, so "off" is a
    # write, not an absence of one.
    serve_config(config_of({"bluesky": {}}))
    manager = FakeManager()
    app_module.set_queue_backend(QueueBackend(manager))

    await app_module._apply_queue_autostart_posture(environment_open=True)

    assert manager.kwargs_for("queue_autostart") == [{"enable": False}]
