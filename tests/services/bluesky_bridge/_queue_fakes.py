"""A scripted queueserver manager, and the bridge backend installed around it."""

from __future__ import annotations

from typing import Any

from osprey.services.bluesky_bridge import app as app_module
from osprey.services.bluesky_bridge.queue_backend import QueueBackend


class FakeManager:
    """A scripted ``REManagerAPI`` stand-in.

    Records every call as ``(method, kwargs)`` and answers from ``responses``.
    A response that is an exception instance is raised instead of returned; a
    response that is a list is consumed one call at a time (the tail repeats),
    which is how the environment-open polling loop gets a state sequence.
    """

    def __init__(self, **responses: Any) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.responses = responses
        self.closed = False

    def _answer(self, method: str) -> Any:
        response = self.responses.get(method, {"success": True, "msg": ""})
        if isinstance(response, list):
            value = response[0]
            if len(response) > 1:
                response.pop(0)
        else:
            value = response
        if isinstance(value, Exception):
            raise value
        return value

    def __getattr__(self, method: str) -> Any:
        async def call(**kwargs: Any) -> Any:
            self.calls.append((method, kwargs))
            return self._answer(method)

        return call

    async def close(self) -> None:
        self.closed = True

    def kwargs_for(self, method: str) -> list[dict[str, Any]]:
        return [kwargs for name, kwargs in self.calls if name == method]

    def method_names(self) -> list[str]:
        return [name for name, _ in self.calls]


def status_doc(**overrides: Any) -> dict[str, Any]:
    """A manager status document: idle, environment open, nothing running."""
    doc = {
        "success": True,
        "manager_state": "idle",
        "worker_environment_exists": True,
        "items_in_queue": 0,
        "items_in_history": 0,
        "running_item_uid": None,
        "plan_queue_uid": "q-1",
        "plan_history_uid": "h-1",
        "queue_stop_pending": False,
        "queue_autostart_enabled": False,
    }
    doc.update(overrides)
    return doc


def install_backend(manager: Any) -> QueueBackend:
    backend = QueueBackend(manager)
    app_module.set_queue_backend(backend)
    return backend


def install_abort_backend(manager: Any) -> QueueBackend:
    """Install a backend whose abort pause window closes fast enough to test."""
    backend = QueueBackend(manager, abort_pause_polls=2, abort_poll_interval=0)
    app_module.set_queue_backend(backend)
    return backend
