"""A stub optimizer behind an ``httpx.MockTransport``, for the optimizer tool suites."""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from osprey.mcp_server.control_system.target_banner import TargetSituation
from osprey.mcp_server.optimizer import server_context

BASE_URL = "http://optimizer.test:9"

#: The session's control target the stub reports, distinct from every baseline.
SESSION_TARGET = "standin"


class StubOptimizer:
    """Records each request and answers it with ``respond`` (default: 200 ``{}``)."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.respond: Callable[[httpx.Request], httpx.Response] = lambda _r: httpx.Response(
            200, json={}
        )

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.respond(request)


@pytest.fixture
def optimizer(monkeypatch) -> StubOptimizer:
    stub = StubOptimizer()
    real_client = httpx.AsyncClient

    def client(**kwargs):
        return real_client(transport=httpx.MockTransport(stub.handle), **kwargs)

    monkeypatch.setenv("OPTIMIZER_URL", BASE_URL)
    monkeypatch.setattr(server_context, "load_osprey_config", dict)
    monkeypatch.setattr(server_context.httpx, "AsyncClient", client)
    monkeypatch.setattr(
        server_context,
        "resolve_target_situation",
        lambda: TargetSituation(control_target=SESSION_TARGET, baseline_target="va"),
    )
    return stub
