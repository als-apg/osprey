"""Stand a fake in for one service kind behind ``ConnectorFactory.create_service``.

``create_service`` builds all four service kinds, so a test that fakes one kind
routes that kind's specs to its fake and every other kind to whatever answered
before: the real factory, or another kind's fake.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import pytest


def patch_create_service(
    monkeypatch: pytest.MonkeyPatch, kind: str, create: Callable[..., Awaitable[Any]]
) -> None:
    """Route ``create_service`` calls whose ``spec.kind`` is *kind* to *create*.

    Args:
        monkeypatch: Undoes the patch at the end of the test.
        kind: The connectors kind: ``directory``, ``alarm_state``, ``alarm_log``
            or ``put_log``.
        create: Called as ``create_service`` is, with the spec and its keywords.
    """
    from osprey_connectors.factory import ConnectorFactory

    previous = ConnectorFactory.create_service

    async def routed(spec: Any, **kwargs: Any) -> Any:
        if spec.kind == kind:
            return await create(spec, **kwargs)
        return await previous(spec, **kwargs)

    monkeypatch.setattr(ConnectorFactory, "create_service", routed)
