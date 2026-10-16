"""The history app's lifespan keeps a store the factory built.

A served app runs its lifespan (uvicorn and ``TestClient`` used as a context
manager both do). A store :func:`create_app` built from a ``transport`` must
survive that startup; only an app with no store opens its own connection, and
only that connection is closed at shutdown.
"""

from __future__ import annotations

import httpx
from fastapi.testclient import TestClient

from osprey.history.app import create_app
from osprey.history.store import HistoryStore


def _handler(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, text="")


def test_the_lifespan_keeps_a_store_the_factory_built() -> None:
    app = create_app(store_url="http://store.test:8123", transport=httpx.MockTransport(_handler))
    built = app.state.store

    with TestClient(app):
        assert app.state.store is built

    assert app.state.store is built


def test_the_lifespan_opens_a_store_when_the_factory_built_none() -> None:
    app = create_app(store_url="http://store.test:8123", store_password="pw")
    assert getattr(app.state, "store", None) is None

    with TestClient(app):
        opened = app.state.store
        assert isinstance(opened, HistoryStore)
        client = opened._client
        assert not client.is_closed

    assert client.is_closed
