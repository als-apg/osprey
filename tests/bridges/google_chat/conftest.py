"""Shared fixtures for the Google Chat adapter suite."""

import pytest

from osprey.bridges.google_chat import client as client_module


@pytest.fixture(autouse=True)
def chat_sleeps(monkeypatch):
    """Record the client's pacing and rate-limit waits instead of sleeping them.

    Every test that posts more than once through a real ``ChatClient`` would otherwise
    wait :data:`~osprey.bridges.google_chat.client.WRITE_INTERVAL` per message.
    """
    waits: list[float] = []
    monkeypatch.setattr(client_module, "_sleep", waits.append)
    return waits
