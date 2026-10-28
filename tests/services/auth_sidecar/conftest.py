"""Fixtures shared by the auth sidecar suites."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from osprey.interfaces._serving import run_app_server
from osprey.services.auth_sidecar import audit
from osprey_connectors.identity import AUDIT_IDENTITY_ENV, TERMINAL_USER_ENV
from tests.services.auth_sidecar._helpers import FakeClock
from tests.services.auth_sidecar.mock_idp import MockIdP


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def zone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The sidecar's own bound audit subdirectory, as compose gives it.

    Also pins the identity ladder to what the sidecar's container env supplies,
    so a developer's local account can never become the actor in an assertion.
    """
    directory = tmp_path / "var" / "audit" / "sidecar"
    directory.mkdir(parents=True)
    monkeypatch.setenv(audit.AUDIT_DIR_ENV, str(directory))
    monkeypatch.setenv(AUDIT_IDENTITY_ENV, "sidecar")
    monkeypatch.delenv(TERMINAL_USER_ENV, raising=False)
    return directory


@pytest.fixture(scope="module")
def idp() -> Iterator[MockIdP]:
    """One stub provider, listening on a real port for the whole module.

    Served once rather than per test: the sidecar reaches it over TCP, so it
    cannot be an ASGI object handed to a client, and standing a server up for
    each test costs more than resetting one.
    """
    provider = MockIdP()
    with run_app_server(provider.app) as base_url:
        provider.issuer = base_url
        yield provider
