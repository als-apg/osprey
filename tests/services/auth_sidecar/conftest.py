"""Fixtures shared by the auth sidecar suites."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from osprey.interfaces._serving import run_app_server
from osprey.services.auth_sidecar.revocation import STATE_DIR_ENV
from osprey_connectors.identity import AUDIT_IDENTITY_ENV, TERMINAL_USER_ENV
from tests.conftest import SpoolZone
from tests.services.auth_sidecar._helpers import SIDECAR_IDENTITY, FakeClock, SidecarZone
from tests.services.auth_sidecar.mock_idp import MockIdP


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def zone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spool_zone: SpoolZone) -> SidecarZone:
    """The sidecar's own spool directory, as compose gives it; not yet created.

    Also binds the service's state directory (``OSPREY_STATE_DIR``) and pins
    the identity ladder to what the sidecar's container env supplies, so a
    developer's local account can never become the actor in an assertion.
    """
    state = tmp_path / "var" / "state" / SIDECAR_IDENTITY
    state.mkdir(parents=True)
    monkeypatch.setenv(STATE_DIR_ENV, str(state))
    monkeypatch.setenv(AUDIT_IDENTITY_ENV, SIDECAR_IDENTITY)
    monkeypatch.delenv(TERMINAL_USER_ENV, raising=False)
    return SidecarZone(spool_zone.spool_for(SIDECAR_IDENTITY), state, spool_zone)


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
