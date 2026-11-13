"""The session database's connects are bounded, and a server that is not there says so."""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Iterator
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

from tests import _session_database
from tests._session_database import DEV_SERVER_URL_5432, DEV_SERVER_URL_5433, own_database

pytest.importorskip("psycopg")


@pytest.fixture
def silent_server() -> Iterator[int]:
    """A port on the loopback that takes a TCP connection and never answers it.

    The kernel completes the handshake into the listen backlog, so the client's
    connect succeeds and its startup packet is never read: only the client's
    own deadline ends the wait.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(8)
        yield listener.getsockname()[1]


@pytest.mark.parametrize("url", [DEV_SERVER_URL_5432, DEV_SERVER_URL_5433], ids=["5432", "5433"])
def test_the_dev_servers_are_named_by_address_so_no_resolver_is_asked(url):
    ipaddress.ip_address(urlsplit(url).hostname or "")


# The hang-breaker the bound is measured against: psycopg's default connect
# wait is 130 s, so an unbounded connect to the silent server runs into it.
@pytest.mark.timeout(60)
def test_a_configured_server_that_does_not_answer_fails_in_one_line(silent_server, monkeypatch):
    monkeypatch.setattr(_session_database, "CONNECT_TIMEOUT_S", 2)
    request = SimpleNamespace(addfinalizer=lambda _fn: None)
    url = f"postgresql://ariel:ariel@127.0.0.1:{silent_server}/ariel"

    with pytest.raises(pytest.fail.Exception) as caught:
        own_database(request, url, "ariel_test")  # type: ignore[arg-type]

    message = str(caught.value)
    assert "\n" not in message, message
    assert f"127.0.0.1:{silent_server}" in message
    assert "CREATE DATABASE" in message
    assert "connect_timeout 2 s" in message
