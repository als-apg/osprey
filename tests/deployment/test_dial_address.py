"""One rule for every host-side dial of a published port.

``deployment.bind_address`` says where a service *listens*. Turning it into an
address to *connect to* is a separate question, and three modules used to answer
it three ways: the provisioner and the health category dialed the bind literal
(unreachable for a wildcard on some platforms), the qmd sidecar always dialed
loopback (wrong for a pinned interface), and the host-port preflight had the
rule the others should have had. These tests hold every reader to that one rule.
"""

from __future__ import annotations

from urllib.parse import urlsplit

import httpx
import pytest

from osprey.deployment import openobserve_provision as provision
from osprey.deployment.host_ports import _probe_host
from osprey.deployment.qmd_service import (
    QMDServiceConfig,
    dial_address,
    dial_host,
    is_loopback_bind,
)
from osprey.health.core.openobserve import openobserve

#: Every spelling of "listen on every interface", with the loopback address of
#: the family it belongs to.
WILDCARDS = {"": "127.0.0.1", "0.0.0.0": "127.0.0.1", "*": "127.0.0.1", "::": "::1"}

#: Addresses that name one interface and are therefore dialed as written.
PINNED = ["10.0.0.7", "192.168.1.4", "myhost.example.org", "127.0.0.1"]


@pytest.mark.parametrize(("bind", "loopback"), WILDCARDS.items())
def test_wildcard_dials_its_family_loopback(bind: str, loopback: str) -> None:
    assert dial_host(bind) == loopback


@pytest.mark.parametrize("bind", PINNED)
def test_pinned_interface_is_dialed_as_written(bind: str) -> None:
    """Not "always loopback": a pinned bind publishes on that interface only."""
    assert dial_host(bind) == bind


def test_ipv6_literal_is_bracketed_for_a_url_authority() -> None:
    assert dial_address("::") == "[::1]"
    assert dial_address("::1") == "[::1]"
    assert dial_address("0.0.0.0") == "127.0.0.1"


def test_probe_host_is_the_same_rule() -> None:
    """The preflight's prober and the URL builders cannot drift apart."""
    for bind, loopback in WILDCARDS.items():
        assert _probe_host(bind) == loopback


@pytest.mark.parametrize(("bind", "loopback"), WILDCARDS.items())
def test_no_wildcard_survives_into_a_dialed_url(bind: str, loopback: str) -> None:
    """Parity: every URL built for a wildcard bind names a loopback address."""
    config = {
        "deployed_services": ["openobserve"],
        "services": {"openobserve": {"port": 15080}},
        "deployment": {"bind_address": bind},
    }
    urls = [
        provision.store_base_url(config),
        QMDServiceConfig(port=8180, bind_address=bind).base_url,
        _healthz_url(config),
    ]
    expected = f"[{loopback}]" if ":" in loopback else loopback
    for url in urls:
        assert urlsplit(url).netloc.rsplit(":", 1)[0] == expected, url


def _healthz_url(config: dict) -> str:
    """The URL the openobserve health category actually requests."""
    captured: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(str(request.url))
        return httpx.Response(200)

    import asyncio

    asyncio.run(openobserve(config, transport=httpx.MockTransport(handler))())
    return captured[0]


@pytest.mark.parametrize("bind", ["127.0.0.1", "::1", "localhost", "127.0.0.2"])
def test_loopback_spellings_are_not_exposed(bind: str) -> None:
    assert is_loopback_bind(bind) is True


@pytest.mark.parametrize("bind", ["0.0.0.0", "::", "", "*", "10.0.0.7", "myhost.example"])
def test_everything_else_is_exposed(bind: str) -> None:
    assert is_loopback_bind(bind) is False
