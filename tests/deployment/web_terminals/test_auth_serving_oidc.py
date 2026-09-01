"""One OIDC login through the rendered perimeter, where account and subject differ.

Every other test that puts a real request through the real nginx into the real
sidecar runs under ``auth.method: password`` — the serving suite beside this
file and the full chain in ``tests/e2e/test_full_chain_auth.py``. There the
roster account and the login subject are the same string, so each of their
header assertions compares a value with itself, and a conflation of
``X-Osprey-Auth-Account`` and ``X-Osprey-Auth-Subject`` anywhere between
``/verify`` and the terminal app is invisible to all of them. It was invisible
once: the audit middleware compared the provider's subject against the
container's roster user, and every request of a healthy OIDC deployment carried
a mismatch marker until someone read a log.

Under OIDC the two values differ, and this module drives exactly one such login
so the upstream's own view of the two headers is asserted, once, against values
that are not equal. That is the whole scope. The handshake itself — discovery,
the nonce, the ID token's signature and claims, identity-to-roster mapping in
every failure shape — is proven in ``tests/services/auth_sidecar/test_oidc_flow.py``
and is not re-proved by standing up containers; what the ledger then writes is
``tests/interfaces/test_http_audit_emitters.py``'s.

The stack is :func:`~tests.deployment.web_terminals.test_auth_serving.serving_stack`
with ``oidc=True``: the same rendered artifacts and the same replay, plus that
flow test's own stub provider served from a fourth container inside the shared
network namespace, so one ``http://127.0.0.1:<port>`` issuer resolves identically
for the sidecar dialling discovery from inside and for the client walking the
redirect from the host.
"""

from __future__ import annotations

from collections.abc import Iterator

import httpx
import pytest

from osprey.services.auth_sidecar.identity_headers import ACCOUNT_HEADER, SUBJECT_HEADER
from osprey.services.auth_sidecar.routes.oidc import CALLBACK_PATH, LOGIN_PATH
from tests.deployment.web_terminals import test_auth_serving as serving
from tests.deployment.web_terminals.test_auth_serving import OIDC_SUBJECTS, Stack, serving_stack

pytestmark = serving.pytestmark

USER = "alice"
"""The one login walked. Her roster mapping is the subject the stub provider
asserts by default, so hers is the handshake that completes."""

PROVIDER_SUBJECT = OIDC_SUBJECTS[USER]

TOP_LEVEL_NAVIGATION = {
    "Sec-Fetch-Site": "cross-site",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Dest": "document",
}
"""What a browser sends when the provider redirects it back: a cross-site
top-level GET, the request shape the Lax state cookie is admitted on."""


@pytest.fixture(scope="module")
def stack(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Stack]:
    """The rendered perimeter in ``oidc`` mode, with the provider beside it."""
    with serving_stack(tmp_path_factory.mktemp("auth-serving-oidc"), oidc=True) as running:
        yield running


def test_an_oidc_login_forwards_the_roster_account_and_the_provider_subject_as_two_headers(
    stack: Stack,
) -> None:
    """The one assertion password mode cannot make.

    Walked, never assembled: the login route's redirect is followed to the
    provider with a real request, and the provider's redirect is handed back
    through nginx as a browser would hand it. What the upstream then receives
    is read from the stub's transcript — the app's view, not the client's.
    """
    assert stack.idp_issuer is not None
    with stack.client() as client:
        start = client.get(LOGIN_PATH, params={"user": USER})
        assert start.status_code == 302, start.text
        to_provider = start.headers["location"]
        assert to_provider.startswith(stack.idp_issuer), to_provider

        # The provider is a listening server the client reaches from the host,
        # on the same issuer string the sidecar dialled from inside.
        with httpx.Client(follow_redirects=False, timeout=15) as browser:
            handoff = browser.get(to_provider)
        assert handoff.status_code == 302, handoff.text
        back = handoff.headers["location"]
        assert back.startswith(f"{stack.base_url}{CALLBACK_PATH}"), back

        landed = client.get(back, headers=TOP_LEVEL_NAVIGATION)
        # 303 See Other: the callback consumed the flow and sends the browser on.
        assert landed.status_code == 303, landed.text
        assert landed.headers["location"].startswith(f"/u/{USER}/"), landed.headers["location"]

        report = client.get(f"/u/{USER}/identity").json()

    headers = report["headers"]
    # The account names the container's own user: the upstream nginx dialled is
    # the one rendered for that roster entry.
    assert report["port"] == stack.upstream_ports[USER], report
    assert headers.get(ACCOUNT_HEADER.lower()) == USER, report
    assert headers.get(SUBJECT_HEADER.lower()) == PROVIDER_SUBJECT, report
    assert headers[ACCOUNT_HEADER.lower()] != headers[SUBJECT_HEADER.lower()], report
