"""A scripted Authlib client and a forged in-flight handshake, for the OIDC callback suites."""

from __future__ import annotations

import json
from base64 import b64encode
from typing import Any

import httpx
import itsdangerous
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.services.auth_sidecar.app import STATE_COOKIE_NAME
from osprey.services.auth_sidecar.routes.oidc import CALLBACK_PATH, PENDING_FLOW_SESSION_KEY
from tests.services.auth_sidecar._helpers import STATE_SECRET

EXTERNAL_ORIGIN = "https://terminals.example.org"
ALICE_SUBJECT = "idp|alice"
FLOW_STATE = "handshake-state-value"
#: Stands in for the compact JWT a token response carries. Never parsed here:
#: Authlib does that, and the route only asks whether one was present at all.
ID_TOKEN = "header.payload.signature"
IDP_AUTHORIZE_URL = "https://idp.example.org/authorize?state=" + FLOW_STATE


class FakeOIDCClient:
    """Stands in for Authlib's Starlette client at ``app.state.oidc_client``.

    Returns the claims the test wants the callback to see. It records whether
    the token exchange ran at all (``exchanged``: a callback that refuses
    *before* the exchange is proved by it staying false) and the keyword
    arguments the route handed it (``token_kwargs``, where ``claims_options`` is
    asserted on).
    """

    def __init__(self, userinfo: dict[str, Any] | None = None, *, id_token: bool = True) -> None:
        # `id_token` is what Authlib parsed `userinfo` FROM: the real client
        # writes that key only when the token response carried one, so a
        # stand-in that always supplies claims without it models a token the
        # route is entitled to trust and this one is not. `id_token=False` is
        # the OAuth2-provider shape.
        self.token: dict[str, Any] = {
            "userinfo": userinfo if userinfo is not None else {"sub": ALICE_SUBJECT}
        }
        if id_token:
            self.token["id_token"] = ID_TOKEN
        self.exchanged = False
        self.token_kwargs: dict[str, Any] = {}

    async def create_authorization_url(self, _redirect_uri: str | None = None) -> dict[str, Any]:
        return {"url": IDP_AUTHORIZE_URL, "state": FLOW_STATE, "nonce": "nonce-value"}

    async def save_authorize_data(self, request: Any, **kwargs: Any) -> None:  # noqa: ARG002 - the OIDC client signature
        return None

    async def authorize_access_token(self, request: Any, **kwargs: Any) -> dict[str, Any]:  # noqa: ARG002 - the OIDC client signature
        self.exchanged = True
        self.token_kwargs = kwargs
        return self.token


def pending_state(user: str) -> str:
    """A forged state cookie carrying one in-flight handshake for ``user``."""
    signer = itsdangerous.TimestampSigner(STATE_SECRET)
    data = {PENDING_FLOW_SESSION_KEY: {"state": FLOW_STATE, "user": user, "next": f"/u/{user}/"}}
    return signer.sign(b64encode(json.dumps(data).encode("utf-8"))).decode("utf-8")


def callback(app: FastAPI, user: str = "alice") -> httpx.Response:
    """Walk the callback leg for ``user`` with a handshake already in flight."""
    with TestClient(app, base_url=EXTERNAL_ORIGIN) as client:
        client.cookies.set(STATE_COOKIE_NAME, pending_state(user))
        return client.get(
            CALLBACK_PATH,
            params={"code": "auth-code", "state": FLOW_STATE},
            follow_redirects=False,
        )
