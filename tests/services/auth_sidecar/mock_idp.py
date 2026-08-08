"""A stub OpenID provider, complete enough for a real Authlib handshake.

``test_oidc.py`` substitutes Authlib at the route boundary, which pins what
:mod:`osprey.services.auth_sidecar.routes.oidc` decides but proves nothing about
the library underneath it: discovery, the nonce, the ID token's signature and
its ``iss``/``aud``/``exp`` claims are all validated inside Authlib, and a fake
client short-circuits every one of them. This module is the other half — a real
IdP, served over a real socket, whose tokens are really signed, so
``test_oidc_flow.py`` can drive the handshake the deployment will actually run.

**It is a provider, not a mock.** The four endpoints an OIDC code flow touches
are all here (:data:`DISCOVERY_PATH`, :data:`AUTHORIZE_PATH`,
:data:`TOKEN_PATH`, :data:`JWKS_PATH`), it checks the client credentials it is
sent, and it binds each authorization code to the ``nonce`` and ``redirect_uri``
of the leg that created it. A token this provider issues verifies against the
JWKS it publishes — which is what makes a *deliberately* broken token
(:attr:`MockIdP.sign_with_intruder_key` and friends) evidence of a rejection
path rather than of a fixture that never worked.

**Every way it can lie is one attribute.** The knobs on :class:`MockIdP` change
one property of the response and nothing else, so a test that flips
:attr:`MockIdP.nonce_claim` differs from the passing case in exactly the value
under test. :meth:`MockIdP.reset` restores all of them, which is what lets one
served instance back a whole module of tests.

There is a :data:`USERINFO_PATH` too, and it answers with an identity mapped to
a *different* roster user on purpose: the sidecar reads identity only from the
verified ID token, and a test can prove that by watching this endpoint go
untouched (:attr:`MockIdP.userinfo_requests`).
"""

from __future__ import annotations

import base64
import secrets
import time
from typing import Any
from urllib.parse import urlencode

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from joserfc import jwt
from joserfc.jwk import KeySet, RSAKey

DISCOVERY_PATH = "/.well-known/openid-configuration"
AUTHORIZE_PATH = "/authorize"
TOKEN_PATH = "/token"
JWKS_PATH = "/jwks.json"
USERINFO_PATH = "/userinfo"

SIGNING_KEY_ID = "mock-idp-signing-key"
"""``kid`` of the published key.

The intruder key carries the same one deliberately: a different ``kid`` would be
refused as an unknown key (Authlib even re-fetches the JWKS for that case),
which is a different failure from the one a signature test means to provoke.
"""

DISCOVERY_OK = "ok"
DISCOVERY_WITHOUT_AUTHORIZATION_ENDPOINT = "no-authorization-endpoint"
DISCOVERY_AS_HTML = "html"
"""Discovery-document postures. The last two are what a mistyped issuer or a
captive portal actually serves: a reachable document that is not usable."""

DEFAULT_CLIENT_ID = "osprey-terminals"
DEFAULT_CLIENT_SECRET = "client-secret-value"


def _client_credentials(request: Request, form: dict[str, str]) -> tuple[str, str]:
    """The client id and secret presented on a token request.

    Both RFC 6749 forms are read — Authlib defaults to ``client_secret_basic``,
    but a provider that only understood one of them would silently pass a client
    configured for the other.
    """
    header = request.headers.get("authorization", "")
    if header.lower().startswith("basic "):
        decoded = base64.b64decode(header.split(" ", 1)[1]).decode("latin1")
        client_id, _, client_secret = decoded.partition(":")
        return client_id, client_secret
    return form.get("client_id", ""), form.get("client_secret", "")


class MockIdP:
    """A stub OpenID provider whose responses a test can bend one at a time.

    Attributes:
        issuer: The provider's own origin, assigned once it is being served —
            every URL it publishes and the ``iss`` it stamps into tokens are
            built from it, so discovery and the token agree by construction
            until a test says otherwise.
        subject: The identity asserted in the ``sub`` claim.
        extra_claims: Additional ID-token claims, e.g. ``preferred_username``.
        discovery: One of the ``DISCOVERY_*`` postures.
        sign_with_intruder_key: Sign with a key the JWKS does not publish.
        issuer_claim: Override the token's ``iss``.
        audience_claim: Override the token's ``aud``.
        nonce_claim: Override the token's ``nonce`` — otherwise it echoes the
            nonce the authorize leg was given, which is what a correct provider
            does.
        token_lifetime: Seconds between the token's ``iat`` and its ``exp``.
        issued_at_offset: Seconds to shift ``iat`` by; negative values age a
            token past Authlib's 120-second leeway.
        omit_id_token: Answer the token endpoint with an access token only —
            an OAuth2 provider where an OIDC one was configured.
        authorize_error: Send the browser back with ``error=<this>`` instead of
            a code, the shape of a denied consent screen.
        authorize_requests: Query parameters of each authorize call, in order.
        token_requests: Form fields of each token call, in order.
        userinfo_requests: How many times the userinfo endpoint was called.
    """

    def __init__(
        self,
        *,
        client_id: str = DEFAULT_CLIENT_ID,
        client_secret: str = DEFAULT_CLIENT_SECRET,
    ) -> None:
        """Build the provider and its FastAPI app.

        Args:
            client_id: The one client this provider will issue tokens to.
            client_secret: That client's secret.
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.issuer = ""
        self._key = RSAKey.generate_key(
            2048, parameters={"kid": SIGNING_KEY_ID, "use": "sig", "alg": "RS256"}
        )
        self._intruder_key = RSAKey.generate_key(
            2048, parameters={"kid": SIGNING_KEY_ID, "use": "sig", "alg": "RS256"}
        )
        self._flows: dict[str, dict[str, str]] = {}
        self.reset()
        self.app = self._build_app()

    # --- knobs ---------------------------------------------------------------

    def reset(self) -> None:
        """Restore every knob and forget every recorded request.

        Called between tests so one served provider can back a whole module: a
        leftover ``nonce_claim`` from an earlier test would otherwise fail the
        next one somewhere far from its cause.
        """
        self.subject = "idp|alice"
        self.extra_claims: dict[str, Any] = {}
        self.discovery = DISCOVERY_OK
        self.sign_with_intruder_key = False
        self.issuer_claim: str | None = None
        self.audience_claim: str | None = None
        self.nonce_claim: str | None = None
        self.token_lifetime = 300
        self.issued_at_offset = 0
        self.omit_id_token = False
        self.authorize_error: str | None = None
        self._flows.clear()
        self.authorize_requests: list[dict[str, str]] = []
        self.token_requests: list[dict[str, str]] = []
        self.userinfo_requests = 0

    @property
    def discovery_url(self) -> str:
        """Where the sidecar will look for the discovery document."""
        return f"{self.issuer}{DISCOVERY_PATH}"

    # --- endpoints -----------------------------------------------------------

    def _discovery_document(self) -> dict[str, Any]:
        return {
            "issuer": self.issuer,
            "authorization_endpoint": f"{self.issuer}{AUTHORIZE_PATH}",
            "token_endpoint": f"{self.issuer}{TOKEN_PATH}",
            "jwks_uri": f"{self.issuer}{JWKS_PATH}",
            "userinfo_endpoint": f"{self.issuer}{USERINFO_PATH}",
            "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code"],
            "subject_types_supported": ["public"],
            "id_token_signing_alg_values_supported": ["RS256"],
            "scopes_supported": ["openid", "profile", "email"],
            "token_endpoint_auth_methods_supported": [
                "client_secret_basic",
                "client_secret_post",
            ],
        }

    def _id_token(self, flow: dict[str, str]) -> str:
        """Sign an ID token for a completed authorization ``flow``."""
        issued_at = int(time.time()) + self.issued_at_offset
        claims: dict[str, Any] = {
            "iss": self.issuer_claim if self.issuer_claim is not None else self.issuer,
            "sub": self.subject,
            "aud": self.audience_claim if self.audience_claim is not None else self.client_id,
            "iat": issued_at,
            "exp": issued_at + self.token_lifetime,
            "nonce": self.nonce_claim if self.nonce_claim is not None else flow.get("nonce", ""),
            **self.extra_claims,
        }
        key = self._intruder_key if self.sign_with_intruder_key else self._key
        return jwt.encode({"alg": "RS256", "kid": SIGNING_KEY_ID}, claims, key)

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="Mock OpenID Provider", docs_url=None, redoc_url=None, openapi_url=None)

        @app.get(DISCOVERY_PATH)
        async def discovery() -> Any:
            if self.discovery == DISCOVERY_AS_HTML:
                # A captive portal or a plain 200 from a web server: reachable,
                # served, and not JSON.
                return HTMLResponse("<html><body>sign in to the guest network</body></html>")
            document = self._discovery_document()
            if self.discovery == DISCOVERY_WITHOUT_AUTHORIZATION_ENDPOINT:
                document.pop("authorization_endpoint")
            return JSONResponse(document)

        @app.get(JWKS_PATH)
        async def jwks() -> Any:
            return JSONResponse(KeySet([self._key]).as_dict(private=False))

        @app.get(AUTHORIZE_PATH)
        async def authorize(request: Request) -> Any:
            params = dict(request.query_params)
            self.authorize_requests.append(params)
            target = params.get("redirect_uri", "")
            state = params.get("state", "")

            if self.authorize_error is not None:
                returned = {"error": self.authorize_error, "state": state}
            else:
                code = f"authorization-code-{secrets.token_hex(8)}"
                self._flows[code] = {
                    "nonce": params.get("nonce", ""),
                    "redirect_uri": target,
                }
                returned = {"code": code, "state": state}

            separator = "&" if "?" in target else "?"
            return RedirectResponse(f"{target}{separator}{urlencode(returned)}", status_code=302)

        @app.post(TOKEN_PATH)
        async def token(request: Request) -> Any:
            async with request.form() as submitted:
                form = {key: str(value) for key, value in submitted.items()}
            self.token_requests.append(form)

            if _client_credentials(request, form) != (self.client_id, self.client_secret):
                return JSONResponse({"error": "invalid_client"}, status_code=401)

            flow = self._flows.pop(form.get("code", ""), None)
            if flow is None:
                return JSONResponse({"error": "invalid_grant"}, status_code=400)

            body: dict[str, Any] = {
                "access_token": f"access-token-{secrets.token_hex(8)}",
                "token_type": "Bearer",
                "expires_in": 3600,
                "scope": "openid profile email",
            }
            if not self.omit_id_token:
                body["id_token"] = self._id_token(flow)
            return JSONResponse(body)

        @app.get(USERINFO_PATH)
        async def userinfo() -> Any:
            # Deliberately not `self.subject`: the sidecar must never reach here,
            # and an identical answer would hide it if it did.
            self.userinfo_requests += 1
            return JSONResponse({"sub": "idp|carol"})

        return app
