"""Resolving an authorization role out of the validated ID token's group claim.

The sidecar's trust model is pinned and this file does not widen it: the group
claim is read from the token Authlib already validated, never from the userinfo
endpoint, and never from anything the browser sent. What is under test here is
the *value contract* real providers impose — Entra sends ``groups`` as a GUID
array, Keycloak as a name array, and a facility that maps one app role sends a
bare string — and the fail-closed rules that keep an ambiguous membership from
quietly granting a privilege.

**Every refusal is a category, not a message.** Each denial below is audited
under its own reason constant, and none of the records or log lines may carry a
claim value: the operator reading the ledger is pointed at their own config,
which is where the offending value already lives.

The load-bearing assertion, in several shapes: a login that cannot be resolved
to exactly one role is refused, and the granted role never depends on the order
the values happen to arrive in.
"""

from __future__ import annotations

import json
import logging
from base64 import b64encode
from collections.abc import Iterator
from typing import Any

import httpx
import itsdangerous
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.interfaces._serving import run_app_server
from osprey.services.auth_sidecar import audit
from osprey.services.auth_sidecar.app import STATE_COOKIE_NAME, AuthSettings, create_app
from osprey.services.auth_sidecar.routes.oidc import (
    CALLBACK_PATH,
    ENV_ROLE_CLAIM,
    ENV_ROLE_MAP,
    PENDING_FLOW_SESSION_KEY,
    REASON_AMBIGUOUS_ROLE_CLAIM,
    REASON_IDENTITY_MISMATCH,
    REASON_MISSING_ROLE_CLAIM,
    REASON_NO_ASSERTED_IDENTITY,
    REASON_UNMAPPED_ROLE_CLAIM,
    REASON_UNMAPPED_USER,
    REASON_UNSAFE_ROLE,
    RoleBinding,
    _claims_options,
)
from osprey.services.auth_sidecar.sessions import SESSION_COOKIE_NAME, SessionCodec, SessionState
from tests.services.auth_sidecar.mock_idp import MockIdP

SESSION_SECRET = "session-secret-value"
STATE_SECRET = "state-secret-value"
EXTERNAL_ORIGIN = "https://terminals.example.org"
SESSION_LIFETIME = 3600

ALICE_SUBJECT = "idp|alice"
CAROL_SUBJECT = "idp|carol"
FLOW_STATE = "handshake-state-value"

#: Stands in for the compact JWT a token response carries. Never parsed here —
#: Authlib does that, and the route only asks whether one was present at all.
ID_TOKEN = "header.payload.signature"
IDP_AUTHORIZE_URL = "https://idp.example.org/authorize?state=" + FLOW_STATE

GROUP_CLAIM = "groups"
OPERATOR_GROUP = "als-operators"
EXPERT_GROUP = "als-experts"
SECOND_OPERATOR_GROUP = "als-operators-oncall"
CLAIM_MAP = {
    OPERATOR_GROUP: "operator",
    SECOND_OPERATOR_GROUP: "operator",
    EXPERT_GROUP: "expert",
}

OIDC_ENV = {
    "OSPREY_AUTH_METHOD": "oidc",
    "OSPREY_AUTH_SESSION_SECRET": SESSION_SECRET,
    "OSPREY_AUTH_STATE_SECRET": STATE_SECRET,
    "OSPREY_AUTH_SESSION_LIFETIME": str(SESSION_LIFETIME),
    "OSPREY_AUTH_USERS": "alice,bob,carol",
    "OSPREY_AUTH_OIDC_ISSUER": "https://idp.example.org",
    "OSPREY_AUTH_OIDC_CLIENT_ID": "client-id",
    "OSPREY_AUTH_OIDC_CLIENT_SECRET": "client-secret-value",
    "OSPREY_AUTH_OIDC_SUBJECT_ALICE": ALICE_SUBJECT,
    "OSPREY_AUTH_OIDC_SUBJECT_CAROL": CAROL_SUBJECT,
    "OSPREY_AUTH_EXTERNAL_ORIGIN": EXTERNAL_ORIGIN,
    "OSPREY_AUTH_TLS_ENABLED": "true",
}


class FakeOIDCClient:
    """Stands in for Authlib at ``app.state.oidc_client``.

    Returns the claims the test wants the callback to see and records the
    keyword arguments the route hands ``authorize_access_token`` — which is
    where ``claims_options`` is asserted on.
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

    async def create_authorization_url(self, redirect_uri: str | None = None) -> dict[str, Any]:
        return {"url": IDP_AUTHORIZE_URL, "state": FLOW_STATE, "nonce": "nonce-value"}

    async def save_authorize_data(self, request: Any, **kwargs: Any) -> None:
        return None

    async def authorize_access_token(self, request: Any, **kwargs: Any) -> dict[str, Any]:
        self.exchanged = True
        self.token_kwargs = kwargs
        return self.token


def _claims(**extra: Any) -> dict[str, Any]:
    """An ID token's claims for alice, plus whatever the test adds."""
    return {"sub": ALICE_SUBJECT, **extra}


def _app(
    *,
    userinfo: dict[str, Any] | None = None,
    binding: RoleBinding | None = RoleBinding(claim=GROUP_CLAIM, claim_map=CLAIM_MAP),
    env: dict[str, str] | None = None,
    client: FakeOIDCClient | None = None,
) -> FastAPI:
    """An OIDC-mode sidecar with Authlib replaced and a role binding installed.

    ``binding=None`` leaves ``app.state`` untouched, which is the shape a
    deployment with no ``claims:`` stanza runs in.
    """
    app = create_app(env if env is not None else OIDC_ENV)
    app.state.oidc_client = client if client is not None else FakeOIDCClient(userinfo)
    if binding is not None:
        app.state.role_binding = binding
    return app


def _pending(user: str) -> str:
    """A forged state cookie carrying one in-flight handshake for ``user``."""
    signer = itsdangerous.TimestampSigner(STATE_SECRET)
    data = {PENDING_FLOW_SESSION_KEY: {"state": FLOW_STATE, "user": user, "next": f"/u/{user}/"}}
    return signer.sign(b64encode(json.dumps(data).encode("utf-8"))).decode("utf-8")


def _callback(app: FastAPI, user: str = "alice") -> httpx.Response:
    """Walk the callback leg for ``user`` with a handshake already in flight."""
    with TestClient(app, base_url=EXTERNAL_ORIGIN) as client:
        client.cookies.set(STATE_COOKIE_NAME, _pending(user))
        return client.get(
            CALLBACK_PATH,
            params={"code": "auth-code", "state": FLOW_STATE},
            follow_redirects=False,
        )


def _session_from(response: httpx.Response) -> SessionState:
    """The auth session the response re-issued."""
    codec = SessionCodec(SESSION_SECRET, max_age=SESSION_LIFETIME)
    return codec.decode(response.cookies[SESSION_COOKIE_NAME])


def _role_of(response: httpx.Response, user: str = "alice") -> str:
    """The role the re-issued session carries for ``user``."""
    entry = _session_from(response).entry(user)
    assert entry is not None
    return entry.role


def _log(caplog: pytest.LogCaptureFixture) -> str:
    """Only this service's log records, never the test client's request log."""
    return "\n".join(
        record.getMessage() for record in caplog.records if record.name.startswith("osprey.")
    )


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Capture what the sidecar hands its audit seam."""
    written: list[Any] = []
    monkeypatch.setattr(audit, "write_envelope", written.append)
    return written


# --- the value contract ------------------------------------------------------


class TestClaimValueShapes:
    """What a provider may send in the group claim, and what it resolves to."""

    def test_a_string_claim_resolves_a_role(self) -> None:
        """A facility mapping a single app role sends a bare string."""
        response = _callback(_app(userinfo=_claims(groups=OPERATOR_GROUP)))
        assert response.status_code == 303
        assert _role_of(response) == "operator"

    def test_a_list_claim_resolves_a_role(self) -> None:
        """Entra sends an array of GUIDs, Keycloak an array of names."""
        response = _callback(_app(userinfo=_claims(groups=["unrelated", OPERATOR_GROUP])))
        assert response.status_code == 303
        assert _role_of(response) == "operator"

    def test_values_outside_the_map_are_ignored_not_refused(self) -> None:
        """A real directory membership is mostly groups this deployment has
        never heard of; only the intersection decides."""
        response = _callback(
            _app(userinfo=_claims(groups=["everyone", "building-6", EXPERT_GROUP, "vpn-users"]))
        )
        assert response.status_code == 303
        assert _role_of(response) == "expert"

    def test_two_values_naming_the_same_role_are_one_role(self) -> None:
        """The rule is one distinct *role*, not one matching value: a user in
        both the operators group and the on-call operators group holds exactly
        the privilege both of them name."""
        response = _callback(_app(userinfo=_claims(groups=[OPERATOR_GROUP, SECOND_OPERATOR_GROUP])))
        assert response.status_code == 303
        assert _role_of(response) == "operator"

    def test_a_list_keeps_its_string_entries_when_the_provider_mixes_types(self) -> None:
        """A stray non-string entry does not blind the intersection to the
        strings beside it — and cannot itself match a key."""
        response = _callback(_app(userinfo=_claims(groups=[7, None, OPERATOR_GROUP])))
        assert response.status_code == 303
        assert _role_of(response) == "operator"

    def test_matching_is_exact_not_case_folded(self) -> None:
        """Group identifiers are opaque; folding case would map a value the
        operator never wrote."""
        response = _callback(_app(userinfo=_claims(groups=[OPERATOR_GROUP.upper()])))
        assert response.status_code == 403


class TestAmbiguityFailsClosed:
    """More than one distinct role is a refusal, never a pick."""

    def _response(self) -> httpx.Response:
        return _callback(_app(userinfo=_claims(groups=[OPERATOR_GROUP, EXPERT_GROUP])))

    def test_two_roles_refuse_the_login(self) -> None:
        response = self._response()
        assert response.status_code == 403
        assert SESSION_COOKIE_NAME not in response.cookies

    def test_the_order_the_values_arrive_in_does_not_grant_a_role(self) -> None:
        """First-match-wins would make the granted privilege depend on YAML key
        order — the accidental-escalation case this contract exists to close.
        Both orderings refuse, and neither unlocks anything."""
        reversed_order = _callback(_app(userinfo=_claims(groups=[EXPERT_GROUP, OPERATOR_GROUP])))
        assert self._response().status_code == 403
        assert reversed_order.status_code == 403
        assert SESSION_COOKIE_NAME not in reversed_order.cookies

    def test_the_refusal_has_its_own_category(self, recorded: list[Any]) -> None:
        self._response()
        assert [record.to_dict()["reason"] for record in recorded] == [REASON_AMBIGUOUS_ROLE_CLAIM]

    def test_the_record_names_the_roles_and_never_the_claim_values(
        self, recorded: list[Any]
    ) -> None:
        """Role names are this deployment's own config identifiers, so they are
        the actionable half. The group values are the IdP's and stay out."""
        self._response()
        record = recorded[0].to_dict()
        assert "expert" in (record["detail"] or "")
        assert "operator" in (record["detail"] or "")
        assert OPERATOR_GROUP not in json.dumps(record)
        assert EXPERT_GROUP not in json.dumps(record)

    def test_no_role_is_recorded_on_an_ambiguous_refusal(self, recorded: list[Any]) -> None:
        """Naming one of the two would claim the login resolved to it."""
        self._response()
        assert recorded[0].to_dict().get("role") in (None, "")


class TestEmptyIntersection:
    """A membership this deployment maps nothing to."""

    def _response(self) -> httpx.Response:
        return _callback(_app(userinfo=_claims(groups=["building-6", "vpn-users"])))

    def test_the_login_is_refused(self) -> None:
        response = self._response()
        assert response.status_code == 403
        assert SESSION_COOKIE_NAME not in response.cookies

    def test_the_refusal_has_its_own_category(self, recorded: list[Any]) -> None:
        self._response()
        assert [record.to_dict()["reason"] for record in recorded] == [REASON_UNMAPPED_ROLE_CLAIM]

    def test_an_empty_map_refuses_every_login(self, recorded: list[Any]) -> None:
        """A ``claims:`` stanza whose map lost its entries authorizes nobody —
        it does not fall back to the roleless login."""
        response = _callback(
            _app(
                userinfo=_claims(groups=[OPERATOR_GROUP]),
                binding=RoleBinding(claim=GROUP_CLAIM, claim_map={}),
            )
        )
        assert response.status_code == 403
        assert [record.to_dict()["reason"] for record in recorded] == [REASON_UNMAPPED_ROLE_CLAIM]

    def test_the_record_never_carries_the_claim_values(self, recorded: list[Any]) -> None:
        self._response()
        assert "building-6" not in json.dumps(recorded[0].to_dict())

    def test_the_log_line_never_carries_the_claim_values(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="osprey"):
            self._response()
        logged = _log(caplog)
        assert "alice" in logged
        assert "building-6" not in logged


class TestMissingClaim:
    """The claim never arrived — Entra's group overage among the causes."""

    @pytest.mark.parametrize(
        ("claims", "why"),
        [
            ({}, "the claim is simply absent"),
            ({"groups": ""}, "an empty string is not a membership"),
            ({"groups": []}, "an empty array is not a membership"),
            ({"groups": [7, None]}, "no usable string entry"),
            ({"groups": {"value": OPERATOR_GROUP}}, "an object is not a value shape we accept"),
            ({"groups": 7}, "a number is not a value shape we accept"),
            (
                {"_claim_names": {"groups": "src1"}, "_claim_sources": {"src1": {}}},
                "Entra group overage: groups is replaced by a pointer to Graph",
            ),
        ],
    )
    def test_the_login_is_refused(
        self, claims: dict[str, Any], why: str, recorded: list[Any]
    ) -> None:
        """The CATEGORY is asserted per row, not just the status.

        Every one of these shapes must arrive as "the claim never came", not as
        "it came and mapped to nothing": a value contract that quietly coerced a
        number or an object into a string would still 403 — under
        ``unmapped_role_claim`` — and a status-only assertion would call that a
        pass.
        """
        response = _callback(_app(userinfo=_claims(**claims)))
        assert response.status_code == 403, why
        assert SESSION_COOKIE_NAME not in response.cookies
        assert [record.to_dict()["reason"] for record in recorded] == [REASON_MISSING_ROLE_CLAIM], (
            why
        )

    def test_the_refusal_has_its_own_category(self, recorded: list[Any]) -> None:
        _callback(_app(userinfo=_claims()))
        assert [record.to_dict()["reason"] for record in recorded] == [REASON_MISSING_ROLE_CLAIM]

    def test_the_record_names_the_claim_the_deployment_expected(self, recorded: list[Any]) -> None:
        """A claim *name* is config, and it is the one thing that makes this
        refusal actionable at the IdP."""
        _callback(_app(userinfo=_claims()))
        assert recorded[0].to_dict()["detail"] == GROUP_CLAIM

    def test_the_diagnostic_names_the_claims_that_did_arrive(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The overage case is only diagnosable from the claim names: seeing
        ``_claim_names`` where ``groups`` should be is the whole signal."""
        with caplog.at_level(logging.WARNING, logger="osprey"):
            _callback(_app(userinfo=_claims(_claim_names={"groups": "src1"})))
        logged = _log(caplog)
        assert "_claim_names" in logged
        assert GROUP_CLAIM in logged

    def test_the_diagnostic_carries_names_and_not_values(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.DEBUG, logger="osprey"):
            _callback(_app(userinfo=_claims(groups=[OPERATOR_GROUP], email="alice@example.org")))
        logged = _log(caplog)
        assert "email" in logged
        assert "alice@example.org" not in logged

    def test_a_binding_with_no_claim_name_refuses_rather_than_reading_nothing(
        self, recorded: list[Any]
    ) -> None:
        """Half a stanza cannot resolve anything, so it denies rather than
        degrading to the roleless login the deployment did not ask for."""
        response = _callback(
            _app(
                userinfo=_claims(groups=[OPERATOR_GROUP]),
                binding=RoleBinding(claim="", claim_map=CLAIM_MAP),
            )
        )
        assert response.status_code == 403
        assert [record.to_dict()["reason"] for record in recorded] == [REASON_MISSING_ROLE_CLAIM]

    def test_the_record_names_the_variable_the_binding_is_missing(
        self, recorded: list[Any]
    ) -> None:
        """Half a stanza and a genuine group overage share a category, so the
        record has to separate them: this one names the env variable that was
        never set, which points the operator at their own render rather than at
        their IdP."""
        _callback(
            _app(
                userinfo=_claims(groups=[OPERATOR_GROUP]),
                binding=RoleBinding(claim="", claim_map=CLAIM_MAP),
            )
        )
        assert recorded[0].to_dict()["detail"] == ENV_ROLE_CLAIM


class TestNoBindingIsInert:
    """A deployment with no ``claims:`` stanza logs in exactly as before."""

    def test_the_login_succeeds_with_no_role(self) -> None:
        response = _callback(_app(userinfo=_claims(groups=[OPERATOR_GROUP]), binding=None))
        assert response.status_code == 303
        assert _role_of(response) == ""

    def test_no_refusal_is_audited(self, recorded: list[Any]) -> None:
        """The login itself is recorded — every login is — but as the success it
        is: a membership that would be ambiguous under a binding refuses nothing
        when no role is bound."""
        _callback(_app(userinfo=_claims(groups=[EXPERT_GROUP, OPERATOR_GROUP]), binding=None))
        assert [record.to_dict()["decision"] for record in recorded] == ["allowed"]

    def test_a_membership_that_would_be_ambiguous_is_not_even_read(self) -> None:
        """Nothing about the token's groups matters when no role is bound."""
        response = _callback(
            _app(userinfo=_claims(groups=[EXPERT_GROUP, OPERATOR_GROUP]), binding=None)
        )
        assert response.status_code == 303


@pytest.fixture(scope="module")
def idp() -> Iterator[MockIdP]:
    """One stub provider on a real port, shared by the whole-handshake tests."""
    provider = MockIdP()
    with run_app_server(provider.app) as base_url:
        provider.issuer = base_url
        yield provider


# --- the trust model ---------------------------------------------------------


class TestTheGroupClaimIsNotAValidationOption:
    """``claims_options`` REPLACES Authlib's default; adding to it removes."""

    def test_the_options_name_only_the_issuer_and_the_audience(self) -> None:
        settings = AuthSettings.from_env(OIDC_ENV)
        assert set(_claims_options(settings)) == {"iss", "aud"}

    def test_the_callback_does_not_ask_authlib_to_validate_the_group_claim(self) -> None:
        client = FakeOIDCClient(_claims(groups=[OPERATOR_GROUP]))
        _callback(_app(client=client))
        options = client.token_kwargs["claims_options"]
        assert GROUP_CLAIM not in options
        assert set(options) == {"iss", "aud"}


class TestTheRoleComesFromTheIdToken:
    """Against a real signing provider: the userinfo endpoint stays untouched."""

    def _sidecar(self, idp: MockIdP) -> FastAPI:
        env = {
            **OIDC_ENV,
            "OSPREY_AUTH_OIDC_ISSUER": idp.issuer,
            "OSPREY_AUTH_OIDC_CLIENT_ID": idp.client_id,
            "OSPREY_AUTH_OIDC_CLIENT_SECRET": idp.client_secret,
            "OSPREY_AUTH_EXTERNAL_ORIGIN": "https://testserver",
        }
        app = create_app(env)
        app.state.role_binding = RoleBinding(claim=GROUP_CLAIM, claim_map=CLAIM_MAP)
        return app

    def _log_in(self, app: FastAPI) -> httpx.Response:
        with TestClient(app, base_url="https://testserver") as client:
            login = client.get("/auth/oidc/login", params={"user": "alice"}, follow_redirects=False)
            assert login.status_code == 302, login.text
            with httpx.Client(follow_redirects=False, timeout=10.0) as browser:
                handoff = browser.get(login.headers["location"])
            assert handoff.status_code == 302, handoff.text
            return client.get(
                handoff.headers["location"],
                follow_redirects=False,
                headers={
                    "Sec-Fetch-Site": "cross-site",
                    "Sec-Fetch-Mode": "navigate",
                    "Sec-Fetch-Dest": "document",
                },
            )

    def test_a_signed_token_carrying_the_group_claim_resolves_the_role(self, idp: MockIdP) -> None:
        idp.reset()
        idp.extra_claims = {GROUP_CLAIM: [OPERATOR_GROUP]}
        response = self._log_in(self._sidecar(idp))
        assert response.status_code == 303, response.text
        assert _role_of(response) == "operator"
        assert idp.userinfo_requests == 0

    def test_a_token_without_the_group_claim_fails_closed(self, idp: MockIdP) -> None:
        """And the userinfo endpoint is not consulted to rescue it — which is
        precisely what the Entra overage case would need it to do."""
        idp.reset()
        response = self._log_in(self._sidecar(idp))
        assert response.status_code == 403
        assert idp.userinfo_requests == 0


# --- the claims must be ones Authlib actually validated ----------------------


class TestTheClaimsMustComeFromAnIdToken:
    """``token["userinfo"]`` is trustworthy only when an ID token was parsed.

    Authlib writes that key with the parsed, validated ID token when — and only
    when — the token response carried an ``id_token``. Without one it never
    writes it at all, so whatever the token endpoint's own JSON body happened to
    carry under that name would flow through untouched: unsigned, unchecked
    against ``iss``/``aud``, and, since this task, deciding a ROLE.
    """

    def _forged(self) -> httpx.Response:
        """A token response with no ID token, carrying its own ``userinfo``.

        The shape an attacker-controlled or simply non-OIDC token endpoint can
        produce: a correct-looking claims object that nothing ever validated.
        """
        client = FakeOIDCClient(
            {"sub": ALICE_SUBJECT, GROUP_CLAIM: [OPERATOR_GROUP]}, id_token=False
        )
        client.token["access_token"] = "opaque"
        return _callback(_app(client=client))

    def test_unvalidated_claims_do_not_log_anyone_in(self) -> None:
        response = self._forged()
        assert response.status_code == 502
        assert SESSION_COOKIE_NAME not in response.cookies

    def test_no_session_is_minted_from_them(self) -> None:
        """The refusal lands before the claims are read at all, so the mapped
        group never becomes a privilege — the property the module's prose
        asserts, and this is what enforces it. No auth session at all — not a
        roleless one either; the only cookie the response touches is the
        handshake state it expires."""
        response = self._forged()
        assert SESSION_COOKIE_NAME not in response.cookies
        assert SESSION_COOKIE_NAME not in response.headers.get("set-cookie", "")

    def test_the_control_still_logs_in(self) -> None:
        """The same claims, arriving inside a parsed ID token, resolve the role
        they always did: the guard rejects the token SHAPE, not the claims."""
        response = _callback(_app(userinfo=_claims(groups=[OPERATOR_GROUP])))
        assert response.status_code == 303
        assert _role_of(response) == "operator"


class TestOnlyTheValidatedClaimsAreRead:
    """The token endpoint's own JSON body is not a claim source. Neither is the
    request.

    :class:`TestTheClaimsMustComeFromAnIdToken` pins the case where Authlib
    wrote no ``userinfo`` at all. This one pins the narrower and more dangerous
    shape: a token response that *does* carry a valid ID token — so the callback
    proceeds — while ALSO carrying identity and group claims at the top level of
    its JSON body, where nothing signed them and ``_claims_options`` never
    looked. A hostile or substituted token endpoint controls that body; it does
    not control the ID token.

    Both halves are pinned separately, because a regression can widen one
    without the other: reading the identity from the raw body, and merging the
    raw body into the claims the role is resolved from.
    """

    @staticmethod
    def _client(*, userinfo: dict[str, Any], **top_level: Any) -> FakeOIDCClient:
        """A validated-looking token whose JSON body carries ``top_level`` too."""
        client = FakeOIDCClient(userinfo)
        client.token.update(top_level)
        return client

    def test_an_identity_claim_only_in_the_raw_body_asserts_nobody(
        self, recorded: list[Any]
    ) -> None:
        """The validated claims carry neither claim; the raw body carries both.

        Refused for the *identity*, which is the first thing the callback reads
        — so a reader that fell back to the token body would not merely resolve
        a role it should not, it would log in an account nobody proved.
        """
        client = self._client(userinfo={}, **{"sub": ALICE_SUBJECT, GROUP_CLAIM: [OPERATOR_GROUP]})

        response = _callback(_app(client=client))

        assert response.status_code == 403
        assert SESSION_COOKIE_NAME not in response.cookies
        assert [record.to_dict()["reason"] for record in recorded] == [REASON_NO_ASSERTED_IDENTITY]

    def test_a_group_claim_only_in_the_raw_body_resolves_no_role(self, recorded: list[Any]) -> None:
        """The role half on its own: the ID token proves who this is and says
        nothing about groups, while the raw body offers a mapped group. The
        login is refused for the missing claim rather than granted the body's
        role."""
        client = self._client(userinfo=_claims(), **{GROUP_CLAIM: [OPERATOR_GROUP]})

        response = _callback(_app(client=client))

        assert response.status_code == 403
        assert SESSION_COOKIE_NAME not in response.cookies
        assert [record.to_dict()["reason"] for record in recorded] == [REASON_MISSING_ROLE_CLAIM]

    def test_the_browsers_own_request_cannot_supply_a_role(self, recorded: list[Any]) -> None:
        """The other direction a claim source can widen in: the caller's query
        string. The callback URL is whatever the browser navigates to, so a role
        read from it would be a privilege the person granted themselves."""
        app = _app(client=FakeOIDCClient(_claims()))
        with TestClient(app, base_url=EXTERNAL_ORIGIN) as client:
            client.cookies.set(STATE_COOKIE_NAME, _pending("alice"))
            response = client.get(
                CALLBACK_PATH,
                params={
                    "code": "auth-code",
                    "state": FLOW_STATE,
                    "role": "operator",
                    GROUP_CLAIM: OPERATOR_GROUP,
                },
                follow_redirects=False,
            )

        assert response.status_code == 403
        assert SESSION_COOKIE_NAME not in response.cookies
        assert [record.to_dict()["reason"] for record in recorded] == [REASON_MISSING_ROLE_CLAIM]

    def test_the_control_admits_the_same_claims_from_the_id_token(self) -> None:
        """The same two values, this time inside the validated token: admitted,
        so what the three tests above pin is the SOURCE and not the values."""
        response = _callback(_app(userinfo=_claims(groups=[OPERATOR_GROUP])))
        assert response.status_code == 303
        assert _role_of(response) == "operator"


# --- what a resolved role must survive ---------------------------------------


class TestAnUncarryableRoleFailsClosed:
    """A role the identity header cannot carry refuses the login, not 500s."""

    def _response(self, role: str) -> httpx.Response:
        return _callback(
            _app(
                userinfo=_claims(groups=[OPERATOR_GROUP]),
                binding=RoleBinding(claim=GROUP_CLAIM, claim_map={OPERATOR_GROUP: role}),
            )
        )

    @pytest.mark.parametrize(
        ("role", "why"),
        [
            ("bediener", "the control: an ASCII role is carried"),
        ],
    )
    def test_an_ascii_role_is_carried(self, role: str, why: str) -> None:
        response = self._response(role)
        assert response.status_code == 303, why
        assert _role_of(response) == role

    @pytest.mark.parametrize(
        ("role", "why"),
        [
            ("bediener-für-orbit", "non-ASCII: latin-1 carries it, nginx does not"),
            ("operator\r\nX-Osprey-Auth-Subject: root", "header injection"),
            (" operator", "an HTTP parser would strip the space it was granted with"),
        ],
    )
    def test_it_is_refused_rather_than_minted(self, role: str, why: str) -> None:
        response = self._response(role)
        assert response.status_code == 403, why
        assert SESSION_COOKIE_NAME not in response.cookies

    def test_the_refusal_has_its_own_category_and_hides_the_value(
        self, recorded: list[Any]
    ) -> None:
        self._response("bediener-für-orbit")
        record = recorded[0].to_dict()
        assert record["reason"] == REASON_UNSAFE_ROLE
        assert "für" not in json.dumps(record, ensure_ascii=False)

    def _poisoned(self) -> httpx.Response:
        """A map where one mapped value carries CR/LF and a second group is also
        mapped — so the intersection holds two roles and the ambiguity branch,
        which names the roles it found, would be the one to build the record."""
        return _callback(
            _app(
                userinfo=_claims(groups=[OPERATOR_GROUP, EXPERT_GROUP]),
                binding=RoleBinding(
                    claim=GROUP_CLAIM,
                    claim_map={
                        OPERATOR_GROUP: "operator\r\nX-Osprey-Auth-Role: admin",
                        EXPERT_GROUP: "expert",
                    },
                ),
            )
        )

    def test_an_uncarryable_candidate_is_caught_before_the_ambiguity_check(
        self, recorded: list[Any]
    ) -> None:
        """Header safety is a property of the MAP, not of whichever entry
        happens to survive the intersection: a table naming a role the boundary
        cannot carry is refused as that, whatever else the token asserted."""
        response = self._poisoned()
        assert response.status_code == 403
        assert SESSION_COOKIE_NAME not in response.cookies
        assert [record.to_dict()["reason"] for record in recorded] == [REASON_UNSAFE_ROLE]

    def test_no_control_character_reaches_the_record(self, recorded: list[Any]) -> None:
        """The ambiguity branch writes the role names it found into `detail`,
        and the envelope bounds that field's LENGTH but validates no charset. A
        candidate that never passed the header gate is exactly the string that
        must not get there."""
        self._poisoned()
        record = recorded[0].to_dict()
        # Asserted on the FIELD, not on a JSON dump: `json.dumps` escapes a real
        # CR into a two-character `\r`, so a dump would read as clean while the
        # record carried the control character.
        detail = record.get("detail") or ""
        assert "\r" not in detail
        assert "\n" not in detail
        assert "admin" not in detail
        assert "admin" not in json.dumps(record)


# --- every other denial the callback can reach -------------------------------


class TestTheOtherDenialCategories:
    """The identity half of the callback is audited on the same terms."""

    def test_a_user_with_no_mapped_identity(self, recorded: list[Any]) -> None:
        response = _callback(_app(userinfo=_claims(groups=[OPERATOR_GROUP])), "bob")
        assert response.status_code == 403
        assert [record.to_dict()["reason"] for record in recorded] == [REASON_UNMAPPED_USER]

    def test_a_token_asserting_no_identity(self, recorded: list[Any]) -> None:
        response = _callback(_app(userinfo={"groups": [OPERATOR_GROUP]}))
        assert response.status_code == 403
        assert [record.to_dict()["reason"] for record in recorded] == [REASON_NO_ASSERTED_IDENTITY]

    def test_an_identity_mapped_to_a_different_user(self, recorded: list[Any]) -> None:
        response = _callback(_app(userinfo={"sub": CAROL_SUBJECT, "groups": [OPERATOR_GROUP]}))
        assert response.status_code == 403
        assert [record.to_dict()["reason"] for record in recorded] == [REASON_IDENTITY_MISMATCH]

    def test_the_mismatch_record_never_carries_the_asserted_subject(
        self, recorded: list[Any]
    ) -> None:
        """The subject that was refused is somebody else's account identifier;
        the record names the user whose card was clicked and stops there."""
        _callback(_app(userinfo={"sub": CAROL_SUBJECT, "groups": [OPERATOR_GROUP]}))
        record = recorded[0].to_dict()
        assert record["subject"] == "alice"
        assert CAROL_SUBJECT not in json.dumps(record)

    def test_identity_is_settled_before_the_role_is(self, recorded: list[Any]) -> None:
        """A refused identity is refused as one, whatever its groups say — the
        role question is only asked about a login that got past 'who'."""
        _callback(_app(userinfo={"sub": CAROL_SUBJECT, "groups": [EXPERT_GROUP, OPERATOR_GROUP]}))
        assert [record.to_dict()["reason"] for record in recorded] == [REASON_IDENTITY_MISMATCH]

    def test_a_successful_login_records_no_refusal(self, recorded: list[Any]) -> None:
        """The denial seam is reached only from the branches that deny: a login
        that resolved its role cleanly files one record, and it is the success.
        """
        response = _callback(_app(userinfo=_claims(groups=[OPERATOR_GROUP])))
        assert response.status_code == 303
        assert len(recorded) == 1
        record = recorded[0].to_dict()
        assert record["decision"] == "allowed"
        assert record["role"] == "operator"

    def test_the_audit_seam_never_costs_the_refusal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An unwritable ledger degrades the trail, never the decision."""

        def boom(envelope: Any) -> None:
            raise OSError("the ledger is unwritable")

        monkeypatch.setattr(audit, "write_envelope", boom)
        response = _callback(_app(userinfo=_claims(groups=[EXPERT_GROUP, OPERATOR_GROUP])))
        assert response.status_code == 403

    def test_every_refused_login_leaves_the_session_locked(self) -> None:
        for claims in (
            _claims(),
            _claims(groups=["building-6"]),
            _claims(groups=[EXPERT_GROUP, OPERATOR_GROUP]),
        ):
            response = _callback(_app(userinfo=claims))
            assert SESSION_COOKIE_NAME not in response.cookies


# --- how the binding reaches the sidecar -------------------------------------


class TestRoleBindingFromEnv:
    """The two variables the rendered compose file exports."""

    def test_an_absent_binding_is_inert(self) -> None:
        assert RoleBinding.from_env({}).configured is False

    def test_the_claim_and_map_are_parsed(self) -> None:
        binding = RoleBinding.from_env(
            {
                ENV_ROLE_CLAIM: GROUP_CLAIM,
                ENV_ROLE_MAP: json.dumps(CLAIM_MAP, separators=(",", ":")),
            }
        )
        assert binding.configured is True
        assert binding.claim == GROUP_CLAIM
        assert dict(binding.claim_map) == CLAIM_MAP

    def test_surrounding_whitespace_is_not_part_of_the_claim_name(self) -> None:
        binding = RoleBinding.from_env({ENV_ROLE_CLAIM: "  groups  ", ENV_ROLE_MAP: "{}"})
        assert binding.claim == GROUP_CLAIM

    @pytest.mark.parametrize(
        ("raw", "why"),
        [
            ("not json at all", "a hand-edited compose file"),
            ("[1, 2]", "valid JSON, wrong shape"),
            ('"operator"', "valid JSON, wrong shape"),
            ("", "the variable is present but empty"),
        ],
    )
    def test_an_unusable_map_denies_rather_than_disabling_the_binding(
        self, raw: str, why: str
    ) -> None:
        """The claim name is what says 'this deployment binds roles'. A map that
        could not be read leaves that true and every intersection empty, which
        is a refusal — never a silent downgrade to the roleless login."""
        binding = RoleBinding.from_env({ENV_ROLE_CLAIM: GROUP_CLAIM, ENV_ROLE_MAP: raw})
        assert binding.configured is True, why
        assert dict(binding.claim_map) == {}

    def test_entries_that_are_not_string_to_string_are_dropped(self) -> None:
        binding = RoleBinding.from_env(
            {
                ENV_ROLE_CLAIM: GROUP_CLAIM,
                ENV_ROLE_MAP: json.dumps({OPERATOR_GROUP: "operator", "bad": 7, "worse": None}),
            }
        )
        assert dict(binding.claim_map) == {OPERATOR_GROUP: "operator"}

    def test_a_map_without_a_claim_name_is_still_configured(self) -> None:
        """Half a stanza denies; it does not read as 'no roles here'."""
        binding = RoleBinding.from_env({ENV_ROLE_MAP: json.dumps(CLAIM_MAP)})
        assert binding.configured is True

    def test_the_process_environment_is_what_the_container_runs_on(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No test double: the deployed sidecar has these in its own env, and
        the callback must pick them up with nothing installed on ``app.state``."""
        monkeypatch.setenv(ENV_ROLE_CLAIM, GROUP_CLAIM)
        monkeypatch.setenv(ENV_ROLE_MAP, json.dumps({OPERATOR_GROUP: "operator"}))
        response = _callback(_app(userinfo=_claims(groups=[OPERATOR_GROUP]), binding=None))
        assert response.status_code == 303
        assert _role_of(response) == "operator"

    def test_the_environment_binding_also_refuses(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV_ROLE_CLAIM, GROUP_CLAIM)
        monkeypatch.setenv(ENV_ROLE_MAP, json.dumps({OPERATOR_GROUP: "operator"}))
        response = _callback(_app(userinfo=_claims(groups=["building-6"]), binding=None))
        assert response.status_code == 403
