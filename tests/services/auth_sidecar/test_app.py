"""Tests for the auth sidecar's FastAPI application factory.

Three things are pinned here, in rough order of how much damage getting them
wrong would do: the app refuses everything but ``/health`` when a required
setting is missing (fail closed), the OIDC state cookie carries the pinned
attributes rather than Starlette's defaults, and :class:`AuthSettings` reads the
documented env-var names — including the per-user suffix mapping the credential
provisioner writes with.

Settings are always passed to ``create_app`` as an explicit mapping, so a
variable left over in the real process environment can never make a test pass.
"""

from __future__ import annotations

import logging
import sys
import time
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from fastapi import APIRouter, FastAPI, Request, WebSocket
from fastapi.testclient import TestClient
from starlette.middleware.sessions import SessionMiddleware
from starlette.websockets import WebSocketDisconnect

from osprey.services.auth_sidecar import app as app_mod
from osprey.services.auth_sidecar.app import (
    DEFAULT_OIDC_CLIENT_ID_ENV,
    DEFAULT_OIDC_CLIENT_SECRET_ENV,
    DEFAULT_SESSION_LIFETIME,
    HEALTH_PATH,
    STATE_COOKIE_MAX_AGE,
    STATE_COOKIE_NAME,
    WEBSOCKET_REFUSAL_CODE,
    AuthSettings,
    create_app,
    get_attempt_throttle,
    get_revocation_store,
    get_session_codec,
    get_settings,
)
from osprey.services.auth_sidecar.exceptions import InvalidSessionError
from osprey.services.auth_sidecar.revocation import RevocationStore
from osprey.services.auth_sidecar.sessions import (
    SessionCodec,
    SessionState,
)
from osprey.services.auth_sidecar.throttle import AttemptThrottle

SESSION_SECRET = "session-secret-value"
STATE_SECRET = "state-secret-value"

PASSWORD_ENV = {
    "OSPREY_AUTH_METHOD": "password",
    "OSPREY_AUTH_SESSION_SECRET": SESSION_SECRET,
    "OSPREY_AUTH_USERS": "alice,bob",
    "OSPREY_AUTH_PW_HASH_ALICE": "scrypt$16384$8$1$c2FsdA$aGFzaA",
    "OSPREY_AUTH_EXTERNAL_ORIGIN": "https://terminals.example.org",
    "OSPREY_AUTH_TLS_ENABLED": "true",
}

OIDC_ENV = {
    "OSPREY_AUTH_METHOD": "oidc",
    "OSPREY_AUTH_SESSION_SECRET": SESSION_SECRET,
    "OSPREY_AUTH_STATE_SECRET": STATE_SECRET,
    "OSPREY_AUTH_USERS": "alice",
    "OSPREY_AUTH_OIDC_ISSUER": "https://idp.example.org",
    "OSPREY_AUTH_OIDC_CLIENT_ID": "client-id",
    "OSPREY_AUTH_OIDC_CLIENT_SECRET": "client-secret",
    "OSPREY_AUTH_EXTERNAL_ORIGIN": "https://terminals.example.org",
    "OSPREY_AUTH_TLS_ENABLED": "true",
}


def _probe_app(env: dict[str, str]) -> FastAPI:
    """An app plus one throwaway ``/auth`` route that touches the session.

    The state cookie only reaches the wire once something writes to the session,
    and this factory deliberately ships no such route yet — the OIDC task's
    login route will be the first. The probe stands in for it so the pinned
    cookie attributes are asserted on a real ``Set-Cookie`` header rather than
    on the middleware's constructor arguments.
    """
    app = create_app(env)

    @app.get("/auth/_probe")
    async def probe(request: Request) -> dict[str, str]:
        request.session["handshake"] = "in-flight"
        return {"status": "ok"}

    return app


class TestHealth:
    """``/health`` answers in every configuration state."""

    def test_configured_app_reports_ok(self) -> None:
        with TestClient(create_app(PASSWORD_ENV)) as client:
            response = client.get(HEALTH_PATH)
        assert response.status_code == 200
        assert response.json() == {
            "status": "ok",
            "service": "auth-sidecar",
            "method": "password",
            "configured": True,
        }

    def test_unconfigured_app_still_answers(self) -> None:
        with TestClient(create_app({})) as client:
            response = client.get(HEALTH_PATH)
        assert response.status_code == 200
        assert response.json()["configured"] is False

    def test_body_carries_no_secret(self) -> None:
        with TestClient(create_app(OIDC_ENV)) as client:
            body = client.get(HEALTH_PATH).text
        for secret in (SESSION_SECRET, STATE_SECRET, "client-secret"):
            assert secret not in body


class TestFailClosed:
    """A missing required setting refuses everything but the healthcheck."""

    @pytest.mark.parametrize("path", ["/auth/login", "/verify", "/openapi.json", "/does-not-exist"])
    def test_unconfigured_refuses_every_other_path(self, path: str) -> None:
        with TestClient(create_app({})) as client:
            response = client.get(path)
        assert response.status_code == 503
        assert "not configured" in response.json()["detail"]

    def test_refusal_sets_no_cookie(self) -> None:
        """A refused request must not hand the browser any state to carry.

        Built from the one configuration where this could actually fail: a
        state secret IS set, so SessionMiddleware is installed and able to mint
        cookies, while the OIDC settings around it are incomplete, so the app
        is unservable. The probe route writes to the session, which is what
        would emit a ``Set-Cookie`` — the guard sitting outside the session
        middleware is what stops the request from ever reaching it.
        """
        env = dict(OIDC_ENV)
        del env["OSPREY_AUTH_OIDC_ISSUER"]
        del env["OSPREY_AUTH_OIDC_CLIENT_ID"]
        app = _probe_app(env)
        assert any(middleware.cls is SessionMiddleware for middleware in app.user_middleware)

        with TestClient(app) as client:
            response = client.get("/auth/_probe")
        assert response.status_code == 503
        assert "set-cookie" not in {name.lower() for name in response.headers}

    def test_refusal_names_no_secret(self) -> None:
        env = dict(OIDC_ENV)
        del env["OSPREY_AUTH_STATE_SECRET"]
        with TestClient(create_app(env)) as client:
            body = client.get("/auth/login").text
        assert SESSION_SECRET not in body
        assert "client-secret" not in body

    def test_configured_app_lets_requests_through_to_routing(self) -> None:
        with TestClient(create_app(PASSWORD_ENV)) as client:
            response = client.get("/does-not-exist")
        assert response.status_code == 404

    def test_method_none_is_not_a_servable_sidecar_mode(self) -> None:
        env = dict(PASSWORD_ENV, OSPREY_AUTH_METHOD="none")
        with TestClient(create_app(env)) as client:
            assert client.get(HEALTH_PATH).json()["configured"] is False
            assert client.get("/auth/login").status_code == 503

    def test_unknown_method_refuses_rather_than_crashing_the_factory(self) -> None:
        env = dict(PASSWORD_ENV, OSPREY_AUTH_METHOD="kerberos")
        app = create_app(env)
        with TestClient(app) as client:
            assert client.get("/auth/login").status_code == 503

    def test_missing_settings_are_logged_by_name_only(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        env = dict(OIDC_ENV)
        del env["OSPREY_AUTH_OIDC_CLIENT_SECRET"]
        with caplog.at_level(logging.ERROR, logger=app_mod.__name__):
            create_app(env)
        assert DEFAULT_OIDC_CLIENT_SECRET_ENV in caplog.text
        assert "client-secret" not in caplog.text

    def test_cleartext_origin_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        env = dict(
            PASSWORD_ENV,
            OSPREY_AUTH_TLS_ENABLED="false",
            OSPREY_AUTH_EXTERNAL_ORIGIN="http://terminals.example.org",
        )
        with caplog.at_level(logging.WARNING, logger=app_mod.__name__):
            create_app(env)
        assert "cleartext" in caplog.text

    def test_https_origin_without_the_tls_flag_is_a_loud_mismatch(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An https origin with TLS off issues Secure-less cookies on a TLS site.

        The generic cleartext warning must not stand in for this: it reads as
        "you meant to deploy without TLS", which is the opposite diagnosis.
        """
        env = dict(
            PASSWORD_ENV,
            OSPREY_AUTH_TLS_ENABLED="false",
            OSPREY_AUTH_EXTERNAL_ORIGIN="https://terminals.example.org",
        )
        with caplog.at_level(logging.WARNING, logger=app_mod.__name__):
            create_app(env)
        assert "one of the two settings is wrong" in caplog.text
        assert "cleartext" not in caplog.text

    def test_cleartext_origin_with_the_tls_flag_on_is_a_loud_mismatch(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The other direction, and the more confusing failure of the two.

        Secure cookies over http are discarded by the browser without comment,
        so every login appears to succeed and unlocks nothing. Silence here
        leaves an operator debugging a login loop with no signal at all.
        """
        env = dict(
            PASSWORD_ENV,
            OSPREY_AUTH_TLS_ENABLED="true",
            OSPREY_AUTH_EXTERNAL_ORIGIN="http://terminals.example.org",
        )
        with caplog.at_level(logging.WARNING, logger=app_mod.__name__):
            create_app(env)
        assert "one of the two settings is wrong" in caplog.text
        assert "discard Secure cookies" in caplog.text

    def test_no_cookie_warning_when_tls_is_on(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger=app_mod.__name__):
            create_app(PASSWORD_ENV)
        assert "Secure attribute" not in caplog.text

    def test_unrecognized_tls_flag_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        env = dict(PASSWORD_ENV, OSPREY_AUTH_TLS_ENABLED="maybe")
        with caplog.at_level(logging.WARNING, logger=app_mod.__name__):
            app = create_app(env)
        assert "neither true nor false" in caplog.text
        assert app.state.settings.tls_enabled is False

    def test_empty_roster_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        env = dict(PASSWORD_ENV, OSPREY_AUTH_USERS="")
        with caplog.at_level(logging.WARNING, logger=app_mod.__name__):
            create_app(env)
        assert "empty roster" in caplog.text


class TestRosterCollisions:
    """Two usernames keying one credential refuse the whole sidecar.

    There is no per-user answer available here: the sidecar cannot tell which of
    ``alice-b`` and ``alice_b`` the single ``OSPREY_AUTH_PW_HASH_ALICE_B`` entry
    belongs to, so serving either one would be a guess about whose terminal a
    password opens.
    """

    COLLIDING = "alice-b,alice_b"

    def test_a_collision_makes_the_app_unservable(self) -> None:
        env = dict(PASSWORD_ENV, OSPREY_AUTH_USERS=self.COLLIDING)
        settings = AuthSettings.from_env(env)
        assert settings.missing_requirements() == ()
        assert settings.roster_collisions() == {"ALICE_B": ["alice-b", "alice_b"]}
        assert settings.configured is False

    def test_a_collision_refuses_traffic(self) -> None:
        env = dict(PASSWORD_ENV, OSPREY_AUTH_USERS=self.COLLIDING)
        with TestClient(create_app(env)) as client:
            assert client.get("/auth/login").status_code == 503
            assert client.get(HEALTH_PATH).json()["configured"] is False

    def test_the_colliding_usernames_are_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        env = dict(PASSWORD_ENV, OSPREY_AUTH_USERS=self.COLLIDING)
        with caplog.at_level(logging.ERROR, logger=app_mod.__name__):
            create_app(env)
        assert "alice-b" in caplog.text
        assert "alice_b" in caplog.text
        assert "OSPREY_AUTH_PW_HASH_ALICE_B" in caplog.text

    def test_a_distinct_roster_is_unaffected(self) -> None:
        assert AuthSettings.from_env(PASSWORD_ENV).roster_collisions() == {}


class TestSecretsNeverRender:
    """No secret-bearing field appears in the dataclass repr.

    One ``logger.debug("settings=%s", settings)`` or one traceback frame is all
    it takes for a default repr to write the session signing key into a log
    aggregator, so the exclusion is pinned rather than left to reviewer
    vigilance.
    """

    def test_secrets_are_absent_from_repr(self) -> None:
        env = dict(
            OIDC_ENV,
            OSPREY_AUTH_USERS="alice",
            OSPREY_AUTH_PW_HASH_ALICE="scrypt$16384$8$1$c2FsdA$aGFzaA",
        )
        rendered = repr(AuthSettings.from_env(env))
        for secret in (SESSION_SECRET, STATE_SECRET, "client-secret", "aGFzaA"):
            assert secret not in rendered

    def test_configuration_stays_visible(self) -> None:
        env = dict(OIDC_ENV, OSPREY_AUTH_OIDC_SUBJECT_ALICE="alice@example.org")
        rendered = repr(AuthSettings.from_env(env))
        assert "alice@example.org" in rendered
        assert "https://idp.example.org" in rendered
        assert DEFAULT_OIDC_CLIENT_ID_ENV in rendered

    def test_password_hashes_are_still_readable_by_code(self) -> None:
        settings = AuthSettings.from_env(PASSWORD_ENV)
        assert settings.password_hash("alice") == PASSWORD_ENV["OSPREY_AUTH_PW_HASH_ALICE"]


class TestWebsocketRefusal:
    """The guard covers websocket scopes, not just HTTP.

    ``BaseHTTPMiddleware`` dispatches ``http`` scopes only and passes everything
    else through, so an unconfigured sidecar would complete a websocket
    handshake past a gate that believes it is refusing all traffic. The sidecar
    ships no websocket route today — which is why this is pinned now, since the
    failure would otherwise arrive with whoever adds the first one.
    """

    @staticmethod
    def _app_with_socket(env: dict[str, str]) -> FastAPI:
        app = create_app(env)

        @app.websocket("/ws")
        async def socket(websocket: WebSocket) -> None:
            await websocket.accept()
            await websocket.send_text("reached the route")

        return app

    def test_unconfigured_websocket_never_completes_its_handshake(self) -> None:
        with TestClient(self._app_with_socket({})) as client:
            with pytest.raises(WebSocketDisconnect) as caught:
                with client.websocket_connect("/ws"):
                    pass
        assert caught.value.code == WEBSOCKET_REFUSAL_CODE

    def test_configured_websocket_connects(self) -> None:
        with TestClient(self._app_with_socket(PASSWORD_ENV)) as client:
            with client.websocket_connect("/ws") as socket:
                assert socket.receive_text() == "reached the route"

    def test_lifespan_still_runs_unconfigured(self) -> None:
        """Startup/shutdown are not http/websocket scopes and must pass through.

        Refusing them would leave an unconfigured app unable to boot far enough
        to answer the healthcheck that explains why it is refusing.
        """
        with TestClient(create_app({})) as client:
            assert client.get(HEALTH_PATH).status_code == 200


class TestRequirements:
    """Which gaps take the whole sidecar down, and which are per-user."""

    def test_password_mode_needs_only_a_session_secret(self) -> None:
        env = dict(PASSWORD_ENV)
        del env["OSPREY_AUTH_SESSION_SECRET"]
        assert AuthSettings.from_env(env).missing_requirements() == ("OSPREY_AUTH_SESSION_SECRET",)

    def test_password_mode_does_not_require_the_state_secret(self) -> None:
        env = dict(PASSWORD_ENV)
        env.pop("OSPREY_AUTH_STATE_SECRET", None)
        assert AuthSettings.from_env(env).configured is True

    def test_oidc_mode_requires_the_full_handshake_surface(self) -> None:
        assert set(
            AuthSettings.from_env({"OSPREY_AUTH_METHOD": "oidc"}).missing_requirements()
        ) == {
            DEFAULT_OIDC_CLIENT_ID_ENV,
            DEFAULT_OIDC_CLIENT_SECRET_ENV,
            "OSPREY_AUTH_EXTERNAL_ORIGIN",
            "OSPREY_AUTH_OIDC_ISSUER",
            "OSPREY_AUTH_SESSION_SECRET",
            "OSPREY_AUTH_STATE_SECRET",
        }

    def test_missing_requirements_are_reported_in_a_stable_order(self) -> None:
        missing = AuthSettings.from_env({"OSPREY_AUTH_METHOD": "oidc"}).missing_requirements()
        assert list(missing) == sorted(missing)

    def test_indirected_credentials_are_reported_under_their_resolved_names(self) -> None:
        """Naming the default would send an operator to set a variable nothing reads."""
        env = dict(OIDC_ENV)
        del env["OSPREY_AUTH_OIDC_CLIENT_ID"]
        env["OSPREY_AUTH_OIDC_CLIENT_ID_ENV"] = "FACILITY_IDP_CLIENT"

        settings = AuthSettings.from_env(env)
        assert settings.missing_requirements() == ("FACILITY_IDP_CLIENT",)
        assert settings.oidc_client_id_var == "FACILITY_IDP_CLIENT"
        assert settings.oidc_client_secret_var == DEFAULT_OIDC_CLIENT_SECRET_ENV

    def test_resolved_names_reach_the_startup_log(self, caplog: pytest.LogCaptureFixture) -> None:
        env = dict(OIDC_ENV)
        del env["OSPREY_AUTH_OIDC_CLIENT_ID"]
        env["OSPREY_AUTH_OIDC_CLIENT_ID_ENV"] = "FACILITY_IDP_CLIENT"
        with caplog.at_level(logging.ERROR, logger=app_mod.__name__):
            create_app(env)
        assert "FACILITY_IDP_CLIENT" in caplog.text
        assert DEFAULT_OIDC_CLIENT_ID_ENV not in caplog.text

    @pytest.mark.parametrize(
        ("overrides", "expected"),
        [
            ({"session_secret": "   "}, "OSPREY_AUTH_SESSION_SECRET"),
            ({"session_lifetime": 0}, "OSPREY_AUTH_SESSION_LIFETIME"),
            ({"session_lifetime": -1}, "OSPREY_AUTH_SESSION_LIFETIME"),
            ({"method": "oidc", "state_secret": "   "}, "OSPREY_AUTH_STATE_SECRET"),
        ],
        ids=["blank-secret", "zero-lifetime", "negative-lifetime", "blank-state-secret"],
    )
    def test_codec_preconditions_are_requirements_not_parser_luck(
        self, overrides: dict[str, Any], expected: str
    ) -> None:
        """Settings the codec cannot be built from must report as unservable.

        Constructed directly rather than through ``from_env``: the parser
        normalizes both of these away, so routing through it would only prove
        the parser works, not that ``configured`` implies a constructible
        codec.
        """
        kwargs: dict[str, Any] = {"method": "password", "session_secret": SESSION_SECRET}
        kwargs.update(overrides)
        settings = AuthSettings(**kwargs)
        assert expected in settings.missing_requirements()
        assert settings.configured is False

    @pytest.mark.parametrize(
        "env_value", ["   ", "0", "-30", "abc"], ids=["blank", "zero", "negative", "nonsense"]
    )
    def test_the_factory_survives_unusable_session_settings(self, env_value: str) -> None:
        """Whatever the value, the app starts and the healthcheck still answers.

        A secret or lifetime the codec would reject must degrade to a 503
        refusal with a working ``/health``, never to a factory that raises and
        leaves nothing running to report why.
        """
        for name in ("OSPREY_AUTH_SESSION_SECRET", "OSPREY_AUTH_SESSION_LIFETIME"):
            env = dict(PASSWORD_ENV)
            env[name] = env_value
            with TestClient(create_app(env)) as client:
                assert client.get(HEALTH_PATH).status_code == 200

    def test_a_user_without_a_hash_is_denied_individually_not_globally(self) -> None:
        settings = AuthSettings.from_env(PASSWORD_ENV)
        assert settings.configured is True
        assert settings.password_hash("bob") is None
        assert settings.knows_user("bob") is True


class TestSettingsParsing:
    """The documented env-var names, read exactly as documented."""

    def test_explicit_mapping_ignores_the_process_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OSPREY_AUTH_METHOD", "password")
        monkeypatch.setenv("OSPREY_AUTH_SESSION_SECRET", "leaked-from-the-process")
        settings = AuthSettings.from_env({})
        assert settings.method == "none"
        assert settings.session_secret == ""

    def test_process_environment_is_the_default_source(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for name, value in PASSWORD_ENV.items():
            monkeypatch.setenv(name, value)
        assert AuthSettings.from_env().configured is True

    def test_roster_is_ordered_and_deduplicated(self) -> None:
        env = dict(PASSWORD_ENV, OSPREY_AUTH_USERS=" bob , alice ,, bob ")
        assert AuthSettings.from_env(env).users == ("bob", "alice")

    def test_per_user_hash_uses_the_shared_suffix_mapping(self) -> None:
        env = dict(
            PASSWORD_ENV,
            OSPREY_AUTH_USERS="alice-b",
            OSPREY_AUTH_PW_HASH_ALICE_B="stored-hash",
        )
        assert AuthSettings.from_env(env).password_hash("alice-b") == "stored-hash"

    def test_hashes_outside_the_roster_are_not_loaded(self) -> None:
        env = dict(PASSWORD_ENV, OSPREY_AUTH_PW_HASH_MALLORY="stored-hash")
        settings = AuthSettings.from_env(env)
        assert settings.password_hash("mallory") is None
        assert set(settings.password_hashes) == {"alice"}

    def test_per_user_oidc_subject_is_read(self) -> None:
        env = dict(OIDC_ENV, OSPREY_AUTH_OIDC_SUBJECT_ALICE="alice@example.org")
        settings = AuthSettings.from_env(env)
        assert settings.oidc_subject("alice") == "alice@example.org"
        assert settings.oidc_subject("bob") is None

    def test_oidc_credentials_are_read_through_the_named_variables(self) -> None:
        env = dict(OIDC_ENV)
        del env["OSPREY_AUTH_OIDC_CLIENT_ID"]
        env["OSPREY_AUTH_OIDC_CLIENT_ID_ENV"] = "FACILITY_IDP_CLIENT"
        env["FACILITY_IDP_CLIENT"] = "facility-client-id"
        settings = AuthSettings.from_env(env)
        assert settings.oidc_client_id == "facility-client-id"
        assert settings.oidc_client_secret == "client-secret"

    def test_session_lifetime_defaults_and_rejects_nonsense(self) -> None:
        assert AuthSettings.from_env(PASSWORD_ENV).session_lifetime == DEFAULT_SESSION_LIFETIME
        for value in ("0", "-5", "half an hour", ""):
            env = dict(PASSWORD_ENV, OSPREY_AUTH_SESSION_LIFETIME=value)
            assert AuthSettings.from_env(env).session_lifetime == DEFAULT_SESSION_LIFETIME
        env = dict(PASSWORD_ENV, OSPREY_AUTH_SESSION_LIFETIME="3600")
        assert AuthSettings.from_env(env).session_lifetime == 3600

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("true", True), ("1", True), ("YES", True), ("false", False), ("", False)],
    )
    def test_tls_flag_reading(self, value: str, expected: bool) -> None:
        env = dict(PASSWORD_ENV, OSPREY_AUTH_TLS_ENABLED=value)
        assert AuthSettings.from_env(env).tls_enabled is expected

    def test_external_origin_drops_a_trailing_slash(self) -> None:
        env = dict(PASSWORD_ENV, OSPREY_AUTH_EXTERNAL_ORIGIN="https://example.org/")
        assert AuthSettings.from_env(env).external_origin == "https://example.org"

    def test_settings_are_reachable_from_a_request(self) -> None:
        app = create_app(PASSWORD_ENV)

        @app.get("/auth/_settings")
        async def read(request: Request) -> dict[str, Any]:
            return {"users": list(get_settings(request).users)}

        with TestClient(app) as client:
            assert client.get("/auth/_settings").json() == {"users": ["alice", "bob"]}

    def test_settings_are_cached_on_app_state(self) -> None:
        app = create_app(PASSWORD_ENV)
        assert isinstance(app.state.settings, AuthSettings)
        assert app.state.settings.method == "password"


class TestSharedStores:
    """The revocation store and attempt throttle follow the codec's pattern.

    Both are per-process state that only means anything if every route touches
    the one instance: a second revocation store is a logout nothing checks, and
    a second throttle counts every attempt as the first, so neither ever closes
    a window.
    """

    STORES = [
        ("revocation_store", get_revocation_store, RevocationStore, "revocation store"),
        ("attempt_throttle", get_attempt_throttle, AttemptThrottle, "attempt throttle"),
    ]
    IDS = ["revocation-store", "attempt-throttle"]

    @pytest.mark.parametrize(("attribute", "accessor", "kind", "label"), STORES, ids=IDS)
    def test_configured_app_builds_the_store(
        self, attribute: str, accessor: Any, kind: type, label: str
    ) -> None:
        assert isinstance(getattr(create_app(PASSWORD_ENV).state, attribute), kind)

    @pytest.mark.parametrize(("attribute", "accessor", "kind", "label"), STORES, ids=IDS)
    def test_every_route_gets_the_same_instance(
        self, attribute: str, accessor: Any, kind: type, label: str
    ) -> None:
        app = create_app(PASSWORD_ENV)
        seen: list[Any] = []

        @app.get("/auth/_store")
        async def read(request: Request) -> dict[str, bool]:
            seen.append(accessor(request))
            return {"ok": True}

        with TestClient(app) as client:
            client.get("/auth/_store")
            client.get("/auth/_store")

        assert len(seen) == 2
        assert seen[0] is seen[1] is getattr(app.state, attribute)

    @pytest.mark.parametrize(("attribute", "accessor", "kind", "label"), STORES, ids=IDS)
    def test_unconfigured_app_has_no_store(
        self, attribute: str, accessor: Any, kind: type, label: str
    ) -> None:
        assert getattr(create_app({}).state, attribute) is None

    @pytest.mark.parametrize(("attribute", "accessor", "kind", "label"), STORES, ids=IDS)
    def test_accessor_refuses_when_there_is_no_store(
        self, attribute: str, accessor: Any, kind: type, label: str
    ) -> None:
        request = SimpleNamespace(app=SimpleNamespace(state=create_app({}).state))
        with pytest.raises(RuntimeError, match="configuration guard"):
            accessor(request)
        with pytest.raises(RuntimeError, match=label):
            accessor(request)

    def test_the_three_shared_objects_are_distinct(self) -> None:
        """A copy-paste in the factory would otherwise alias two of them."""
        state = create_app(PASSWORD_ENV).state
        assert state.session_codec is not state.revocation_store
        assert state.revocation_store is not state.attempt_throttle

    def test_stores_keep_their_own_clocks(self) -> None:
        """Neither store is wired to the codec's clock.

        The revocation store records absolute epoch expiries taken straight
        from session cookies, and the throttle runs on a monotonic clock so a
        wall-clock step cannot open a window early. Sharing the codec's
        injectable clock would break both properties.
        """
        state = create_app(PASSWORD_ENV).state
        assert state.revocation_store._clock is time.time
        assert state.attempt_throttle._clock is time.monotonic


class TestSessionCodec:
    """One codec per app, shared by every route that touches a session cookie.

    Four route modules each constructing their own would be four chances to
    drift on the signing secret, the maximum age, or — least visibly — the
    clock, so the identity of the shared instance is the thing worth pinning.
    """

    def test_configured_app_builds_a_codec(self) -> None:
        app = create_app(PASSWORD_ENV)
        assert isinstance(app.state.session_codec, SessionCodec)

    def test_every_route_gets_the_same_instance(self) -> None:
        app = create_app(PASSWORD_ENV)
        seen: list[SessionCodec] = []

        @app.get("/auth/_codec")
        async def read(request: Request) -> dict[str, bool]:
            seen.append(get_session_codec(request))
            return {"ok": True}

        with TestClient(app) as client:
            client.get("/auth/_codec")
            client.get("/auth/_codec")

        assert len(seen) == 2
        assert seen[0] is seen[1] is app.state.session_codec

    def test_unconfigured_app_has_no_codec(self) -> None:
        assert create_app({}).state.session_codec is None

    def test_accessor_refuses_when_there_is_no_codec(self) -> None:
        app = create_app({})
        request = SimpleNamespace(app=SimpleNamespace(state=app.state))
        with pytest.raises(RuntimeError, match="configuration guard"):
            get_session_codec(request)  # type: ignore[arg-type]

    def test_codec_signs_with_the_configured_session_secret(self) -> None:
        app = create_app(PASSWORD_ENV)
        codec: SessionCodec = app.state.session_codec
        cookie = codec.encode(codec.new_state())

        assert SessionCodec(SESSION_SECRET).decode(cookie).session_id
        with pytest.raises(InvalidSessionError):
            SessionCodec("a-different-secret").decode(cookie)

    def test_codec_max_age_is_the_configured_session_lifetime(self) -> None:
        env = dict(PASSWORD_ENV, OSPREY_AUTH_SESSION_LIFETIME="60")
        codec: SessionCodec = create_app(env).state.session_codec

        fresh = SessionState(session_id="s", issued_at=time.time())
        assert codec.decode(codec.encode(fresh)).session_id == "s"

        stale = SessionState(session_id="s", issued_at=time.time() - 120)
        with pytest.raises(InvalidSessionError, match="maximum session age"):
            codec.decode(codec.encode(stale))


class TestStateCookie:
    """Authlib's session cookie carries our pins, not Starlette's defaults."""

    def _set_cookie(self, env: dict[str, str]) -> str:
        """The ``Set-Cookie`` header, lowercased.

        Starlette writes the attribute names in lowercase (``path=``,
        ``httponly``); browsers treat them case-insensitively, so the
        assertions do too rather than pinning today's casing.
        """
        with TestClient(_probe_app(env)) as client:
            response = client.get("/auth/_probe")
        assert response.status_code == 200
        return response.headers["set-cookie"].lower()

    def test_name_path_and_lifetime_are_pinned(self) -> None:
        header = self._set_cookie(OIDC_ENV)
        assert header.startswith(f"{STATE_COOKIE_NAME}=")
        assert f"max-age={STATE_COOKIE_MAX_AGE}" in header
        assert "path=/auth" in header
        assert "samesite=lax" in header
        assert "httponly" in header

    def test_lifetime_is_not_starlettes_fortnight(self) -> None:
        assert STATE_COOKIE_MAX_AGE == 300
        assert "max-age=1209600" not in self._set_cookie(OIDC_ENV)

    def test_secure_follows_the_tls_flag(self) -> None:
        assert "secure" in self._set_cookie(OIDC_ENV)
        assert "secure" not in self._set_cookie(
            dict(OIDC_ENV, OSPREY_AUTH_TLS_ENABLED="false", OSPREY_AUTH_EXTERNAL_ORIGIN="http://x")
        )

    def test_state_secret_never_reaches_the_client(self) -> None:
        header = self._set_cookie(OIDC_ENV)
        assert STATE_SECRET not in header
        assert SESSION_SECRET not in header

    def test_app_without_a_state_secret_still_constructs(self) -> None:
        env = dict(PASSWORD_ENV)
        env.pop("OSPREY_AUTH_STATE_SECRET", None)
        with TestClient(create_app(env)) as client:
            assert client.get(HEALTH_PATH).status_code == 200

    def test_no_state_secret_means_no_session_middleware(self) -> None:
        """Without a minted secret the middleware is absent, not stand-in keyed.

        A per-process fallback key would let a route issue state cookies signed
        by a secret that dies with the process — surfacing later as
        unexplainable intermittent handshake failures. Absent middleware instead
        fails at the first line that touches a session, naming the real problem.
        """
        env = dict(PASSWORD_ENV)
        env.pop("OSPREY_AUTH_STATE_SECRET", None)
        app = _probe_app(env)

        assert not any(middleware.cls is SessionMiddleware for middleware in app.user_middleware)
        with TestClient(app) as client, pytest.raises(AssertionError, match="SessionMiddleware"):
            client.get("/auth/_probe")

    def test_a_state_secret_installs_the_middleware(self) -> None:
        app = create_app(OIDC_ENV)
        assert any(middleware.cls is SessionMiddleware for middleware in app.user_middleware)


class TestRouteExtensionPoint:
    """Route modules are picked up when they land, and never faked when they don't."""

    def test_absent_route_modules_are_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A name with no module behind it leaves the app with only ``/health``.

        The list is replaced rather than left alone: asserting on the real route
        surface would make this test fail every time a route module lands, which
        says nothing about the skip behaviour it exists to pin.
        """
        monkeypatch.setattr(app_mod, "_ROUTE_MODULES", ("never_landing",))
        paths = create_app(PASSWORD_ENV).openapi()["paths"]
        assert set(paths) == {HEALTH_PATH}

    def test_schema_endpoints_are_not_served(self) -> None:
        """A security perimeter publishes no schema of its own routes.

        ``app.openapi()`` still builds one in-process (the assertions above rely
        on it); only the served endpoints are gone.
        """
        with TestClient(create_app(PASSWORD_ENV)) as client:
            for path in ("/openapi.json", "/docs", "/redoc"):
                assert client.get(path).status_code == 404

    def test_a_landed_route_module_is_included(self, monkeypatch: pytest.MonkeyPatch) -> None:
        router = APIRouter()

        @router.get("/verify")
        async def verify() -> dict[str, str]:
            return {"status": "ok"}

        module = ModuleType("osprey.services.auth_sidecar.routes.verify")
        module.router = router  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, module.__name__, module)

        app = create_app(PASSWORD_ENV)
        assert "/verify" in app.openapi()["paths"]
        with TestClient(app) as client:
            assert client.get("/verify").json() == {"status": "ok"}

    def test_a_broken_route_module_fails_loudly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raise(name: str) -> ModuleType:
            raise ModuleNotFoundError("No module named 'joserfc'", name="joserfc")

        monkeypatch.setattr(app_mod, "import_module", _raise)
        with pytest.raises(ModuleNotFoundError, match="joserfc"):
            create_app(PASSWORD_ENV)
