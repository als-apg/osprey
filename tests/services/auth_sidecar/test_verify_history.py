"""The history page's ``auth_request`` target and the sign-in paths that feed it.

``/verify-history`` resolves one person out of the session cookie and maps them
to one store view user. Every row of the mapping table is pinned here, with the
default selection's tie-break, the selector, the headers, reader
re-validation, and the carriers the sidecar reads its answers from. Cookies are
minted with a client-side codec, as in ``test_verify.py``; the app is always
built from an explicit environment, and the roster role table — which the
sidecar reads from its process environment — is installed directly.
"""

from __future__ import annotations

from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.services.auth_sidecar.app import AuthSettings, create_app
from osprey.services.auth_sidecar.passwords import generation_tag, hash_password
from osprey.services.auth_sidecar.return_to import acceptable_return_to, safe_return_to
from osprey.services.auth_sidecar.routes import entry
from osprey.services.auth_sidecar.routes.history import (
    CHOICES_HEADER,
    ENTRY_HEADER,
    PERSON_HEADER,
    ROLE_HEADER,
    SCOPE_HEADER,
    TABLES_HEADER,
    VIEW_HEADER,
    VIEW_TABLES,
)
from osprey.services.auth_sidecar.routes.recheck import ROLE_SOURCE_CLAIM, RosterRoles
from osprey.services.auth_sidecar.sessions import (
    SESSION_COOKIE_NAME,
    SessionCodec,
    SessionState,
    UnlockedUser,
)

SECRET = "session-secret-value"
LIFETIME = 3600
HISTORY_ORIGIN = "https://terminals.example.org:8443"
ALL_TABLES = "otel_logs,otel_traces,audit_ledger"

HASHES = {name: hash_password(f"{name}-password") for name in ("alice", "bob", "desk")}

PASSWORD_ENV = {
    "OSPREY_AUTH_METHOD": "password",
    "OSPREY_AUTH_SESSION_SECRET": SECRET,
    "OSPREY_AUTH_SESSION_LIFETIME": str(LIFETIME),
    "OSPREY_AUTH_USERS": "alice,bob,desk",
    "OSPREY_AUTH_PW_HASH_ALICE": HASHES["alice"],
    "OSPREY_AUTH_PW_HASH_BOB": HASHES["bob"],
    "OSPREY_AUTH_PW_HASH_DESK": HASHES["desk"],
    "OSPREY_AUTH_EXTERNAL_ORIGIN": "https://terminals.example.org",
    "OSPREY_AUTH_TLS_ENABLED": "true",
    "OSPREY_AUTH_HISTORY_ORIGIN": HISTORY_ORIGIN,
    "OSPREY_AUTH_ROSTER_HISTORY_ALICE": "own",
    "OSPREY_AUTH_ROSTER_HISTORY_BOB": "own",
    "OSPREY_AUTH_ROSTER_HISTORY_DESK": "own",
    "OSPREY_AUTH_READERS": "carol",
    "OSPREY_AUTH_READER_TABLES_CAROL": "otel_logs",
}

OIDC_ENV = {
    **{k: v for k, v in PASSWORD_ENV.items() if not k.startswith("OSPREY_AUTH_PW_HASH_")},
    "OSPREY_AUTH_METHOD": "oidc",
    "OSPREY_AUTH_STATE_SECRET": "state-secret-value",
    "OSPREY_AUTH_OIDC_ISSUER": "https://idp.example.org",
    "OSPREY_AUTH_OIDC_CLIENT_ID": "client-id",
    "OSPREY_AUTH_OIDC_CLIENT_SECRET": "client-secret",
    "OSPREY_AUTH_OIDC_CLAIM": "email",
    "OSPREY_AUTH_OIDC_SUBJECT_ALICE": "alice@example.org",
    "OSPREY_AUTH_OIDC_SUBJECT_BOB": "bob@example.org",
    "OSPREY_AUTH_READERS": "carol,ext",
    "OSPREY_AUTH_READER_TABLES_EXT": "otel_logs,audit_ledger",
    "OSPREY_AUTH_READER_SUBJECT_EXT": "ext@partner.org",
}


def _app(env: dict[str, str], roles: dict[str, str] | None = None) -> FastAPI:
    app = create_app(env)
    app.state.roster_roles = RosterRoles(roles=roles or {})
    return app


def _entry(
    username: str,
    *,
    ttl: float = 600.0,
    tag_of: str | None = None,
    subject: str = "",
    role: str = "",
    role_source: str = "",
    opener: str = "",
    admitted: str = "",
) -> UnlockedUser:
    """An entry; ``tag_of`` names whose password hash the tag is minted from."""
    hash_owner = tag_of if tag_of is not None else (opener or username)
    stored = HASHES.get(hash_owner)
    return UnlockedUser(
        username=username,
        expires_at=SessionCodec(SECRET).now() + ttl,
        generation_tag=generation_tag(stored) if stored and not subject else "",
        oidc_subject=subject,
        role=role,
        role_source=role_source,
        opener=opener,
        admitted_identity=admitted,
    )


def _cookie(*entries: UnlockedUser, session_id: str = "sid") -> str:
    codec = SessionCodec(SECRET)
    return codec.encode(SessionState(session_id=session_id, issued_at=codec.now(), users=entries))


def _ask(app: FastAPI, cookie: str | None, original_uri: str = "/history/") -> httpx.Response:
    headers = {"X-Original-URI": original_uri}
    if cookie is not None:
        headers["Cookie"] = f"{SESSION_COOKIE_NAME}={cookie}"
    with TestClient(app) as client:
        return client.get("/verify-history", headers=headers)


def _view(response: httpx.Response) -> tuple[int, str | None, str | None]:
    return (
        response.status_code,
        response.headers.get(VIEW_HEADER),
        response.headers.get(PERSON_HEADER),
    )


# --- carriers -----------------------------------------------------------------


class TestCarriers:
    def test_every_carrier_is_parsed(self) -> None:
        settings = AuthSettings.from_env(
            {
                **OIDC_ENV,
                "OSPREY_AUTH_ROSTER_HISTORY_ALICE": "all",
                "OSPREY_AUTH_ROLE_HISTORY_SHIFT_LEAD": "all",
                "OSPREY_AUTH_ROLE_HISTORY_VIEWER": "own",
            }
        )
        assert settings.history_origin == HISTORY_ORIGIN
        assert settings.user_history_all("alice")
        assert not settings.user_history_all("bob")
        assert settings.role_history_all("shift-lead")
        assert not settings.role_history_all("viewer")
        assert not settings.role_history_all("")
        assert settings.readers == ("carol", "ext")
        assert settings.tables_for_reader("carol") == ("otel_logs",)
        assert settings.tables_for_reader("ext") == ("otel_logs", "audit_ledger")
        assert settings.reader_subject("ext") == "ext@partner.org"
        assert settings.reader_subject("carol") is None

    def test_absent_carriers_mean_no_history_page(self) -> None:
        settings = AuthSettings.from_env({"OSPREY_AUTH_METHOD": "password"})
        assert settings.history_origin == ""
        assert settings.readers == ()
        assert not settings.history_all_users
        assert not settings.history_all_roles

    def test_the_origin_loses_a_trailing_slash_and_values_are_trimmed(self) -> None:
        settings = AuthSettings.from_env(
            {
                "OSPREY_AUTH_USERS": "alice",
                "OSPREY_AUTH_HISTORY_ORIGIN": f" {HISTORY_ORIGIN}/ ",
                "OSPREY_AUTH_ROSTER_HISTORY_ALICE": " ALL ",
                "OSPREY_AUTH_READERS": " x , ,x, y",
                "OSPREY_AUTH_READER_TABLES_X": "otel_logs, audit_ledger",
            }
        )
        assert settings.history_origin == HISTORY_ORIGIN
        assert settings.user_history_all("alice")
        assert settings.readers == ("x", "y")
        assert settings.tables_for_reader("x") == ("otel_logs", "audit_ledger")
        assert settings.tables_for_reader("y") == ()

    def test_the_view_tables_are_the_stores(self) -> None:
        from osprey.deployment.telemetry_store.accounts import DEFAULT_READER_TABLES

        assert VIEW_TABLES == DEFAULT_READER_TABLES


# --- the mapping table ----------------------------------------------------------


class TestOwnCard:
    def test_own_maps_to_the_persons_view(self) -> None:
        response = _ask(_app(PASSWORD_ENV), _cookie(_entry("alice")))
        assert _view(response) == (200, "view_alice", "alice")
        assert response.headers[SCOPE_HEADER] == "own"
        assert response.headers[TABLES_HEADER] == ALL_TABLES

    def test_a_reader_named_like_the_person_takes_precedence(self) -> None:
        env = {
            **PASSWORD_ENV,
            "OSPREY_AUTH_READERS": "alice",
            "OSPREY_AUTH_READER_TABLES_ALICE": "otel_logs",
        }
        env["OSPREY_AUTH_ROSTER_HISTORY_ALICE"] = "all"
        response = _ask(_app(env), _cookie(_entry("alice")))
        assert _view(response) == (200, "view__reader_alice", "alice")
        assert response.headers[SCOPE_HEADER] == "reader"
        assert response.headers[TABLES_HEADER] == "otel_logs"

    def test_all_on_the_roster_entry_maps_to_view_all(self) -> None:
        env = {**PASSWORD_ENV, "OSPREY_AUTH_ROSTER_HISTORY_ALICE": "all"}
        response = _ask(_app(env), _cookie(_entry("alice")))
        assert _view(response) == (200, "view__all", "alice")
        assert response.headers[SCOPE_HEADER] == "all"
        assert response.headers[TABLES_HEADER] == ALL_TABLES

    def test_all_on_the_current_roster_role_maps_to_view_all(self) -> None:
        env = {**PASSWORD_ENV, "OSPREY_AUTH_ROLE_HISTORY_LEAD": "all"}
        response = _ask(_app(env, {"alice": "lead"}), _cookie(_entry("alice")))
        assert _view(response) == (200, "view__all", "alice")
        assert response.headers[ROLE_HEADER] == "lead"

    def test_a_reassigned_role_takes_effect_on_the_next_request(self) -> None:
        env = {**PASSWORD_ENV, "OSPREY_AUTH_ROLE_HISTORY_LEAD": "all"}
        app = _app(env, {"alice": "lead"})
        cookie = _cookie(_entry("alice", role="lead", role_source="roster"))
        assert _view(_ask(app, cookie))[1] == "view__all"
        app.state.roster_roles = RosterRoles(roles={"alice": "viewer"})
        assert _view(_ask(app, cookie)) == (200, "view_alice", "alice")

    def test_the_stored_roster_role_is_never_read(self) -> None:
        env = {**PASSWORD_ENV, "OSPREY_AUTH_ROLE_HISTORY_LEAD": "all"}
        cookie = _cookie(_entry("alice", role="lead", role_source="roster"))
        assert _view(_ask(_app(env, {}), cookie))[1] == "view_alice"

    def test_a_claim_granted_role_is_read_from_the_session(self) -> None:
        env = {**OIDC_ENV, "OSPREY_AUTH_ROLE_HISTORY_LEAD": "all"}
        cookie = _cookie(
            _entry("alice", subject="alice@example.org", role="lead", role_source=ROLE_SOURCE_CLAIM)
        )
        response = _ask(_app(env), cookie)
        assert _view(response) == (200, "view__all", "alice")
        assert response.headers[ROLE_HEADER] == "lead"


class TestSharedCards:
    SHARED = {**PASSWORD_ENV, "OSPREY_AUTH_ROSTER_ACCESS_DESK": "any"}

    def test_a_sign_in_as_a_shared_card_with_all_maps_to_view_all(self) -> None:
        env = {**OIDC_ENV, "OSPREY_AUTH_ROSTER_ACCESS_DESK": '["self","roster"]'}
        env["OSPREY_AUTH_OIDC_SUBJECT_DESK"] = "desk@example.org"
        env["OSPREY_AUTH_ROSTER_HISTORY_DESK"] = "all"
        response = _ask(_app(env), _cookie(_entry("desk", subject="desk@example.org")))
        assert _view(response) == (200, "view__all", "desk")

    def test_a_sign_in_as_a_shared_card_through_its_role_maps_to_view_all(self) -> None:
        env = {**OIDC_ENV, "OSPREY_AUTH_ROSTER_ACCESS_DESK": '["self","roster"]'}
        env["OSPREY_AUTH_OIDC_SUBJECT_DESK"] = "desk@example.org"
        env["OSPREY_AUTH_ROLE_HISTORY_OPERATOR"] = "all"
        app = _app(env, {"desk": "operator"})
        response = _ask(app, _cookie(_entry("desk", subject="desk@example.org")))
        assert _view(response) == (200, "view__all", "desk")

    def test_a_sign_in_as_a_shared_card_without_all_lists_nothing(self) -> None:
        env = {**OIDC_ENV, "OSPREY_AUTH_ROSTER_ACCESS_DESK": '["self","roster"]'}
        env["OSPREY_AUTH_OIDC_SUBJECT_DESK"] = "desk@example.org"
        response = _ask(_app(env), _cookie(_entry("desk", subject="desk@example.org")))
        assert response.status_code == 403
        assert VIEW_HEADER not in response.headers

    def test_an_opener_is_mapped_by_their_own_row_never_the_cards_role(self) -> None:
        env = {**self.SHARED, "OSPREY_AUTH_ROLE_HISTORY_OPERATOR": "all"}
        app = _app(env, {"desk": "operator"})
        cookie = _cookie(_entry("desk", opener="bob", role="operator", role_source="roster"))
        response = _ask(app, cookie)
        assert _view(response) == (200, "view_bob", "bob")
        assert response.headers[ENTRY_HEADER] == "desk"

    def test_an_openers_own_all_still_widens_their_view(self) -> None:
        env = {**self.SHARED, "OSPREY_AUTH_ROSTER_HISTORY_BOB": "all"}
        response = _ask(_app(env), _cookie(_entry("desk", opener="bob")))
        assert _view(response) == (200, "view__all", "bob")


class TestRuleAdmitted:
    ENV = {**OIDC_ENV, "OSPREY_AUTH_ROSTER_ACCESS_DESK": '["domain:example.org"]'}

    def test_a_rule_admitted_roster_person_gets_their_own_row(self) -> None:
        response = _ask(_app(self.ENV), _cookie(_entry("desk", admitted="bob@example.org")))
        assert _view(response) == (200, "view_bob", "bob")

    def test_a_rule_admitted_stranger_is_refused(self) -> None:
        response = _ask(_app(self.ENV), _cookie(_entry("desk", admitted="zed@example.org")))
        assert response.status_code == 403

    def test_an_identity_two_persons_share_is_a_stranger(self) -> None:
        env = {**self.ENV, "OSPREY_AUTH_OIDC_SUBJECT_BOB": "alice@example.org"}
        response = _ask(_app(env), _cookie(_entry("desk", admitted="alice@example.org")))
        assert response.status_code == 403


# --- readers --------------------------------------------------------------------


class TestReaders:
    def test_a_reader_entry_maps_to_its_view_user(self) -> None:
        response = _ask(_app(OIDC_ENV), _cookie(_entry("reader:ext", subject="ext@partner.org")))
        assert _view(response) == (200, "view__reader_ext", "ext")
        assert response.headers[SCOPE_HEADER] == "reader"
        assert response.headers[TABLES_HEADER] == "otel_logs,audit_ledger"
        assert response.headers[ENTRY_HEADER] == "reader:ext"

    def test_a_withdrawn_reader_is_refused_on_the_next_request(self) -> None:
        cookie = _cookie(_entry("reader:ext", subject="ext@partner.org"))
        env = {**OIDC_ENV, "OSPREY_AUTH_READERS": "carol"}
        assert _ask(_app(env), cookie).status_code == 401

    def test_a_reader_whose_subject_was_removed_is_refused_not_skipped(self) -> None:
        cookie = _cookie(_entry("reader:ext", subject="ext@partner.org"))
        env = {k: v for k, v in OIDC_ENV.items() if k != "OSPREY_AUTH_READER_SUBJECT_EXT"}
        assert _ask(_app(env), cookie).status_code == 401

    def test_a_reader_whose_subject_changed_is_refused(self) -> None:
        cookie = _cookie(_entry("reader:ext", subject="ext@partner.org"))
        env = {**OIDC_ENV, "OSPREY_AUTH_READER_SUBJECT_EXT": "other@partner.org"}
        assert _ask(_app(env), cookie).status_code == 401

    def test_a_reader_entry_is_refused_under_password_login(self) -> None:
        cookie = _cookie(_entry("reader:carol", subject="carol@x.org"))
        env = {**PASSWORD_ENV, "OSPREY_AUTH_READER_SUBJECT_CAROL": "carol@x.org"}
        assert _ask(_app(env), cookie).status_code == 401

    def test_a_withdrawn_reader_beside_a_valid_card_leaves_the_card(self) -> None:
        cookie = _cookie(
            _entry("reader:ext", subject="ext@partner.org"),
            _entry("alice", subject="alice@example.org"),
        )
        env = {**OIDC_ENV, "OSPREY_AUTH_READERS": "carol"}
        response = _ask(_app(env), cookie)
        assert _view(response) == (200, "view_alice", "alice")
        assert response.headers[CHOICES_HEADER] == "alice=alice"

    def test_a_reader_a_domain_rule_also_admits_is_still_a_reader(self) -> None:
        env = {**OIDC_ENV, "OSPREY_AUTH_ROSTER_ACCESS_DESK": '["domain:partner.org"]'}
        session = (
            _entry("desk", admitted="ext@partner.org"),
            _entry("reader:ext", subject="ext@partner.org"),
        )
        # One card-less login mints both with one expiry.
        same = [
            UnlockedUser(**{**_asdict(e), "expires_at": session[0].expires_at}) for e in session
        ]
        response = _ask(_app(env), _cookie(*same))
        assert _view(response) == (200, "view__reader_ext", "ext")
        assert response.headers[CHOICES_HEADER] == "reader:ext=ext"

    def test_verify_refuses_a_reader_key(self) -> None:
        cookie = _cookie(_entry("reader:ext", subject="ext@partner.org"))
        with TestClient(_app(OIDC_ENV)) as client:
            response = client.get(
                "/verify",
                params={"user": "reader:ext"},
                headers={"Cookie": f"{SESSION_COOKIE_NAME}={cookie}"},
            )
        assert response.status_code == 401


def _asdict(entry: UnlockedUser) -> dict[str, object]:
    return {name: getattr(entry, name) for name in UnlockedUser.__slots__}


# --- selection --------------------------------------------------------------------


def _at(entry: UnlockedUser, expires_at: float) -> UnlockedUser:
    return UnlockedUser(**{**_asdict(entry), "expires_at": expires_at})


class TestSelection:
    def test_the_newest_sign_in_is_the_default(self) -> None:
        now = SessionCodec(SECRET).now()
        cookie = _cookie(_at(_entry("alice"), now + 100), _at(_entry("bob"), now + 200))
        assert _view(_ask(_app(PASSWORD_ENV), cookie))[2] == "bob"

    def test_a_re_login_moves_the_default_without_moving_the_entry(self) -> None:
        now = SessionCodec(SECRET).now()
        codec = SessionCodec(SECRET)
        state = SessionState(session_id="sid", issued_at=now)
        state = state.with_user(
            "alice", expires_at=now + 100, generation_tag=generation_tag(HASHES["alice"])
        )
        state = state.with_user(
            "bob", expires_at=now + 200, generation_tag=generation_tag(HASHES["bob"])
        )
        app = _app(PASSWORD_ENV)
        assert _view(_ask(app, codec.encode(state)))[2] == "bob"
        state = state.with_user(
            "alice", expires_at=now + 300, generation_tag=generation_tag(HASHES["alice"])
        )
        assert [user.username for user in state.users] == ["alice", "bob"]
        assert _view(_ask(app, codec.encode(state)))[2] == "alice"

    def test_the_selector_picks_either_person(self) -> None:
        now = SessionCodec(SECRET).now()
        cookie = _cookie(_at(_entry("alice"), now + 100), _at(_entry("bob"), now + 200))
        app = _app(PASSWORD_ENV)
        response = _ask(app, cookie, "/history/session/abc?as=alice")
        assert _view(response) == (200, "view_alice", "alice")
        assert response.headers[ENTRY_HEADER] == "alice"
        assert response.headers[CHOICES_HEADER] == "bob=bob,alice=alice"
        assert _view(_ask(app, cookie, "/history/?as=bob"))[2] == "bob"

    def test_a_repeated_selector_is_refused(self) -> None:
        cookie = _cookie(_entry("alice"), _entry("bob"))
        assert _ask(_app(PASSWORD_ENV), cookie, "/history/?as=alice&as=bob").status_code == 401
        assert _ask(_app(PASSWORD_ENV), cookie, "/history/?as=alice&as=alice").status_code == 401

    def test_a_selector_naming_no_mappable_entry_is_refused(self) -> None:
        cookie = _cookie(_entry("alice"))
        assert _ask(_app(PASSWORD_ENV), cookie, "/history/?as=bob").status_code == 401
        assert _ask(_app(PASSWORD_ENV), cookie, "/history/?as=").status_code == 401

    def test_ties_go_reader_then_own_card_then_opener_then_rule(self) -> None:
        env = {
            **OIDC_ENV,
            "OSPREY_AUTH_ROSTER_ACCESS_DESK": '["roster"]',
            "OSPREY_AUTH_USERS": "alice,bob,desk,lab",
            "OSPREY_AUTH_ROSTER_ACCESS_LAB": '["domain:example.org"]',
        }
        expiry = SessionCodec(SECRET).now() + 500
        rule = _at(_entry("lab", admitted="bob@example.org"), expiry)
        opener = _at(_entry("desk", opener="bob", subject="bob@example.org"), expiry)
        own = _at(_entry("alice", subject="alice@example.org"), expiry)
        reader = _at(_entry("reader:ext", subject="ext@partner.org"), expiry)
        app = _app(env)
        assert _ask(app, _cookie(rule, opener)).headers[ENTRY_HEADER] == "desk"
        assert _ask(app, _cookie(rule, opener, own)).headers[ENTRY_HEADER] == "alice"
        response = _ask(app, _cookie(rule, opener, own, reader))
        assert response.headers[ENTRY_HEADER] == "reader:ext"
        assert response.headers[CHOICES_HEADER] == ("reader:ext=ext,alice=alice,desk=bob,lab=bob")

    def test_an_entry_that_maps_to_nothing_is_skipped(self) -> None:
        env = {**OIDC_ENV, "OSPREY_AUTH_ROSTER_ACCESS_DESK": '["domain:example.org"]'}
        now = SessionCodec(SECRET).now()
        cookie = _cookie(
            _at(_entry("alice", subject="alice@example.org"), now + 100),
            _at(_entry("desk", admitted="zed@example.org"), now + 900),
        )
        response = _ask(_app(env), cookie)
        assert _view(response) == (200, "view_alice", "alice")
        assert response.headers[CHOICES_HEADER] == "alice=alice"


class TestRefusals:
    def test_no_cookie_is_401(self) -> None:
        assert _ask(_app(PASSWORD_ENV), None).status_code == 401

    def test_a_forged_cookie_is_401(self) -> None:
        forged = SessionCodec("other-secret").encode(
            SessionState(
                session_id="s", issued_at=SessionCodec(SECRET).now(), users=(_entry("alice"),)
            )
        )
        assert _ask(_app(PASSWORD_ENV), forged).status_code == 401

    def test_a_rotated_password_is_401(self) -> None:
        cookie = _cookie(_entry("alice", tag_of="bob"))
        assert _ask(_app(PASSWORD_ENV), cookie).status_code == 401

    def test_a_revoked_session_is_401(self) -> None:
        app = _app(PASSWORD_ENV)
        cookie = _cookie(_entry("alice"), session_id="gone")
        with TestClient(app) as client:
            client.app.state.revocation_store.revoke("gone", SessionCodec(SECRET).now() + 600)
            response = client.get(
                "/verify-history",
                headers={
                    "Cookie": f"{SESSION_COOKIE_NAME}={cookie}",
                    "X-Original-URI": "/history/",
                },
            )
        assert response.status_code == 401


# --- the return-to ----------------------------------------------------------------


class TestReturnTo:
    @pytest.mark.parametrize(
        "target",
        [f"{HISTORY_ORIGIN}/history/", f"{HISTORY_ORIGIN}/history/session/abc?as=alice"],
    )
    def test_the_history_page_is_an_admitted_return_to(self, target: str) -> None:
        assert safe_return_to(target, "alice", history_origin=HISTORY_ORIGIN) == target

    @pytest.mark.parametrize(
        "target",
        [
            f"{HISTORY_ORIGIN}/telemetry/",
            f"{HISTORY_ORIGIN}/history",
            f"{HISTORY_ORIGIN}.evil.org/history/",
            "https://terminals.example.org:9999/history/",
            "http://terminals.example.org:8443/history/",
            f"{HISTORY_ORIGIN}/history/%2f%2fevil",
            f"{HISTORY_ORIGIN}/history/\\evil",
            f"{HISTORY_ORIGIN}/history/\r\nSet-Cookie: x",
        ],
    )
    def test_nothing_else_off_this_origin_is(self, target: str) -> None:
        assert safe_return_to(target, "alice", history_origin=HISTORY_ORIGIN) == "/u/alice/"

    def test_without_a_history_origin_no_absolute_url_is_admitted(self) -> None:
        assert acceptable_return_to(f"{HISTORY_ORIGIN}/history/") is None
        assert acceptable_return_to("/u/alice/x") == "/u/alice/x"

    def test_logout_honours_a_validated_return(self) -> None:
        cookie = _cookie(_entry("reader:ext", subject="ext@partner.org"))
        with TestClient(_app(OIDC_ENV), base_url="https://terminals.example.org") as client:
            client.cookies.set(SESSION_COOKIE_NAME, cookie)
            back = client.get(
                "/auth/logout",
                params={"user": "reader:ext", "return": f"{HISTORY_ORIGIN}/history/"},
                follow_redirects=False,
            )
            elsewhere = client.get(
                "/auth/logout",
                params={"user": "reader:ext", "return": "https://evil.example/"},
                follow_redirects=False,
            )
        assert back.status_code == 303
        assert back.headers["location"] == f"{HISTORY_ORIGIN}/history/"
        surviving = SessionCodec(SECRET).decode(back.cookies[SESSION_COOKIE_NAME])
        assert surviving.entry("reader:ext") is None
        assert elsewhere.headers["location"] == "/"


# --- next through the card-less entry ----------------------------------------------


NEXT = f"{HISTORY_ORIGIN}/history/session/abc"


class TestNext:
    def test_the_oidc_hop_forwards_next(self) -> None:
        with TestClient(_app(OIDC_ENV)) as client:
            response = client.get(entry.ENTRY_PATH, params={"next": NEXT}, follow_redirects=False)
        assert response.status_code == 302
        location = urlsplit(response.headers["location"])
        assert location.path == entry.OIDC_ENTRY_PATH
        assert parse_qs(location.query) == {"next": [NEXT]}

    def test_the_oidc_hop_drops_a_refused_next(self) -> None:
        with TestClient(_app(OIDC_ENV)) as client:
            response = client.get(
                entry.ENTRY_PATH, params={"next": "https://evil.example/"}, follow_redirects=False
            )
        assert response.headers["location"] == entry.OIDC_ENTRY_PATH

    def test_the_password_form_carries_next_and_the_sign_in_returns_there(self) -> None:
        app = _app(PASSWORD_ENV)
        with TestClient(app, base_url="https://terminals.example.org") as client:
            page = client.get(entry.ENTRY_PATH, params={"next": NEXT})
            assert f'name="next" value="{NEXT}"' in page.text
            response = client.post(
                entry.ENTRY_PATH,
                data={"username": "alice", "password": "alice-password", "next": NEXT},
                headers={"Origin": "https://terminals.example.org"},
                follow_redirects=False,
            )
        assert response.status_code == 303
        assert response.headers["location"] == NEXT

    def test_a_failed_password_sign_in_keeps_next(self) -> None:
        with TestClient(_app(PASSWORD_ENV), base_url="https://terminals.example.org") as client:
            response = client.post(
                entry.ENTRY_PATH,
                data={"username": "alice", "password": "wrong", "next": NEXT},
                headers={"Origin": "https://terminals.example.org"},
                follow_redirects=False,
            )
        assert response.status_code == 401
        assert f'name="next" value="{NEXT}"' in response.text


# --- reader entries from the card-less OIDC callback --------------------------------

from tests.services.auth_sidecar.mock_idp import MockIdP  # noqa: E402
from tests.services.auth_sidecar.test_oidc_flow import (  # noqa: E402
    ENTRY_PATH as OIDC_ENTRY_PATH,
)
from tests.services.auth_sidecar.test_oidc_flow import (  # noqa: E402
    PENDING_FLOW_SESSION_KEY,
    _browser,
    _env,
    _reset_idp,  # noqa: F401 - the autouse fixture, re-exported to this module
    _return_to_sidecar,
    _session_from,
    _state_payload,
    _visit_identity_provider,
    idp,  # noqa: F401 - the module-scoped provider fixture
)

EXT_SUBJECT = "idp|ext"


def _reader_sidecar(provider: MockIdP, **overrides: str) -> FastAPI:
    carriers = {
        "OSPREY_AUTH_HISTORY_ORIGIN": HISTORY_ORIGIN,
        "OSPREY_AUTH_READERS": "ext",
        "OSPREY_AUTH_READER_TABLES_EXT": "otel_logs",
        "OSPREY_AUTH_READER_SUBJECT_EXT": EXT_SUBJECT,
    }
    return create_app(_env(provider, **{**carriers, **overrides}))


def _card_less(client: TestClient, next_target: str | None = None) -> httpx.Response:
    params = {} if next_target is None else {"next": next_target}
    start = client.get(OIDC_ENTRY_PATH, params=params, follow_redirects=False)
    assert start.status_code == 302, start.text
    handoff = _visit_identity_provider(start.headers["location"])
    assert handoff.status_code == 302, handoff.text
    return _return_to_sidecar(client, handoff.headers["location"])


class TestReaderSignIn:
    def test_the_pending_flow_keeps_next(self, idp: MockIdP) -> None:  # noqa: F811
        with _browser(_reader_sidecar(idp)) as client:
            client.get(OIDC_ENTRY_PATH, params={"next": NEXT}, follow_redirects=False)
            pending = _state_payload(client)[PENDING_FLOW_SESSION_KEY]
        assert pending["next"] == NEXT

    def test_a_reader_with_no_card_is_returned_to_next(self, idp: MockIdP) -> None:  # noqa: F811
        idp.subject = EXT_SUBJECT
        with _browser(_reader_sidecar(idp)) as client:
            response = _card_less(client, NEXT)
            history = client.get("/verify-history", headers={"X-Original-URI": "/history/"})
        assert response.status_code == 303
        assert response.headers["location"] == NEXT
        minted = _session_from(response).entry("reader:ext")
        assert minted is not None and minted.oidc_subject == EXT_SUBJECT
        assert _view(history) == (200, "view__reader_ext", "ext")

    def test_a_reader_with_no_card_and_no_next_lands_on_the_history_page(
        self,
        idp: MockIdP,  # noqa: F811
    ) -> None:
        idp.subject = EXT_SUBJECT
        with _browser(_reader_sidecar(idp)) as client:
            response = _card_less(client)
        assert response.status_code == 303
        assert response.headers["location"] == f"{HISTORY_ORIGIN}/history/"

    def test_a_reader_a_domain_rule_admits_gets_both_entries(self, idp: MockIdP) -> None:  # noqa: F811
        idp.subject = "ext@partner.org"
        app = _reader_sidecar(
            idp,
            OSPREY_AUTH_OIDC_CLAIM="sub",
            OSPREY_AUTH_READER_SUBJECT_EXT="ext@partner.org",
            OSPREY_AUTH_ROSTER_ACCESS_BOB='["domain:partner.org"]',
        )
        with _browser(app) as client:
            response = _card_less(client, NEXT)
            history = client.get("/verify-history", headers={"X-Original-URI": "/history/"})
        assert response.status_code == 303
        assert response.headers["location"] == NEXT
        state = _session_from(response)
        assert {user.username for user in state.users} == {"bob", "reader:ext"}
        assert _view(history) == (200, "view__reader_ext", "ext")

    def test_an_identity_no_card_and_no_reader_admits_is_still_refused(
        self,
        idp: MockIdP,  # noqa: F811
    ) -> None:
        idp.subject = "idp|nobody"
        with _browser(_reader_sidecar(idp)) as client:
            response = _card_less(client, NEXT)
        assert response.status_code == 403

    def test_a_card_sign_in_honours_next_to_the_history_page(self, idp: MockIdP) -> None:  # noqa: F811
        with _browser(_reader_sidecar(idp)) as client:
            response = _card_less(client, NEXT)
        assert response.status_code == 303
        assert response.headers["location"] == NEXT
        assert _session_from(response).entry("alice") is not None
