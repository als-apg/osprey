"""Tests for the card-less sign-in at ``/auth/enter``.

The property carrying the weight is equivalence, not a new rule: for every
roster card, the card-less login mints an entry if and only if the card login
for that card, driven by the same credential, would mint one, and the two
entries agree field for field except the clock-stamped expiry. The matrix test
drives both flows over the same access rules and compares what they minted.

Around it: the form names nobody, every credential refusal is one refusal, the
card-less form and the card form share one throttle window per name, and a card
the identity matrix refuses is filed and left out rather than failing the
login.
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from osprey.audit import writer
from osprey.deployment.web_terminals.personas import env_var_suffix
from osprey.services.auth_sidecar import audit
from osprey.services.auth_sidecar.app import create_app
from osprey.services.auth_sidecar.passwords import hash_password
from osprey.services.auth_sidecar.routes import entry
from osprey.services.auth_sidecar.routes.login import DENIAL_MESSAGE, LOGIN_PATH
from osprey.services.auth_sidecar.routes.recheck import ENV_ROSTER_ROLE_PREFIX
from osprey.services.auth_sidecar.sessions import (
    SessionCodec,
    UnlockedUser,
)
from osprey.utils.identity import AUDIT_IDENTITY_ENV, TERMINAL_USER_ENV

SESSION_SECRET = "session-secret-value"
SESSION_LIFETIME = 3600
EXTERNAL_ORIGIN = "https://terminals.example.org"

ALICE_PASSWORD = "alice-password"
BOB_PASSWORD = "bob-password"
CAROL_PASSWORD = "carol-password"
ALICE_HASH = hash_password(ALICE_PASSWORD)

PASSWORD_ENV = {
    "OSPREY_AUTH_METHOD": "password",
    "OSPREY_AUTH_SESSION_SECRET": SESSION_SECRET,
    "OSPREY_AUTH_SESSION_LIFETIME": str(SESSION_LIFETIME),
    "OSPREY_AUTH_USERS": "alice,bob,carol",
    "OSPREY_AUTH_PW_HASH_ALICE": ALICE_HASH,
    "OSPREY_AUTH_PW_HASH_BOB": hash_password(BOB_PASSWORD),
    "OSPREY_AUTH_PW_HASH_CAROL": hash_password(CAROL_PASSWORD),
    "OSPREY_AUTH_EXTERNAL_ORIGIN": EXTERNAL_ORIGIN,
    "OSPREY_AUTH_TLS_ENABLED": "true",
}

OIDC_ENV = {
    "OSPREY_AUTH_METHOD": "oidc",
    "OSPREY_AUTH_SESSION_SECRET": SESSION_SECRET,
    "OSPREY_AUTH_STATE_SECRET": "state-secret-value",
    "OSPREY_AUTH_SESSION_LIFETIME": str(SESSION_LIFETIME),
    "OSPREY_AUTH_USERS": "alice,bob",
    "OSPREY_AUTH_OIDC_ISSUER": "https://idp.example.org",
    "OSPREY_AUTH_OIDC_CLIENT_ID": "client-id",
    "OSPREY_AUTH_OIDC_CLIENT_SECRET": "client-secret-value",
    "OSPREY_AUTH_EXTERNAL_ORIGIN": EXTERNAL_ORIGIN,
    "OSPREY_AUTH_TLS_ENABLED": "true",
}

BROWSER_ACCEPT = {"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}
JSON_ACCEPT = {"Accept": "application/json"}


def _access_env(**rules: str) -> dict[str, str]:
    """The default roster with ``OSPREY_AUTH_ROSTER_ACCESS_<USER>`` set per rule."""
    return {
        **PASSWORD_ENV,
        **{
            f"OSPREY_AUTH_ROSTER_ACCESS_{env_var_suffix(user)}": rule
            for user, rule in rules.items()
        },
    }


def _client(env: dict[str, str] | None = None) -> TestClient:
    """A test client over a sidecar built from ``env``, addressed over https."""
    return TestClient(
        create_app(env if env is not None else PASSWORD_ENV), base_url="https://testserver"
    )


def _post(client: TestClient, path: str, data: dict[str, object], **kwargs: Any) -> httpx.Response:
    """POST a form the way a browser on this deployment would, with its ``Origin``."""
    headers = {"Origin": EXTERNAL_ORIGIN, **dict(kwargs.pop("headers", {}) or {})}
    return client.post(path, data=data, headers=headers, follow_redirects=False, **kwargs)


def _enter(
    client: TestClient, username: str = "alice", password: str = ALICE_PASSWORD
) -> httpx.Response:
    """One card-less sign-in attempt."""
    return _post(client, entry.ENTRY_PATH, {"username": username, "password": password})


def _card_login(
    client: TestClient, card: str, *, opener: str | None, password: str = ALICE_PASSWORD
) -> httpx.Response:
    """One card-form attempt, naming the opener where the card's rule carries ``roster``."""
    data: dict[str, object] = {"user": card, "next": "", "password": password}
    if opener is not None:
        data["username"] = opener
    return _post(client, LOGIN_PATH, data)


def _entries(response: httpx.Response) -> dict[str, UnlockedUser]:
    """The session entries the response's cookie carries, keyed by card."""
    header = response.headers.get("set-cookie")
    if not header:
        return {}
    raw = header.split(";", 1)[0].split("=", 1)[1]
    state = SessionCodec(SESSION_SECRET, max_age=SESSION_LIFETIME).decode(raw)
    return {user.username: user for user in state.users}


def _without_expiry(user: UnlockedUser) -> UnlockedUser:
    """``user`` with its clock-stamped expiry blanked, for field-for-field comparison."""
    return dataclasses.replace(user, expires_at=0.0)


def _links(body: str) -> list[str]:
    """Every link target inside the page's card list."""
    listed = re.search(r'<ul class="login-cards">(.*?)</ul>', body, flags=re.DOTALL)
    return re.findall(r'href="([^"]+)"', listed.group(1)) if listed else []


def _inputs(body: str) -> list[str]:
    """Every ``<input>`` tag in the page, as raw markup."""
    return re.findall(r"<input[^>]*>", body, flags=re.IGNORECASE | re.DOTALL)


@pytest.fixture
def zone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The sidecar's own bound audit subdirectory, as compose gives it."""
    directory = tmp_path / "var" / "audit" / "sidecar"
    directory.mkdir(parents=True)
    monkeypatch.setenv(audit.AUDIT_DIR_ENV, str(directory))
    monkeypatch.setenv(AUDIT_IDENTITY_ENV, "sidecar")
    monkeypatch.delenv(TERMINAL_USER_ENV, raising=False)
    return directory


def _records(zone: Path) -> list[dict[str, Any]]:
    """Every ledger record filed so far, in append order."""
    path = zone / f"{audit.SURFACE}{writer.LEDGER_SUFFIX}"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]


def _throttle_window(client: TestClient, key: str) -> float:
    """How long the app's login throttle holds ``key`` shut."""
    return client.app.state.attempt_throttle.retry_after(key)  # type: ignore[attr-defined]


# --- The form ---------------------------------------------------------------


def test_the_sign_in_page_asks_for_a_username_and_a_password() -> None:
    response = _client().get(entry.ENTRY_PATH)

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    fields = _inputs(response.text)
    assert sorted(re.search(r'name="([^"]+)"', field).group(1) for field in fields) == [  # type: ignore[union-attr]
        "password",
        "username",
    ]
    assert not any('type="hidden"' in field for field in fields)
    assert f'action="{entry.ENTRY_PATH}"' in response.text


def test_the_sign_in_page_names_nobody_on_the_roster() -> None:
    body = _client().get(entry.ENTRY_PATH).text

    for name in ("alice", "bob", "carol"):
        assert name not in body


# --- What a verified credential opens ---------------------------------------


def test_one_card_sends_the_person_to_their_own_terminal() -> None:
    client = _client()
    response = _enter(client)

    assert response.status_code == 303
    assert response.headers["location"] == "/u/alice/"
    entries = _entries(response)
    assert list(entries) == ["alice"]
    assert entries["alice"].opener == ""
    assert entries["alice"].generation_tag
    assert client.get("/verify", params={"user": "alice"}).status_code == 200


def test_an_own_and_a_shared_card_are_listed_and_both_open() -> None:
    client = _client(_access_env(carol="any"))
    response = _enter(client)

    assert response.status_code == 200
    assert _links(response.text) == ["/u/alice/", "/u/carol/"]
    listed = re.search(r'<ul class="login-cards">(.*?)</ul>', response.text, flags=re.DOTALL)
    assert listed is not None
    items = re.findall(r"<li>(.*?)</li>", listed.group(1), flags=re.DOTALL)
    assert "shared" not in items[0]
    assert "shared" in items[1]
    entries = _entries(response)
    assert entries["alice"].opener == ""
    assert entries["carol"].opener == "alice"
    assert client.get("/verify", params={"user": "alice"}).status_code == 200
    assert client.get("/verify", params={"user": "carol"}).status_code == 200


def test_a_roster_card_of_ones_own_is_opened_as_its_opener() -> None:
    response = _enter(_client(_access_env(alice="any")))

    assert response.status_code == 303
    entries = _entries(response)
    assert list(entries) == ["alice"]
    assert entries["alice"].opener == "alice"


RULES = {
    "own": "own",
    "any": "any",
    "self-and-roster": '["self","roster"]',
    "domain": '["domain:lab.example"]',
    "user": '["user:x@lab.example"]',
    "unreadable": "not a rule",
}


@pytest.mark.parametrize("card", ["alice", "carol"])
@pytest.mark.parametrize("rule", list(RULES), ids=list(RULES))
def test_the_sign_in_opens_exactly_the_cards_the_card_login_opens(card: str, rule: str) -> None:
    env = _access_env(**{card: RULES[rule]})

    by_card: dict[str, UnlockedUser] = {}
    for name in ("alice", "bob", "carol"):
        client = _client(env)
        shared = "roster" in client.app.state.settings.access(name)  # type: ignore[attr-defined]
        response = _card_login(client, name, opener="alice" if shared else None)
        if response.status_code == 303:
            by_card.update(_entries(response))

    response = _enter(_client(env))
    card_less = _entries(response) if response.status_code in (200, 303) else {}

    assert set(card_less) == set(by_card)
    assert {name: _without_expiry(user) for name, user in card_less.items()} == {
        name: _without_expiry(user) for name, user in by_card.items()
    }


# --- Refusals ----------------------------------------------------------------


def test_an_unknown_name_and_a_wrong_password_are_one_refusal(zone: Path) -> None:
    unknown = _enter(_client(), username="mally", password=ALICE_PASSWORD)
    wrong = _enter(_client(), username="alice", password="not-it")

    assert unknown.status_code == wrong.status_code == 401
    assert DENIAL_MESSAGE in unknown.text
    assert DENIAL_MESSAGE in wrong.text
    assert unknown.text.replace("mally", "alice") == wrong.text
    assert [(record["subject"], record["reason"]) for record in _records(zone)] == [
        ("mally", audit.REASON_BAD_CREDENTIAL),
        ("alice", audit.REASON_BAD_CREDENTIAL),
    ]


def test_the_sign_in_and_the_card_share_one_throttle_window() -> None:
    client = _client()
    assert _enter(client, password="not-it").status_code == 401
    assert _card_login(client, "alice", opener=None).status_code == 429

    client = _client()
    assert _card_login(client, "alice", opener=None, password="not-it").status_code == 401
    assert _enter(client).status_code == 429


def test_a_throttled_attempt_is_not_evaluated(zone: Path) -> None:
    client = _client()
    assert _enter(client, password="not-it").status_code == 401
    window = _throttle_window(client, "alice")
    filed = len(_records(zone))

    response = _enter(client)

    assert response.status_code == 429
    assert response.headers["retry-after"]
    assert len(_records(zone)) == filed
    assert _throttle_window(client, "alice") <= window


def test_a_cross_site_post_never_reaches_the_throttle() -> None:
    client = _client()
    response = _post(
        client,
        entry.ENTRY_PATH,
        {"username": "alice", "password": "not-it"},
        headers={"Origin": "https://evil.example"},
    )

    assert response.status_code == 400
    assert _throttle_window(client, "alice") == 0


@pytest.mark.parametrize(
    ("data", "headers", "status"),
    [
        ({"password": "x"}, BROWSER_ACCEPT, 303),
        ({"password": "x"}, JSON_ACCEPT, 400),
        ({"username": ["alice", "bob"], "password": "x"}, BROWSER_ACCEPT, 400),
        ({"username": "", "password": "x"}, BROWSER_ACCEPT, 303),
    ],
    ids=["missing-browser", "missing-json", "repeated", "empty"],
)
def test_a_form_without_one_username_is_not_an_attempt(
    data: dict[str, object], headers: dict[str, str], status: int
) -> None:
    client = _client()
    response = _post(client, entry.ENTRY_PATH, data, headers=headers)

    assert response.status_code == status
    if status == 303:
        assert response.headers["location"] == entry.ENTRY_PATH
    for name in ("alice", "bob", ""):
        assert _throttle_window(client, name) == 0


def test_an_over_long_username_is_never_looked_up(zone: Path) -> None:
    padded = "alice" + "x" * 200
    response = _enter(_client(), username=padded, password=ALICE_PASSWORD)

    assert response.status_code == 401
    assert [record["subject"] for record in _records(zone)] == [padded[: entry.MAX_USERNAME_LENGTH]]


def test_a_person_no_card_admits_is_refused_and_filed(zone: Path) -> None:
    client = _client(_access_env(alice='["domain:lab.example"]'))
    response = _enter(client)

    assert response.status_code == 403
    assert "No terminal for this account" in response.text
    assert "set-cookie" not in response.headers
    assert [(record["subject"], record["reason"]) for record in _records(zone)] == [
        ("alice", audit.REASON_NO_CARD)
    ]
    assert _throttle_window(client, "alice") == 0


def test_a_card_the_recheck_refuses_is_left_out_and_filed(
    zone: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(f"{ENV_ROSTER_ROLE_PREFIX}{env_var_suffix('carol')}", "öperator")
    response = _enter(_client(_access_env(carol="any")))

    assert response.status_code == 303
    assert response.headers["location"] == "/u/alice/"
    assert list(_entries(response)) == ["alice"]
    records = [(record["subject"], record["reason"]) for record in _records(zone)]
    assert ("carol", audit.REASON_UNSAFE_ROLE) in records
    assert ("alice", audit.REASON_PASSWORD_LOGIN) in records
    assert len(records) == 2


def test_success_is_filed_once_per_opened_card(zone: Path) -> None:
    assert _enter(_client(_access_env(carol="any"))).status_code == 200

    records = _records(zone)
    assert [(record["subject"], record["reason"]) for record in records] == [
        ("alice", audit.REASON_PASSWORD_LOGIN),
        ("carol", audit.REASON_PASSWORD_LOGIN),
    ]
    assert "detail" not in records[0]
    assert records[1]["detail"] == "opener=alice"


# --- Where the routes apply ---------------------------------------------------


def test_the_entry_routes_answer_404_where_they_do_not_apply() -> None:
    assert _enter(_client(OIDC_ENV)).status_code == 404

    unconfigured = _client({**PASSWORD_ENV, "OSPREY_AUTH_METHOD": "none"})
    response = unconfigured.get(entry.ENTRY_PATH, follow_redirects=False)
    assert response.status_code != 200
    assert 'name="password"' not in response.text


def test_oidc_deployments_are_handed_to_the_oidc_entry() -> None:
    response = _client(OIDC_ENV).get(entry.ENTRY_PATH, follow_redirects=False)

    assert response.status_code == 302
    assert response.headers["location"] == entry.OIDC_ENTRY_PATH


def test_the_oidc_entry_path_is_the_one_the_oidc_routes_serve() -> None:
    from osprey.services.auth_sidecar.routes import oidc

    assert entry.OIDC_ENTRY_PATH == oidc.ENTRY_PATH
