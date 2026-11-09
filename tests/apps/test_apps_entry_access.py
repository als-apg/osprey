"""``access:`` on a login's ``apps.<name>`` entry: who sees and opens the panel.

The entry is authored in the words a shared roster card's ``access`` uses and
read by the same parser. The build resolves it as a shared card's set is
expanded at render — ``self``, ``roster`` and ``tier:`` spelled out to the
identities the sidecar forwards — and projects the result onto the login's
``web.panels.<id>.access`` as ``user:``/``domain:`` members only. The terminal
compares the forwarded subject against that set: ``GET /api/panels`` leaves out
a panel the caller may not open, and the panel proxy answers it exactly as an
unpublished panel. An entry without the key is every user of the login, and the
tier alone decides writes.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from osprey.apps.manifest import parse_manifest
from osprey.apps.projection import project_login
from osprey.apps.rules import AppSelection, LoginSelection
from osprey.cli.build_profile_load import load_profile_document, reject_unknown_kind_keys
from osprey.cli.profile_root import ProfileKind
from osprey.deployment.web_terminals.lint import lint_web_terminals
from osprey.deployment.web_terminals.roster import (
    app_access_refusal,
    expand_login_principals,
    resolve_cards,
)
from osprey.interfaces.web_auth import WebCredentials
from osprey.interfaces.web_terminal.app import _load_panel_config
from osprey.services.auth_sidecar.identity_headers import canonical_identity
from osprey_connectors.errors import BuildProfileError
from tests._renders import tier_render
from tests.interfaces._asgi import OPERATOR_SECRET, PANEL_TOKEN
from tests.interfaces.web_terminal._proxy_fakes import panel_app

SUBJECT_HEADER = "X-Osprey-Auth-Subject"

_MANIFEST = {"name": "knob", "containers": {"knob": {"runtime": "osprey"}}}

_PANEL_MANIFEST = {
    "name": "dial",
    "containers": {"dial": {"runtime": "osprey"}},
    "panel": {"label": "Dial", "container": "dial", "entry": "/"},
}


def _write_yaml(path: Path, body: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return path


def _repo(tmp_path: Path, root: dict[str, Any] | None = None) -> Path:
    _write_yaml(tmp_path / "apps" / "knob" / "app.yml", _MANIFEST)
    _write_yaml(
        tmp_path / "profile.yml",
        {"name": "site", "data": "data", "apps": {"knob": {}}, **(root or {})},
    )
    (tmp_path / "data").mkdir(exist_ok=True)
    return tmp_path


# ---------------------------------------------------------------------------
# Schema: the login spells it, the one parser reads it
# ---------------------------------------------------------------------------


def test_a_login_may_spell_access() -> None:
    reject_unknown_kind_keys(
        {"name": "ro", "apps": {"knob": {"shown": True, "access": ["user:a@example.org"]}}},
        ProfileKind.TIER,
        "tiers/ro.yml",
    )


def test_a_login_entry_carries_its_access(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    layer = _write_yaml(
        repo / "tiers" / "ro.yml",
        {"name": "ro", "apps": {"knob": {"access": ["tier:admin", "user:x@example.org"]}}},
    )

    profile = load_profile_document(layer).profile

    assert profile.apps["knob"].access == ["tier:admin", "user:x@example.org"]


def test_an_entry_without_access_carries_none(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    layer = _write_yaml(repo / "tiers" / "ro.yml", {"name": "ro", "apps": {"knob": {}}})

    assert load_profile_document(layer).profile.apps["knob"].access is None


@pytest.mark.parametrize(
    ("access", "named"), [(["group:ops"], "'group:ops'"), ("everyone", "'everyone'"), ([], "empty")]
)
def test_an_unreadable_access_is_refused_naming_login_app_and_value(
    tmp_path: Path, access: Any, named: str
) -> None:
    repo = _repo(tmp_path)
    layer = _write_yaml(
        repo / "tiers" / "ro.yml", {"name": "ro", "apps": {"knob": {"access": access}}}
    )

    with pytest.raises(BuildProfileError) as excinfo:
        load_profile_document(layer)

    message = str(excinfo.value)
    assert "login 'ro' apps.knob.access" in message
    assert named in message


def test_a_refusal_is_the_parsers_one_sentence_naming_the_login_entry(tmp_path: Path) -> None:
    _write_yaml(
        tmp_path / "apps" / "knob-example" / "app.yml",
        {**_MANIFEST, "name": "knob-example"},
    )
    _write_yaml(
        tmp_path / "profile.yml", {"name": "site", "data": "data", "apps": {"knob-example": {}}}
    )
    (tmp_path / "data").mkdir()
    layer = _write_yaml(
        tmp_path / "tiers" / "readonly.yml",
        {"name": "readonly", "apps": {"knob-example": {"access": ["group:ops"]}}},
    )

    with pytest.raises(BuildProfileError) as excinfo:
        load_profile_document(layer)

    (refusal,) = [line for line in str(excinfo.value).splitlines() if "group:ops" in line]
    assert refusal.lstrip(" -").startswith("login 'readonly' apps.knob-example.access")
    assert "modules.web_terminals.shared" not in str(excinfo.value)


# ---------------------------------------------------------------------------
# Lint: the twin of the shared card's access check
# ---------------------------------------------------------------------------


def _web_terminals(method: str = "oidc", claim: str | None = None) -> dict[str, Any]:
    web_terminals: dict[str, Any] = {
        "enabled": True,
        "default_tier": "viewer",
        "renders": {
            "viewer": tier_render("viewer", "demo-viewer"),
            "admin": tier_render("admin", "demo-admin"),
        },
        "users": [
            {"name": "alice", "index": 0, "tier": "admin", "oidc_subject": "alice@example.org"},
            {"name": "bob", "index": 1, "tier": "viewer", "oidc_subject": "bob@example.org"},
        ],
        "auth": {"method": method, "allow_insecure_http": True},
    }
    if method == "oidc":
        web_terminals["auth"]["oidc"] = {"issuer": "https://sso.example.org"}
        if claim is not None:
            web_terminals["auth"]["oidc"]["claim"] = claim
    return web_terminals


def _lint_codes(tmp_path: Path, access: Any, *, method: str = "oidc") -> list[str]:
    _write_yaml(
        tmp_path / "tiers" / "admin.yml", {"name": "admin", "apps": {"knob": {"access": access}}}
    )
    config = {
        "facility": {"prefix": "demo"},
        "system": {"timezone": "UTC"},
        "registry": {"url": "registry.example.org/demo"},
        "deploy": {"host": "demo", "fqdn": "demo.example.org"},
        "modules": {"web_terminals": _web_terminals(method)},
    }
    findings = lint_web_terminals(config, rendered_project=False, profile_root=tmp_path)
    return [f.code for f in findings if "apps.knob.access" in f.message]


def test_lint_refuses_an_unreadable_app_access(tmp_path: Path) -> None:
    assert _lint_codes(tmp_path, ["group:ops"]) == ["web_terminals.invalid_app_access"]


@pytest.mark.parametrize("method", ["none", "token"])
def test_lint_refuses_a_set_under_a_method_that_forwards_no_identity(
    tmp_path: Path, method: str
) -> None:
    assert _lint_codes(tmp_path, ["user:x@example.org"], method=method) == [
        "web_terminals.access_principal_without_idp"
    ]


def test_lint_refuses_a_domain_only_set_under_password(tmp_path: Path) -> None:
    assert _lint_codes(tmp_path, ["domain:example.org"], method="password") == [
        "web_terminals.access_principal_without_idp"
    ]


@pytest.mark.parametrize("method", ["oidc", "password"])
def test_lint_refuses_a_tier_that_admits_nobody(tmp_path: Path, method: str) -> None:
    assert _lint_codes(tmp_path, ["tier:ghost"], method=method) == [
        "web_terminals.tier_principal_admits_nobody"
    ]


@pytest.mark.parametrize(
    ("method", "access"),
    [
        ("oidc", ["self", "tier:admin", "domain:example.org"]),
        ("password", ["user:alice"]),
        ("password", ["self", "tier:viewer"]),
    ],
)
def test_lint_accepts_a_set_a_browser_caller_can_match(
    tmp_path: Path, method: str, access: list[str]
) -> None:
    assert _lint_codes(tmp_path, access, method=method) == []


# ---------------------------------------------------------------------------
# Resolved at build: only user: and domain: members remain
# ---------------------------------------------------------------------------


def _cards(method: str = "oidc") -> list[dict[str, Any]]:
    return resolve_cards(_web_terminals(method), {}, strict=False)


@pytest.mark.parametrize(
    ("principals", "expected"),
    [
        ({"self"}, {"user:alice@example.org"}),
        ({"roster"}, {"user:alice@example.org", "user:bob@example.org"}),
        ({"tier:viewer"}, {"user:bob@example.org"}),
        ({"user:x@example.org", "domain:lab.org"}, {"user:x@example.org", "domain:lab.org"}),
    ],
)
def test_roster_relative_members_are_spelled_out_under_oidc(
    principals: set[str], expected: set[str]
) -> None:
    resolved = expand_login_principals(
        frozenset(principals), _cards(), login="admin", method="oidc"
    )

    assert resolved == frozenset(expected)


def test_under_password_a_person_is_their_roster_name() -> None:
    resolved = expand_login_principals(
        frozenset({"self", "roster"}), _cards("password"), login="viewer", method="password"
    )

    assert resolved == frozenset({"user:alice", "user:bob"})


def test_under_password_a_tier_is_its_peoples_roster_names() -> None:
    resolved = expand_login_principals(
        frozenset({"tier:admin"}), _cards("password"), login="viewer", method="password"
    )

    assert resolved == frozenset({"user:alice"})


def test_members_take_the_canonical_spelling_of_the_mapped_claim() -> None:
    web_terminals = _web_terminals(claim="email")
    web_terminals["users"][0]["oidc_subject"] = "Alice@Example.org"
    cards = resolve_cards(web_terminals, {}, strict=False)

    resolved = expand_login_principals(
        frozenset({"self", "user:Carol@Lab.org"}),
        cards,
        login="admin",
        method="oidc",
        claim="email",
    )

    assert resolved == frozenset({"user:alice@example.org", "user:carol@lab.org"})


def test_under_sub_a_member_keeps_its_spelling() -> None:
    resolved = expand_login_principals(
        frozenset({"user:Carol@Lab.org"}), _cards(), login="admin", method="oidc", claim=None
    )

    assert resolved == frozenset({"user:Carol@Lab.org"})


@pytest.mark.parametrize(
    ("resolved", "method", "code"),
    [
        (frozenset({"user:a@example.org"}), None, "access_principal_without_idp"),
        (frozenset({"domain:lab.org"}), "password", "access_principal_without_idp"),
        (frozenset(), "oidc", "tier_principal_admits_nobody"),
        (frozenset({"user:a@example.org"}), "oidc", None),
        (frozenset({"domain:lab.org"}), "oidc", None),
        (frozenset({"user:alice"}), "password", None),
    ],
)
def test_a_set_no_browser_caller_can_match_is_refused(
    resolved: frozenset[str], method: str | None, code: str | None
) -> None:
    refusal = app_access_refusal(resolved, login="ro", app="dial", method=method)

    assert (refusal[0] if refusal else None) == code
    if refusal:
        assert refusal[1].startswith("login 'ro' apps.dial.access admits no browser caller")


def test_without_a_login_wall_nobody_is_forwarded() -> None:
    resolved = expand_login_principals(
        frozenset({"self", "roster", "domain:lab.org"}), _cards(), login="admin", method=None
    )

    assert resolved == frozenset({"domain:lab.org"})


def _panel_repo(tmp_path: Path, admin_access: Any) -> Path:
    _write_yaml(
        tmp_path / "apps" / "dial" / "app.yml",
        _PANEL_MANIFEST,
    )
    (tmp_path / "data").mkdir()
    web_terminals = _web_terminals()
    web_terminals.pop("renders")
    _write_yaml(
        tmp_path / "profile.yml",
        {
            "name": "site",
            "data": "data",
            "apps": {"dial": {}},
            "config": {f"modules.web_terminals.{k}": v for k, v in web_terminals.items()},
        },
    )
    _write_yaml(
        tmp_path / "tiers" / "admin.yml",
        {"name": "admin", "apps": {"dial": {"access": admin_access}}},
    )
    _write_yaml(tmp_path / "tiers" / "viewer.yml", {"name": "viewer"})
    return tmp_path


def test_the_build_resolves_each_logins_entry(tmp_path: Path) -> None:
    from osprey.apps.loader import ROOT_LOGIN, check_deployment

    logins = check_deployment(_panel_repo(tmp_path, ["self", "domain:lab.org"])).logins

    assert logins["admin"].apps["dial"].access == json.dumps(
        ["domain:lab.org", "user:alice@example.org"], separators=(",", ":")
    )
    assert logins["viewer"].apps["dial"].access is None
    assert logins[ROOT_LOGIN].apps["dial"].access is None


def test_the_build_refuses_a_set_no_browser_caller_can_match(tmp_path: Path) -> None:
    from osprey.apps.loader import check_deployment

    with pytest.raises(BuildProfileError, match="login 'admin' apps.dial.access admits no"):
        check_deployment(_panel_repo(tmp_path, ["tier:ghost"]))


def test_a_login_gets_the_apps_skill_only_when_it_may_see_the_app(tmp_path: Path) -> None:
    """``access:`` narrows who on the login opens the panel, not what its agent knows.

    The agent works on the panel token and is judged by the login alone, so a login
    whose ``access:`` restricts the application still carries its skill; a login that
    does not see the application at all does not.
    """
    from osprey.apps.loader import check_deployment
    from osprey.apps.projection import login_skills

    repo = _panel_repo(tmp_path, ["self"])
    _write_yaml(repo / "apps" / "dial" / "app.yml", {**_PANEL_MANIFEST, "skills": ["dial-ops"]})
    skill = repo / "apps" / "dial" / "skills" / "dial-ops" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("---\nname: dial-ops\ndescription: Turn the dial.\n---\n", encoding="utf-8")
    _write_yaml(
        repo / "tiers" / "viewer.yml", {"name": "viewer", "apps": {"dial": {"shown": False}}}
    )

    checked = check_deployment(repo)

    assert checked.logins["admin"].apps["dial"].access is not None
    assert login_skills(checked.manifests, checked.logins["admin"]) == [("dial", "dial-ops")]
    assert login_skills(checked.manifests, checked.logins["viewer"]) == []


# ---------------------------------------------------------------------------
# Projection: the panel block carries the resolved set
# ---------------------------------------------------------------------------


def _project(access: str | None) -> dict[str, Any]:
    manifest = parse_manifest(dict(_PANEL_MANIFEST), app="dial")
    selection = LoginSelection(
        audience="person",
        apps={"dial": AppSelection(shown=True, access=access)},
        tier="never",
        shared_cards=(),
    )
    return project_login([manifest], {"services": {"dial": {"port": 18950}}}, selection)


def test_the_panel_block_carries_the_resolved_set() -> None:
    wire = json.dumps(["user:alice@example.org"], separators=(",", ":"))

    assert _project(wire)["web"]["panels"]["dial"]["access"] == wire


def test_an_entry_without_access_projects_no_key() -> None:
    assert "access" not in _project(None)["web"]["panels"]["dial"]


# ---------------------------------------------------------------------------
# The terminal: who is listed, who is proxied
# ---------------------------------------------------------------------------

PANEL_ID = "dial"
ACCESS = ["domain:lab.org", "user:alice@example.org"]


def _load(spec: dict[str, Any]) -> list[dict]:
    from unittest.mock import patch

    with patch(
        "osprey_connectors.workspace.load_osprey_config",
        return_value={"web": {"panels": {PANEL_ID: spec}}},
    ):
        _enabled, custom, _default = _load_panel_config()
    return custom


def test_the_terminal_reads_the_access_set() -> None:
    (panel,) = _load({"url": "http://127.0.0.1:9300", "access": json.dumps(ACCESS)})

    assert panel["access"] == tuple(ACCESS)


@pytest.mark.parametrize("value", ["any", '["tier:admin"]', '["self"]', "[1]", "not json"])
def test_an_access_set_the_build_would_not_write_refuses_at_boot(value: str) -> None:
    with pytest.raises(ValueError, match=rf"web\.panels\.{PANEL_ID}\.access"):
        _load({"url": "http://127.0.0.1:9300", "access": value})


def _panel(access: list[str] | None) -> dict[str, Any]:
    panel: dict[str, Any] = {
        "id": PANEL_ID,
        "label": "Dial",
        "url": "http://127.0.0.1:9300",
        "configDefined": True,
        "app": "dial",
        "agentRoutes": [{"method": "GET", "path": "/api/state"}],
    }
    if access is not None:
        panel["access"] = tuple(access)
    return panel


@pytest.fixture
def workspace_dir(tmp_path: Path) -> Path:
    """An empty watched directory for the terminal."""
    ws = tmp_path / "_agent_data"
    ws.mkdir()
    return ws


@pytest.fixture
def restricted(workspace_dir):
    yield from panel_app(workspace_dir, [_panel(ACCESS)])


@pytest.fixture
def open_panel(workspace_dir):
    yield from panel_app(workspace_dir, [_panel(None)])


def _listed(client: TestClient, subject: str | None) -> dict[str, Any]:
    headers = {SUBJECT_HEADER: subject} if subject else {}
    return client.get("/api/panels", headers=headers).json()


@pytest.mark.parametrize("subject", ["alice@example.org", "carol@lab.org"])
def test_an_admitted_caller_is_listed_the_panel(restricted, subject: str) -> None:
    _app, client = restricted

    payload = _listed(client, subject)

    (row,) = [cp for cp in payload["custom"] if cp["id"] == PANEL_ID]
    assert "access" not in row


@pytest.mark.parametrize("subject", ["bob@example.org", "x@sub.lab.org", None])
def test_any_other_caller_is_not_listed_the_panel(restricted, subject: str | None) -> None:
    _app, client = restricted

    payload = _listed(client, subject)

    assert PANEL_ID not in [cp["id"] for cp in payload["custom"]]
    assert PANEL_ID not in payload["visible"]
    assert PANEL_ID not in payload["loopback_ports"]


def test_a_panel_without_access_is_listed_to_everyone(open_panel) -> None:
    _app, client = open_panel

    assert PANEL_ID in [cp["id"] for cp in _listed(client, None)["custom"]]


def _stub_upstream(app) -> list[str]:
    from unittest.mock import AsyncMock

    import httpx

    sent: list[str] = []

    async def fake_request(*, method, url, headers, content, follow_redirects=True):  # noqa: ARG001
        sent.append(url)
        return httpx.Response(200, json={"ok": True}, headers={"content-type": "application/json"})

    app.state.proxy_client.request = AsyncMock(side_effect=fake_request)
    return sent


def test_a_withheld_panel_is_answered_as_an_unpublished_one(restricted) -> None:
    app, client = restricted
    sent = _stub_upstream(app)

    withheld = client.get(f"/panel/{PANEL_ID}/api/state", headers={SUBJECT_HEADER: "bob@x.org"})
    unpublished = client.get("/panel/nosuch/api/state", headers={SUBJECT_HEADER: "bob@x.org"})

    assert withheld.status_code == unpublished.status_code == 404
    assert withheld.text == f"Panel '{PANEL_ID}' is not available"
    assert sent == []


def test_an_admitted_caller_is_proxied(restricted) -> None:
    app, client = restricted
    sent = _stub_upstream(app)

    resp = client.get(f"/panel/{PANEL_ID}/api/state", headers={SUBJECT_HEADER: "carol@lab.org"})

    assert resp.status_code == 200
    assert sent == ["http://127.0.0.1:9300/api/state"]


def test_a_withheld_panel_socket_is_closed_as_an_unpublished_one(restricted) -> None:
    _app, client = restricted

    with pytest.raises(WebSocketDisconnect) as excinfo:
        with client.websocket_connect(
            f"/panel/{PANEL_ID}/ws", headers={SUBJECT_HEADER: "bob@x.org"}
        ) as ws:
            ws.receive_text()

    assert excinfo.value.code == 4004


@pytest.mark.no_auth_seam
def test_the_agent_on_the_panel_token_is_judged_by_the_login_alone(restricted) -> None:
    app, _client = restricted
    app.state.web_credentials = WebCredentials(
        operator_secret=OPERATOR_SECRET, panel_token=PANEL_TOKEN
    )
    sent = _stub_upstream(app)
    client = TestClient(app, client=("127.0.0.1", 54321))

    resp = client.get(
        f"/panel/{PANEL_ID}/api/state", headers={"authorization": f"Bearer {PANEL_TOKEN}"}
    )

    assert resp.status_code == 200
    assert sent == ["http://127.0.0.1:9300/api/state"]


# ---------------------------------------------------------------------------
# One canonical spelling: what the sidecar forwards is what the build wrote
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("claim", "expected"),
    [("email", "alice@example.org"), ("sub", "Alice@Example.org"), (None, "Alice@Example.org")],
)
def test_the_canonical_spelling_follows_the_claims_rule(claim: str | None, expected: str) -> None:
    assert canonical_identity("Alice@Example.org", claim=claim) == expected


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        (SimpleNamespace(admitted_identity="Carol@Lab.org"), "carol@lab.org"),
        (
            SimpleNamespace(admitted_identity="", oidc_subject="Alice@Example.org"),
            "alice@example.org",
        ),
    ],
)
def test_the_sidecar_forwards_the_canonical_spelling(entry: Any, expected: str) -> None:
    from osprey.services.auth_sidecar.routes.verify import _subject_for

    settings = SimpleNamespace(method="oidc", oidc_claim="email")

    assert _subject_for("alice", entry, settings) == expected  # type: ignore[arg-type]


def test_under_password_the_sidecar_forwards_the_roster_name() -> None:
    from osprey.services.auth_sidecar.routes.verify import _subject_for

    entry = SimpleNamespace(admitted_identity="", opener="")
    settings = SimpleNamespace(method="password", oidc_claim="email")

    assert _subject_for("alice", entry, settings) == "alice"  # type: ignore[arg-type]
