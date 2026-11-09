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

from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.cli.build_profile_load import load_profile_document, reject_unknown_kind_keys
from osprey.cli.profile_root import ProfileKind
from osprey.deployment.web_terminals.lint import lint_web_terminals
from osprey.deployment.web_terminals.roster import (
    app_access_refusal,
    expand_login_principals,
    resolve_cards,
)
from osprey_connectors.errors import BuildProfileError
from tests._renders import tier_render

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
