"""The roster as one reader sees it: people plus the shared app cards.

``roster_entries`` is the single reader of ``modules.web_terminals``' roster:
the people the profile lists under ``users:`` (normalized exactly as
``normalize_users`` normalizes them) and one derived, in-memory entry per name
under ``shared:``. The derived entries take the top port slots, counting down
from ``INDEX_MAX`` in ``shared:`` order, so the people below them keep the slots
they always had.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from osprey.deployment.web_terminals import roster
from osprey.deployment.web_terminals.roster import (
    authored_people,
    freeze_user_indices,
    is_app_entry,
    login_titles_in,
    normalize_users,
    roster_entries,
    roster_people,
)
from osprey.port_layout import INDEX_MAX, default_port
from tests._tree_scan import product_sources

REPO_ROOT = Path(__file__).resolve().parents[3]

TITLES = {
    "logbook": "Logbook Research",
    "knowledge": "Facility Knowledge",
    "channel-finder": "Channel Finder",
}

PEOPLE: list[Any] = [
    "alice",
    {"name": "bob", "index": 1, "display_name": "Bob's terminal", "theme": "desy"},
    {"name": "carol", "index": 2, "role": "operator", "tier": "readwrite"},
]


def _subtree(**extra: Any) -> dict[str, Any]:
    return {"enabled": True, "users": list(PEOPLE), **extra}


# ── The people half ──────────────────────────────────────────────────────────


def test_people_are_byte_identical_to_normalize_users() -> None:
    subtree = _subtree(shared=["logbook", "knowledge"])
    entries = roster_entries(subtree, shared_titles=TITLES)
    people = [entry for entry in entries if not is_app_entry(entry)]
    assert people == normalize_users(PEOPLE)
    # No marker is added to a person: a roster without `shared:` reads exactly
    # as it did before app cards existed.
    assert all("kind" not in entry for entry in people)


def test_roster_without_shared_is_the_people_list() -> None:
    assert roster_entries(_subtree()) == normalize_users(PEOPLE)


def test_people_come_first_in_roster_order() -> None:
    entries = roster_entries(_subtree(shared=["logbook"]), shared_titles=TITLES)
    assert [entry["name"] for entry in entries] == ["alice", "bob", "carol", "logbook"]


# ── The derived app entries ──────────────────────────────────────────────────


def test_each_shared_name_derives_one_app_entry() -> None:
    subtree = _subtree(shared=["logbook", "knowledge", "channel-finder"])
    apps = [entry for entry in roster_entries(subtree, shared_titles=TITLES) if is_app_entry(entry)]
    assert apps == [
        {
            "kind": "app",
            "name": "logbook",
            "app": "logbook",
            "index": INDEX_MAX,
            "display_name": "Logbook Research",
            "access": "any",
        },
        {
            "kind": "app",
            "name": "knowledge",
            "app": "knowledge",
            "index": INDEX_MAX - 1,
            "display_name": "Facility Knowledge",
            "access": "any",
        },
        {
            "kind": "app",
            "name": "channel-finder",
            "app": "channel-finder",
            "index": INDEX_MAX - 2,
            "display_name": "Channel Finder",
            "access": "any",
        },
    ]


def test_slot_follows_position_in_shared() -> None:
    entries = roster_entries(_subtree(shared=["knowledge", "logbook"]), shared_titles=TITLES)
    slots = {entry["name"]: entry["index"] for entry in entries if is_app_entry(entry)}
    assert slots == {"knowledge": INDEX_MAX, "logbook": INDEX_MAX - 1}


def test_app_entry_is_shared_with_the_whole_roster() -> None:
    entries = roster_entries(_subtree(shared=["logbook"]), shared_titles=TITLES)
    (app,) = [entry for entry in entries if is_app_entry(entry)]
    assert roster.entry_is_shared(app)
    assert roster.access_wire_value(roster.resolve_access_principals(app)) == "any"


def test_title_falls_back_to_the_app_name() -> None:
    entries = roster_entries(_subtree(shared=["logbook"]))
    (app,) = [entry for entry in entries if is_app_entry(entry)]
    assert app["display_name"] == "logbook"


def test_blank_title_falls_back_to_the_app_name() -> None:
    entries = roster_entries(_subtree(shared=["logbook"]), shared_titles={"logbook": ""})
    (app,) = [entry for entry in entries if is_app_entry(entry)]
    assert app["display_name"] == "logbook"


def test_unreadable_shared_items_are_skipped_without_moving_slots() -> None:
    entries = roster_entries(_subtree(shared=[42, "", "knowledge"]), shared_titles=TITLES)
    apps = [entry for entry in entries if is_app_entry(entry)]
    assert [(entry["name"], entry["index"]) for entry in apps] == [("knowledge", INDEX_MAX - 2)]


@pytest.mark.parametrize("shared", [None, "logbook", {"logbook": True}, []])
def test_shared_that_is_not_a_list_derives_nothing(shared: Any) -> None:
    assert roster_entries(_subtree(shared=shared)) == normalize_users(PEOPLE)


@pytest.mark.parametrize("subtree", [None, [], "users", {}])
def test_absent_subtree_has_no_roster(subtree: Any) -> None:
    assert roster_entries(subtree) == []


def test_apps_are_listed_even_without_people() -> None:
    entries = roster_entries({"shared": ["logbook"]}, shared_titles=TITLES)
    assert [entry["name"] for entry in entries] == ["logbook"]


# ── strict threading ─────────────────────────────────────────────────────────


def test_strict_is_threaded_into_the_people_normalizer() -> None:
    subtree = {
        "users": [{"name": "dave", "index": 0, "access": "everyone?"}],
        "shared": ["logbook"],
    }
    with pytest.raises(ValueError):
        roster_entries(subtree)
    lenient = roster_entries(subtree, strict=False)
    assert [entry["name"] for entry in lenient] == ["logbook"]


# ── People-only readers ──────────────────────────────────────────────────────


def test_roster_people_never_lists_an_app() -> None:
    subtree = _subtree(shared=["logbook"])
    assert roster_people(subtree) == normalize_users(PEOPLE)


def test_authored_people_is_the_raw_list_as_written() -> None:
    subtree = _subtree(shared=["logbook"])
    authored = authored_people(subtree)
    assert authored == PEOPLE
    # A copy: a reader that appends to it does not edit the profile.
    authored.append("eve")
    assert subtree["users"] == PEOPLE


@pytest.mark.parametrize("users", [None, "alice", {"alice": 0}])
def test_authored_people_of_a_non_list_is_empty(users: Any) -> None:
    assert authored_people({"users": users}) == []


def test_freeze_user_indices_stays_people_only() -> None:
    frozen = freeze_user_indices(PEOPLE)
    assert [entry["name"] for entry in frozen] == ["alice", "bob", "carol"]


def test_bare_list_drift_warning_reads_people_only() -> None:
    from osprey.deployment.web_terminals.lint import lint_web_terminals

    def _codes(subtree: dict[str, Any]) -> list[str]:
        config = {"modules": {"web_terminals": subtree}}
        return [f.code for f in lint_web_terminals(config) if "bare_list" in f.code]

    bare = {"enabled": True, "users": ["alice", "bob"]}
    assert _codes(bare) == _codes({**bare, "shared": ["logbook", "knowledge"]})


# ── Shared-card titles from the repo's tiers/ ────────────────────────────────


def test_login_titles_in_reads_each_login_files_name(tmp_path: Path) -> None:
    apps = tmp_path / "tiers"
    apps.mkdir()
    (apps / "logbook.yml").write_text("# comment\nname: Logbook Research\n", encoding="utf-8")
    (apps / "knowledge.yml").write_text("name: Facility Knowledge\n", encoding="utf-8")
    (apps / "untitled.yml").write_text("claude_md_template: x.j2\n", encoding="utf-8")
    (apps / "broken.yml").write_text("name: [unterminated\n", encoding="utf-8")
    (apps / "notes.txt").write_text("name: ignored\n", encoding="utf-8")
    assert login_titles_in(apps) == {
        "logbook": "Logbook Research",
        "knowledge": "Facility Knowledge",
    }


def test_login_titles_in_a_missing_directory_is_empty(tmp_path: Path) -> None:
    assert login_titles_in(tmp_path / "tiers") == {}


def test_bundled_shared_logins_carry_their_titles() -> None:
    bundled = REPO_ROOT / "src" / "osprey" / "profiles" / "presets" / "tiers"
    titles = login_titles_in(bundled)
    assert {name: titles[name] for name in TITLES} == TITLES


# ── Readers that must see the app cards ──────────────────────────────────────


def _rendered(**extra: Any) -> dict[str, Any]:
    return {"modules": {"web_terminals": {"enabled": True, "users": ["alice"], **extra}}}


def test_host_port_preflight_sees_the_app_slots() -> None:
    from osprey.deployment.host_ports import derive_web_terminal_bindings

    config = _rendered(shared=["logbook"])
    services = {binding.service for binding in derive_web_terminal_bindings(config)}
    assert any("logbook" in service for service in services)
    ports = {binding.host_port for binding in derive_web_terminal_bindings(config)}
    without = {binding.host_port for binding in derive_web_terminal_bindings(_rendered())}
    assert without < ports


def test_liveness_probes_cover_the_app_cards() -> None:
    from osprey.cli.deploy_scaffold_templates import _web_probes

    profile = {
        "config": {
            "deployment.port_base": 20000,
            "modules.web_terminals": {
                "enabled": True,
                "users": ["alice"],
                "shared": ["logbook", "knowledge"],
            },
        }
    }
    terminals = {
        probe.label: probe.port for probe in _web_probes(profile) if probe.label != "landing page"
    }
    web0 = default_port("web", 0, base=20000)
    assert terminals == {
        "terminal (alice)": web0,
        "terminal (logbook)": web0 + INDEX_MAX,
        "terminal (knowledge)": web0 + INDEX_MAX - 1,
    }


def test_liveness_probes_keep_a_person_without_an_index_out() -> None:
    from osprey.cli.deploy_scaffold_templates import _web_probes

    profile = {
        "config": {
            "deployment.port_base": 20000,
            "modules.web_terminals": {"enabled": True, "users": ["alice", {"name": "bob"}]},
        }
    }
    labels = {probe.label for probe in _web_probes(profile)}
    # An object entry without an integer index has no port of its own: the
    # normalizer drops it, and so does every reader built on it.
    assert labels == {"landing page", "terminal (alice)"}


def test_deploy_summary_lists_app_cards_as_shared() -> None:
    from osprey.deployment import deploy_summary

    config = _rendered(shared=["logbook"], auth={"method": "password"})
    assert deploy_summary._shared_cards(config) == ["logbook"]
    assert ("logbook", INDEX_MAX) in deploy_summary._roster(config)


def test_token_logins_include_the_app_cards() -> None:
    from osprey.deployment.deploy_summary import token_login_users

    config = _rendered(shared=["logbook"], auth={"method": "token"})
    assert token_login_users(config) == ["alice", "logbook"]


def test_terminal_secrets_are_minted_for_app_cards(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from osprey.deployment.web_terminals import provision

    seen: list[list[str]] = []

    class _Result:
        missing: list[str] = []

    def _ensure(_root: Any, names: list[str]) -> _Result:
        seen.append(list(names))
        return _Result()

    monkeypatch.setattr(provision, "ensure_terminal_secrets", _ensure)
    provision._provision_terminal_secrets(
        _rendered(shared=["logbook"])["modules"]["web_terminals"], str(tmp_path)
    )
    assert seen == [["alice", "logbook"]]


# ── The grep gate: the roster has one reader ─────────────────────────────────

#: The module that owns the roster: the only place its keys may be read raw.
_HELPER_MODULE = "src/osprey/deployment/web_terminals/roster.py"

#: Files that may still read a ``users`` key directly, each with why. Two kinds:
#: a ``users`` key that is not the web-terminal roster at all, and readers not
#: yet routed through the helper, which this list names so it can only shrink.
_ALLOWED: dict[str, str] = {
    "src/osprey/services/auth_sidecar/sessions.py": "a session payload, not the roster",
    "src/osprey/cli/users_cmd.py": "edits the authored roster in place (people only)",
    "src/osprey/health/core/web_terminals.py": "not yet routed through roster_entries",
    "src/osprey/cli/profile_card.py": "not yet routed through roster_entries",
    "src/osprey/cli/build_profile_model.py": "not yet routed through roster_entries",
    "src/osprey/deployment/status_display.py": "not yet routed through roster_entries",
}

_RAW_USERS_READ = re.compile(
    r"""\.get\(\s*["']users["']|\[\s*["']users["']\s*\]|\bnormalize_users\("""
)


def _offenders() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for path, text in product_sources(REPO_ROOT):
        if path == _HELPER_MODULE:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            code = line.split("#", 1)[0]
            if _RAW_USERS_READ.search(code):
                found.setdefault(path, []).append(f"{number}: {line.strip()}")
    return found


def test_no_reader_outside_the_helper_reads_the_roster_raw() -> None:
    stray = {path: lines for path, lines in _offenders().items() if path not in _ALLOWED}
    assert stray == {}, (
        "read modules.web_terminals' roster through roster_entries (cards), "
        "roster_people or authored_people (people) in roster.py, not raw:\n"
        + "\n".join(f"{path}\n  " + "\n  ".join(lines) for path, lines in stray.items())
    )


def test_the_allowance_list_only_shrinks() -> None:
    stale = sorted(path for path in _ALLOWED if path not in _offenders())
    assert stale == [], f"no longer reads the roster raw; drop it from _ALLOWED: {stale}"
