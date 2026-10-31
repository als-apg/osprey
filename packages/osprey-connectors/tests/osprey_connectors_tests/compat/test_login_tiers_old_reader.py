"""An old container's owner rung facing a login-tier map that spells ``asked``.

A shared container (Bluesky queue, dispatch worker) whose image still carries
osprey-connectors 2026.10.0 reads the ``control_system.login_tiers`` map a newer
framework build stages. That release knows two tier words, and its rung reads a
map holding any other value as unreadable as a whole. So one ``asked`` entry
refuses every login's writes ``owner_unknown``, those mapped ``allowed``
included, until the container's connectors are upgraded. This is fail-closed,
not a loosening, and it is why the connectors release goes out before the
framework that renders ``asked``.

``_released_owner_tier_term`` is that release's ``posture_store.owner_tier_term``
(tag ``osprey-connectors-v2026.10.0``), frozen here with its tree branch reduced
to the tree-bound answer a shared container gives.
"""

from __future__ import annotations

from typing import Any

import pytest

from osprey_connectors import posture_store

#: The two values a 2026.10.0 map entry may carry.
_RELEASED_LOGIN_TIERS = frozenset({"allowed", "never"})


def _released_owner_tier_term(section: Any, owner: Any) -> str:
    try:
        carries = isinstance(section, dict) and "login_tiers" in section
        if not carries:
            return "unknown"  # a shared container: the tree is bound
        login_tiers = section["login_tiers"]
        if not isinstance(login_tiers, dict) or not all(
            isinstance(login, str)
            and login
            and isinstance(tier, str)
            and tier in _RELEASED_LOGIN_TIERS
            for login, tier in login_tiers.items()
        ):
            return "unknown"
        if not isinstance(owner, str) or not owner:
            return "unknown"
        tier = login_tiers.get(owner)
    except Exception:
        return "unknown"
    if tier == "allowed":
        return "allowed"
    if tier == "never":
        return "never"
    return "unknown"


MIXED = {"type": "epics", "writes": "allowed", "login_tiers": {"alice": "allowed", "bob": "asked"}}
TWO_WORDS = {
    "type": "epics",
    "writes": "allowed",
    "login_tiers": {"alice": "allowed", "carol": "never"},
}


@pytest.mark.parametrize("login", ["alice", "bob"])
def test_one_asked_entry_makes_the_old_reader_refuse_every_login(login):
    assert _released_owner_tier_term(MIXED, login) == "unknown"


def test_the_old_reader_reads_a_two_word_map_as_before():
    assert _released_owner_tier_term(TWO_WORDS, "alice") == "allowed"
    assert _released_owner_tier_term(TWO_WORDS, "carol") == "never"


def test_this_package_reads_each_entry_of_the_same_map(monkeypatch):
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, "/nonexistent")
    assert posture_store.owner_tier_term(MIXED, "alice") == "allowed"
    assert posture_store.owner_tier_term(MIXED, "bob") == "asked"
