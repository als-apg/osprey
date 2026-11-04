"""The vocabulary names the beam-lifetime and operation-mode signal roles."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

VOCABULARY_JSON = (
    Path(__file__).resolve().parents[2] / "src/osprey/facility/schema/_generated/vocabulary.json"
)

LIFETIME_AND_MODE_ROLES = {
    "beam_lifetime_readback": ["beam lifetime", "lifetime"],
    "operation_mode_status": [
        "operation mode",
        "operating mode",
        "machine mode",
        "beam mode",
        "ops mode",
        "machine state",
    ],
}


def _roles() -> dict[str, dict]:
    vocabulary = json.loads(VOCABULARY_JSON.read_text(encoding="utf-8"))
    return {record["name"]: record for record in vocabulary["signal_roles"]}


@pytest.mark.parametrize("role", sorted(LIFETIME_AND_MODE_ROLES))
def test_each_role_is_in_the_generated_vocabulary_with_a_description(role: str) -> None:
    roles = _roles()

    assert role in roles
    assert roles[role]["description"].strip()


@pytest.mark.parametrize("role", sorted(LIFETIME_AND_MODE_ROLES))
def test_each_role_carries_its_channel_finder_aliases(role: str) -> None:
    assert _roles()[role]["aliases"] == LIFETIME_AND_MODE_ROLES[role]


def test_no_alias_of_these_roles_is_claimed_by_another_role() -> None:
    roles = _roles()
    others = {
        alias.casefold()
        for name, record in roles.items()
        if name not in LIFETIME_AND_MODE_ROLES
        for alias in record.get("aliases") or []
    }

    for aliases in LIFETIME_AND_MODE_ROLES.values():
        assert not others & {alias.casefold() for alias in aliases}


def test_the_validator_accepts_these_roles() -> None:
    from osprey.facility.validate import signal_roles

    assert set(LIFETIME_AND_MODE_ROLES) <= signal_roles()
