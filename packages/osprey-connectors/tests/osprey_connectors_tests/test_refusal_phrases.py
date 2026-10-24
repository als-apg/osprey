"""The refusals contract: every reason has a phrase that names who can lift it."""

from __future__ import annotations

import pytest

from osprey_connectors import refusals
from osprey_connectors.refusals import KIND_WORDS, REASONS, STORE_PATH_MARKERS, refusal_phrase
from osprey_connectors_tests._harness import REFUSAL_BAN_LIST

LIFTERS = {
    "switch_off": "turn it on from the chip in the web terminal",
    "tier_never": "deployment admin",
    "owner_unknown": "deployment admin",
    "owner_unscheduled": "ask the deployment admin",
    "owner_foreign": "only that login can write here",
    "readonly_run": "the deployment is running read-only",
    "launch_pinned": "run it again without the pin",
    "stopped_since": "turn it on again from the chip",
    "target_unknown": "retry, then ask the deployment admin",
    "record_unavailable": "retry, then ask the deployment admin",
    "switching": "retry, then ask the deployment admin",
}


def test_public_names_are_listed():
    assert set(refusals.__all__) == {
        "KIND_WORDS",
        "REASONS",
        "STORE_PATH_MARKERS",
        "refusal_phrase",
    }


def test_reasons_are_the_closed_set():
    assert len(set(REASONS)) == len(REASONS)
    assert set(REASONS) == set(LIFTERS)


@pytest.mark.parametrize("reason", REASONS)
def test_every_reason_has_a_phrase_naming_its_lifter(reason):
    assert LIFTERS[reason] in refusal_phrase(reason)


def test_readonly_run_says_the_chip_cannot_lift_it():
    assert "chip cannot lift" in refusal_phrase("readonly_run")


@pytest.mark.parametrize("kind", [None, *KIND_WORDS])
@pytest.mark.parametrize("reason", REASONS)
def test_no_phrase_names_a_key_file_variable_or_build_command(reason, kind):
    assert REFUSAL_BAN_LIST.search(refusal_phrase(reason, kind)) is None


@pytest.mark.parametrize("kind", KIND_WORDS)
@pytest.mark.parametrize("reason", REASONS)
def test_kind_adds_the_machine_words(reason, kind):
    phrase = refusal_phrase(reason, kind)
    assert KIND_WORDS[kind] in phrase
    assert LIFTERS[reason] in phrase
    assert phrase.startswith(refusal_phrase(reason))


def test_unknown_kind_leaves_the_phrase_alone():
    assert refusal_phrase("switch_off", "nonsense") == refusal_phrase("switch_off")


def test_unknown_reason_raises():
    with pytest.raises(ValueError):
        refusal_phrase("nope")


def test_ban_list_catches_what_it_names():
    for bad in (
        "writes_enabled",
        "control_system.writes",
        "profile.yml",
        "config.yml",
        "osprey build",
        "OSPREY_CONTROL_OWNER",
        "SOME_ENV_VAR",
    ):
        assert REFUSAL_BAN_LIST.search(f"set {bad} first")


def test_store_path_markers_are_path_spellings():
    assert STORE_PATH_MARKERS == ("control_target", "control_context.json")
