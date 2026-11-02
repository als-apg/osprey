"""The refusals contract: every reason has a phrase that ends with who can lift it.

The hook twin restates the phrase table and :func:`refusal_text`; both are pinned
here against the package, word for word.
"""

from __future__ import annotations

import importlib.util
import sys

import pytest

from osprey_connectors import refusals
from osprey_connectors.refusals import (
    KIND_WORDS,
    REASONS,
    STORE_PATH_MARKERS,
    refusal_phrase,
    refusal_text,
)
from osprey_connectors_tests._harness import REFUSAL_BAN_LIST
from osprey_connectors_tests._paths import REPO_ROOT

_TWIN_PATH = REPO_ROOT / "src/osprey/templates/claude_code/claude/hooks/osprey_target_state.py"

LIFTERS = {
    "switch_off": "turn it on from the chip in the web terminal",
    "tier_never": "deployment admin",
    "session_held": "the operator must answer",
    "owner_unknown": "deployment admin",
    "owner_unscheduled": "ask the deployment admin",
    "owner_foreign": "only that login can write here",
    "readonly_run": "the chip cannot lift this — ask the deployment admin",
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
        "refusal_text",
    }


def test_reasons_are_the_closed_set():
    assert len(set(REASONS)) == len(REASONS)
    assert set(REASONS) == set(LIFTERS)


@pytest.mark.parametrize("reason", REASONS)
def test_every_reason_has_a_phrase_naming_its_lifter(reason):
    assert LIFTERS[reason] in refusal_phrase(reason)


@pytest.mark.parametrize("reason", REASONS)
def test_every_phrase_ends_with_who_can_lift_it(reason):
    """What stopped the write comes first; the last clause is the lifter."""
    phrase = refusal_phrase(reason)
    assert phrase.endswith(f"{LIFTERS[reason]}.")
    assert len(phrase) > len(LIFTERS[reason]) + 1


def test_readonly_run_says_the_chip_cannot_lift_it():
    phrase = refusal_phrase("readonly_run")
    assert "the deployment is running read-only" in phrase
    assert "the chip cannot lift this" in phrase


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


# -- refusal_text: the phrase plus a machine-readable trailer -----------------


@pytest.mark.parametrize("kind", [None, *KIND_WORDS, "nonsense"])
@pytest.mark.parametrize("reason", REASONS)
def test_refusal_text_is_the_phrase_then_the_trailer_line(reason, kind):
    text = refusal_text(reason, kind)
    assert text == f"{refusal_phrase(reason, kind)}\n[osprey-refusal: {reason}]"


@pytest.mark.parametrize("reason", REASONS)
def test_refusal_text_ends_with_the_trailer(reason):
    text = refusal_text(reason, "real")
    last_line = text.splitlines()[-1]
    assert last_line == f"[osprey-refusal: {reason}]"
    assert text.endswith(last_line)


@pytest.mark.parametrize("kind", [None, *KIND_WORDS])
@pytest.mark.parametrize("reason", REASONS)
def test_refusal_text_names_no_key_file_variable_or_build_command(reason, kind):
    assert REFUSAL_BAN_LIST.search(refusal_text(reason, kind)) is None
    for marker in STORE_PATH_MARKERS:
        assert marker not in refusal_text(reason, kind)


def test_refusal_text_refuses_an_unknown_reason():
    with pytest.raises(ValueError, match="unknown refusal reason"):
        refusal_text("nope")


@pytest.mark.parametrize("reason", REASONS)
def test_a_detail_follows_the_phrase_and_the_trailer_stays_last(reason):
    text = refusal_text(reason, "real", detail="channel_write was refused.")
    assert text == (
        f"{refusal_phrase(reason, 'real')} channel_write was refused.\n[osprey-refusal: {reason}]"
    )


def test_an_empty_detail_adds_nothing():
    assert refusal_text("switch_off", detail="") == refusal_text("switch_off")
    assert refusal_text("switch_off", detail=None) == refusal_text("switch_off")


def test_refusal_text_default_kind_is_none():
    assert refusal_text("switch_off") == refusal_text("switch_off", None)


# -- the twin restates the same words ------------------------------------------


@pytest.fixture
def twin(monkeypatch):
    """The shipped stdlib hook twin, loaded by path; its ``sys.path`` insert is undone."""
    monkeypatch.setattr(sys, "path", list(sys.path))
    spec = importlib.util.spec_from_file_location("_refusal_phrases_twin", _TWIN_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_twin_phrase_table_matches(twin):
    assert twin._PHRASES == refusals._PHRASES


@pytest.mark.parametrize("kind", [None, *KIND_WORDS, "nonsense"])
@pytest.mark.parametrize("reason", REASONS)
def test_the_twin_refusal_text_matches(twin, reason, kind):
    assert twin.refusal_text(reason, kind) == refusal_text(reason, kind)


def test_the_twin_exports_refusal_text_and_refuses_an_unknown_reason(twin):
    assert "refusal_text" in twin.__all__
    with pytest.raises(ValueError, match="unknown refusal reason"):
        twin.refusal_text("nope")


@pytest.mark.parametrize("detail", [None, "", "queue_start was refused."])
@pytest.mark.parametrize("reason", REASONS)
def test_the_twin_refusal_text_matches_with_a_detail(twin, reason, detail):
    assert twin.refusal_text(reason, "rehearsal", detail=detail) == refusal_text(
        reason, "rehearsal", detail=detail
    )
