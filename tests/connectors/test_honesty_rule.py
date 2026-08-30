"""The question every refusal site asks — asked the way each site's reader reads.

There are two kinds of config, read by different readers, and the guard has to
resolve a key exactly as the reader it guards resolves it. A build profile's
``config:`` block reaches the rendered project through the emitter, which honors
the dotted spelling and a nested mapping alike, so both are live there. A
rendered ``config.yml`` is read by ``ConfigBuilder`` and ``MCPServerConfig``,
which walk nested sections only — a top-level ``archiver.type:`` line in that
file configures nothing.

Getting that backwards is not a cosmetic difference: it is a running VA+mock
stack. The bypass tests below are the regression, and each one states the live
value the readers would have resolved.

The run-time question is the third: a session asking to be pointed at the
virtual accelerator is asking for a pairing the config does not yet have, so it
is judged against the target's control system and the config's own archiver.

All three are asked of both machines a deployment stands up for itself — the
virtual accelerator and the live stand-in — because neither has a past anybody
recorded. The last section is the stand-in half of each.
"""

from __future__ import annotations

from typing import Any

import pytest

from osprey.connectors.honesty import (
    VA_MOCK_ARCHIVER_WHY,
    pairing_for_target,
    pairing_in_profile,
    pairing_in_rendered_config,
)
from osprey.connectors.types import MOCK_ARCHIVER

VA = "virtual_accelerator"
STANDIN = "live_standin"


def _flat(control_system: str | None = None, archiver: Any = ...) -> dict[str, Any]:
    """The dotted spelling — canonical in a build profile's ``config:`` block."""
    config: dict[str, Any] = {}
    if control_system is not None:
        config["control_system.type"] = control_system
    if archiver is not ...:
        config["archiver.type"] = archiver
    return config


def _nested(control_system: str | None = None, archiver: Any = ...) -> dict[str, Any]:
    """The nested spelling — the only live one in a rendered ``config.yml``."""
    config: dict[str, Any] = {}
    if control_system is not None:
        config["control_system"] = {"type": control_system, "writes_enabled": False}
    if archiver is not ...:
        config["archiver"] = {"type": archiver}
    return config


# ---------------------------------------------------------------------------
# A build profile's config: block — both spellings are live
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spell", [_flat, _nested], ids=["flat", "nested"])
def test_a_profile_pairing_a_va_with_the_mock_is_caught_in_either_spelling(spell: Any) -> None:
    assert pairing_in_profile(spell(VA, MOCK_ARCHIVER)).is_invented_history


@pytest.mark.parametrize("spell", [_flat, _nested], ids=["flat", "nested"])
def test_a_profile_that_names_no_archiver_has_named_the_mock(spell: Any) -> None:
    """The emitter writes no archiver.type, and the factory then falls back."""
    verdict = pairing_in_profile(spell(VA))

    assert verdict.is_invented_history
    assert "unset" in verdict.archiver_phrase


@pytest.mark.parametrize("spell", [_flat, _nested], ids=["flat", "nested"])
def test_a_profile_with_a_store_is_honest_in_either_spelling(spell: Any) -> None:
    assert not pairing_in_profile(spell(VA, "mongodb_archiver")).is_invented_history


def test_a_profile_that_spells_the_archiver_twice_and_differently_fails_closed() -> None:
    """Both spellings reach the same rendered leaf, so the profile has stated
    the archive twice and is free to be wrong once. The build is not the thing
    that should pick a winner — the same stance va_archiver_errors takes on a
    duplicated connection key."""
    verdict = pairing_in_profile(
        {"control_system.type": VA, "archiver.type": "mongodb_archiver"}
        | {"archiver": {"type": MOCK_ARCHIVER}}
    )

    assert verdict.is_invented_history
    assert "twice" in verdict.archiver_phrase


def test_a_blank_archiver_type_in_a_profile_is_the_mock() -> None:
    """YAML's bare `archiver.type:` is None, and the factory falls back for it
    exactly as for an absent key."""
    assert pairing_in_profile(
        {"control_system.type": VA, "archiver.type": None}
    ).is_invented_history


# ---------------------------------------------------------------------------
# A rendered config.yml — nested sections only, and the bypasses that proves
# ---------------------------------------------------------------------------


def test_a_rendered_va_with_the_nested_mock_is_caught() -> None:
    assert pairing_in_rendered_config(_nested(VA, MOCK_ARCHIVER)).is_invented_history


def test_a_rendered_va_with_no_archiver_section_is_caught() -> None:
    assert pairing_in_rendered_config(_nested(VA)).is_invented_history


def test_a_rendered_va_with_a_nested_store_is_honest() -> None:
    assert not pairing_in_rendered_config(_nested(VA, "mongodb_archiver")).is_invented_history


def test_a_flat_archiver_line_cannot_excuse_a_nested_mock() -> None:
    """BYPASS REGRESSION. ConfigBuilder and MCPServerConfig walk nested sections
    only, so the flat line configures nothing while the nested mock is what the
    factory builds. Reading the flat key as the archiver would wave through a
    running VA+mock stack."""
    config = _nested(VA, MOCK_ARCHIVER) | {"archiver.type": "mongodb_archiver"}

    assert pairing_in_rendered_config(config).is_invented_history


def test_a_flat_only_archiver_line_is_not_an_archiver() -> None:
    """BYPASS REGRESSION. With no `archiver:` section the factory finds no type
    and falls back to the mock, however real the top-level line looks."""
    config = _nested(VA) | {"archiver.type": "mongodb_archiver"}
    verdict = pairing_in_rendered_config(config)

    assert verdict.is_invented_history
    assert "configures nothing" in verdict.archiver_phrase
    assert "mongodb_archiver" in verdict.archiver_phrase


def test_a_flat_control_system_line_cannot_excuse_a_nested_virtual_accelerator() -> None:
    """BYPASS REGRESSION, the same hole on the other key: the live control system
    is the nested one, so a flat 'mock' line does not make this deployment a
    simulation that claims nothing."""
    config = _nested(VA, MOCK_ARCHIVER) | {"control_system.type": "mock"}

    assert pairing_in_rendered_config(config).is_invented_history


def test_a_flat_only_control_system_is_not_a_virtual_accelerator() -> None:
    """The mirror of the rule, and not a bypass: with no `control_system:`
    section the factory falls back to the mock, so this deployment is a mock
    machine with a mock archive — the honest storeless pairing, whatever the
    inert line says."""
    assert not pairing_in_rendered_config(_flat(VA, MOCK_ARCHIVER)).is_invented_history


# ---------------------------------------------------------------------------
# Negative space, shared by both readings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("judge", [pairing_in_profile, pairing_in_rendered_config])
@pytest.mark.parametrize(
    "config",
    [
        pytest.param(_nested("mock", MOCK_ARCHIVER), id="mock-with-mock"),
        pytest.param(_nested("epics", MOCK_ARCHIVER), id="epics-with-mock"),
        pytest.param(_nested("epics", "epics_archiver"), id="hardware"),
        pytest.param({}, id="says-nothing"),
    ],
)
def test_every_other_pairing_is_left_alone(judge: Any, config: dict[str, Any]) -> None:
    """The rule is about a simulation inventing its own past — not about which
    store a facility runs, and not a general ban on the mock archiver."""
    assert not judge(config).is_invented_history


@pytest.mark.parametrize("judge", [pairing_in_profile, pairing_in_rendered_config])
def test_a_config_that_is_not_a_mapping_reads_as_unset(judge: Any) -> None:
    """A malformed config has its own error elsewhere; this is not the place to
    compete for it."""
    assert not judge(None).is_invented_history
    assert not judge(["control_system.type: virtual_accelerator"]).is_invented_history


def test_a_set_type_is_named_as_written() -> None:
    assert pairing_in_profile({"archiver.type": MOCK_ARCHIVER}).archiver_phrase == repr(
        MOCK_ARCHIVER
    )


def test_the_shared_explanation_says_what_is_wrong() -> None:
    """Every site quotes this; it has to carry the reason on its own — and it is
    quoted about either machine, so it has to name both."""
    assert "virtual accelerator" in VA_MOCK_ARCHIVER_WHY
    assert "stand-in" in VA_MOCK_ARCHIVER_WHY
    assert "mock archiver" in VA_MOCK_ARCHIVER_WHY


# ---------------------------------------------------------------------------
# A prospective target — the pairing a switch would create
# ---------------------------------------------------------------------------


def test_a_va_target_beside_the_mock_archiver_is_refused() -> None:
    """Nothing has been switched yet — the config still says 'epics' — so the
    pairing to judge is the one asking for the simulator would produce."""
    verdict = pairing_for_target(_nested("epics", MOCK_ARCHIVER), "va")

    assert verdict.is_invented_history
    assert verdict.archiver_phrase == repr(MOCK_ARCHIVER)


def test_a_va_target_with_no_archiver_section_is_refused() -> None:
    """Unset counts as mock here for the reason it does everywhere: the factory
    falls back, so a deployment that named no store still has the synthesizing
    one waiting for the session that switches."""
    verdict = pairing_for_target(_nested("epics"), "va")

    assert verdict.is_invented_history
    assert "unset" in verdict.archiver_phrase


def test_a_va_target_beside_a_store_is_allowed() -> None:
    config = _nested("epics", "mongodb_archiver")

    assert not pairing_for_target(config, "va").is_invented_history


def test_a_live_target_beside_the_mock_archiver_is_allowed() -> None:
    """The rule is about a simulation inventing its own past. A session on the
    real machine reading a synthesized history is a different complaint, and one
    this module does not make."""
    config = _nested("epics", MOCK_ARCHIVER)

    assert not pairing_for_target(config, "live").is_invented_history


def test_a_live_target_off_a_simulated_baseline_is_allowed_beside_the_mock_archiver() -> None:
    """The deployment this predicate exists for: built for the simulator, and
    switching to the one real machine its connector table names."""
    config = _nested(VA, MOCK_ARCHIVER)
    config["control_system"]["connector"] = {"epics": {"address": "gw"}}

    assert not pairing_for_target(config, "live").is_invented_history


def test_a_va_target_reads_the_archiver_nested_only() -> None:
    """BYPASS REGRESSION, inherited from the rendered reading this shares: the
    flat line configures nothing, so it cannot excuse the nested mock."""
    config = _nested("epics", MOCK_ARCHIVER) | {"archiver.type": "mongodb_archiver"}

    assert pairing_for_target(config, "va").is_invented_history


def test_a_va_target_on_a_config_that_names_no_archiver_at_all_is_refused() -> None:
    """Fail closed: 'va' is the virtual accelerator on every deployment, and a
    file with no archiver section has the mock, empty or malformed alike."""
    assert pairing_for_target({}, "va").is_invented_history
    assert pairing_for_target(None, "va").is_invented_history


@pytest.mark.parametrize("target", ["production", "", None], ids=["unknown", "blank", "none"])
def test_a_target_that_is_not_a_target_has_no_pairing_to_judge(target: Any) -> None:
    """The resolver's refusal is propagated, not answered around: reporting a
    target nobody can be switched to as 'allowed' would put the verdict on a
    session that cannot exist."""
    with pytest.raises(ValueError, match="Unknown control target"):
        pairing_for_target(_nested("epics", MOCK_ARCHIVER), target)


def test_a_live_target_with_no_derivable_machine_propagates_the_refusal() -> None:
    """A simulated baseline with no live connector block names no real machine,
    so there is no pairing here either — and the caller hears which fact is
    missing rather than a verdict about it."""
    with pytest.raises(ValueError, match="has no control system"):
        pairing_for_target(_nested(VA, "mongodb_archiver"), "live")


# ---------------------------------------------------------------------------
# The live stand-in — the same rule's other machine
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spell", [_flat, _nested], ids=["flat", "nested"])
def test_a_profile_pairing_the_standin_with_the_mock_is_caught(spell: Any) -> None:
    """A stand-in's past is as invented as the simulator's: it is a soft IOC this
    deployment stood up for itself, so there is no history of it to have kept."""
    assert pairing_in_profile(spell(STANDIN, MOCK_ARCHIVER)).is_invented_history


@pytest.mark.parametrize("spell", [_flat, _nested], ids=["flat", "nested"])
def test_a_standin_profile_that_names_no_archiver_has_named_the_mock(spell: Any) -> None:
    """The fallback is the common way in here too."""
    verdict = pairing_in_profile(spell(STANDIN))

    assert verdict.is_invented_history
    assert "unset" in verdict.archiver_phrase


@pytest.mark.parametrize("spell", [_flat, _nested], ids=["flat", "nested"])
def test_a_standin_profile_with_a_store_is_honest(spell: Any) -> None:
    assert not pairing_in_profile(spell(STANDIN, "mongodb_archiver")).is_invented_history


def test_a_rendered_standin_with_the_nested_mock_is_caught() -> None:
    assert pairing_in_rendered_config(_nested(STANDIN, MOCK_ARCHIVER)).is_invented_history


def test_a_rendered_standin_with_no_archiver_section_is_caught() -> None:
    assert pairing_in_rendered_config(_nested(STANDIN)).is_invented_history


def test_a_rendered_standin_with_a_nested_store_is_honest() -> None:
    assert not pairing_in_rendered_config(_nested(STANDIN, "mongodb_archiver")).is_invented_history


def test_a_flat_archiver_line_cannot_excuse_a_nested_mock_beside_the_standin() -> None:
    """BYPASS REGRESSION, inherited whole: the rendered reading is nested-only
    whichever invented-history machine is being judged."""
    config = _nested(STANDIN, MOCK_ARCHIVER) | {"archiver.type": "mongodb_archiver"}

    assert pairing_in_rendered_config(config).is_invented_history


def test_a_standin_target_beside_the_mock_archiver_is_refused() -> None:
    """Nothing has been switched yet — the config still says 'epics' — so the
    pairing to judge is the one asking for the stand-in would produce."""
    verdict = pairing_for_target(_nested("epics", MOCK_ARCHIVER), "standin")

    assert verdict.is_invented_history
    assert verdict.archiver_phrase == repr(MOCK_ARCHIVER)


def test_a_standin_target_with_no_archiver_section_is_refused() -> None:
    verdict = pairing_for_target(_nested("epics"), "standin")

    assert verdict.is_invented_history
    assert "unset" in verdict.archiver_phrase


def test_a_standin_target_beside_a_store_is_allowed() -> None:
    config = _nested("epics", "mongodb_archiver")

    assert not pairing_for_target(config, "standin").is_invented_history


def test_a_standin_target_reads_the_archiver_nested_only() -> None:
    """BYPASS REGRESSION: the flat line configures nothing on this target either."""
    config = _nested("epics", MOCK_ARCHIVER) | {"archiver.type": "mongodb_archiver"}

    assert pairing_for_target(config, "standin").is_invented_history


def test_a_live_target_is_still_allowed_beside_the_mock_on_a_standin_deployment() -> None:
    """The widening must not reach 'live'. A deployment running the stand-in
    still has one real machine, and a session on it reading a synthesized history
    is the different complaint this module has never made."""
    config = _nested(STANDIN, MOCK_ARCHIVER)
    config["control_system"]["connector"] = {"epics": {"address": "gw"}}

    assert not pairing_for_target(config, "live").is_invented_history
