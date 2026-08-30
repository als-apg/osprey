"""The one place a session target becomes a connector type — and its refusals.

A session target (``live`` / ``va`` / ``standin``) is a run-time argument; a
connector type (``epics`` / ``virtual_accelerator`` / ``live_standin`` / …) is
what a config selects. Each target names a machine — the facility's own, the
virtual accelerator, the soft-IOC stand-in the deployment runs itself. Every
holder
that follows the session target — the connector-host parent, its child, an
executor sandbox — has to make that translation, and any holder making it
privately is free to route somewhere the roster never claimed. So the
translation is pinned here, including the cases where it must refuse rather than
answer: an unnamed target and a deployment with no derivable live machine both
have to fail loudly, because the failure mode of guessing is a tool call landing
on hardware nobody selected.
"""

from __future__ import annotations

from typing import Any

import pytest

from osprey_connectors.types import (
    CHANNEL_ACCESS_TYPES,
    CONTROL_TARGETS,
    DOOCS,
    EPICS,
    INVENTED_HISTORY_TYPES,
    LIVE_STANDIN,
    MOCK,
    STANDIN_TYPES,
    TARGET_LIVE,
    TARGET_STANDIN,
    TARGET_VA,
    VIRTUAL_ACCELERATOR,
    baseline_target,
    resolve_control_system_type,
    resolve_target,
)


def _section(control_system_type: Any = ..., connector: Any = ...) -> dict[str, Any]:
    """A ``control_system:`` section as the rendered config.yml carries it."""
    section: dict[str, Any] = {}
    if control_system_type is not ...:
        section["type"] = control_system_type
    if connector is not ...:
        section["connector"] = connector
    return section


# ---------------------------------------------------------------------------
# The target vocabulary
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_the_three_targets_are_live_va_and_standin():
    assert (TARGET_LIVE, TARGET_VA, TARGET_STANDIN) == ("live", "va", "standin")
    assert CONTROL_TARGETS == ["live", "va", "standin"]


@pytest.mark.unit
def test_the_stand_in_is_a_type_of_its_own_whose_history_is_invented():
    """Keyed apart from ``epics``, and grouped with the VA for the archive rule."""
    assert LIVE_STANDIN == "live_standin"
    assert STANDIN_TYPES == (LIVE_STANDIN,)
    assert INVENTED_HISTORY_TYPES == (VIRTUAL_ACCELERATOR, LIVE_STANDIN)


@pytest.mark.unit
def test_channel_access_is_spoken_by_epics_the_va_and_the_stand_in():
    """The one class the queue worker executes plans against; the rest browse."""
    assert CHANNEL_ACCESS_TYPES == (EPICS, VIRTUAL_ACCELERATOR, LIVE_STANDIN)
    assert MOCK not in CHANNEL_ACCESS_TYPES
    assert DOOCS not in CHANNEL_ACCESS_TYPES


# ---------------------------------------------------------------------------
# va — the simulator is the simulator on every deployment
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "section",
    [
        _section(EPICS),
        _section(VIRTUAL_ACCELERATOR),
        _section(MOCK),
        _section(),
        None,
    ],
    ids=["epics-baseline", "va-baseline", "mock-baseline", "no-type", "no-section"],
)
def test_va_resolves_to_the_virtual_accelerator_whatever_the_baseline_is(section: Any):
    assert resolve_target(section, TARGET_VA) == VIRTUAL_ACCELERATOR


@pytest.mark.unit
def test_the_resolved_type_is_the_connector_sub_block_key():
    """The factory reads ``connector.<resolved type>``, so the type IS the key."""
    section = _section(
        EPICS,
        {"epics": {"address": "gw"}, "virtual_accelerator": {"timeout": 5.0}},
    )

    assert section["connector"][resolve_target(section, TARGET_VA)] == {"timeout": 5.0}
    assert section["connector"][resolve_target(section, TARGET_LIVE)] == {"address": "gw"}


# ---------------------------------------------------------------------------
# standin — a third machine, reached through a block of its own
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "section",
    [
        _section(EPICS),
        _section(VIRTUAL_ACCELERATOR),
        _section(LIVE_STANDIN),
        _section(MOCK),
        _section(),
        None,
    ],
    ids=[
        "epics-baseline",
        "va-baseline",
        "standin-baseline",
        "mock-baseline",
        "no-type",
        "no-section",
    ],
)
def test_standin_resolves_to_the_stand_in_block_whatever_the_baseline_is(section: Any):
    assert resolve_target(section, TARGET_STANDIN) == LIVE_STANDIN


# ---------------------------------------------------------------------------
# live — the deployment's own control system, when it has one
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_live_on_an_epics_baseline_is_that_baseline():
    assert resolve_target(_section(EPICS), TARGET_LIVE) == EPICS


@pytest.mark.unit
def test_live_is_protocol_neutral():
    """Nothing here knows which control system a facility runs."""
    assert resolve_target(_section(DOOCS), TARGET_LIVE) == DOOCS


@pytest.mark.unit
def test_live_passes_an_unknown_baseline_type_through_unjudged():
    """A typo reaches the factory's "Unknown … type" error, as it does today."""
    assert resolve_target(_section("epcis"), TARGET_LIVE) == "epcis"


# ---------------------------------------------------------------------------
# live on a simulated baseline — derived from a configured block, or refused
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "baseline", [VIRTUAL_ACCELERATOR, MOCK], ids=["va-baseline", "mock-baseline"]
)
def test_live_on_a_simulated_baseline_is_the_one_configured_live_block(baseline: str):
    section = _section(
        baseline,
        {
            "virtual_accelerator": {"timeout": 5.0},
            "mock": {"noise_level": 0.0},
            "epics": {"gateways": {"read_only": {"address": "gw"}}},
        },
    )

    assert resolve_target(section, TARGET_LIVE) == EPICS


@pytest.mark.unit
def test_live_on_a_simulated_baseline_with_no_live_block_refuses():
    section = _section(VIRTUAL_ACCELERATOR, {"virtual_accelerator": {"timeout": 5.0}})

    with pytest.raises(ValueError) as excinfo:
        resolve_target(section, TARGET_LIVE)

    message = str(excinfo.value)
    assert "control_system.connector" in message
    assert VIRTUAL_ACCELERATOR in message


@pytest.mark.unit
@pytest.mark.parametrize(
    "connector",
    [{}, None, "epics", ...],
    ids=["empty", "none", "not-a-mapping", "absent"],
)
def test_live_on_a_simulated_baseline_refuses_without_a_connector_table(connector: Any):
    with pytest.raises(ValueError):
        resolve_target(_section(MOCK, connector), TARGET_LIVE)


@pytest.mark.unit
def test_live_refuses_when_two_live_blocks_leave_it_ambiguous():
    section = _section(
        VIRTUAL_ACCELERATOR,
        {"epics": {"address": "gw"}, "doocs": {"address": "gw"}},
    )

    with pytest.raises(ValueError) as excinfo:
        resolve_target(section, TARGET_LIVE)

    message = str(excinfo.value)
    assert DOOCS in message
    assert EPICS in message


@pytest.mark.unit
def test_live_never_falls_back_to_hardware_on_a_bare_config():
    """An empty config resolves to the mock baseline; live has to raise, not guess."""
    for section in ({}, None, _section(), _section(None)):
        assert resolve_control_system_type(section) == MOCK
        with pytest.raises(ValueError):
            resolve_target(section, TARGET_LIVE)


# ---------------------------------------------------------------------------
# live is never the stand-in, and the stand-in never makes live ambiguous
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "baseline", [VIRTUAL_ACCELERATOR, LIVE_STANDIN], ids=["va-baseline", "standin-baseline"]
)
def test_a_stand_in_block_is_not_a_candidate_for_the_live_machine(baseline: str):
    """The shape every deployment running the stand-in has: two live-looking blocks."""
    section = _section(
        baseline,
        {
            "epics": {"gateways": {"read_only": {"address": "gw"}}},
            "live_standin": {"gateways": {"read_only": {"address": "standin"}}},
        },
    )

    assert resolve_target(section, TARGET_LIVE) == EPICS
    assert resolve_target(section, TARGET_STANDIN) == LIVE_STANDIN


@pytest.mark.unit
def test_a_stand_in_baseline_is_never_returned_as_its_own_live_type():
    """``standin`` reaches the stand-in; ``live`` has to name a machine it isn't."""
    section = _section(LIVE_STANDIN, {"live_standin": {"gateways": {"read_only": {}}}})

    assert resolve_target(section, TARGET_STANDIN) == LIVE_STANDIN
    with pytest.raises(ValueError) as excinfo:
        resolve_target(section, TARGET_LIVE)

    message = str(excinfo.value)
    assert "control_system.connector" in message
    assert LIVE_STANDIN in message


# ---------------------------------------------------------------------------
# An unnamed target is a refusal, never a default
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "target",
    [
        None,
        "",
        "   ",
        "LIVE",
        "Va",
        "live ",
        "epics",
        "virtual_accelerator",
        "mock",
        "live_standin",
        0,
        True,
    ],
    ids=[
        "none",
        "blank",
        "whitespace",
        "wrong-case-live",
        "wrong-case-va",
        "padded",
        "connector-type-epics",
        "connector-type-va",
        "connector-type-mock",
        "connector-type-standin",
        "zero",
        "bool",
    ],
)
def test_an_unrecognized_target_raises_and_resolves_to_nothing(target: Any):
    with pytest.raises(ValueError) as excinfo:
        resolve_target(_section(EPICS), target)

    message = str(excinfo.value)
    assert TARGET_LIVE in message
    assert TARGET_VA in message
    assert TARGET_STANDIN in message


# ---------------------------------------------------------------------------
# The baseline resolver is untouched
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_the_no_argument_resolver_keeps_its_mock_fallback():
    assert resolve_control_system_type(None) == MOCK
    assert resolve_control_system_type({}) == MOCK
    assert resolve_control_system_type({"type": None}) == MOCK
    assert resolve_control_system_type({"type": ""}) == MOCK
    assert resolve_control_system_type("not-a-mapping") == MOCK
    assert resolve_control_system_type({"type": EPICS}) == EPICS
    assert resolve_control_system_type({"type": " epics "}) == " epics "


@pytest.mark.unit
@pytest.mark.parametrize(
    ("control_system_type", "expected"),
    [
        (VIRTUAL_ACCELERATOR, TARGET_VA),
        (LIVE_STANDIN, TARGET_STANDIN),
        (EPICS, TARGET_LIVE),
        (DOOCS, TARGET_LIVE),
        (MOCK, TARGET_LIVE),
    ],
    ids=["va", "standin", "epics", "doocs", "mock"],
)
def test_the_baseline_target_is_the_machine_the_section_selects(
    control_system_type: str, expected: str
):
    assert baseline_target(_section(control_system_type)) == expected


@pytest.mark.unit
def test_a_deployment_that_named_no_machine_is_still_on_the_live_target():
    """``live`` may be underivable there; it is still the target it describes."""
    for section in ({}, None, _section(), _section(None)):
        assert baseline_target(section) == TARGET_LIVE


@pytest.mark.unit
def test_resolving_a_target_does_not_mutate_the_section():
    section = _section(VIRTUAL_ACCELERATOR, {"epics": {"address": "gw"}})
    before = {"type": VIRTUAL_ACCELERATOR, "connector": {"epics": {"address": "gw"}}}

    resolve_target(section, TARGET_LIVE)
    resolve_target(section, TARGET_VA)

    assert section == before
