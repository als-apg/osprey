"""The one place a session target becomes a connector type — and its refusals.

A session target (``live`` / ``va``) is a run-time argument; a connector type
(``epics`` / ``virtual_accelerator`` / …) is what a config selects. Every holder
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
    CONTROL_TARGETS,
    DOOCS,
    EPICS,
    MOCK,
    TARGET_LIVE,
    TARGET_VA,
    VIRTUAL_ACCELERATOR,
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
def test_the_only_two_targets_are_live_and_va():
    assert (TARGET_LIVE, TARGET_VA) == ("live", "va")
    assert CONTROL_TARGETS == ["live", "va"]


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
# An unnamed target is a refusal, never a default
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "target",
    [None, "", "   ", "LIVE", "Va", "live ", "epics", "virtual_accelerator", "mock", 0, True],
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
def test_resolving_a_target_does_not_mutate_the_section():
    section = _section(VIRTUAL_ACCELERATOR, {"epics": {"address": "gw"}})
    before = {"type": VIRTUAL_ACCELERATOR, "connector": {"epics": {"address": "gw"}}}

    resolve_target(section, TARGET_LIVE)
    resolve_target(section, TARGET_VA)

    assert section == before
