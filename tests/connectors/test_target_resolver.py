"""The one place a control target becomes a connector type — and its refusals.

A control target (``live`` / ``va`` / ``standin``) is a run-time argument; a
connector type (``epics`` / ``virtual_accelerator`` / ``live_standin`` / …) is
what a config selects. Each target names a machine — the facility's own, the
virtual accelerator, the soft-IOC stand-in the deployment runs itself. Every
holder
that follows the control target — the connector-host parent, its child, an
executor sandbox — has to make that translation, and any holder making it
privately is free to route somewhere the roster never claimed. So the
translation is pinned here, including the cases where it must refuse rather than
answer: an unnamed target and a deployment with no derivable live machine both
have to fail loudly, because the failure mode of guessing is a tool call landing
on hardware nobody selected.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from osprey_connectors.types import (
    CONTROL_TARGETS,
    DOOCS,
    EPICS,
    IN_PROCESS,
    INVENTED_HISTORY_TYPES,
    LIVE_STANDIN,
    ONE_REAL_MACHINE,
    STANDIN_TYPES,
    TARGET_LIVE,
    TARGET_STANDIN,
    TARGET_VA,
    TRANSPORT_CA,
    TRANSPORT_DOOCS,
    TRANSPORT_IN_PROCESS,
    VIRTUAL_ACCELERATOR,
    baseline_target,
    connector_transport,
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


def _in_process(connector: Any = ...) -> dict[str, Any]:
    """The simulator served in process, with *connector* blocks beside its own."""
    blocks = connector if isinstance(connector, dict) else {}
    own = {**blocks.get(VIRTUAL_ACCELERATOR, {}), "serving": IN_PROCESS}
    return _section(VIRTUAL_ACCELERATOR, {**blocks, VIRTUAL_ACCELERATOR: own})


# ---------------------------------------------------------------------------
# The target vocabulary
# ---------------------------------------------------------------------------


def test_the_three_targets_are_live_va_and_standin():
    assert (TARGET_LIVE, TARGET_VA, TARGET_STANDIN) == ("live", "va", "standin")
    assert CONTROL_TARGETS == ["live", "va", "standin"]


def test_the_stand_in_is_a_type_of_its_own_whose_history_is_invented():
    """Keyed apart from ``epics``, and grouped with the VA for the archive rule."""
    assert LIVE_STANDIN == "live_standin"
    assert STANDIN_TYPES == (LIVE_STANDIN,)
    assert INVENTED_HISTORY_TYPES == (VIRTUAL_ACCELERATOR, LIVE_STANDIN)


def test_channel_access_is_spoken_by_epics_the_va_and_the_stand_in():
    """The one wire the queue worker executes plans against; the rest browse."""
    for connector_type in (EPICS, VIRTUAL_ACCELERATOR, LIVE_STANDIN):
        assert connector_transport({"type": connector_type}) == TRANSPORT_CA
    assert connector_transport({"type": DOOCS}) == TRANSPORT_DOOCS
    assert connector_transport(_in_process()) == TRANSPORT_IN_PROCESS


# ---------------------------------------------------------------------------
# va — the simulator is the simulator on every deployment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "section",
    [
        _section(EPICS),
        _section(VIRTUAL_ACCELERATOR),
        _in_process(),
        _section(),
        None,
    ],
    ids=["epics-baseline", "va-baseline", "in-process-baseline", "no-type", "no-section"],
)
def test_va_resolves_to_the_virtual_accelerator_whatever_the_baseline_is(section: Any):
    assert resolve_target(section, TARGET_VA) == VIRTUAL_ACCELERATOR


def test_the_resolved_type_is_the_connector_sub_block_key():
    """The factory reads ``connector.<resolved type>``, so the type IS the key."""
    section = _section(
        EPICS,
        {"epics": {"address": "gw"}, "virtual_accelerator": {"timeout_s": 5.0}},
    )

    assert section["connector"][resolve_target(section, TARGET_VA)] == {"timeout_s": 5.0}
    assert section["connector"][resolve_target(section, TARGET_LIVE)] == {"address": "gw"}


# ---------------------------------------------------------------------------
# standin — a third machine, reached through a block of its own
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "section",
    [
        _section(EPICS),
        _section(VIRTUAL_ACCELERATOR),
        _section(LIVE_STANDIN),
        _in_process(),
        _section(),
        None,
    ],
    ids=[
        "epics-baseline",
        "va-baseline",
        "standin-baseline",
        "in-process-baseline",
        "no-type",
        "no-section",
    ],
)
def test_standin_resolves_to_the_stand_in_block_whatever_the_baseline_is(section: Any):
    assert resolve_target(section, TARGET_STANDIN) == LIVE_STANDIN


# ---------------------------------------------------------------------------
# live — the deployment's own control system, when it has one
# ---------------------------------------------------------------------------


def test_live_on_an_epics_baseline_is_that_baseline():
    assert resolve_target(_section(EPICS), TARGET_LIVE) == EPICS


def test_live_is_protocol_neutral():
    """Nothing here knows which control system a facility runs."""
    assert resolve_target(_section(DOOCS), TARGET_LIVE) == DOOCS


def test_live_passes_an_unknown_baseline_type_through_unjudged():
    """A typo reaches the factory's "Unknown … type" error, as it does today."""
    assert resolve_target(_section("epcis"), TARGET_LIVE) == "epcis"


# ---------------------------------------------------------------------------
# live on a simulated baseline — derived from a configured block, or refused
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "serving", ["served", IN_PROCESS], ids=["va-baseline", "in-process-baseline"]
)
def test_live_on_a_simulated_baseline_is_the_one_configured_live_block(serving: str):
    section = _section(
        VIRTUAL_ACCELERATOR,
        {
            "virtual_accelerator": {"timeout_s": 5.0, "serving": serving},
            "epics": {"gateways": {"read_only": {"address": "gw"}}},
        },
    )

    assert resolve_target(section, TARGET_LIVE) == EPICS


@pytest.mark.parametrize(
    "connector",
    [{}, None, "epics", ...],
    ids=["empty", "none", "not-a-mapping", "absent"],
)
def test_live_on_a_simulated_baseline_refuses_without_a_connector_table(connector: Any):
    with pytest.raises(ValueError):
        resolve_target(_section(None, connector), TARGET_LIVE)


@pytest.mark.parametrize(
    ("connector", "expected_substrings"),
    [
        (
            {"virtual_accelerator": {"timeout_s": 5.0}},
            ["control_system.connector", VIRTUAL_ACCELERATOR, ONE_REAL_MACHINE],
        ),
        (
            {"epics": {"address": "gw"}, "doocs": {"address": "gw"}},
            [DOOCS, EPICS, ONE_REAL_MACHINE],
        ),
    ],
    ids=["no-live-block", "two-live-blocks"],
)
def test_a_live_that_cannot_be_derived_names_the_one_real_machine_limit(
    connector: Any, expected_substrings: list[str]
):
    """Both halves of the refusal are the same topology: a deployment describes
    one real machine, so nought and two are equally underivable. A deployer who
    reads only the error still learns what the product's shape is, and which
    blocks (the missing live one, or the two ambiguous ones) made it so."""
    with pytest.raises(ValueError) as excinfo:
        resolve_target(_section(VIRTUAL_ACCELERATOR, connector), TARGET_LIVE)

    message = str(excinfo.value)
    for expected in expected_substrings:
        assert expected in message


def test_a_second_real_block_beside_a_real_baseline_is_reported(caplog: Any):
    """A facility that writes a second real machine down gets no target for it:
    the baseline is its own live type, so the connector table is never consulted.
    Silence would leave that to be discovered by never being offered the machine."""
    section = _section(EPICS, {"epics": {"address": "ring"}, DOOCS: {"address": "injector"}})

    with caplog.at_level(logging.WARNING, logger="osprey_connectors.types"):
        assert resolve_target(section, TARGET_LIVE) == EPICS

    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert DOOCS in message
    assert ONE_REAL_MACHINE in message


def test_the_stand_in_and_the_simulator_are_not_a_second_real_machine(caplog: Any):
    """The shape every stand-in deployment has, and it is within the limit."""
    section = _section(
        EPICS,
        {
            "epics": {"address": "gw"},
            "virtual_accelerator": {"timeout_s": 5.0},
            "live_standin": {"address": "127.0.0.1"},
        },
    )

    with caplog.at_level(logging.WARNING, logger="osprey_connectors.types"):
        assert resolve_target(section, TARGET_LIVE) == EPICS

    assert caplog.records == []


def test_live_never_falls_back_to_hardware_on_a_bare_config():
    """An empty config resolves to the simulator in process; live has to raise, not guess."""
    for section in ({}, None, _section(), _section(None)):
        assert resolve_control_system_type(section) == VIRTUAL_ACCELERATOR
        with pytest.raises(ValueError):
            resolve_target(section, TARGET_LIVE)


# ---------------------------------------------------------------------------
# live is never the stand-in, and the stand-in never makes live ambiguous
# ---------------------------------------------------------------------------


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
    assert ONE_REAL_MACHINE in message


# ---------------------------------------------------------------------------
# The baseline resolver is untouched
# ---------------------------------------------------------------------------


def test_the_no_argument_resolver_falls_back_to_the_simulator():
    assert resolve_control_system_type(None) == VIRTUAL_ACCELERATOR
    assert resolve_control_system_type({}) == VIRTUAL_ACCELERATOR
    assert resolve_control_system_type({"type": None}) == VIRTUAL_ACCELERATOR
    assert resolve_control_system_type({"type": ""}) == VIRTUAL_ACCELERATOR
    assert resolve_control_system_type("not-a-mapping") == VIRTUAL_ACCELERATOR
    assert resolve_control_system_type({"type": EPICS}) == EPICS
    assert resolve_control_system_type({"type": " epics "}) == " epics "


@pytest.mark.parametrize(
    ("control_system_type", "expected"),
    [
        (VIRTUAL_ACCELERATOR, TARGET_VA),
        (LIVE_STANDIN, TARGET_STANDIN),
        (EPICS, TARGET_LIVE),
        (DOOCS, TARGET_LIVE),
    ],
    ids=["va", "standin", "epics", "doocs"],
)
def test_the_baseline_target_is_the_machine_the_section_selects(
    control_system_type: str, expected: str
):
    assert baseline_target(_section(control_system_type)) == expected


def test_a_deployment_that_named_no_machine_is_on_the_simulator():
    """A section that states no type is the simulator in process, baselined on ``va``."""
    for section in ({}, None, _section(), _section(None), _in_process()):
        assert baseline_target(section) == TARGET_VA


def test_resolving_a_target_does_not_mutate_the_section():
    section = _section(VIRTUAL_ACCELERATOR, {"epics": {"address": "gw"}})
    before = {"type": VIRTUAL_ACCELERATOR, "connector": {"epics": {"address": "gw"}}}

    resolve_target(section, TARGET_LIVE)
    resolve_target(section, TARGET_VA)

    assert section == before
