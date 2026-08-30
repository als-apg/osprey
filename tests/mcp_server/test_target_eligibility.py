"""ELIGIBILITY and VERIFICATION — the two named checks on a control target.

The matrix at the bottom of this file is the load-bearing test. FR-8's live gate
is a conjunction with an exemption, and a conjunction with an exemption is
exactly the shape of rule that reads correctly and behaves wrongly in one corner.
So the sixteen combinations of {baseline live, va} x {strict, permissive limits}
x {acknowledgment set, unset} x {switching away, returning to baseline} are
enumerated with their expected outcome **written out per cell**, not computed:
a test that re-derives the expectation from the same rule the code applies
agrees with the code by construction and proves nothing about either.

The stand-in is a third target with the same shape of gate and a deliberately
different split, so it gets its own enumerated section rather than a widened
matrix: the strict limits posture applies to a switch toward ``live`` *and*
toward ``standin`` (both behave like hardware), while the operator
acknowledgment applies to ``live`` alone, and ``live`` carries one gate the
other two never do — the archive must not already be the stand-in's.

Every case injects ``readonly_run`` rather than letting it default, so no test
depends on the execution mode of the machine running it. ``writes_enabled`` is
injected too, except in the per-target posture matrix below, where the value the
config resolves to for each target is the thing being tested.
"""

from __future__ import annotations

from typing import Any

import pytest

from osprey.mcp_server.control_system import target_eligibility as te
from osprey_connectors.honesty import VA_MOCK_ARCHIVER_WHY

LIVE = "live"
VA = "va"
STANDIN = "standin"

EPICS_TYPE = "epics"
VA_TYPE = "virtual_accelerator"
STANDIN_TYPE = "live_standin"

#: The port this deployment's stand-in soft IOC serves, as
#: ``services.live_standin.port`` projects it and as the stand-in's own gateways
#: dial it. Deliberately not 5064: a stand-in on the Channel Access default
#: would let a block that simply never set a port pass the deployed check.
STANDIN_PORT = 5094

#: The deployment-wide limits keys, spelled out rather than imported from the
#: module under test: the gate builds them off the resolved posture, so a
#: refusal quoting them is the assertion, not a constant the two share.
DEPLOYMENT_WIDE_ENABLED_KEY = "control_system.limits_checking.enabled"
DEPLOYMENT_WIDE_ALLOW_UNLISTED_KEY = "control_system.limits_checking.allow_unlisted_channels"


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------


def _epics_block(**overrides: Any) -> dict[str, Any]:
    block = {
        "timeout": 5.0,
        "probe_channel": "LIVE:PROBE:CHANNEL",
        "gateways": {
            "read_only": {
                "address": "gw.example.org",
                "port": 5064,
                "use_name_server": False,
            },
            "write_access": {
                "address": "gw.example.org",
                "port": 5084,
                "use_name_server": False,
            },
        },
    }
    block.update(overrides)
    return block


def _va_block(**overrides: Any) -> dict[str, Any]:
    block = {
        "timeout": 5.0,
        "probe_channel": "VA:PROBE:CHANNEL",
        "gateways": {
            "read_only": {"address": "localhost", "port": 5074, "use_name_server": True},
            "write_access": {"address": "localhost", "port": 5074, "use_name_server": True},
        },
    }
    block.update(overrides)
    return block


def _standin_block(**overrides: Any) -> dict[str, Any]:
    """The stand-in's own connector block, dialling the co-deployed soft IOC."""
    block = {
        "timeout": 5.0,
        "probe_channel": "STANDIN:PROBE:CHANNEL",
        "gateways": {
            "read_only": {
                "address": "127.0.0.1",
                "port": STANDIN_PORT,
                "use_name_server": False,
            },
            "write_access": {
                "address": "127.0.0.1",
                "port": STANDIN_PORT,
                "use_name_server": False,
            },
        },
    }
    block.update(overrides)
    return block


def _config(
    *,
    control_system_type: str = EPICS_TYPE,
    connector: dict[str, Any] | None = None,
    limits: str = "strict",
    ack: bool = True,
    archiver_type: str | None = "epics_archiver",
    writes_enabled: bool = False,
    target_switch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A rendered config.yml, in the nested shape the MCP server reads."""
    switch: dict[str, Any] = dict(target_switch or {})
    if ack:
        switch[te.ACK_LEAF] = "gw.example.org"

    limits_block = (
        {"enabled": True, "allow_unlisted_channels": False}
        if limits == "strict"
        else {"enabled": False, "allow_unlisted_channels": True}
    )

    config: dict[str, Any] = {
        "control_system": {
            "type": control_system_type,
            "writes_enabled": writes_enabled,
            "limits_checking": limits_block,
            "connector": (
                connector
                if connector is not None
                else {EPICS_TYPE: _epics_block(), VA_TYPE: _va_block()}
            ),
            "target_switch": switch,
        }
    }
    if archiver_type is not None:
        config["archiver"] = {"type": archiver_type}
    return config


def _standin_config(
    *,
    standin_block: dict[str, Any] | None = None,
    standin_port: int | None = STANDIN_PORT,
    recorder: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """A rendered config for a deployment that co-deploys the stand-in.

    Three top-level facts beyond :func:`_config`: the stand-in's own connector
    block, the ``services.live_standin.port`` the build projects when a profile
    asked for a stand-in, and the ``deployed_services`` list whose
    ``archiver_recorder`` entry is what makes the store this deployment's own.
    """
    connector = kwargs.pop("connector", None)
    if connector is None:
        connector = {
            EPICS_TYPE: _epics_block(),
            VA_TYPE: _va_block(),
            STANDIN_TYPE: _standin_block() if standin_block is None else standin_block,
        }
    config = _config(connector=connector, **kwargs)
    if standin_port is not None:
        config["services"] = {"live_standin": {"port": standin_port}}
    config["deployed_services"] = ["mcp_server"] + (["archiver_recorder"] if recorder else [])
    return config


def _eligibility(config: dict[str, Any], target: str, **kwargs: Any) -> te.Eligibility:
    kwargs.setdefault("writes_enabled", False)
    kwargs.setdefault("readonly_run", False)
    return te.evaluate_eligibility(config, target, **kwargs)


def _derive(config: dict[str, Any], target: str, **kwargs: Any) -> te.TargetDerivation:
    kwargs.setdefault("writes_enabled", False)
    kwargs.setdefault("readonly_run", False)
    return te.derive_endpoints(config, target, **kwargs)


# ---------------------------------------------------------------------------
# ELIGIBILITY — one reason per check, in order
# ---------------------------------------------------------------------------


def test_a_fully_configured_target_is_eligible() -> None:
    verdict = _eligibility(_config(), LIVE)

    assert verdict.eligible is True
    assert verdict.reason is None


def test_an_unknown_target_is_ineligible_rather_than_raising() -> None:
    verdict = _eligibility(_config(), "staging")

    assert verdict.eligible is False
    assert verdict.reason == te.REASON_TARGET_UNRESOLVABLE
    assert "staging" in verdict.detail


def test_live_on_an_all_simulated_deployment_is_ineligible_never_guessed() -> None:
    """No connector block names a real machine, so 'live' has nowhere to land."""
    config = _config(
        control_system_type="mock",
        connector={"mock": {"response_delay_ms": 10}, VA_TYPE: _va_block()},
    )

    verdict = _eligibility(config, LIVE)

    assert verdict.eligible is False
    assert verdict.reason == te.REASON_TARGET_UNRESOLVABLE
    assert "connector" in verdict.detail


def test_a_typo_in_the_type_reports_the_missing_block_it_actually_selects() -> None:
    """A mistyped type is passed through verbatim, so the block it names is absent."""
    config = _config(control_system_type="epcis", connector={EPICS_TYPE: _epics_block()})

    verdict = _eligibility(config, LIVE)

    assert verdict.eligible is False
    assert verdict.reason == te.REASON_CONNECTOR_BLOCK_MISSING
    assert "control_system.connector.epcis" in verdict.detail


def test_an_empty_gateways_table_is_ineligible() -> None:
    config = _config(connector={EPICS_TYPE: _epics_block(gateways={})})

    verdict = _eligibility(config, LIVE)

    assert verdict.eligible is False
    assert verdict.reason == te.REASON_GATEWAYS_MISSING
    assert "control_system.connector.epics.gateways" in verdict.detail


def test_a_gateways_table_without_the_selectable_role_is_ineligible() -> None:
    """Writes off selects read_only; a write-only table therefore has no role."""
    config = _config(
        connector={
            EPICS_TYPE: _epics_block(
                gateways={
                    "write_access": {
                        "address": "gw.example.org",
                        "port": 5084,
                        "use_name_server": False,
                    }
                }
            )
        }
    )

    verdict = _eligibility(config, LIVE, writes_enabled=False)

    assert verdict.eligible is False
    assert verdict.reason == te.REASON_SELECTED_ROLE_MISSING
    assert "'read_only'" in verdict.detail


def test_an_empty_gateway_entry_does_not_count_as_the_role() -> None:
    """``connect()`` skips a falsy gateway entirely, so neither does this."""
    config = _config(
        connector={
            EPICS_TYPE: _epics_block(
                gateways={
                    "read_only": {},
                    "write_access": {"address": "gw.example.org", "port": 5084},
                }
            )
        }
    )

    verdict = _eligibility(config, LIVE)

    assert verdict.eligible is False
    assert verdict.reason == te.REASON_SELECTED_ROLE_MISSING


def test_a_target_without_a_probe_channel_is_ineligible_with_that_reason() -> None:
    block = _epics_block()
    block.pop("probe_channel")
    config = _config(connector={EPICS_TYPE: block})

    verdict = _eligibility(config, LIVE)

    assert verdict.eligible is False
    assert verdict.reason == te.REASON_PROBE_CHANNEL_MISSING
    assert "control_system.connector.epics.probe_channel" in verdict.detail


def test_a_blank_probe_channel_counts_as_unset() -> None:
    config = _config(connector={EPICS_TYPE: _epics_block(probe_channel="   ")})

    assert _eligibility(config, LIVE).reason == te.REASON_PROBE_CHANNEL_MISSING


def test_va_with_a_mock_archiver_is_ineligible_and_says_why_up_front() -> None:
    config = _config(archiver_type=None)

    verdict = _eligibility(config, VA)

    assert verdict.eligible is False
    assert verdict.reason == te.REASON_INVENTED_HISTORY
    assert VA_MOCK_ARCHIVER_WHY in verdict.detail
    assert "mock_archiver" in verdict.detail


def test_va_with_a_real_archiver_is_eligible() -> None:
    assert _eligibility(_config(archiver_type="mongodb_archiver"), VA).eligible is True


def test_probe_channel_is_checked_before_the_honesty_precondition() -> None:
    """Check order is part of the contract: the nearest fix is the one reported."""
    block = _va_block()
    block.pop("probe_channel")
    config = _config(archiver_type=None, connector={EPICS_TYPE: _epics_block(), VA_TYPE: block})

    assert _eligibility(config, VA).reason == te.REASON_PROBE_CHANNEL_MISSING


def test_live_without_the_acknowledgment_is_ineligible() -> None:
    verdict = _eligibility(_config(limits="strict", ack=False), LIVE)

    assert verdict.eligible is False
    assert verdict.reason == te.REASON_OPERATOR_ACK_MISSING
    assert te.ACK_KEY in verdict.detail


def test_live_without_strict_limits_is_ineligible() -> None:
    verdict = _eligibility(_config(limits="permissive", ack=True), LIVE)

    assert verdict.eligible is False
    assert verdict.reason == te.REASON_LIMITS_POSTURE
    assert DEPLOYMENT_WIDE_ENABLED_KEY in verdict.detail
    assert DEPLOYMENT_WIDE_ALLOW_UNLISTED_KEY in verdict.detail


def test_live_failing_both_posture_checks_reports_the_limits_one_first() -> None:
    verdict = _eligibility(_config(limits="permissive", ack=False), LIVE)

    assert verdict.reason == te.REASON_LIMITS_POSTURE


@pytest.mark.parametrize(
    "ack_value",
    ["gw.example.org", "your-gateway.example.com", "anything at all"],
    ids=["operator-hostname", "the-shipped-example-value", "free-text"],
)
def test_the_acknowledgment_is_tested_for_presence_only(ack_value: str) -> None:
    """No value is special. The template ships a real-hostname-shaped example, so
    a string test against a known default would be a distinction the config
    cannot carry."""
    config = _config(ack=False, target_switch={te.ACK_LEAF: ack_value})

    assert _eligibility(config, LIVE).eligible is True


@pytest.mark.parametrize("blank", ["", "   ", None], ids=["empty", "whitespace", "null"])
def test_a_blank_acknowledgment_counts_as_unset(blank: Any) -> None:
    config = _config(ack=False, target_switch={te.ACK_LEAF: blank})

    assert _eligibility(config, LIVE).reason == te.REASON_OPERATOR_ACK_MISSING


# ---------------------------------------------------------------------------
# The limits posture is read per connector type
# ---------------------------------------------------------------------------
#
# A deployment with a live machine beside a virtual accelerator holds one limits
# posture per machine, so the gate must read the posture of the type the target
# actually resolves to and quote the key that answered — the per-type line when a
# per-type block spoke, the deployment-wide one otherwise. Quoting the wrong key
# would send an operator to edit a line the per-type block overrides.


def _limits_block(**leaves: Any) -> dict[str, Any]:
    """A ``limits_checking`` block to hang under one connector block."""
    return {"limits_checking": dict(leaves)}


PER_TYPE_ENABLED_KEY = f"control_system.connector.{EPICS_TYPE}.limits_checking.enabled"
PER_TYPE_ALLOW_UNLISTED_KEY = (
    f"control_system.connector.{EPICS_TYPE}.limits_checking.allow_unlisted_channels"
)


def test_a_strict_per_type_block_makes_live_eligible_on_a_permissive_deployment() -> None:
    """The live machine's own block answers, and the deployment-wide relaxation a
    simulator was given does not reach it."""
    config = _config(
        limits="permissive",
        ack=True,
        connector={
            EPICS_TYPE: _epics_block(**_limits_block(enabled=True, allow_unlisted_channels=False)),
            VA_TYPE: _va_block(),
        },
    )

    verdict = _eligibility(config, LIVE)

    assert verdict.eligible is True
    assert verdict.reason is None


def test_a_permissive_per_type_block_refuses_live_and_names_the_per_type_key() -> None:
    """The per-type block overrides whole, so a strict deployment-wide block does
    not rescue it — and the refusal names the line that actually answered."""
    config = _config(
        limits="strict",
        ack=True,
        connector={
            EPICS_TYPE: _epics_block(**_limits_block(enabled=False, allow_unlisted_channels=True)),
            VA_TYPE: _va_block(),
        },
    )

    verdict = _eligibility(config, LIVE)

    assert verdict.eligible is False
    assert verdict.reason == te.REASON_LIMITS_POSTURE
    assert PER_TYPE_ENABLED_KEY in verdict.detail
    assert PER_TYPE_ALLOW_UNLISTED_KEY in verdict.detail
    assert DEPLOYMENT_WIDE_ENABLED_KEY not in verdict.detail


def test_an_incomplete_per_type_block_is_not_strict() -> None:
    """Half a block answers nothing — not the half it states, and not the
    deployment-wide block it overrides. The refusal names the block an operator
    has to finish."""
    config = _config(
        limits="strict",
        ack=True,
        connector={
            EPICS_TYPE: _epics_block(**_limits_block(enabled=True)),
            VA_TYPE: _va_block(),
        },
    )

    verdict = _eligibility(config, LIVE)

    assert verdict.eligible is False
    assert verdict.reason == te.REASON_LIMITS_POSTURE
    assert PER_TYPE_ENABLED_KEY in verdict.detail
    assert PER_TYPE_ALLOW_UNLISTED_KEY in verdict.detail


def test_the_standin_reads_its_own_types_block_not_the_live_machines() -> None:
    """The stand-in is a connector type of its own, so the gate follows the type
    the target resolves to rather than the deployment's live one."""
    config = _standin_config(
        limits="permissive",
        ack=True,
        connector={
            EPICS_TYPE: _epics_block(),
            VA_TYPE: _va_block(),
            STANDIN_TYPE: _standin_block(
                **_limits_block(enabled=True, allow_unlisted_channels=False)
            ),
        },
    )

    assert _eligibility(config, STANDIN).eligible is True

    live = _eligibility(config, LIVE)
    assert live.eligible is False
    assert live.reason == te.REASON_LIMITS_POSTURE
    assert DEPLOYMENT_WIDE_ENABLED_KEY in live.detail


def test_returning_to_live_needs_neither_posture_nor_acknowledgment() -> None:
    config = _config(limits="permissive", ack=False)

    assert _eligibility(config, LIVE, direction=te.DIRECTION_BACK).eligible is True


def test_the_return_exemption_does_not_excuse_the_configuration_checks() -> None:
    """It waives FR-8's posture, not the target's existence."""
    block = _epics_block()
    block.pop("probe_channel")
    config = _config(limits="permissive", ack=False, connector={EPICS_TYPE: block})

    verdict = _eligibility(config, LIVE, direction=te.DIRECTION_BACK)

    assert verdict.eligible is False
    assert verdict.reason == te.REASON_PROBE_CHANNEL_MISSING


def test_va_is_never_gated_on_posture_or_acknowledgment() -> None:
    config = _config(control_system_type=VA_TYPE, limits="permissive", ack=False)

    assert _eligibility(config, VA, direction=te.DIRECTION_AWAY).eligible is True


# ---------------------------------------------------------------------------
# The stand-in as a third target
# ---------------------------------------------------------------------------


def test_the_standin_target_is_eligible_when_it_selects_the_co_deployed_standin() -> None:
    """Stand-in built, block dialling that port, over loopback: all three."""
    verdict = _eligibility(_standin_config(), STANDIN)

    assert verdict.eligible is True
    assert verdict.reason is None


def test_a_standin_block_pointed_at_a_gateway_is_not_this_deployments_standin() -> None:
    """The port is right and the host is a real gateway — off the loopback
    interface is off this host, so it is a machine, not the container."""
    block = _standin_block(
        gateways={"read_only": {"address": "gw.example.org", "port": STANDIN_PORT}}
    )
    verdict = _eligibility(_standin_config(standin_block=block), STANDIN)

    assert verdict.eligible is False
    assert verdict.reason == te.REASON_STANDIN_NOT_DEPLOYED
    assert "services.live_standin.port" in verdict.detail
    assert "gw.example.org" in verdict.detail


def test_a_standin_block_moved_to_another_port_is_not_this_deployments_standin() -> None:
    """Loopback, but not the port the deployment stood its stand-in up on."""
    block = _standin_block(gateways={"read_only": {"address": "127.0.0.1", "port": 5064}})
    verdict = _eligibility(_standin_config(standin_block=block), STANDIN)

    assert verdict.eligible is False
    assert verdict.reason == te.REASON_STANDIN_NOT_DEPLOYED


def test_without_a_services_block_there_is_no_standin_to_switch_to() -> None:
    """The SSH-tunnel case: a real gateway forwarded to loopback satisfies the
    host and the port and nothing else, because the deployment built no
    stand-in at all."""
    verdict = _eligibility(_standin_config(standin_port=None), STANDIN)

    assert verdict.eligible is False
    assert verdict.reason == te.REASON_STANDIN_NOT_DEPLOYED


def test_the_standin_deployed_check_runs_after_the_probe_channel_check() -> None:
    """Check order is part of the contract: the nearest fix is the one reported."""
    block = _standin_block(gateways={"read_only": {"address": "gw.example.org", "port": 5064}})
    block.pop("probe_channel")

    assert _eligibility(_standin_config(standin_block=block), STANDIN).reason == (
        te.REASON_PROBE_CHANNEL_MISSING
    )


def test_the_standin_deployed_check_runs_before_the_honesty_precondition() -> None:
    """A block that reaches no stand-in is reported as that, not as an archive
    pairing it would only create if it reached one."""
    block = _standin_block(gateways={"read_only": {"address": "gw.example.org", "port": 5064}})
    config = _standin_config(standin_block=block, archiver_type=None)

    assert _eligibility(config, STANDIN).reason == te.REASON_STANDIN_NOT_DEPLOYED


def test_the_standin_beside_a_mock_archiver_invents_history_too() -> None:
    """The stand-in has no more recorded past than the virtual accelerator has."""
    verdict = _eligibility(_standin_config(archiver_type=None), STANDIN)

    assert verdict.eligible is False
    assert verdict.reason == te.REASON_INVENTED_HISTORY
    assert VA_MOCK_ARCHIVER_WHY in verdict.detail
    assert "mock_archiver" in verdict.detail


def test_switching_to_the_standin_requires_the_strict_limits_posture() -> None:
    """It is dialled, it refuses out-of-limit writes and it carries
    ``real_machine`` — a rehearsal on a permissive posture rehearses nothing."""
    verdict = _eligibility(_standin_config(limits="permissive", ack=True), STANDIN)

    assert verdict.eligible is False
    assert verdict.reason == te.REASON_LIMITS_POSTURE
    assert DEPLOYMENT_WIDE_ENABLED_KEY in verdict.detail
    assert DEPLOYMENT_WIDE_ALLOW_UNLISTED_KEY in verdict.detail


def test_switching_to_the_standin_needs_no_operator_acknowledgment() -> None:
    """``virtual_accelerator.live_standin`` is the stand-in's acknowledgment: the
    operator already said which machine this is, at build time."""
    verdict = _eligibility(_standin_config(limits="strict", ack=False), STANDIN)

    assert verdict.eligible is True
    assert verdict.reason is None


def test_live_is_refused_while_the_archive_belongs_to_the_standin() -> None:
    """Recording your own store beside a stand-in records the stand-in, and a
    real machine's readings must not land in that store."""
    config = _standin_config(control_system_type=STANDIN_TYPE, recorder=True, ack=True)

    verdict = _eligibility(config, LIVE)

    assert verdict.eligible is False
    assert verdict.reason == te.REASON_ARCHIVE_BELONGS_TO_STANDIN
    assert "archiver_recorder" in verdict.detail
    assert "virtual_accelerator.live_standin" in verdict.detail


def test_live_beside_a_standin_nobody_records_is_eligible() -> None:
    """No recorder, so this deployment writes no store, so no past is its own."""
    config = _standin_config(control_system_type=STANDIN_TYPE, recorder=False, ack=True)

    assert _eligibility(config, LIVE).eligible is True


def test_a_persona_session_is_refused_live_on_the_projected_recorder_fact() -> None:
    """The multi-user half of the same gate, and it has to answer identically.

    A web-terminal persona is an attached render: it lists no services, so the
    host's `deployed_services` spelling of "we record" never reaches it. The
    build projects the recorder's own block instead (the `archiver_recorder`
    Reach Contract), and the persona reads the very store the fact is about —
    so a session behind a persona is refused the live machine exactly as a
    single-user session on the same deployment is.
    """
    config = _standin_config(control_system_type=STANDIN_TYPE, recorder=False, ack=True)
    config["deployed_services"] = []
    config["services"]["archiver_recorder"] = {"path": "./services/archiver_recorder"}

    verdict = _eligibility(config, LIVE)

    assert verdict.eligible is False
    assert verdict.reason == te.REASON_ARCHIVE_BELONGS_TO_STANDIN
    assert "archiver_recorder" in verdict.detail
    # ... and the same render with nothing projected keeps `live` open, so it is
    # the fact that decides and not the persona shape.
    del config["services"]["archiver_recorder"]
    assert _eligibility(config, LIVE).eligible is True


def test_the_missing_acknowledgment_is_reported_before_the_archive_ownership() -> None:
    """Both are true; the operator is told the one that is theirs to answer."""
    config = _standin_config(control_system_type=STANDIN_TYPE, recorder=True, ack=False)

    assert _eligibility(config, LIVE).reason == te.REASON_OPERATOR_ACK_MISSING


def test_the_archive_ownership_gate_is_the_live_machines_alone() -> None:
    """The store holding the stand-in's history is no reason to refuse the
    stand-in — it is exactly the machine that history belongs to."""
    config = _standin_config(control_system_type=STANDIN_TYPE, recorder=True, ack=True)

    assert _eligibility(config, STANDIN, direction=te.DIRECTION_AWAY).eligible is True


def test_va_is_ungated_on_a_standin_baseline() -> None:
    config = _standin_config(
        control_system_type=STANDIN_TYPE, recorder=True, limits="permissive", ack=False
    )

    assert _eligibility(config, VA).eligible is True


def test_returning_to_a_standin_baseline_is_exempt_from_the_limits_posture() -> None:
    """The baseline a deployment comes home to may be the stand-in, and coming
    home is never gated: a session stranded on the simulator is the worse
    outcome of the two this gate can produce."""
    config = _standin_config(control_system_type=STANDIN_TYPE, limits="permissive", ack=False)

    assert _eligibility(config, STANDIN, direction=te.DIRECTION_BACK).eligible is True
    assert _eligibility(config, STANDIN, direction=te.DIRECTION_AWAY).reason == (
        te.REASON_LIMITS_POSTURE
    )


def test_the_return_exemption_does_not_excuse_the_standins_deployed_check() -> None:
    """It waives FR-8's posture, not the evidence that the target is the
    stand-in — a block repointed at hardware is refused coming home too."""
    block = _standin_block(gateways={"read_only": {"address": "gw.example.org", "port": 5064}})
    config = _standin_config(
        control_system_type=STANDIN_TYPE, standin_block=block, limits="permissive"
    )

    verdict = _eligibility(config, STANDIN, direction=te.DIRECTION_BACK)

    assert verdict.eligible is False
    assert verdict.reason == te.REASON_STANDIN_NOT_DEPLOYED


# The roster on a live_standin baseline, written out per row rather than
# computed: session on the stand-in it was built for, and the two other machines
# judged from there. The ``live`` row is the one that moves, and it moves for a
# different reason in each cell.
STANDIN_BASELINE_ROSTER = [
    # ack, recorder, live_available, live_reason
    (False, False, False, te.REASON_OPERATOR_ACK_MISSING),
    (False, True, False, te.REASON_OPERATOR_ACK_MISSING),
    (True, False, True, None),
    (True, True, False, te.REASON_ARCHIVE_BELONGS_TO_STANDIN),
]


@pytest.mark.parametrize(
    ("ack", "recorder", "live_available", "live_reason"),
    STANDIN_BASELINE_ROSTER,
    ids=[
        f"{'ack' if cell[0] else 'noack'}-{'recorder' if cell[1] else 'norecorder'}"
        for cell in STANDIN_BASELINE_ROSTER
    ],
)
def test_the_roster_on_a_standin_baseline(
    ack: bool, recorder: bool, live_available: bool, live_reason: str | None
) -> None:
    config = _standin_config(control_system_type=STANDIN_TYPE, ack=ack, recorder=recorder)

    rows = {
        target: te.target_availability(
            config,
            target,
            session_target=STANDIN,
            baseline_target=STANDIN,
            writes_enabled=False,
            readonly_run=False,
        )
        for target in (LIVE, VA, STANDIN)
    }

    # The stand-in is the session's own target and the deployment's baseline.
    assert rows[STANDIN].eligible is True
    assert rows[STANDIN].reason == te.REASON_ALREADY_ACTIVE
    assert rows[VA].available_now is True
    assert (rows[LIVE].available_now, rows[LIVE].reason) == (live_available, live_reason)


def test_the_roster_reports_a_repointed_standin_block_as_not_deployed() -> None:
    """The gateway a persona or an operator moved is the roster's answer, not a
    silently absent row."""
    block = _standin_block(
        gateways={"read_only": {"address": "gw.example.org", "port": STANDIN_PORT}}
    )
    config = _standin_config(control_system_type=STANDIN_TYPE, standin_block=block)

    row = te.target_availability(
        config,
        STANDIN,
        session_target=VA,
        baseline_target=STANDIN,
        writes_enabled=False,
        readonly_run=False,
    )

    assert row.available_now is False
    assert row.reason == te.REASON_STANDIN_NOT_DEPLOYED
    assert row.eligible_from_baseline is False


# ---------------------------------------------------------------------------
# (a) Endpoint derivation
# ---------------------------------------------------------------------------


def test_addr_list_mode_is_derived_from_use_name_server_false() -> None:
    derivation = _derive(_config(), LIVE)

    assert derivation.connector_type == EPICS_TYPE
    assert derivation.endpoints["read_only"] == te.Endpoint("gw.example.org", 5064, "addr_list")


def test_name_server_mode_is_derived_from_use_name_server_true() -> None:
    derivation = _derive(_config(), VA)

    assert derivation.connector_type == VA_TYPE
    assert derivation.endpoints["read_only"] == te.Endpoint("localhost", 5074, "name_server")


def test_read_only_and_write_access_carry_their_own_differing_ports() -> None:
    derivation = _derive(_config(), LIVE)

    assert derivation.endpoints["read_only"].port == 5064
    assert derivation.endpoints["write_access"].port == 5084


def test_an_unset_epics_port_falls_back_to_the_ca_default() -> None:
    config = _config(
        connector={
            EPICS_TYPE: _epics_block(
                gateways={"read_only": {"address": "gw.example.org", "use_name_server": False}}
            )
        }
    )

    assert _derive(config, LIVE).endpoints["read_only"].port == te.DEFAULT_CA_PORT


def test_unset_va_ports_follow_the_deployed_va_service_port(monkeypatch) -> None:
    """The VA is a service this project deploys, so its gateways follow it."""
    from osprey_connectors import config as config_module

    def fake_get_config_value(path: str, default: Any = None, config_path: str | None = None):
        assert path == "services.virtual_accelerator.port", path
        return 5077

    monkeypatch.setattr(config_module, "get_config_value", fake_get_config_value)

    config = _config(
        connector={
            VA_TYPE: _va_block(
                gateways={
                    "read_only": {"address": "localhost", "use_name_server": True},
                    "write_access": {"address": "localhost", "use_name_server": True},
                }
            )
        }
    )

    derivation = _derive(config, VA)

    assert derivation.endpoints["read_only"].port == 5077
    assert derivation.endpoints["write_access"].port == 5077


def test_an_explicit_va_port_wins_over_the_service_port(monkeypatch) -> None:
    from osprey_connectors import config as config_module

    def unexpected_read(*args: Any, **kwargs: Any):
        raise AssertionError("a fully explicit VA config must not read the config file")

    monkeypatch.setattr(config_module, "get_config_value", unexpected_read)

    assert _derive(_config(), VA).endpoints["read_only"].port == 5074


def test_a_pva_row_appears_only_with_both_globs_and_a_gateway() -> None:
    config = _config(
        connector={
            EPICS_TYPE: _epics_block(
                pva_channels=["*:IMAGE*"],
                pva_gateway={"address": "pva.example.org", "use_name_server": True},
            )
        }
    )

    derivation = _derive(config, LIVE)

    assert derivation.endpoints["pva"] == te.Endpoint(
        "pva.example.org", te.DEFAULT_PVA_PORT, "name_server"
    )


def test_pva_globs_without_a_gateway_derive_no_pva_row() -> None:
    """``connect()`` touches no PVA environment variable without a gateway."""
    config = _config(connector={EPICS_TYPE: _epics_block(pva_channels=["*:IMAGE*"])})

    assert "pva" not in _derive(config, LIVE).endpoints


def test_a_pva_gateway_without_globs_derives_no_pva_row() -> None:
    config = _config(
        connector={EPICS_TYPE: _epics_block(pva_gateway={"address": "pva.example.org"})}
    )

    assert "pva" not in _derive(config, LIVE).endpoints


def test_a_plain_epics_block_derives_no_pva_row() -> None:
    assert "pva" not in _derive(_config(), LIVE).endpoints


def test_a_missing_block_derives_empty_endpoints_rather_than_raising() -> None:
    config = _config(connector={EPICS_TYPE: _epics_block()})

    derivation = _derive(config, VA)

    assert derivation.endpoints == {}
    assert derivation.selected_role == "read_only"
    assert derivation.selected_endpoint() is None


# ---------------------------------------------------------------------------
# selected_role — the same inputs connect() uses
# ---------------------------------------------------------------------------


def test_writes_enabled_selects_the_write_access_gateway() -> None:
    assert _derive(_config(), LIVE, writes_enabled=True).selected_role == "write_access"


def test_writes_disabled_stays_on_the_read_only_gateway() -> None:
    assert _derive(_config(), LIVE, writes_enabled=False).selected_role == "read_only"


def test_a_readonly_run_stays_on_the_read_only_gateway_even_with_writes_enabled() -> None:
    derivation = _derive(_config(), LIVE, writes_enabled=True, readonly_run=True)

    assert derivation.selected_role == "read_only"


def test_writes_enabled_without_a_write_gateway_falls_back_to_read_only() -> None:
    config = _config(
        connector={
            EPICS_TYPE: _epics_block(
                gateways={
                    "read_only": {
                        "address": "gw.example.org",
                        "port": 5064,
                        "use_name_server": False,
                    }
                }
            )
        }
    )

    assert _derive(config, LIVE, writes_enabled=True).selected_role == "read_only"


def test_a_readonly_run_moves_the_selected_role_not_the_eligibility() -> None:
    config = _config()

    readwrite = _eligibility(config, LIVE, writes_enabled=True, readonly_run=False)
    readonly = _eligibility(config, LIVE, writes_enabled=True, readonly_run=True)

    assert readwrite.eligible is True
    assert readonly.eligible is True
    assert _derive(config, LIVE, writes_enabled=True, readonly_run=False).selected_role != (
        _derive(config, LIVE, writes_enabled=True, readonly_run=True).selected_role
    )


def test_writes_enabled_defaults_to_the_configs_own_posture() -> None:
    config = _config(writes_enabled=True)

    assert te.derive_endpoints(config, LIVE, readonly_run=False).selected_role == "write_access"


def test_readonly_run_defaults_to_the_processs_execution_mode(monkeypatch) -> None:
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
    config = _config(writes_enabled=True)

    assert te.derive_endpoints(config, LIVE).selected_role == "read_only"


# ---------------------------------------------------------------------------
# Write posture, per target
# ---------------------------------------------------------------------------

#: A block carrying no ``writes_enabled`` leaf at all, which is the tri-state's
#: "inherit the deployment-wide key" and is not any value the leaf could hold.
ABSENT = object()


def _posture_config(*, deployment: bool, live: Any = ABSENT, va: Any = ABSENT) -> dict[str, Any]:
    """A two-target config where each block states its own posture, or doesn't."""
    epics = _epics_block()
    va_block = _va_block()
    if live is not ABSENT:
        epics["writes_enabled"] = live
    if va is not ABSENT:
        va_block["writes_enabled"] = va
    return _config(
        writes_enabled=deployment,
        connector={EPICS_TYPE: epics, VA_TYPE: va_block},
    )


# {deployment-wide posture} x {each block's own posture} x {readonly run}, with
# the role EACH target selects written out per cell. Neither target's expectation
# is derived from the other's: the whole point of the per-type key is that one
# deployment can arm one machine and not the other, and a matrix that computed
# the second answer from the first could not tell the two apart.
#
# These cases deliberately do not inject ``writes_enabled`` — the default seam,
# which is what resolves the target's posture, is what they are about.
POSTURE_MATRIX = {
    # deployment-wide, live leaf, va leaf, readonly run, live role, va role
    "arm-the-simulator-alone": (False, ABSENT, True, False, "read_only", "write_access"),
    "disarm-the-real-machine-alone": (True, False, ABSENT, False, "read_only", "write_access"),
    "arm-the-real-machine-alone": (False, True, ABSENT, False, "write_access", "read_only"),
    "silent-blocks-inherit-a-disarmed-deployment": (
        False,
        ABSENT,
        ABSENT,
        False,
        "read_only",
        "read_only",
    ),
    "silent-blocks-inherit-an-armed-deployment": (
        True,
        ABSENT,
        ABSENT,
        False,
        "write_access",
        "write_access",
    ),
    "both-blocks-disarm-an-armed-deployment": (True, False, False, False, "read_only", "read_only"),
    "a-quoted-true-arms-nothing": (False, ABSENT, "true", False, "read_only", "read_only"),
    "a-readonly-run-collapses-an-armed-simulator": (
        False,
        ABSENT,
        True,
        True,
        "read_only",
        "read_only",
    ),
    "a-readonly-run-collapses-an-armed-deployment": (
        True,
        ABSENT,
        ABSENT,
        True,
        "read_only",
        "read_only",
    ),
}


@pytest.mark.parametrize(
    ("deployment", "live_leaf", "va_leaf", "readonly", "live_role", "va_role"),
    POSTURE_MATRIX.values(),
    ids=list(POSTURE_MATRIX),
)
def test_each_target_selects_the_gateway_its_own_posture_arms(
    deployment: bool,
    live_leaf: Any,
    va_leaf: Any,
    readonly: bool,
    live_role: str,
    va_role: str,
) -> None:
    config = _posture_config(deployment=deployment, live=live_leaf, va=va_leaf)

    live = te.derive_endpoints(config, LIVE, readonly_run=readonly)
    va = te.derive_endpoints(config, VA, readonly_run=readonly)

    assert live.selected_role == live_role
    assert va.selected_role == va_role


def test_posture_decides_which_gateway_a_target_is_required_to_have() -> None:
    """Check 3 is asked per target: a block with only a write gateway is eligible
    exactly when its own posture arms the write role it has."""
    config = _config(
        writes_enabled=False,
        connector={
            EPICS_TYPE: _epics_block(
                writes_enabled=True,
                gateways={"write_access": {"address": "gw.example.org", "port": 5084}},
            ),
            VA_TYPE: _va_block(
                gateways={"write_access": {"address": "localhost", "port": 5074}},
            ),
        },
    )

    live = te.evaluate_eligibility(config, LIVE, readonly_run=False)
    va = te.evaluate_eligibility(config, VA, readonly_run=False)

    assert live.eligible is True
    assert va.eligible is False
    assert va.reason == te.REASON_SELECTED_ROLE_MISSING


def test_the_roster_reads_each_targets_own_posture() -> None:
    config = _config(
        writes_enabled=False,
        connector={
            EPICS_TYPE: _epics_block(
                writes_enabled=True,
                gateways={"write_access": {"address": "gw.example.org", "port": 5084}},
            ),
            VA_TYPE: _va_block(
                gateways={"write_access": {"address": "localhost", "port": 5074}},
            ),
        },
    )

    live = te.target_availability(
        config, LIVE, session_target=VA, baseline_target=LIVE, readonly_run=False
    )
    va = te.target_availability(
        config, VA, session_target=VA, baseline_target=LIVE, readonly_run=False
    )

    assert live.available_now is True
    assert va.eligible is False
    assert va.reason == te.REASON_ALREADY_ACTIVE
    assert va.eligible_from_baseline is False


# ---------------------------------------------------------------------------
# (c) VERIFICATION
# ---------------------------------------------------------------------------


def _report(**overrides: Any) -> dict[str, Any]:
    report = {
        "selected_role": "read_only",
        "mode": "addr_list",
        "host": "gw.example.org",
        "port": 5064,
        "_epics_configured": True,
    }
    report.update(overrides)
    return report


def test_a_matching_child_report_verifies() -> None:
    result = te.verify_child_report(_derive(_config(), LIVE), _report())

    assert result.ok is True
    assert result.field is None


def test_a_matching_child_report_verifies_as_a_tuple_too() -> None:
    report = ("read_only", "addr_list", "gw.example.org", 5064, True)

    assert te.verify_child_report(_derive(_config(), LIVE), report).ok is True


def test_a_port_reported_as_text_still_verifies() -> None:
    """``connect()`` interpolates the port into a string environment value."""
    result = te.verify_child_report(_derive(_config(), LIVE), _report(port="5064"))

    assert result.ok is True


@pytest.mark.parametrize(
    ("field", "wrong_value", "expected"),
    [
        ("selected_role", "write_access", "read_only"),
        ("mode", "name_server", "addr_list"),
        ("host", "other-gateway.example.org", "gw.example.org"),
        ("port", 5084, 5064),
    ],
)
def test_each_mismatched_field_fails_and_names_itself(
    field: str, wrong_value: Any, expected: Any
) -> None:
    result = te.verify_child_report(_derive(_config(), LIVE), _report(**{field: wrong_value}))

    assert result.ok is False
    assert result.field == field
    assert result.expected == expected
    assert result.got == wrong_value


def test_a_child_that_configured_no_epics_gateway_fails_verification() -> None:
    result = te.verify_child_report(_derive(_config(), LIVE), _report(_epics_configured=False))

    assert result.ok is False
    assert result.field == "_epics_configured"
    assert result.expected is True


def test_verification_fails_when_the_derivation_has_no_endpoint_for_the_role() -> None:
    config = _config(connector={EPICS_TYPE: _epics_block()})
    derivation = _derive(config, VA)

    result = te.verify_child_report(derivation, _report(host="localhost"))

    assert result.ok is False
    assert result.field == "endpoints"


def test_a_report_of_the_wrong_shape_is_a_protocol_error() -> None:
    with pytest.raises(ValueError, match="fields"):
        te.verify_child_report(_derive(_config(), LIVE), ("read_only", "addr_list"))


# ---------------------------------------------------------------------------
# Session-relative availability
# ---------------------------------------------------------------------------


def test_the_active_target_reports_already_active() -> None:
    availability = te.target_availability(
        _config(),
        LIVE,
        session_target=LIVE,
        baseline_target=LIVE,
        writes_enabled=False,
        readonly_run=False,
    )

    assert availability.available_now is False
    assert availability.reason == te.REASON_ALREADY_ACTIVE
    # The configuration picture is not shadowed by the no-op.
    assert availability.eligible is True
    assert availability.eligible_from_baseline is True


def test_already_active_does_not_hide_that_the_target_is_misconfigured() -> None:
    block = _epics_block()
    block.pop("probe_channel")
    config = _config(connector={EPICS_TYPE: block})

    availability = te.target_availability(
        config,
        LIVE,
        session_target=LIVE,
        baseline_target=LIVE,
        writes_enabled=False,
        readonly_run=False,
    )

    assert availability.available_now is False
    assert availability.reason == te.REASON_ALREADY_ACTIVE
    assert availability.eligible is False
    assert availability.eligible_from_baseline is False


def test_eligible_from_baseline_is_the_static_view_of_a_live_baseline() -> None:
    """From baseline there is no return in progress, so live is judged as a
    switch toward the live machine — which is what makes the pair of answers
    informative rather than redundant."""
    config = _config(limits="permissive", ack=False)

    availability = te.target_availability(
        config,
        LIVE,
        session_target=VA,
        baseline_target=LIVE,
        writes_enabled=False,
        readonly_run=False,
    )

    assert availability.available_now is True
    assert availability.eligible_from_baseline is False


@pytest.mark.parametrize(
    ("target", "session_target", "baseline_target", "expected"),
    [
        (LIVE, VA, LIVE, te.DIRECTION_BACK),
        (VA, LIVE, VA, te.DIRECTION_BACK),
        (LIVE, LIVE, LIVE, te.DIRECTION_AWAY),
        (LIVE, VA, VA, te.DIRECTION_AWAY),
        (VA, LIVE, LIVE, te.DIRECTION_AWAY),
        (VA, VA, VA, te.DIRECTION_AWAY),
        # A live_standin baseline is a baseline like any other: coming home to
        # it from either of the other two machines is a return.
        (STANDIN, LIVE, STANDIN, te.DIRECTION_BACK),
        (STANDIN, VA, STANDIN, te.DIRECTION_BACK),
        (STANDIN, STANDIN, STANDIN, te.DIRECTION_AWAY),
        (STANDIN, VA, LIVE, te.DIRECTION_AWAY),
        (LIVE, STANDIN, STANDIN, te.DIRECTION_AWAY),
    ],
)
def test_switch_direction(
    target: str, session_target: str, baseline_target: str, expected: str
) -> None:
    assert te.switch_direction(target, session_target, baseline_target) == expected


# ---------------------------------------------------------------------------
# The 16-cell agreement matrix
# ---------------------------------------------------------------------------

# {baseline} x {limits posture} x {acknowledgment} x {direction}, with the
# expected roster answer for BOTH prospective targets written out per cell.
#
# The session sits on the baseline in an "away" cell (so a switch to the other
# target is a switch away) and on the other target in a "back" cell (so a switch
# to the baseline is a return). In every cell one of the two targets is
# therefore the session's own, and reports the no-op.
MATRIX = [
    # baseline, limits, ack, direction, live_available, live_reason, va_available, va_reason
    ("live", "strict", True, "away", False, te.REASON_ALREADY_ACTIVE, True, None),
    ("live", "strict", False, "away", False, te.REASON_ALREADY_ACTIVE, True, None),
    ("live", "permissive", True, "away", False, te.REASON_ALREADY_ACTIVE, True, None),
    ("live", "permissive", False, "away", False, te.REASON_ALREADY_ACTIVE, True, None),
    ("live", "strict", True, "back", True, None, False, te.REASON_ALREADY_ACTIVE),
    ("live", "strict", False, "back", True, None, False, te.REASON_ALREADY_ACTIVE),
    ("live", "permissive", True, "back", True, None, False, te.REASON_ALREADY_ACTIVE),
    ("live", "permissive", False, "back", True, None, False, te.REASON_ALREADY_ACTIVE),
    ("va", "strict", True, "away", True, None, False, te.REASON_ALREADY_ACTIVE),
    (
        "va",
        "strict",
        False,
        "away",
        False,
        te.REASON_OPERATOR_ACK_MISSING,
        False,
        te.REASON_ALREADY_ACTIVE,
    ),
    (
        "va",
        "permissive",
        True,
        "away",
        False,
        te.REASON_LIMITS_POSTURE,
        False,
        te.REASON_ALREADY_ACTIVE,
    ),
    (
        "va",
        "permissive",
        False,
        "away",
        False,
        te.REASON_LIMITS_POSTURE,
        False,
        te.REASON_ALREADY_ACTIVE,
    ),
    ("va", "strict", True, "back", False, te.REASON_ALREADY_ACTIVE, True, None),
    ("va", "strict", False, "back", False, te.REASON_ALREADY_ACTIVE, True, None),
    ("va", "permissive", True, "back", False, te.REASON_ALREADY_ACTIVE, True, None),
    ("va", "permissive", False, "back", False, te.REASON_ALREADY_ACTIVE, True, None),
]


@pytest.mark.parametrize(
    (
        "baseline",
        "limits",
        "ack",
        "direction",
        "live_available",
        "live_reason",
        "va_available",
        "va_reason",
    ),
    MATRIX,
    ids=[f"{cell[0]}-{cell[1]}-{'ack' if cell[2] else 'noack'}-{cell[3]}" for cell in MATRIX],
)
def test_availability_matrix(
    baseline: str,
    limits: str,
    ack: bool,
    direction: str,
    live_available: bool,
    live_reason: str | None,
    va_available: bool,
    va_reason: str | None,
) -> None:
    baseline_target = LIVE if baseline == "live" else VA
    other = VA if baseline_target == LIVE else LIVE
    session_target = baseline_target if direction == "away" else other

    config = _config(
        control_system_type=EPICS_TYPE if baseline_target == LIVE else VA_TYPE,
        limits=limits,
        ack=ack,
    )

    live = te.target_availability(
        config,
        LIVE,
        session_target=session_target,
        baseline_target=baseline_target,
        writes_enabled=False,
        readonly_run=False,
    )
    va = te.target_availability(
        config,
        VA,
        session_target=session_target,
        baseline_target=baseline_target,
        writes_enabled=False,
        readonly_run=False,
    )

    assert (live.available_now, live.reason) == (live_available, live_reason)
    assert (va.available_now, va.reason) == (va_available, va_reason)
