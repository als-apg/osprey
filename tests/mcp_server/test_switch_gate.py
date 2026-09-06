"""The switch gate: one context-free answer, for every surface that asks.

``target_eligibility.evaluate_switch`` answers "may this deployment move to
that target right now", and it is deliberately context-free: no server context,
no files, no process table — every fact arrives as an argument, gathered by
whichever surface is asking. That is what makes the parity test at the bottom
possible at all, and that test is the point of the extraction: the terminal
route and the owning server's tool assemble their arguments from different
sources, and the operator must be told exactly the same thing either way.

The tool half this file used to open with is gone with the function it pinned.
``control_target.switch_gate`` no longer exists — the gate is not a thing only
the controls server can ask — and the tool's side of the contract, that it
reports the verdict rather than paraphrasing it, is pinned from the operator's
side in ``test_control_target_set.py``.

The refusal ladder is ordered, and the order is asserted rather than implied: a
read-only run refuses whatever else is true; an execution in flight refuses
before a probe row is consulted; eligibility — which is where "already active"
and the FR-8 posture gates live — refuses before reachability, because a target
this config could never reach is not a target whose probe row is interesting.

Reachability is the new rung, and it splits three ways on the same absence of
data. With a live controls server that has not swept yet, the honest answer is a
refusal: something is there to ask and it has not answered. With no live server
at all, nothing was ever going to answer, so the switch is allowed and the
reachability is labelled ``not probed yet`` rather than claimed as good.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from osprey.mcp_server.control_system import target_eligibility as te
from osprey_connectors import control_context

# ===========================================================================
# evaluate_switch: the same gate, context-free, as both callers will ask it
# ===========================================================================

LIVE = "live"
VA = "va"

EPICS_TYPE = "epics"
VA_TYPE = "virtual_accelerator"

#: A session key that is not this process's, so ``busy_client`` names it as
#: another session rather than as this one.
OTHER_SESSION = "0123456789abcdef"


# ---------------------------------------------------------------------------
# The world the gate is asked about
# ---------------------------------------------------------------------------


def _config(*, va_probe_channel: str | None = "VA:PROBE:CHANNEL") -> dict[str, Any]:
    """A rendered config whose ``va`` target is eligible from a ``live`` baseline.

    ``va`` is used as the wanted target throughout because it carries none of
    FR-8's posture gates: what this file is testing is the ladder around
    eligibility, not eligibility itself, which has its own suite.
    """
    va_block: dict[str, Any] = {
        "timeout": 5.0,
        "gateways": {
            "read_only": {"address": "localhost", "port": 5074, "use_name_server": True},
            "write_access": {"address": "localhost", "port": 5074, "use_name_server": True},
        },
    }
    if va_probe_channel is not None:
        va_block["probe_channel"] = va_probe_channel
    return {
        "control_system": {
            "type": EPICS_TYPE,
            "writes_enabled": False,
            "limits_checking": {"enabled": True, "allow_unlisted_channels": False},
            "connector": {
                EPICS_TYPE: {
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
                },
                VA_TYPE: va_block,
            },
            "target_switch": {te.ACK_LEAF: "gw.example.org"},
        },
        "archiver": {"type": "epics_archiver"},
    }


def _stamp(*, seconds_ago: float = 0.0) -> str:
    """A wall-clock stamp the prober would have written, ISO-8601 in UTC."""
    return (datetime.now(UTC) - timedelta(seconds=seconds_ago)).isoformat()


def _reachability(
    *rows: tuple[str, str, str],
    published_at: str | None = None,
) -> dict[str, Any]:
    """A report's ``reachability`` block from ``(target, role, state)`` triples.

    Each row is stamped with its own ``probed_at`` so a test can state which
    observation is the newest without depending on dictionary order.
    """
    targets: dict[str, dict[str, Any]] = {}
    for index, (target, role, state) in enumerate(rows):
        targets.setdefault(target, {})[role] = {
            "state": state,
            "probed_at": _stamp(seconds_ago=float(len(rows) - index)),
            "gateway": "localhost:5074",
            "detail": "",
        }
    return {"published_at": published_at or _stamp(), "targets": targets}


def _report(
    pid: int = 4321,
    *,
    reachability: dict[str, Any] | None = None,
    session: str | None = None,
) -> control_context.ServerReport:
    """One live controls server's report, as the gate receives it."""
    return control_context.ServerReport(
        server_pid=pid,
        session=session,
        applied_target=LIVE,
        applied_generation=1,
        reachability=reachability or {},
    )


def _marker(
    *,
    pid: int = 9001,
    session: str | None = OTHER_SESSION,
    surface: str = te.SURFACE_PYTHON_EXECUTOR,
    kernel_id: str | None = None,
    target: str = LIVE,
) -> dict[str, Any]:
    """One in-flight marker, in the schema the executor and the kernel write."""
    return {
        "pid": pid,
        "session": session,
        "surface": surface,
        "kernel_id": kernel_id,
        "target": target,
        "launch_posture": "sandbox",
        "started_at": _stamp(seconds_ago=30),
    }


def _gate(**overrides: Any) -> te.GateVerdict:
    """The gate asked about a switch from ``live`` to ``va``, on a live-baseline."""
    kwargs: dict[str, Any] = {
        "config": _config(),
        "wanted": VA,
        "current_target": LIVE,
        "baseline": LIVE,
        "in_flight": (),
        "reports": (),
        "writes_enabled": False,
    }
    kwargs.update(overrides)
    config = kwargs.pop("config")
    wanted = kwargs.pop("wanted")
    return te.evaluate_switch(config, wanted, **kwargs)


@pytest.fixture(autouse=True)
def _not_a_readonly_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every case states the execution mode, so none inherits the machine's."""
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)


# ---------------------------------------------------------------------------
# (1) A read-only run
# ---------------------------------------------------------------------------


def test_a_readonly_run_is_refused_whatever_else_is_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first rung: a run that mutates nothing cannot mutate the target."""
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")

    verdict = _gate(reports=(_report(reachability=_reachability((VA, "read_only", "reached"))),))

    assert verdict.allowed is False
    assert verdict.reason == te.REASON_READONLY_RUN
    assert verdict.details["reason"] == te.REASON_READONLY_RUN
    assert verdict.details["target"] == VA


def test_the_readonly_refusal_comes_before_the_in_flight_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Order, not merely membership: both are true and read-only is reported."""
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")

    verdict = _gate(in_flight=(_marker(),))

    assert verdict.reason == te.REASON_READONLY_RUN


# ---------------------------------------------------------------------------
# (2) An execution in flight, on any session and any surface
# ---------------------------------------------------------------------------


def test_a_marker_from_another_session_refuses_and_names_the_client() -> None:
    """The refusal is the marker reader's own words, not a second spelling."""
    marker = _marker()
    verdict = _gate(in_flight=(marker,))

    message, suggestions, details = te.in_flight_detail(marker, VA)
    assert verdict.allowed is False
    assert verdict.reason == te.REASON_EXECUTION_IN_FLIGHT
    assert verdict.detail == message
    assert verdict.suggestions == suggestions
    assert verdict.details == details
    assert f"session {te.session_short_key(OTHER_SESSION)}" in " ".join(verdict.suggestions)


def test_a_notebook_kernels_marker_refuses_the_same_switch() -> None:
    """Markers are not filtered by surface: a cell holds the target too."""
    verdict = _gate(
        in_flight=(
            _marker(surface=te.SURFACE_NOTEBOOK_KERNEL, kernel_id="abcdef01", session="kernel:x"),
        )
    )

    assert verdict.reason == te.REASON_EXECUTION_IN_FLIGHT
    assert verdict.details["surface"] == te.SURFACE_NOTEBOOK_KERNEL
    assert verdict.details["kernel_id"] == "abcdef01"


def test_the_oldest_marker_is_the_one_named() -> None:
    """One refusal, and it names the run the caller listed first."""
    first = _marker(pid=9001, session=OTHER_SESSION)
    second = _marker(pid=9002, session="fedcba9876543210")

    verdict = _gate(in_flight=(first, second))

    assert verdict.details["executor_pid"] == 9001


def test_a_marker_refuses_before_reachability_is_consulted() -> None:
    """A busy client is a better answer than a probe row the operator cannot act on."""
    verdict = _gate(
        in_flight=(_marker(),),
        reports=(_report(reachability=_reachability((VA, "read_only", "down"))),),
    )

    assert verdict.reason == te.REASON_EXECUTION_IN_FLIGHT
    assert verdict.reachability == ""


def test_a_marker_that_is_not_a_mapping_states_nothing() -> None:
    """Residue in the marker list is ignored, never crashed on."""
    verdict = _gate(in_flight=("not a marker", None), reports=())

    assert verdict.allowed is True


# ---------------------------------------------------------------------------
# (3) Eligibility, which is where "already active" and FR-8 live
# ---------------------------------------------------------------------------


def test_the_target_the_session_is_already_on_is_refused() -> None:
    """Switching to where you stand is a no-op, and that is the reason given."""
    verdict = _gate(wanted=LIVE, current_target=LIVE, baseline=LIVE)

    assert verdict.allowed is False
    assert verdict.reason == te.REASON_ALREADY_ACTIVE


def test_an_ineligible_target_is_refused_in_the_eligibility_modules_words() -> None:
    """One reason, one spelling: the roster and the gate say the same sentence."""
    config = _config(va_probe_channel=None)

    verdict = _gate(config=config)

    availability = te.target_availability(
        config, VA, LIVE, LIVE, writes_enabled=False, readonly_run=False
    )
    assert verdict.reason == te.REASON_PROBE_CHANNEL_MISSING
    assert verdict.detail == availability.detail
    assert verdict.details == availability.as_dict()


def test_eligibility_is_answered_before_reachability() -> None:
    """A target that could never work is not reported as merely unreachable."""
    verdict = _gate(
        config=_config(va_probe_channel=None),
        reports=(_report(reachability=_reachability((VA, "read_only", "down"))),),
    )

    assert verdict.reason == te.REASON_PROBE_CHANNEL_MISSING
    assert verdict.reachability == ""


# ---------------------------------------------------------------------------
# (4) Reachability, read off the live fleet's reports
# ---------------------------------------------------------------------------


def test_a_target_the_fleet_measured_as_down_is_refused() -> None:
    """Positive evidence of a gateway that did not answer refuses the switch."""
    verdict = _gate(reports=(_report(reachability=_reachability((VA, "read_only", "down"))),))

    assert verdict.allowed is False
    assert verdict.reason == te.REASON_TARGET_UNREACHABLE
    assert verdict.reachability == te.REACHABILITY_DOWN
    assert verdict.details["reachability"] == te.REACHABILITY_DOWN
    assert "localhost:5074" in verdict.detail


def test_a_target_the_fleet_reached_is_allowed() -> None:
    """The ordinary case: a probe answered and the switch may proceed."""
    verdict = _gate(reports=(_report(reachability=_reachability((VA, "read_only", "reached"))),))

    assert verdict.allowed is True
    assert verdict.reason == ""
    assert verdict.reachability == te.REACHABILITY_REACHED


def test_the_newest_observation_is_the_one_that_decides() -> None:
    """Two servers disagree; the fleet's answer is the one measured last."""
    stale = _report(
        1111,
        reachability={
            "published_at": _stamp(seconds_ago=600),
            "targets": {
                VA: {
                    "read_only": {
                        "state": "down",
                        "probed_at": _stamp(seconds_ago=600),
                        "gateway": "localhost:5074",
                        "detail": "",
                    }
                }
            },
        },
    )
    fresh = _report(
        2222,
        reachability={
            "published_at": _stamp(),
            "targets": {
                VA: {
                    "read_only": {
                        "state": "reached",
                        "probed_at": _stamp(),
                        "gateway": "localhost:5074",
                        "detail": "",
                    }
                }
            },
        },
    )

    assert _gate(reports=(stale, fresh)).allowed is True
    assert _gate(reports=(fresh, stale)).allowed is True


def test_another_targets_down_row_does_not_refuse_this_one() -> None:
    """Reachability is per target; the live machine being down is not va's problem."""
    verdict = _gate(
        reports=(
            _report(
                reachability=_reachability(
                    (LIVE, "read_only", "down"), (VA, "read_only", "reached")
                )
            ),
        )
    )

    assert verdict.allowed is True
    assert verdict.reachability == te.REACHABILITY_REACHED


def test_a_live_server_that_has_not_probed_yet_earns_its_own_refusal() -> None:
    """Something is there to answer and it has not: distinct from unreachable."""
    verdict = _gate(reports=(_report(),))

    assert verdict.allowed is False
    assert verdict.reason == te.REASON_REACHABILITY_UNKNOWN
    assert verdict.reason != te.REASON_TARGET_UNREACHABLE
    assert verdict.reachability == te.REACHABILITY_NOT_PROBED
    assert "4321" in verdict.detail


def test_a_report_with_rows_for_other_targets_only_is_still_unprobed_here() -> None:
    """A sweep that never reached this target has said nothing about it."""
    verdict = _gate(reports=(_report(reachability=_reachability((LIVE, "read_only", "reached"))),))

    assert verdict.reason == te.REASON_REACHABILITY_UNKNOWN


def test_zero_live_servers_allows_the_switch_and_says_it_was_never_probed() -> None:
    """Nothing was going to answer, so the switch proceeds and claims nothing."""
    verdict = _gate(reports=())

    assert verdict.allowed is True
    assert verdict.reason == ""
    assert verdict.reachability == te.REACHABILITY_NOT_PROBED
    assert verdict.details["reachability"] == te.REACHABILITY_NOT_PROBED
    assert te.REACHABILITY_NOT_PROBED in verdict.detail


def test_a_target_whose_only_rows_decline_to_claim_is_allowed() -> None:
    """``not_applicable`` is a decision, not a failure: CA search is UDP."""
    verdict = _gate(
        reports=(_report(reachability=_reachability((VA, "read_only", "not_applicable"))),)
    )

    assert verdict.allowed is True
    assert verdict.reachability == te.REACHABILITY_NOT_APPLICABLE


def test_a_row_whose_state_is_unreadable_claims_nothing() -> None:
    """Fail closed on nonsense: an unreadable row is not evidence of reach."""
    verdict = _gate(reports=(_report(reachability=_reachability((VA, "read_only", "banana"))),))

    assert verdict.allowed is False
    assert verdict.reason == te.REASON_REACHABILITY_UNKNOWN


def test_a_degraded_reachability_block_is_not_a_crash() -> None:
    """Every shape a broken report can take reads as "nothing measured"."""
    for block in ({"targets": "nonsense"}, {"targets": {VA: "nonsense"}}, {"targets": {}}):
        verdict = _gate(reports=(_report(reachability=block),))
        assert verdict.reason == te.REASON_REACHABILITY_UNKNOWN


def test_a_row_with_no_probed_at_still_counts_as_a_measurement() -> None:
    """An undated row is the oldest thing there is, never a row that says nothing."""
    undated = _report(
        reachability={
            "targets": {VA: {"read_only": {"state": "down", "gateway": "localhost:5074"}}}
        }
    )

    assert _gate(reports=(undated,)).reason == te.REASON_TARGET_UNREACHABLE


# ---------------------------------------------------------------------------
# Context-freedom: the gate reads nothing it was not handed
# ---------------------------------------------------------------------------


def test_the_gate_never_reads_the_posture_store(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every fact arrives as an argument; a store read here would be a file read."""
    from osprey_connectors import posture_store

    def _explode(*args: Any, **kwargs: Any) -> bool:
        raise AssertionError("evaluate_switch read the posture store")

    monkeypatch.setattr(posture_store, "effective_writes", _explode)

    assert _gate(writes_enabled=None).allowed is True


def test_the_gate_never_reads_the_control_context_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The record's target arrives as ``current_target``, never off the disk."""

    def _explode(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("evaluate_switch read the control-context record")

    monkeypatch.setattr(control_context, "read_record", _explode)

    assert _gate().allowed is True


def test_the_published_reachability_vocabulary_is_the_probers_own() -> None:
    """Replicated rather than imported (the prober imports this module), so pinned."""
    from osprey.mcp_server.control_system import endpoint_prober

    assert te.REACHABILITY_REACHED == endpoint_prober.STATE_REACHED
    assert te.REACHABILITY_DOWN == endpoint_prober.STATE_DOWN
    assert te.REACHABILITY_NOT_APPLICABLE == endpoint_prober.STATUS_NOT_APPLICABLE


# ---------------------------------------------------------------------------
# The shared parity test: one world, two callers, one verdict
# ---------------------------------------------------------------------------
#
# The two callers do not share a line of code. The terminal route gathers its
# facts from the shared agent-data root in one worker-thread hop; the server
# tool has them in hand already, off the connector-host manager it owns. What
# follows builds ONE world on disk, has each caller shape assemble its own
# arguments from it, and asserts the two verdicts are the same object value —
# not merely the same reason, because the operator reads the detail and the
# suggestions too.


class _Hosts:
    """The connector-host manager, as far as the switch tool's gate call sees it."""

    def __init__(self, target: str, baseline: str) -> None:
        self._target = target
        self.baseline = baseline

    def active_target(self) -> str:
        return self._target


class _Config:
    """``context.config``, whose ``raw`` is the rendered mapping."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self.raw = raw


class _Context:
    """The controls server context the tool resolves before it gates."""

    def __init__(self, config: dict[str, Any], hosts: _Hosts) -> None:
        self.config = _Config(config)
        self.connector_hosts = hosts


def _write_marker(root: Any, marker: dict[str, Any]) -> None:
    """Plant one in-flight marker where both callers' readers will find it."""
    from osprey.mcp_server.control_system import target_state

    path = (
        root
        / control_context.STATE_DIR_NAME
        / f"{target_state.INFLIGHT_FILE_PREFIX}{marker['pid']}{target_state.INFLIGHT_FILE_SUFFIX}"
    )
    path.write_text(json.dumps(marker), encoding="utf-8")


def _route_caller_verdict(config: dict[str, Any], wanted: str, baseline: str) -> te.GateVerdict:
    """The terminal route's shape: everything read out of the shared root.

    This is what ``_target_request_facts`` gathers under ``run_in_threadpool``
    — the record for the deployment's current target, the live reports, the
    markers — and hands to the gate. It holds no connector and knows no
    manager, which is the whole reason the gate may not ask for one.
    """
    from osprey.mcp_server.control_system import target_state

    record = control_context.read_record()
    return te.evaluate_switch(
        config,
        wanted,
        current_target=(record.target if record is not None else baseline),
        baseline=baseline,
        in_flight=target_state.in_flight_executions(),
        reports=control_context.live_reports(),
        writes_enabled=False,
    )


def _tool_caller_verdict(config: dict[str, Any], wanted: str, baseline: str) -> te.GateVerdict:
    """The server tool's shape: the manager it owns answers where it is."""
    from osprey.mcp_server.control_system import target_state

    record = control_context.read_record()
    context = _Context(config, _Hosts(record.target if record is not None else baseline, baseline))
    hosts = context.connector_hosts
    return te.evaluate_switch(
        context.config.raw,
        wanted,
        current_target=hosts.active_target(),
        baseline=hosts.baseline,
        in_flight=target_state.in_flight_executions(),
        reports=control_context.live_reports(),
        writes_enabled=False,
    )


#: ``(name, wanted, plant)`` — one rung of the ladder each, so parity is pinned
#: on the refusals and on the allowed answers alike.
PARITY_WORLDS = [
    "reachable",
    "unreachable",
    "never_probed_with_a_live_server",
    "no_live_server",
    "execution_in_flight",
    "already_active",
]


@pytest.mark.parametrize("world", PARITY_WORLDS)
def test_both_callers_get_identical_verdicts(
    world: str,
    control_context_root: Any,
    write_control_context: Any,
    write_server_report: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SC-80's last item: one world, two caller shapes, one verdict."""
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)
    root = control_context_root
    config = _config()
    wanted = LIVE if world == "already_active" else VA
    write_control_context(root, target=LIVE, generation=3)

    live_pid = os.getpid()
    if world == "reachable":
        write_server_report(
            root, live_pid, reachability=_reachability((VA, "read_only", "reached"))
        )
    elif world == "unreachable":
        write_server_report(root, live_pid, reachability=_reachability((VA, "read_only", "down")))
    elif world == "never_probed_with_a_live_server":
        write_server_report(root, live_pid)
    elif world == "execution_in_flight":
        write_server_report(
            root, live_pid, reachability=_reachability((VA, "read_only", "reached"))
        )
        _write_marker(root, _marker(pid=live_pid))
    elif world == "already_active":
        write_server_report(
            root, live_pid, reachability=_reachability((LIVE, "read_only", "reached"))
        )
    control_context.invalidate_cache()

    route = _route_caller_verdict(config, wanted, LIVE)
    tool = _tool_caller_verdict(config, wanted, LIVE)

    assert route == tool
    assert route.reason == tool.reason
    assert route.detail == tool.detail
    assert route.suggestions == tool.suggestions
    assert route.details == tool.details


@pytest.mark.parametrize(
    ("world", "allowed", "reason"),
    [
        ("reachable", True, ""),
        ("unreachable", False, te.REASON_TARGET_UNREACHABLE),
        ("never_probed_with_a_live_server", False, te.REASON_REACHABILITY_UNKNOWN),
        ("no_live_server", True, ""),
        ("execution_in_flight", False, te.REASON_EXECUTION_IN_FLIGHT),
        ("already_active", False, te.REASON_ALREADY_ACTIVE),
    ],
)
def test_each_parity_world_is_the_verdict_it_claims_to_be(
    world: str,
    allowed: bool,
    reason: str,
    control_context_root: Any,
    write_control_context: Any,
    write_server_report: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The parity test proves agreement; this proves they agree on the right thing."""
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)
    root = control_context_root
    config = _config()
    wanted = LIVE if world == "already_active" else VA
    write_control_context(root, target=LIVE, generation=3)

    live_pid = os.getpid()
    if world == "reachable":
        write_server_report(
            root, live_pid, reachability=_reachability((VA, "read_only", "reached"))
        )
    elif world == "unreachable":
        write_server_report(root, live_pid, reachability=_reachability((VA, "read_only", "down")))
    elif world == "never_probed_with_a_live_server":
        write_server_report(root, live_pid)
    elif world == "execution_in_flight":
        write_server_report(
            root, live_pid, reachability=_reachability((VA, "read_only", "reached"))
        )
        _write_marker(root, _marker(pid=live_pid))
    elif world == "already_active":
        write_server_report(
            root, live_pid, reachability=_reachability((LIVE, "read_only", "reached"))
        )
    control_context.invalidate_cache()

    verdict = _route_caller_verdict(config, wanted, LIVE)

    assert verdict.allowed is allowed
    assert verdict.reason == reason
