"""The ``control_target`` roster: correct before anything has been switched.

The roster's whole value is that an operator can ask "where am I, and where
could I go" without that question doing anything. So the properties pinned here
are mostly negative — nothing spawned, nothing written, nothing connected to —
plus the one positive property that makes it usable on a fresh session: a
target nobody has ever activated is judged from configuration alone, and says
so with the same reason string the switch would refuse with.

Reachability is the part config cannot answer. A row therefore carries
``endpoint_tcp`` only where the background prober measured one, and the
distinction between "not measured", "measured and fine", and "measured too long
ago to still stand" is pinned explicitly: a roster that spelled those the same
way would be misreporting the only thing it is for.
"""

from __future__ import annotations

import json

import pytest

from osprey.mcp_server.control_system import target_state
from osprey.mcp_server.control_system.connector_host_manager import (
    _display_name,
    target_display_metadata,
)
from osprey.mcp_server.control_system.target_eligibility import (
    ACK_LEAF,
    REASON_ALREADY_ACTIVE,
    REASON_ARCHIVE_BELONGS_TO_STANDIN,
    REASON_OPERATOR_ACK_MISSING,
    REASON_PROBE_CHANNEL_MISSING,
    REASON_STANDIN_NOT_DEPLOYED,
    REASON_TARGET_UNRESOLVABLE,
    target_availability,
)
from osprey.mcp_server.control_system.tools import control_target
from osprey_connectors.standin import ARCHIVER_RECORDER_SERVICE
from tests.mcp_server import test_switch_lifecycle as switch_suite
from tests.mcp_server.conftest import assert_raises_error, extract_response_dict, get_tool_fn
from tests.mcp_server.test_control_target_set import config_with_gateways, install_context

LIVE_PROBE = switch_suite.LIVE_PROBE
VA_PROBE = switch_suite.VA_PROBE
raw_config = switch_suite.raw_config
started_on = switch_suite.started_on

# Fixtures shared with the switch-lifecycle suite; see the note in
# test_control_target_set.py for why they are rebound rather than imported.
child_environment = switch_suite.child_environment
fixture_dir = switch_suite.fixture_dir
make_manager = switch_suite.make_manager
state_root = switch_suite.state_root

ROSTER = get_tool_fn(control_target.control_target)


@pytest.fixture
def no_prober(monkeypatch):
    """A deployment whose endpoint prober never started."""
    from osprey.mcp_server.control_system import server as server_mod

    monkeypatch.setattr(server_mod, "_prober", None)


class StubProber:
    """A prober with a fixed snapshot, and no loop behind it."""

    probe_interval_s = 30.0
    staleness_threshold_s = 90.0

    def __init__(self, snapshot):
        self._snapshot = snapshot
        self.snapshot_calls = 0

    def snapshot(self):
        self.snapshot_calls += 1
        return self._snapshot


def install_prober(monkeypatch, snapshot):
    from osprey.mcp_server.control_system import server as server_mod

    prober = StubProber(snapshot)
    monkeypatch.setattr(server_mod, "_prober", prober)
    return prober


STANDIN_PORT = 5074
#: The port a tunnel forwards a *real* gateway to. Loopback, and nothing else.
TUNNELLED_PORT = 5064
#: The simulator's own gateway port, for the cases that need ``va`` to be a
#: usable destination rather than merely a described one. Stated explicitly so
#: no derivation ever reaches ``services.virtual_accelerator.port`` and reads
#: the config file of whatever project the test happens to run in.
VA_GATEWAY_PORT = 5065
REAL_GATEWAY_HOST = "gw.example.org"
STANDIN_PROBE = "STANDIN:BEAM:CURRENT"

#: A real archiver, so the honesty rule has no invented past to object to. The
#: stand-in's connector type is one of the two whose history is invented, so a
#: fixture that said nothing about its archiver would refuse ``standin`` for a
#: reason no case here is about.
REAL_ARCHIVER = "mongodb_archiver"


def standin_config(
    *,
    standin_port=STANDIN_PORT,
    gateway_host="localhost",
    gateway_port=STANDIN_PORT,
    live_host=REAL_GATEWAY_HOST,
    live_port=TUNNELLED_PORT,
    deployed=True,
    writes_enabled=False,
    baseline_type="virtual_accelerator",
    va_gateways=False,
    strict_limits=False,
    acknowledged=False,
    recorder=False,
    archiver_type=REAL_ARCHIVER,
):
    """A deployment that stood a stand-in up beside its facility's machine.

    Three targets, three connector blocks: ``virtual_accelerator`` (the sandbox,
    and this deployment's own baseline type), ``epics`` (the facility's authored
    machine — the only entry that is neither simulated nor the stand-in, which
    is what makes ``live`` resolve to it), and ``live_standin`` (the second
    virtual accelerator, dialled on loopback at the stand-in's port).

    *standin_port*, *gateway_host* and *gateway_port* are the three conjuncts of
    the stand-in predicate, so a case can fail exactly one of them:
    ``standin_port=None`` omits the ``services:`` block entirely, *gateway_host*
    moves the stand-in's endpoint off loopback, and *gateway_port* moves it off
    the stand-in's port. *live_host* and *live_port* place the facility's own
    gateway, which is how a case can ask what ``live`` is called when its
    endpoint happens to look exactly like the stand-in's.

    The display cases below need nothing more than that. The roster cases need
    the rest of a deployment an operator could actually switch on, and take it
    from the same builder rather than a second copy of these three blocks:
    *baseline_type* is ``control_system.type`` — ``live_standin`` for a
    deployment baselined on its own stand-in — *va_gateways* gives the simulator
    a gateway pair so it is a destination and not merely a description, and
    *strict_limits*, *acknowledged* and *recorder* set the three FR-8 facts the
    live family is gated on (the limits posture, the operator acknowledgment,
    and whether an ``archiver_recorder`` makes the store the stand-in's).
    """
    gateway = {"address": gateway_host, "port": gateway_port, "use_name_server": True}
    live_gateway = {"address": live_host, "port": live_port, "use_name_server": True}
    va_block = {"probe_channel": VA_PROBE}
    if va_gateways:
        va_gateway = {"address": "127.0.0.1", "port": VA_GATEWAY_PORT, "use_name_server": True}
        va_block["gateways"] = {
            "read_only": dict(va_gateway),
            "write_access": dict(va_gateway),
        }
    control_system = {
        "type": baseline_type,
        "writes_enabled": writes_enabled,
        "connector": {
            "epics": {
                "probe_channel": LIVE_PROBE,
                "gateways": {
                    "read_only": dict(live_gateway),
                    "write_access": dict(live_gateway),
                },
            },
            "live_standin": {
                "probe_channel": STANDIN_PROBE,
                "gateways": {
                    "read_only": dict(gateway),
                    "write_access": dict(gateway),
                },
            },
            "virtual_accelerator": va_block,
        },
    }
    if strict_limits:
        control_system["limits_checking"] = {"enabled": True, "allow_unlisted_channels": False}
    if acknowledged:
        control_system["target_switch"] = {ACK_LEAF: REAL_GATEWAY_HOST}
    raw = {"control_system": control_system, "archiver": {"type": archiver_type}}
    if standin_port is not None:
        raw["services"] = {"live_standin": {"port": standin_port}}
        if deployed:
            raw["deployed_services"] = ["virtual_accelerator", "live_standin"]
    if recorder:
        raw.setdefault("deployed_services", []).append(ARCHIVER_RECORDER_SERVICE)
    return raw


def without_a_standin(**kwargs):
    """The same deployment, before anybody stood a stand-in up.

    Two targets and two blocks: the facility's ``epics`` machine and the
    simulator. Neither the ``live_standin`` connector block nor the
    ``services.live_standin`` port the build projects for it exists, which is
    the shape every deployment in this repository had before the stand-in was a
    target at all.
    """
    raw = standin_config(**kwargs)
    del raw["control_system"]["connector"]["live_standin"]
    raw.pop("services", None)
    return raw


# ----------------------------------------------- correct before any switch


class TestCorrectBeforeAnySwitch:
    async def test_a_fresh_session_reports_both_targets_from_config_alone(
        self, make_manager, monkeypatch, no_prober
    ):
        """CF-1: no child has ever run, no probe has ever been taken.

        Every verdict here comes from configuration, and that is enough to
        answer the question the roster exists for.
        """
        manager = make_manager(raw=config_with_gateways())
        install_context(manager, monkeypatch)

        payload = extract_response_dict(await ROSTER())
        rows = payload["access_details"]["targets"]

        assert set(rows) == {"live", "va"}
        assert payload["summary"]["target"] == "live"
        assert payload["summary"]["generation"] == 0
        assert payload["summary"]["connector_host_alive"] is False
        # The session is on live, so live is unavailable *because it is active*
        # and va is available on the strength of config alone.
        assert rows["live"]["active"] is True
        assert rows["live"]["available_now"] is False
        assert rows["live"]["reason"] == REASON_ALREADY_ACTIVE
        assert rows["va"]["available_now"] is True
        assert rows["va"]["eligible_from_baseline"] is True
        assert payload["summary"]["switchable_targets"] == ["va"]

    async def test_an_unconfigured_target_reports_the_switchs_own_reason(
        self, make_manager, monkeypatch, no_prober
    ):
        """The roster and the refusal are one function, not two agreeing ones."""
        raw = config_with_gateways(va_probe=None)
        manager = make_manager(raw=raw)
        install_context(manager, monkeypatch)

        rows = extract_response_dict(await ROSTER())["access_details"]["targets"]

        assert rows["va"]["available_now"] is False
        assert rows["va"]["reason"] == REASON_PROBE_CHANNEL_MISSING
        assert rows["va"]["eligible"] is False
        # Verbatim, so an operator reading the roster and an agent reading the
        # refusal are told the same thing.
        with assert_raises_error(error_type=control_target.ERROR_REFUSED) as ctx:
            await get_tool_fn(control_target.control_target_set)(target="va")
        assert ctx["envelope"]["error_message"] == rows["va"]["detail"]

    async def test_rows_carry_the_probe_channel_and_the_real_machine_flag(
        self, make_manager, monkeypatch, no_prober
    ):
        """Both come from the state file's display metadata, not a second derivation.

        Compared against that metadata rather than against literals: the point
        is not what this fixture's flags happen to be, it is that the roster
        and the state file the prompt hook reads cannot disagree about which
        target is the real machine.
        """
        raw = config_with_gateways()
        manager = make_manager(raw=raw)
        install_context(manager, monkeypatch)
        display = target_display_metadata(raw)

        rows = extract_response_dict(await ROSTER())["access_details"]["targets"]

        assert rows["va"]["probe_channel"] == VA_PROBE
        assert rows["va"]["real_machine"] == display["va"]["real_machine"] is False
        assert rows["live"]["real_machine"] == display["live"]["real_machine"]
        assert rows["live"]["label"] == display["live"]["label"]
        assert rows["live"]["connector_type"].endswith("MockConnector")

    async def test_writes_permitted_follows_the_deployment_posture_no_type_states(
        self, make_manager, monkeypatch, no_prober
    ):
        """Neither connector block says anything, so both rows inherit the global key."""
        manager = make_manager(raw=config_with_gateways())
        install_context(manager, monkeypatch)

        rows = extract_response_dict(await ROSTER())["access_details"]["targets"]
        assert [row["writes_permitted"] for row in rows.values()] == [False, False]

        raw = config_with_gateways()
        raw["control_system"]["writes_enabled"] = True
        writable = make_manager(raw=raw)
        install_context(writable, monkeypatch)

        rows = extract_response_dict(await ROSTER())["access_details"]["targets"]
        assert all(row["writes_permitted"] for row in rows.values())

    async def test_each_row_carries_its_own_targets_posture(
        self, make_manager, monkeypatch, no_prober
    ):
        """A deployment arms its simulator alone, and the two rows say so separately.

        The write posture the roster reports is per connector type, so the row
        for ``va`` and the row for ``live`` are two answers and not one flag
        printed twice.
        """
        raw = config_with_gateways()
        raw["control_system"]["writes_enabled"] = False
        raw["control_system"]["connector"]["virtual_accelerator"]["writes_enabled"] = True
        manager = make_manager(raw=raw)
        install_context(manager, monkeypatch)

        rows = extract_response_dict(await ROSTER())["access_details"]["targets"]

        assert rows["va"]["writes_permitted"] is True
        assert rows["live"]["writes_permitted"] is False

    async def test_a_readonly_run_reports_writes_as_not_permitted(
        self, make_manager, monkeypatch, no_prober
    ):
        """The run's own claim counts, not only the deployment's posture."""
        raw = config_with_gateways()
        raw["control_system"]["writes_enabled"] = True
        manager = make_manager(raw=raw)
        install_context(manager, monkeypatch)
        monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")

        rows = extract_response_dict(await ROSTER())["access_details"]["targets"]

        assert not any(row["writes_permitted"] for row in rows.values())

    async def test_a_readonly_run_collapses_an_armed_target_too(
        self, make_manager, monkeypatch, no_prober
    ):
        """Per-target posture does not survive a read-only run — no row is armed."""
        raw = config_with_gateways()
        raw["control_system"]["writes_enabled"] = False
        raw["control_system"]["connector"]["virtual_accelerator"]["writes_enabled"] = True
        manager = make_manager(raw=raw)
        install_context(manager, monkeypatch)
        monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")

        rows = extract_response_dict(await ROSTER())["access_details"]["targets"]

        assert rows["va"]["writes_permitted"] is False
        assert rows["live"]["writes_permitted"] is False


# ------------------------------------------------------- reachability rows


class TestReachabilityRows:
    async def test_without_a_prober_no_row_claims_a_reachability(
        self, make_manager, monkeypatch, no_prober
    ):
        """Absent measurement is absent, not "down" and not "ok"."""
        manager = make_manager(raw=config_with_gateways())
        install_context(manager, monkeypatch)

        payload = extract_response_dict(await ROSTER())

        assert payload["access_details"]["endpoint_probe"]["running"] is False
        for row in payload["access_details"]["targets"].values():
            for endpoint in row["endpoints"].values():
                # The derived half is still there — config is knowable without
                # touching anything.
                assert endpoint["host"] == "127.0.0.1"
                assert "endpoint_tcp" not in endpoint

    async def test_a_measured_row_carries_the_probers_observation(self, make_manager, monkeypatch):
        manager = make_manager(raw=config_with_gateways())
        install_context(manager, monkeypatch)
        prober = install_prober(
            monkeypatch,
            {
                "va": {
                    "read_only": {
                        "endpoint_tcp": "ok",
                        "last_status": "ok",
                        "gateway": "127.0.0.1:5064",
                        "probed_at": "2026-08-22T10:00:00+00:00",
                        "detail": "",
                    }
                }
            },
        )

        payload = extract_response_dict(await ROSTER())

        va_row = payload["access_details"]["targets"]["va"]["endpoints"]["read_only"]
        assert va_row["endpoint_tcp"] == "ok"
        assert va_row["probed_at"] == "2026-08-22T10:00:00+00:00"
        # Config and measurement are merged, not one replacing the other.
        assert va_row["mode"] == "addr_list"
        assert payload["access_details"]["endpoint_probe"]["running"] is True
        assert payload["access_details"]["endpoint_probe"]["probe_interval_s"] == 30.0
        assert prober.snapshot_calls == 1
        # A target the prober has no row for keeps its derived endpoint only.
        live_row = payload["access_details"]["targets"]["live"]["endpoints"]["read_only"]
        assert "endpoint_tcp" not in live_row

    async def test_staleness_surfaces_with_what_was_last_seen(self, make_manager, monkeypatch):
        """A stalled prober is visible without destroying its last observation."""
        manager = make_manager(raw=config_with_gateways())
        install_context(manager, monkeypatch)
        install_prober(
            monkeypatch,
            {
                "va": {
                    "read_only": {
                        "endpoint_tcp": "stale",
                        "last_status": "ok",
                        "gateway": "127.0.0.1:5064",
                        "probed_at": "2026-08-22T09:00:00+00:00",
                        "detail": "",
                    }
                }
            },
        )

        rows = extract_response_dict(await ROSTER())["access_details"]["targets"]
        endpoint = rows["va"]["endpoints"]["read_only"]

        assert endpoint["endpoint_tcp"] == "stale"
        assert endpoint["last_status"] == "ok"


# ---------------------------------------------------------- after a switch


class TestAfterASwitch:
    async def test_the_roster_reflects_the_active_target_and_generation(
        self, make_manager, monkeypatch, no_prober
    ):
        manager = await started_on(make_manager, "live")
        install_context(manager, monkeypatch)
        await manager.switch("va")

        payload = extract_response_dict(await ROSTER())
        rows = payload["access_details"]["targets"]

        assert payload["summary"]["target"] == "va"
        assert payload["summary"]["generation"] == 1
        assert payload["summary"]["connector_host_alive"] is True
        assert rows["va"]["active"] is True
        assert rows["va"]["available_now"] is False
        assert rows["va"]["reason"] == REASON_ALREADY_ACTIVE
        assert rows["live"]["active"] is False


# --------------------------------------------------------- side-effect-free


class TestSideEffectFree:
    async def test_a_roster_call_starts_nothing_and_writes_nothing(
        self, make_manager, monkeypatch, no_prober, state_root
    ):
        """The whole point: asking the question must not answer it by acting."""
        manager = make_manager(raw=config_with_gateways())
        install_context(manager, monkeypatch)

        async def refuse(*args, **kwargs):
            raise AssertionError("the roster must not start or switch a connector host")

        monkeypatch.setattr(manager, "start", refuse)
        monkeypatch.setattr(manager, "switch", refuse)
        monkeypatch.setattr(manager, "respawn_same_target", refuse)

        before = json.dumps(target_state.read(), sort_keys=True)
        state_dir = target_state.state_dir()
        before_files = sorted(p.name for p in state_dir.iterdir())

        await ROSTER()
        await ROSTER()

        assert manager.has_child() is False
        assert manager.is_started() is False
        assert json.dumps(target_state.read(), sort_keys=True) == before
        assert sorted(p.name for p in state_dir.iterdir()) == before_files

    async def test_the_roster_does_not_emit_a_switch_activity_event(
        self, make_manager, monkeypatch, no_prober
    ):
        """Reporting is not an attempt, so nothing is reported as one."""
        manager = make_manager(raw=config_with_gateways())
        install_context(manager, monkeypatch)
        calls: list[dict] = []

        async def record(**kwargs):
            calls.append(kwargs)

        monkeypatch.setattr(control_target, "notify_target_switch_async", record)

        await ROSTER()

        assert calls == []


# ------------------------------------------------------------- degradation


class TestDegradation:
    async def test_without_a_server_context_the_roster_says_so(self, monkeypatch):
        """No session to describe is reported as that, not as an empty roster."""
        from osprey.mcp_server.control_system import server_context as server_context_mod

        monkeypatch.setattr(server_context_mod, "_registry", None)

        with assert_raises_error(error_type=control_target.ERROR_UNAVAILABLE) as ctx:
            await ROSTER()

        assert ctx["envelope"]["details"]["reason"] == control_target.REASON_CONTEXT_UNAVAILABLE

    async def test_an_underivable_target_gets_no_row_and_is_still_refused(
        self, make_manager, monkeypatch, no_prober
    ):
        """A deployment that never named its real machine has no 'live' row.

        A refusal is not a slot (FR-4). ``live`` here resolves to nothing at
        all — there is no connector block naming a real machine — so the roster
        offers no row for it rather than a row describing a machine this
        deployment has never been told about. The roster enumerates what a
        session can be pointed at, and an entry for an unresolvable target
        would read as "here is a machine, currently unavailable".

        Nothing is lost by the row's absence, which is the other half of this
        test: an agent that asks for the target anyway is still refused, in the
        eligibility module's own words, with the reason that names what is
        missing.
        """
        # A virtual-accelerator deployment that names no real machine: the
        # session is baselined on 'va', so 'live' is judged as a destination
        # and its unresolvability is the reason rather than being shadowed by
        # "you are already there".
        raw = config_with_gateways()
        va_block = raw["control_system"]["connector"]["virtual_accelerator"]
        raw["control_system"]["type"] = "virtual_accelerator"
        raw["control_system"]["connector"] = {"virtual_accelerator": va_block}
        manager = make_manager(raw=raw)
        install_context(manager, monkeypatch)

        rows = extract_response_dict(await ROSTER())["access_details"]["targets"]

        assert "live" not in rows
        assert set(rows) == {"va"}

        # The refusal still travels through target_availability, so the switch
        # answers an unresolvable target with a reason rather than a traceback
        # — and with the same words the roster would have carried had there
        # been a row to carry them.
        expected = target_availability(raw, "live", manager.active_target(), manager.baseline)
        assert expected.reason == REASON_TARGET_UNRESOLVABLE

        with assert_raises_error(error_type=control_target.ERROR_REFUSED) as ctx:
            await get_tool_fn(control_target.control_target_set)(target="live")

        assert ctx["envelope"]["details"] == expected.as_dict()
        assert ctx["envelope"]["details"]["reason"] == REASON_TARGET_UNRESOLVABLE


# ------------------------------------------ the three-target roster (SC-4)


class TestThreeTargetRoster:
    """SC-4: a deployment baselined on its stand-in reports all three rows.

    The roster is the surface an operator asks "where am I, and where could I
    go" on, so a deployment that stood a stand-in up beside its facility's
    machine has to answer for three machines and not two — and answer for each
    of them separately, because the three are gated differently. The stand-in
    is where the session already is; the simulator is a plain destination; and
    ``live`` is the one target that carries the whole FR-8 ritual, including
    the gate the stand-in creates for it — a deployment recording its own store
    beside a stand-in is recording the stand-in, and a real machine's readings
    must not land in that store.

    Every row here is read through the tool rather than through
    :func:`target_availability`, because the row and the refusal being one
    function is the property the roster is worth anything for, and the tool is
    where a reader meets it.
    """

    async def test_a_standin_baselined_deployment_reports_all_three_rows(
        self, make_manager, monkeypatch, no_prober
    ):
        """Three machines, three rows, and the session sitting on the stand-in.

        ``control_system.type: live_standin`` is a legitimate baseline: the
        deployment's own machine is the soft IOC it stands up, so that is where
        a session starts and that is the row marked active. ``live`` still means
        the facility's authored ``epics`` block — the stand-in never renames it
        — and the simulator is untouched beside both.
        """
        raw = standin_config(
            baseline_type="live_standin",
            va_gateways=True,
            strict_limits=True,
            acknowledged=True,
        )
        manager = make_manager(raw=raw)
        install_context(manager, monkeypatch)

        payload = extract_response_dict(await ROSTER())
        rows = payload["access_details"]["targets"]

        assert set(rows) == {"live", "va", "standin"}
        assert payload["summary"]["target"] == "standin"
        assert payload["summary"]["baseline_target"] == "standin"

        # The stand-in is where the session is, so it is unavailable for the one
        # truthful reason — not because anything about it is misconfigured.
        assert rows["standin"]["active"] is True
        assert rows["standin"]["is_baseline"] is True
        assert rows["standin"]["available_now"] is False
        assert rows["standin"]["reason"] == REASON_ALREADY_ACTIVE
        assert rows["standin"]["eligible"] is True
        assert rows["standin"]["label"] == "LIVE MACHINE (stand-in)"
        assert rows["standin"]["real_machine"] is True
        assert rows["standin"]["connector_type"] == "live_standin"
        assert rows["standin"]["probe_channel"] == STANDIN_PROBE

        # ``live`` is the facility's own machine, reached through its own block.
        assert rows["live"]["active"] is False
        assert rows["live"]["label"] == "LIVE MACHINE"
        assert rows["live"]["real_machine"] is True
        assert rows["live"]["connector_type"] == "epics"
        assert rows["live"]["endpoints"]["read_only"]["host"] == REAL_GATEWAY_HOST

        assert rows["va"]["real_machine"] is False
        assert rows["va"]["available_now"] is True

    async def test_the_live_row_wants_the_acknowledgment_the_standin_does_not(
        self, make_manager, monkeypatch, no_prober
    ):
        """The stand-in's equivalent was said at build time; live's is not.

        Both machines are behind the strict limits posture, which this
        deployment has. What separates them is the acknowledgment — the
        operator saying the configured gateways really are this facility's —
        and it is the live machine's alone, so an unacknowledged deployment
        reports the stand-in usable and the facility's machine not.
        """
        raw = standin_config(
            baseline_type="live_standin",
            va_gateways=True,
            strict_limits=True,
            acknowledged=False,
        )
        manager = make_manager(raw=raw)
        install_context(manager, monkeypatch)

        rows = extract_response_dict(await ROSTER())["access_details"]["targets"]

        assert rows["live"]["available_now"] is False
        assert rows["live"]["reason"] == REASON_OPERATOR_ACK_MISSING
        # The stand-in met the same limits posture and is not asked for an
        # acknowledgment at all.
        assert rows["standin"]["eligible"] is True

    async def test_a_recorded_standin_store_closes_the_live_row(
        self, make_manager, monkeypatch, no_prober
    ):
        """Acknowledged, strict, and still refused: the archive is the stand-in's.

        This is the gate the stand-in creates for ``live`` and for nothing else.
        The deployment runs the recorder, so the store it writes holds the
        stand-in's synthesized past; selecting the facility's real machine
        would splice real readings onto it in one store nothing afterwards can
        tell apart.
        """
        raw = standin_config(
            baseline_type="live_standin",
            va_gateways=True,
            strict_limits=True,
            acknowledged=True,
            recorder=True,
        )
        assert ARCHIVER_RECORDER_SERVICE in raw["deployed_services"]
        manager = make_manager(raw=raw)
        install_context(manager, monkeypatch)

        rows = extract_response_dict(await ROSTER())["access_details"]["targets"]

        assert rows["live"]["available_now"] is False
        assert rows["live"]["reason"] == REASON_ARCHIVE_BELONGS_TO_STANDIN
        assert ARCHIVER_RECORDER_SERVICE in rows["live"]["detail"]
        # Nothing else moved: the recorder is a fact about the store, not about
        # the machines beside it.
        assert rows["standin"]["eligible"] is True
        assert rows["va"]["available_now"] is True

    async def test_an_acknowledged_deployment_that_records_nothing_offers_live(
        self, make_manager, monkeypatch, no_prober
    ):
        """The same deployment without the recorder: all three rows are usable.

        The positive case the three refusals above are only meaningful against
        — stop recording and the facility's machine is available from a
        stand-in baseline, which is the point of standing a stand-in up beside
        it rather than instead of it.
        """
        raw = standin_config(
            baseline_type="live_standin",
            va_gateways=True,
            strict_limits=True,
            acknowledged=True,
            recorder=False,
        )
        manager = make_manager(raw=raw)
        install_context(manager, monkeypatch)

        payload = extract_response_dict(await ROSTER())
        rows = payload["access_details"]["targets"]

        assert rows["live"]["available_now"] is True
        assert rows["live"]["reason"] is None
        assert rows["va"]["available_now"] is True
        assert payload["summary"]["switchable_targets"] == ["live", "va"]

    async def test_a_standin_block_off_loopback_is_a_row_that_refuses(
        self, make_manager, monkeypatch, no_prober
    ):
        """The target exists, so it gets a row; the endpoint is not the container.

        A ``live_standin`` block repointed at a gateway on another host is a
        configured target — hence a row — but not the stand-in this deployment
        co-deploys, so the row says so with the reason a switch would refuse
        with. That distinction is exactly what an absent row could not express.
        """
        raw = standin_config(
            baseline_type="epics",
            gateway_host=REAL_GATEWAY_HOST,
            va_gateways=True,
            strict_limits=True,
            acknowledged=True,
        )
        manager = make_manager(raw=raw)
        install_context(manager, monkeypatch)

        rows = extract_response_dict(await ROSTER())["access_details"]["targets"]

        assert "standin" in rows
        assert rows["standin"]["available_now"] is False
        assert rows["standin"]["reason"] == REASON_STANDIN_NOT_DEPLOYED
        assert rows["standin"]["connector_type"] == "live_standin"

    async def test_a_deployment_with_no_standin_has_no_standin_row(
        self, make_manager, monkeypatch, no_prober
    ):
        """Widening the vocabulary must not grow a row on a two-target deployment.

        No ``control_system.connector.live_standin`` block means no stand-in
        target, and the roster of such a deployment enumerates exactly what it
        always did. A ``standin`` row here would describe a soft IOC nobody
        stood up.
        """
        raw = without_a_standin(va_gateways=True, strict_limits=True, acknowledged=True)
        manager = make_manager(raw=raw)
        install_context(manager, monkeypatch)

        rows = extract_response_dict(await ROSTER())["access_details"]["targets"]

        assert set(rows) == {"live", "va"}
        assert "standin" not in rows


# ------------------------------------------------- display: the three labels
#
# Everything below is the display section: what :func:`target_display_metadata`
# names each target and which postures it calls a real machine. Roster rows are
# asserted further up, through the tool.


class TestLiveStandinLabel:
    """What each of the three targets is called, and which one may be a stand-in.

    Pinned on :func:`target_display_metadata` rather than through the tool,
    because that function is the single writer every reader of the label — the
    roster row, the prompt hook, the approval banner, the web badge — renders
    from. A case that failed here would reach all of them at once.

    The parenthesis belongs to ``standin`` alone. ``live`` names the facility's
    authored machine and is never renamed, whatever its endpoint happens to
    look like; ``standin`` earns the parenthesis only when its endpoint really
    is the container this deployment stood up, and is otherwise described as
    the real machine it would be dialling (the eligibility gate is what refuses
    that target, not the label).

    ``real_machine`` is asserted beside the label precisely because it does
    *not* move: the stand-in keeps the real machine's whole ritual, and only
    the name an operator is shown changes. The ``va`` row is asserted for the
    same reason — the simulator is described by its own branch, which the
    stand-in must not disturb.
    """

    def test_every_target_gets_a_slot(self):
        """One entry per state-file slot: a missing target is an empty banner."""
        metadata = target_display_metadata(standin_config())

        assert set(metadata) == set(target_state.TARGET_NAMES)
        assert set(metadata) == {"live", "va", "standin"}

    def test_a_deployed_standin_is_named_as_one(self):
        metadata = target_display_metadata(standin_config())

        assert metadata["standin"]["label"] == "LIVE MACHINE (stand-in)"
        assert metadata["standin"]["real_machine"] is True
        assert metadata["standin"]["endpoint"] == f"localhost:{STANDIN_PORT}"
        assert metadata["standin"]["probe_channel"] == STANDIN_PROBE

    def test_the_facility_machine_is_never_named_a_standin(self):
        """``live`` is the authored ``epics`` block, and it keeps its own name."""
        metadata = target_display_metadata(standin_config())

        assert metadata["live"]["label"] == "LIVE MACHINE"
        assert metadata["live"]["real_machine"] is True
        assert metadata["live"]["endpoint"] == f"{REAL_GATEWAY_HOST}:{TUNNELLED_PORT}"

    def test_a_facility_gateway_on_the_standins_port_is_still_the_live_machine(self):
        """An endpoint that looks like the stand-in's does not rename ``live``.

        The one case where the endpoint predicate alone would answer "stand-in"
        for the facility's own target: a gateway reached over loopback on the
        port the stand-in also serves. Telling an operator that the machine in
        front of them is only a stand-in when it is not is the expensive
        mistake, so the parenthesis is the ``standin`` target's alone.
        """
        metadata = target_display_metadata(
            standin_config(live_host="localhost", live_port=STANDIN_PORT)
        )

        assert metadata["live"]["label"] == "LIVE MACHINE"
        assert metadata["live"]["real_machine"] is True
        assert metadata["standin"]["label"] == "LIVE MACHINE (stand-in)"

    def test_a_deployment_that_stood_no_standin_up_does_not_name_one(self):
        """No ``services.live_standin`` block: the first conjunct fails.

        The target still gets its slot and is described as what it would be
        dialling. Refusing it is ``standin_not_deployed``'s job, not the
        label's.
        """
        metadata = target_display_metadata(standin_config(standin_port=None))

        assert metadata["standin"]["label"] == "LIVE MACHINE"
        assert metadata["standin"]["real_machine"] is True
        assert metadata["live"]["label"] == "LIVE MACHINE"

    def test_a_leftover_standin_block_does_not_rename_a_moved_endpoint(self):
        """The label follows the endpoint, never the stale ``services:`` block."""
        metadata = target_display_metadata(standin_config(gateway_port=TUNNELLED_PORT))

        assert metadata["standin"]["label"] == "LIVE MACHINE"
        assert metadata["standin"]["real_machine"] is True

    def test_an_ssh_tunnel_to_loopback_is_still_the_live_machine(self):
        """Loopback alone proves nothing: the operator is one hop from hardware."""
        metadata = target_display_metadata(
            standin_config(standin_port=None, gateway_port=TUNNELLED_PORT)
        )

        assert metadata["standin"]["label"] == "LIVE MACHINE"
        assert metadata["standin"]["endpoint"] == f"localhost:{TUNNELLED_PORT}"

    def test_a_standin_endpoint_off_loopback_is_not_this_hosts_container(self):
        """The port matches, the host does not: another machine on that port."""
        metadata = target_display_metadata(standin_config(gateway_host=REAL_GATEWAY_HOST))

        assert metadata["standin"]["label"] == "LIVE MACHINE"

    def test_a_persona_render_says_the_same_word_as_the_deployment(self):
        """Multi-user parity: the projected port is the whole evidence.

        An attached project's render carries ``services: {}`` except for the
        keys its reach contract projects, and no ``deployed_services`` at all.
        One word of label for a single-user session and a different word for
        the same machine seen through a persona would be the bug.
        """
        raw = standin_config(deployed=False)
        assert "deployed_services" not in raw
        assert raw["services"] == {"live_standin": {"port": STANDIN_PORT}}

        metadata = target_display_metadata(raw)

        assert metadata["standin"]["label"] == "LIVE MACHINE (stand-in)"
        assert metadata["standin"]["real_machine"] is True

    def test_an_armed_deployment_names_the_standin_through_its_write_gateway(self):
        """Arming writes selects the other gateway row, and both point at it."""
        metadata = target_display_metadata(standin_config(writes_enabled=True))

        assert metadata["standin"]["label"] == "LIVE MACHINE (stand-in)"
        assert metadata["standin"]["real_machine"] is True
        assert metadata["standin"]["endpoint"] == f"localhost:{STANDIN_PORT}"

    def test_the_simulator_is_described_by_its_own_branch(self):
        """``va`` is untouched by the stand-in, in every case above and here."""
        for raw in (
            standin_config(),
            standin_config(standin_port=None),
            standin_config(gateway_host=REAL_GATEWAY_HOST),
        ):
            metadata = target_display_metadata(raw)

            assert metadata["va"]["label"] == "virtual accelerator (simulation)"
            assert metadata["va"]["real_machine"] is False
            assert metadata["va"]["probe_channel"] == VA_PROBE

    def test_real_machine_marks_both_hardware_postures(self):
        """A stand-in is a real-machine posture: only the simulator is not.

        ``real_machine`` is what every strict limit, approval prompt and banner
        keys off, so an operator on the stand-in meets the ritual hardware
        gets.
        """
        metadata = target_display_metadata(standin_config())

        assert metadata["live"]["real_machine"] is True
        assert metadata["standin"]["real_machine"] is True
        assert metadata["va"]["real_machine"] is False


class TestDisplayName:
    """The operator-facing word minted beside each label, and who may rename it.

    ``display_name`` walks the same branches as the label, from the same
    inputs, at the same single writer — so the word on the chip and the
    identity line the prompt hook renders cannot describe different machines.
    Unlike the label, a deployment may rename it per target via
    ``control_system.target_display_names``; only a non-empty string does,
    because an empty name on the chip is a target the operator cannot tell
    apart from the others.
    """

    def test_defaults_follow_the_label_branches(self):
        """Real machine, Rehearsal, Simulator — one word per derivation."""
        metadata = target_display_metadata(standin_config())

        assert metadata["live"]["display_name"] == "Real machine"
        assert metadata["standin"]["display_name"] == "Rehearsal"
        assert metadata["va"]["display_name"] == "Simulator"

    def test_an_underivable_live_target_is_still_the_real_machine(self):
        """The "not set up" nuance is the reader's to render, not the name's."""
        metadata = target_display_metadata({"control_system": {"type": "virtual_accelerator"}})

        assert metadata["live"]["label"] == "live machine (not configured)"
        assert metadata["live"]["display_name"] == "Real machine"

    def test_a_simulated_connector_is_a_demo(self):
        """The branch mirrors the label's "live target on a simulated connector".

        Asserted on the helper directly: :func:`resolve_target` never answers a
        live-family target with a simulated type today, so the branch is
        reachable only the way the label's own simulated branch is — kept so
        the two names cannot diverge if that ever changes.
        """
        assert _display_name({}, "live", "mock") == "Demo"
        assert _display_name({}, "live", "virtual_accelerator") == "Demo"

    def test_a_standin_that_fails_the_predicate_is_the_real_machine(self):
        """Same conjuncts as the parenthesis: no deployed stand-in, no Rehearsal."""
        metadata = target_display_metadata(standin_config(standin_port=None))

        assert metadata["standin"]["label"] == "LIVE MACHINE"
        assert metadata["standin"]["display_name"] == "Real machine"

    def test_a_configured_name_wins_verbatim_stripped(self):
        raw = standin_config()
        raw["control_system"]["target_display_names"] = {
            "live": "  Storage ring ",
            "standin": "Shadow ring",
            "va": "Digital twin",
        }

        metadata = target_display_metadata(raw)

        assert metadata["live"]["display_name"] == "Storage ring"
        assert metadata["standin"]["display_name"] == "Shadow ring"
        assert metadata["va"]["display_name"] == "Digital twin"

    def test_an_empty_or_blank_override_falls_back_to_the_default(self):
        raw = standin_config()
        raw["control_system"]["target_display_names"] = {"live": "", "standin": "   ", "va": None}

        metadata = target_display_metadata(raw)

        assert metadata["live"]["display_name"] == "Real machine"
        assert metadata["standin"]["display_name"] == "Rehearsal"
        assert metadata["va"]["display_name"] == "Simulator"

    def test_an_override_never_touches_the_label(self):
        """The identity line is the safety surface; the name is cosmetic."""
        raw = standin_config()
        raw["control_system"]["target_display_names"] = {"standin": "Shadow ring"}

        metadata = target_display_metadata(raw)

        assert metadata["standin"]["display_name"] == "Shadow ring"
        assert metadata["standin"]["label"] == "LIVE MACHINE (stand-in)"
        assert metadata["standin"]["real_machine"] is True


# ------------------------------------------------------ the limits posture


class TestLimitsPostureRows:
    """``limits_strict``: per target, for the same reason ``writes_permitted`` is.

    Limits checking is per connector type, so a deployment can relax unlisted
    channels on its simulator while its live machine refuses them. A single
    flag for the deployment would tell an operator standing on hardware what is
    true of the sandbox next to it.
    """

    async def test_each_row_carries_its_own_targets_limits_posture(
        self, make_manager, monkeypatch, no_prober
    ):
        """A permissive simulator beside two strict machines: three answers, not one.

        The deployment-wide block is strict and only the simulator's own block
        relaxes it, so ``live`` and ``standin`` — neither of which wrote a block
        — inherit the strict deployment-wide posture and the simulator answers
        from its own.
        """
        raw = standin_config(
            baseline_type="live_standin",
            va_gateways=True,
            strict_limits=True,
            acknowledged=True,
        )
        raw["control_system"]["connector"]["virtual_accelerator"]["limits_checking"] = {
            "enabled": True,
            "allow_unlisted_channels": True,
        }
        manager = make_manager(raw=raw)
        install_context(manager, monkeypatch)

        rows = extract_response_dict(await ROSTER())["access_details"]["targets"]

        assert rows["live"]["limits_strict"] is True
        assert rows["standin"]["limits_strict"] is True
        assert rows["va"]["limits_strict"] is False

    async def test_a_deployment_that_states_no_posture_is_not_strict(
        self, make_manager, monkeypatch, no_prober
    ):
        """Silence is not a guarantee: no block anywhere means no row is strict.

        A deployment that never configured limits checking has refused nothing,
        and a row saying otherwise would advertise a promise no config line
        backs.
        """
        manager = make_manager(raw=config_with_gateways())
        install_context(manager, monkeypatch)

        rows = extract_response_dict(await ROSTER())["access_details"]["targets"]

        assert [row["limits_strict"] for row in rows.values()] == [False, False]

    def test_an_underivable_row_carries_the_deployment_wide_posture(self):
        """``live`` on a mock deployment resolves to no type, so no block can answer.

        The row is still there — the baseline always gets one — and the posture
        it carries is the deployment-wide block, which is the only one such a
        deployment has ever had. The simulator's own permissive block sits
        beside it and does not answer for it.
        """
        raw = {
            "control_system": {
                "type": "mock",
                "limits_checking": {"enabled": True, "allow_unlisted_channels": False},
                "connector": {
                    "mock": {"probe_channel": LIVE_PROBE},
                    "virtual_accelerator": {
                        "probe_channel": VA_PROBE,
                        "limits_checking": {"enabled": True, "allow_unlisted_channels": True},
                    },
                },
            },
            "archiver": {"type": REAL_ARCHIVER},
        }

        rows = control_target.target_rows(raw, session_target="live", baseline="live")

        assert rows["live"]["endpoints"] == {}
        assert rows["live"]["limits_strict"] is True
        assert rows["va"]["limits_strict"] is False
