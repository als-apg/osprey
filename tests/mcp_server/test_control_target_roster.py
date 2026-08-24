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
from osprey.mcp_server.control_system.target_eligibility import (
    REASON_ALREADY_ACTIVE,
    REASON_PROBE_CHANNEL_MISSING,
)
from osprey.mcp_server.control_system.tools import control_target
from tests.mcp_server import test_switch_lifecycle as switch_suite
from tests.mcp_server.conftest import assert_raises_error, extract_response_dict, get_tool_fn
from tests.mcp_server.test_control_target_set import config_with_gateways, install_context

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
        from osprey.mcp_server.control_system.connector_host_manager import (
            target_display_metadata,
        )

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

    async def test_writes_permitted_follows_the_deployment_posture(
        self, make_manager, monkeypatch, no_prober
    ):
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

    async def test_an_underivable_target_still_gets_a_row(
        self, make_manager, monkeypatch, no_prober
    ):
        """A deployment that never named its real machine has no 'live' endpoint.

        The row is still present, with the endpoints table empty and the reason
        naming what is missing — an absent row would read as "no such target",
        which is a different and untrue claim.
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

        assert rows["live"]["endpoints"] == {}
        assert rows["live"]["available_now"] is False
        assert rows["live"]["reason"] == "target_unresolvable"
        assert "connector_type" not in rows["live"]
