"""The run-level pin: a switch turned on never reaches a run that started with it off.

The per-target switch in the control-context record is read at WRITE time,
which is what lets an operator turn a target's writes off while a session is
already mid-conversation. That direction is the point of the feature. The other
direction is a hazard: a script is already running inside the sandbox, and
turning the switch on under it
would hand it write access to a machine nobody re-consented to while it works.

So the executor pins what it launched under. ``OSPREY_LAUNCH_POSTURE`` is
stamped into the sandbox environment and recorded in the in-flight marker, and
inside that sandbox the write verdict reads it as its last term
(``launch_pin_term``):

* the switch goes OFF -> the record read refuses on the next write (immediate);
* the switch goes ON -> the pin still refuses (the run stays off until it ends).

How the pin combines with the switch is pinned by the write-verdict case table
(rows ``pin-*``, ``order-record-beats-pin``, ``order-switch-beats-pin``); this
module covers the stamp's format and what the executor stamps.

The 409 the posture route raises while a marker is live is a courtesy on top of
this, not the barrier.

The executor stamps the pin from the write verdict for the run's target, asked
with the pin itself ignored: a switch turned off reaches a run that is already
going at its next write, and a switch turned on reaches only the next run.
"""

from __future__ import annotations

import pytest

from osprey.audit import posture
from osprey.mcp_server.python_executor import executor as host_executor
from osprey_connectors import control_context, posture_store
from osprey_connectors_tests._harness import pin_identity, state_dir_under, write_control_context
from osprey_connectors_tests._write_fakes import turn_switch

#: A session key of the shape the web terminal writes.
SESSION_KEY = "4f1c2a7e-0000-4000-8000-000000000042"

#: A deployment whose tier allows writes on every target, so the tier never
#: decides anything here — this module is about the switch and the pin.
ARMED_TIER = {
    "type": "epics",
    "writes": "allowed",
    "connector": {
        "epics": {"prefix": "RING:"},
        "virtual_accelerator": {"prefix": "VA:"},
        "live_standin": {"prefix": "SIM:"},
    },
}


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    """A scratch agent-data root, stamped, with every posture env name cleared."""
    monkeypatch.setenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, str(tmp_path))
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)
    monkeypatch.delenv(posture_store.LAUNCH_POSTURE_ENV_VAR, raising=False)
    monkeypatch.setenv("OSPREY_POSTURE_SESSION", SESSION_KEY)
    posture_store.invalidate_cache()
    yield tmp_path
    posture_store.invalidate_cache()


# ---------------------------------------------------------------------------
# The stamp's wire format
# ---------------------------------------------------------------------------


class TestStampFormat:
    """``<target>=<posture>``, written in one place and read as the pin term."""

    def test_the_stamp_is_one_target_equals_posture_pair(self):
        assert posture_store.launch_posture_stamp("standin", "sandbox") == "standin=sandbox"
        assert posture_store.launch_posture_stamp("va", "writes") == "va=writes"

    def test_no_target_is_spelled_as_every_target(self, monkeypatch):
        """A run that cannot name its machine is pinned for all of them."""
        stamp = posture_store.launch_posture_stamp(None, "sandbox")
        monkeypatch.setenv(posture_store.LAUNCH_POSTURE_ENV_VAR, stamp)

        assert stamp == f"{posture_store.LAUNCH_POSTURE_ALL_TARGETS}=sandbox"
        for target in posture_store.CONTROL_TARGETS:
            assert posture_store.launch_pin_term(target) == "pinned", target


# ---------------------------------------------------------------------------
# What the executor stamps, and what the marker records
# ---------------------------------------------------------------------------


class TestExecutorStampsThePin:
    """``_launch_posture`` asks the write verdict once, at launch, with the pin ignored."""

    @pytest.fixture(autouse=True)
    def _tier(self, monkeypatch):
        monkeypatch.setattr(posture, "control_system_section", lambda: dict(ARMED_TIER))

    def test_a_switched_on_target_launches_writes(self, data_root):
        write_control_context(data_root, target="standin", switch={"standin": "asked"})

        assert host_executor._launch_posture("standin") == "standin=writes"

    def test_a_switched_off_target_launches_sandboxed(self, data_root):
        write_control_context(data_root, target="standin", switch={"va": "asked"})

        assert host_executor._launch_posture("standin") == "standin=sandbox"

    @pytest.mark.usefixtures("data_root")
    def test_no_record_launches_sandboxed(self):
        """No record is every switch off."""
        assert host_executor._launch_posture("standin") == "standin=sandbox"

    def test_a_tier_that_allows_nothing_launches_sandboxed(self, data_root, monkeypatch):
        monkeypatch.setattr(posture, "control_system_section", lambda: {"type": "epics"})
        write_control_context(data_root, target="live", switch={"live": "asked"})

        assert host_executor._launch_posture("live") == "live=sandbox"

    def test_the_pin_this_process_inherited_is_ignored(self, data_root, monkeypatch):
        """The pin is asked with the pin ignored, so it never merely echoes itself."""
        write_control_context(data_root, target="standin", switch={"standin": "asked"})
        monkeypatch.setenv(posture_store.LAUNCH_POSTURE_ENV_VAR, "*=sandbox")

        assert host_executor._launch_posture("standin") == "standin=writes"

    def test_the_record_the_operator_switched_is_the_one_the_launch_reads(
        self, data_root, monkeypatch
    ):
        """The switch travels from the record on disk to the posture stamped.

        The record lives one directory per identity, so a launch that resolved
        a different name than the operator's writer used would find no record
        at all — every switch off, sandboxed. The identity is pinned here so
        both ends name one directory.
        """
        identity = pin_identity(monkeypatch)
        write_control_context(data_root, target="standin", switch={"standin": "asked"})

        stamp = host_executor._launch_posture("standin")

        state_dir = state_dir_under(data_root)
        assert state_dir.name == identity
        assert (state_dir / control_context.RECORD_FILENAME).exists()
        assert stamp == posture_store.launch_posture_stamp("standin", posture_store.POSTURE_WRITES)

    def test_an_unknowable_target_is_pinned_sandboxed_everywhere(self, data_root):
        """No machine to ask the verdict about: every target, sandboxed."""
        write_control_context(data_root, target="standin", switch={"standin": "asked"})

        assert host_executor._launch_posture(None) == "*=sandbox"

    @pytest.mark.usefixtures("data_root")
    def test_an_unreachable_verdict_fails_closed(self, monkeypatch):
        """Every way of not being able to answer costs a readwrite run its writes."""

        def boom(*args, **kwargs):
            raise RuntimeError("verdict exploded")

        monkeypatch.setattr(posture_store, "write_verdict", boom)

        assert host_executor._launch_posture("standin") == "standin=sandbox"


class TestASwitchAgainstARunningSandbox:
    """Off reaches the running sandbox at its next write; on reaches only the next run.

    The sandbox's own reference monitor asks
    :func:`~osprey_connectors.posture_store.write_verdict` on every write, with
    the launch pin in force — :data:`~osprey_connectors.posture_store.LAUNCH_POSTURE_ENV_VAR`
    stamped from :func:`_launch_posture` exactly as the executor stamps it.
    """

    @pytest.fixture(autouse=True)
    def _tier(self, monkeypatch):
        monkeypatch.setattr(posture, "control_system_section", lambda: dict(ARMED_TIER))

    @staticmethod
    def _launch(monkeypatch, target: str) -> str:
        """Launch a run on *target*: compute the pin and stamp it as the sandbox sees it."""
        monkeypatch.delenv(posture_store.LAUNCH_POSTURE_ENV_VAR, raising=False)
        posture_store.invalidate_cache()
        stamp = host_executor._launch_posture(target)
        monkeypatch.setenv(posture_store.LAUNCH_POSTURE_ENV_VAR, stamp)
        posture_store.invalidate_cache()
        return stamp

    @staticmethod
    def _next_write(target: str) -> posture_store.WriteVerdict:
        posture_store.invalidate_cache()
        control_context.invalidate_cache()
        return posture_store.write_verdict(ARMED_TIER, target)

    def test_an_off_record_refuses_the_next_write_of_a_running_sandbox(
        self, data_root, monkeypatch
    ):
        write_control_context(data_root, target="standin", switch={"standin": "asked"})
        assert self._launch(monkeypatch, "standin") == "standin=writes"
        assert self._next_write("standin").allowed is True

        turn_switch("standin", "off")

        verdict = self._next_write("standin")
        assert verdict.allowed is False
        assert verdict.reason == "switch_off"

    def test_turning_it_on_reaches_only_the_next_run(self, data_root, monkeypatch):
        write_control_context(data_root, target="standin", switch={"va": "asked"})
        assert self._launch(monkeypatch, "standin") == "standin=sandbox"

        turn_switch("standin", "asked")

        # The run already going stays held off by its pin ...
        verdict = self._next_write("standin")
        assert verdict.allowed is False
        assert verdict.reason == "launch_pinned"
        # ... and the next run is launched with writes, and writes.
        assert self._launch(monkeypatch, "standin") == "standin=writes"
        assert self._next_write("standin").allowed is True


class TestMarkerCarriesThePin:
    """The in-flight marker states what the run launched under."""

    def test_the_marker_record_carries_the_launch_posture(self, tmp_path, monkeypatch):
        # Arrange
        from osprey.mcp_server.control_system import target_state

        root = tmp_path / "var" / "agent_data"
        monkeypatch.setattr(target_state, "resolve_shared_data_root", lambda: root)
        monkeypatch.delenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, raising=False)
        (root / target_state.STATE_DIR_NAME).mkdir(parents=True)

        # Act
        with host_executor._in_flight_marker("standin", "standin=sandbox"):
            live = target_state.in_flight_executions()

        # Assert
        assert len(live) == 1
        assert live[0]["target"] == "standin"
        assert live[0]["launch_posture"] == "standin=sandbox"
        assert target_state.in_flight_executions() == []


# ---------------------------------------------------------------------------
# The name is a wire contract
# ---------------------------------------------------------------------------


def test_the_stamp_is_not_pinnable_from_a_server_spec():
    """A spec that could set it could hand a narrowed run the writes posture."""
    from osprey.registry.mcp import NON_PINNABLE_AUDIT_MARKERS

    assert posture_store.LAUNCH_POSTURE_ENV_VAR == "OSPREY_LAUNCH_POSTURE"
    assert posture_store.LAUNCH_POSTURE_ENV_VAR in NON_PINNABLE_AUDIT_MARKERS


def test_the_executor_and_the_store_spell_one_name():
    """Two modules, one environment variable — imported, so it cannot drift."""
    assert host_executor.ENV_LAUNCH_POSTURE is posture_store.LAUNCH_POSTURE_ENV_VAR
    # Never cleared with the routing stamp: absence would read as "unpinned",
    # and the unstamped run is the one that needs the pin most.
    assert host_executor.ENV_LAUNCH_POSTURE not in host_executor._STAMP_ENV_NAMES
