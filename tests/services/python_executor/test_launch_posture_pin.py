"""The run-level pin: a widen never reaches a run that started narrow.

The per-(session, target) posture store is read at WRITE time, which is what
lets an operator narrow a target for a session that is already mid-conversation.
That direction is the point of the feature. The other direction is a hazard: a
script is already running inside the sandbox, and widening the store under it
would hand it write access to a machine nobody re-consented to while it works.

So the executor pins what it launched under. ``OSPREY_LAUNCH_POSTURE`` is
stamped into the sandbox environment and recorded in the in-flight marker, and
inside that sandbox the store rule becomes the AND of the pin and the live
store:

* the store NARROWS -> the store read refuses on the next write (immediate);
* the store WIDENS -> the pin still refuses (the run stays narrow until it ends).

The 409 the posture route raises while a marker is live is a courtesy on top of
this, not the barrier.

This module also pins the case the session-posture clamp cannot cover: a session
whose control target is unknowable. ``audit.posture.posture()`` degrades to the
environment answer there, and no spawn site stamps a per-target narrowing into
the environment, so the executor's deployment gate has to ask the store itself —
most restrictively, because it cannot say which machine the run is about.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from osprey.mcp_server.python_executor import executor as host_executor
from osprey.mcp_server.python_executor.tools import _execution_gates as gates
from osprey_connectors import session_store

pytestmark = pytest.mark.unit

#: A session key of the shape the web terminal writes.
SESSION_KEY = "4f1c2a7e-0000-4000-8000-000000000042"

#: A deployment armed for every target, so the ceiling never decides anything
#: here — this module is about the store and the pin.
ARMED = {
    "type": "epics",
    "writes_enabled": True,
    "connector": {
        "epics": {"prefix": "RING:"},
        "virtual_accelerator": {"prefix": "VA:"},
        "live_standin": {"prefix": "SIM:"},
    },
}


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    """A scratch agent-data root, stamped, with every posture env name cleared."""
    monkeypatch.setenv(session_store.AGENT_DATA_ROOT_ENV_VAR, str(tmp_path))
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)
    monkeypatch.delenv(session_store.LAUNCH_POSTURE_ENV_VAR, raising=False)
    monkeypatch.setenv("OSPREY_POSTURE_SESSION", SESSION_KEY)
    session_store.invalidate_cache()
    yield tmp_path
    session_store.invalidate_cache()


def write_store(root: Path, payload) -> None:
    """Write the posture store both the gate and the sandbox read."""
    path = root / session_store.STATE_DIR_NAME / session_store.STORE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    session_store.invalidate_cache()


# ---------------------------------------------------------------------------
# The stamp's wire format
# ---------------------------------------------------------------------------


class TestStampFormat:
    """``<target>=<posture>``, and only ``sandbox`` ever does anything."""

    def test_the_stamp_is_one_target_equals_posture_pair(self):
        assert session_store.launch_posture_stamp("standin", "sandbox") == "standin=sandbox"
        assert session_store.launch_posture_stamp("va", "writes") == "va=writes"

    def test_no_target_is_spelled_as_every_target(self):
        """A run that cannot name its machine is pinned for all of them."""
        stamp = session_store.launch_posture_stamp(None, "sandbox")

        assert stamp == f"{session_store.LAUNCH_POSTURE_ALL_TARGETS}=sandbox"
        assert set(session_store.parse_launch_posture(stamp)) == set(session_store.CONTROL_TARGETS)

    def test_writes_narrows_nothing(self):
        """Recorded for the marker, inert for the decision — nothing may widen."""
        assert session_store.parse_launch_posture("standin=writes") == {}
        assert session_store.parse_launch_posture("*=writes") == {}

    @pytest.mark.parametrize(
        "raw",
        [None, "", "   ", "standin", "=sandbox", "standin=locked", "standin sandbox"],
        ids=[
            "absent",
            "empty",
            "blank",
            "no-separator",
            "no-target",
            "unknown-posture",
            "space-separated",
        ],
    )
    def test_an_unparseable_stamp_is_inert(self, raw):
        """Every process that is not an executor sandbox is untouched by this term."""
        assert session_store.parse_launch_posture(raw) == {}


# ---------------------------------------------------------------------------
# The pin inside the sandbox
# ---------------------------------------------------------------------------


class TestLaunchPinAnddedWithTheStore:
    """``store_permits`` / ``effective_writes`` inside a launched sandbox."""

    def test_a_run_launched_narrow_stays_narrow_after_a_widen(self, data_root, monkeypatch):
        """The whole point: the operator widened, the running script did not follow."""
        # Arrange — the run launched while standin was narrowed...
        monkeypatch.setenv(session_store.LAUNCH_POSTURE_ENV_VAR, "standin=sandbox")
        # ...and the operator has since turned writes back on (an entry that
        # is gone IS the writes posture — nothing is ever stored to widen).
        write_store(data_root, {SESSION_KEY: {"live": "sandbox"}})

        # Act / Assert
        assert session_store.store_permits(SESSION_KEY, "standin") is False
        assert session_store.effective_writes(ARMED, SESSION_KEY, "standin") is False

    def test_a_run_launched_writes_refuses_immediately_after_a_narrow(self, data_root, monkeypatch):
        """The other direction lands at once — the store read follows the operator."""
        # Arrange
        monkeypatch.setenv(session_store.LAUNCH_POSTURE_ENV_VAR, "standin=writes")
        write_store(data_root, {SESSION_KEY: {"standin": "sandbox"}})

        # Act / Assert
        assert session_store.store_permits(SESSION_KEY, "standin") is False
        assert session_store.effective_writes(ARMED, SESSION_KEY, "standin") is False

    def test_a_run_launched_writes_still_writes_while_nothing_narrows_it(
        self, data_root, monkeypatch
    ):
        """The pin refuses; it never grants, and it never refuses on its own."""
        # Arrange
        monkeypatch.setenv(session_store.LAUNCH_POSTURE_ENV_VAR, "standin=writes")
        write_store(data_root, {SESSION_KEY: {"live": "sandbox"}})

        # Act / Assert
        assert session_store.store_permits(SESSION_KEY, "standin") is True
        assert session_store.effective_writes(ARMED, SESSION_KEY, "standin") is True

    def test_the_pin_names_one_target_and_leaves_the_others_alone(self, data_root, monkeypatch):
        """A narrowed stand-in does not sandbox the virtual accelerator."""
        # Arrange
        monkeypatch.setenv(session_store.LAUNCH_POSTURE_ENV_VAR, "standin=sandbox")
        write_store(data_root, {})

        # Act / Assert
        assert session_store.store_permits(SESSION_KEY, "va") is True
        assert session_store.store_permits(SESSION_KEY, "standin") is False

    def test_an_all_targets_pin_covers_every_target(self, data_root, monkeypatch):
        """The unknowable-target launch: most restrictive, on every machine."""
        # Arrange
        monkeypatch.setenv(session_store.LAUNCH_POSTURE_ENV_VAR, "*=sandbox")
        write_store(data_root, {})

        # Act / Assert
        for target in session_store.CONTROL_TARGETS:
            assert session_store.store_permits(SESSION_KEY, target) is False
        assert session_store.store_permits(SESSION_KEY, None) is False

    def test_the_pin_holds_without_a_session_key(self, data_root, monkeypatch):
        """It is a fact about the RUN, not about the session.

        The store clause is skipped with no key — nothing addressed the session —
        but the run still launched narrow, and a session key that vanished from
        the environment mid-run must not be a way to shed the pin.
        """
        # Arrange
        monkeypatch.delenv("OSPREY_POSTURE_SESSION", raising=False)
        monkeypatch.setenv(session_store.LAUNCH_POSTURE_ENV_VAR, "standin=sandbox")
        write_store(data_root, {})

        # Act / Assert
        assert session_store.store_permits(None, "standin") is False
        assert session_store.store_permits(None, "va") is True

    def test_no_stamp_leaves_the_store_rule_exactly_as_it_was(self, data_root):
        """Every process outside a sandbox: the term is not merely inert, it is absent."""
        # Arrange
        write_store(data_root, {SESSION_KEY: {"standin": "sandbox"}})

        # Act / Assert
        assert session_store.launch_permits("standin") is True
        assert session_store.store_permits(SESSION_KEY, "standin") is False
        assert session_store.store_permits(SESSION_KEY, "va") is True

    def test_the_pin_refuses_before_the_store_is_read(self, data_root, monkeypatch):
        """A narrow run does not need a readable store to keep refusing.

        The pin is one environment read, deliberately ahead of the file read, so
        a sandbox on a store that has become unreadable still honours the
        narrowing it launched under.
        """
        # Arrange — a store path that cannot resolve at all.
        monkeypatch.setenv(session_store.AGENT_DATA_ROOT_ENV_VAR, "   ")
        monkeypatch.setattr(
            session_store, "resolve_shared_data_root", lambda: (_ for _ in ()).throw(OSError())
        )
        monkeypatch.setenv(session_store.LAUNCH_POSTURE_ENV_VAR, "standin=sandbox")
        session_store.invalidate_cache()

        # Act / Assert
        assert session_store.store_path() is None
        assert session_store.store_permits(SESSION_KEY, "standin") is False


# ---------------------------------------------------------------------------
# What the executor stamps, and what the marker records
# ---------------------------------------------------------------------------


class TestExecutorStampsThePin:
    """``_launch_posture`` reads the store once, at launch."""

    def test_an_unnarrowed_target_launches_writes(self, data_root):
        write_store(data_root, {})

        assert host_executor._launch_posture("standin") == "standin=writes"

    def test_a_narrowed_target_launches_sandboxed(self, data_root):
        write_store(data_root, {SESSION_KEY: {"standin": "sandbox"}})

        assert host_executor._launch_posture("standin") == "standin=sandbox"

    def test_an_unknowable_target_takes_the_most_restrictive_entry(self, data_root):
        """One narrowing anywhere in the session pins a run that names no target."""
        write_store(data_root, {SESSION_KEY: {"live": "sandbox"}})

        assert host_executor._launch_posture(None) == "*=sandbox"

    def test_an_unknowable_target_on_an_unnarrowed_session_launches_writes(self, data_root):
        write_store(data_root, {"another-session": {"live": "sandbox"}})

        assert host_executor._launch_posture(None) == "*=writes"

    def test_an_unreadable_store_fails_closed(self, data_root, monkeypatch):
        """Every way of not being able to answer costs a readwrite run its writes."""

        def boom(*args, **kwargs):
            raise RuntimeError("store exploded")

        monkeypatch.setattr(session_store, "store_permits", boom)

        assert host_executor._launch_posture("standin") == "standin=sandbox"


class TestMarkerCarriesThePin:
    """The in-flight marker states what the run launched under."""

    def test_the_marker_record_carries_the_launch_posture(self, tmp_path, monkeypatch):
        # Arrange
        from osprey.mcp_server.control_system import target_state

        root = tmp_path / "var" / "agent_data"
        monkeypatch.setattr(target_state, "resolve_shared_data_root", lambda: root)
        monkeypatch.delenv(session_store.AGENT_DATA_ROOT_ENV_VAR, raising=False)
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
# The executor's own gate
# ---------------------------------------------------------------------------


class TestDeploymentGateStoreTerm:
    """``enforce_deployment_writes_gate`` asks the store, not only the config."""

    @staticmethod
    def _arm_the_deployment(monkeypatch):
        """Make the deployment half say yes, so only the store can refuse."""
        monkeypatch.setattr(
            "osprey.services.python_executor.execution.control.get_execution_control_config",
            lambda target=None: None,
        )

    def test_an_unknowable_target_refuses_a_narrowed_session(self, data_root, monkeypatch):
        """The routed case: ``posture()`` degrades to the env answer, this does not.

        With no controls-server record there is no session target to resolve, so
        the session clamp sees the environment's writes posture and lets the run
        through. The store still holds a narrowing for this session, and the most
        restrictive rule is what refuses it.
        """
        # Arrange
        self._arm_the_deployment(monkeypatch)
        write_store(data_root, {SESSION_KEY: {"standin": "sandbox"}})

        # Act / Assert
        with pytest.raises(Exception) as excinfo:
            gates.enforce_deployment_writes_gate("readwrite", None)

        message = str(excinfo.value)
        assert "Writes are off" in message
        assert "control-target chip in the header" in message
        assert "could not be identified" in message

    def test_a_named_narrowed_target_is_refused_and_names_itself(self, data_root, monkeypatch):
        # Arrange
        self._arm_the_deployment(monkeypatch)
        write_store(data_root, {SESSION_KEY: {"standin": "sandbox"}})

        # Act / Assert
        with pytest.raises(Exception) as excinfo:
            gates.enforce_deployment_writes_gate("readwrite", "standin")

        message = str(excinfo.value)
        assert "'standin'" in message
        assert "control-target chip in the header" in message

    def test_an_unnarrowed_target_is_untouched(self, data_root, monkeypatch):
        # Arrange
        self._arm_the_deployment(monkeypatch)
        write_store(data_root, {SESSION_KEY: {"standin": "sandbox"}})

        # Act / Assert — no raise
        gates.enforce_deployment_writes_gate("readwrite", "va")

    def test_a_readonly_run_is_never_gated_here(self, data_root, monkeypatch):
        # Arrange
        self._arm_the_deployment(monkeypatch)
        write_store(data_root, {SESSION_KEY: "sandbox"})

        # Act / Assert — no raise
        gates.enforce_deployment_writes_gate("readonly", None)

    def test_a_session_that_narrowed_nothing_is_untouched(self, data_root, monkeypatch):
        # Arrange
        self._arm_the_deployment(monkeypatch)
        write_store(data_root, {"another-session": {"standin": "sandbox"}})

        # Act / Assert — no raise
        gates.enforce_deployment_writes_gate("readwrite", None)

    def test_an_unreadable_store_does_not_wedge_every_run(self, data_root, monkeypatch):
        """This gate degrades; the sandbox's reference monitor is the barrier."""
        # Arrange
        self._arm_the_deployment(monkeypatch)

        def boom(*args, **kwargs):
            raise RuntimeError("store exploded")

        monkeypatch.setattr("osprey_connectors.session_store.store_permits", boom)

        # Act / Assert — no raise
        gates.enforce_deployment_writes_gate("readwrite", "standin")


# ---------------------------------------------------------------------------
# The name is a wire contract
# ---------------------------------------------------------------------------


def test_the_stamp_is_not_pinnable_from_a_server_spec():
    """A spec that could set it could hand a narrowed run the writes posture."""
    from osprey.registry.mcp import NON_PINNABLE_AUDIT_MARKERS

    assert session_store.LAUNCH_POSTURE_ENV_VAR == "OSPREY_LAUNCH_POSTURE"
    assert session_store.LAUNCH_POSTURE_ENV_VAR in NON_PINNABLE_AUDIT_MARKERS


def test_the_executor_and_the_store_spell_one_name():
    """Two modules, one environment variable — imported, so it cannot drift."""
    assert host_executor.ENV_LAUNCH_POSTURE is session_store.LAUNCH_POSTURE_ENV_VAR
    # Never cleared with the routing stamp: absence would read as "unpinned",
    # and the unstamped run is the one that needs the pin most.
    assert host_executor.ENV_LAUNCH_POSTURE not in host_executor._STAMP_ENV_NAMES
