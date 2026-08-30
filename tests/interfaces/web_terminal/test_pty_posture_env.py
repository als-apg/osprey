"""Pool env-fingerprint tests: an environment change always reaches the child.

A PTY child's environment is fixed at ``execvp`` time, so anything delivered
through it can only be changed by killing the child and spawning a new one.
:meth:`PtyRegistry.get_or_create_session` used to hand a warm pooled entry
straight back after an LRU bump, which meant a caller could believe it had
launched a child under an environment that child never saw.

**The per-target write posture is no longer one of those things.** It is
recorded in the posture store and read live by every write-time gate, so a
narrowing lands on a session already mid-conversation and the spawn seams stamp
only the anchors a child needs to find that store — the key and the agent-data
root, never ``OSPREY_EXECUTION_MODE`` (pinned in ``test_posture_source_pin.py``,
which owns the spawn-seam harness). What is left here is the general backstop,
and it still matters: a deployment-wide readonly marker arriving through
``hooks_env``, a rotated panel token, or any privilege-bearing name a later
change adds must reach the child rather than being reattached around.

These tests assert the *child's own environment*, not the registry's
bookkeeping: every spawned child writes what it sees in ``OSPREY_EXECUTION_MODE``
into a file, one line per exec, so the file is a durable record of what each
generation of the child actually ran under. A test that only asked the registry
what it thinks it spawned could not have caught the bug that motivated this
work.

The two properties under test:

* **Safety** — an environment change never fails to reach the child. A warm
  entry whose fingerprint differs from the caller's is terminated and respawned.
* **Liveness** — a mere reconnect never kills a running session. The names that
  legitimately differ between two connections to one session
  (:data:`POOL_FINGERPRINT_EXCLUDED_ENV`) are excluded from the fingerprint.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from osprey.interfaces.web_terminal.pty_manager import (
    EMPTY_ENV_FINGERPRINT,
    POOL_FINGERPRINT_EXCLUDED_ENV,
    PtyRegistry,
    env_fingerprint,
)

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="PTY not available on Windows")


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #


def _reporting_command(report: Path) -> list[str]:
    """A child that records the execution-mode marker it was handed, then waits.

    Appends one line per exec — ``readonly``, ``writes``, or ``<unset>`` — so a
    respawn under the same key leaves two lines and a reattach leaves one. The
    ``exec sleep`` keeps the process running long enough to be reattached (and
    to be observed dying when it is replaced).
    """
    return [
        "/bin/sh",
        "-c",
        f'printf "%s\\n" "${{OSPREY_EXECUTION_MODE:-<unset>}}" >> "{report}"; exec sleep 30',
    ]


def _child_env_lines(report: Path, expected: int, timeout: float = 5.0) -> list[str]:
    """Wait for ``expected`` child generations to report, then return their lines."""
    deadline = time.monotonic() + timeout
    lines: list[str] = []
    while time.monotonic() < deadline:
        if report.exists():
            lines = report.read_text().split()
            if len(lines) >= expected:
                return lines
        time.sleep(0.02)
    raise AssertionError(
        f"child env report never reached {expected} line(s) within {timeout}s; got {lines!r}"
    )


@pytest.fixture
def registry():
    """A registry whose real children are always cleaned up."""
    reg = PtyRegistry(max_background=5)
    try:
        yield reg
    finally:
        reg.cleanup_all()


@pytest.fixture
def report(tmp_path: Path) -> Path:
    return tmp_path / "child_env.txt"


#: A per-connection env overlay of the shape ``_build_extra_env`` produces:
#: constants, the panel token, and the three names that vary per connection.
#: ``execution_mode`` stands for a *deployment-wide* marker — the kind
#: ``hooks_env`` may inject — not for a session posture, which no longer
#: travels in the environment at all.
def _connection_env(
    *,
    execution_mode: str | None = None,
    session_id: str | None = None,
    telemetry_id: str | None = None,
    started_at: str = "2026-08-23T00:00:00+00:00",
) -> dict[str, str]:
    env = {
        "OSPREY_WEB_UX": "expert",
        "OSPREY_PANEL_TOKEN": "panel-token-constant",
    }
    if session_id:
        env["OSPREY_SESSION_ID"] = session_id
    if telemetry_id:
        env["OSPREY_TELEMETRY_SESSION_ID"] = telemetry_id
        env["OSPREY_TELEMETRY_SESSION_START"] = started_at
    if execution_mode:
        env["OSPREY_EXECUTION_MODE"] = execution_mode
    return env


# --------------------------------------------------------------------------- #
# env_fingerprint
# --------------------------------------------------------------------------- #


class TestEnvFingerprint:
    def test_none_and_empty_agree(self):
        assert env_fingerprint(None) == env_fingerprint({}) == EMPTY_ENV_FINGERPRINT

    def test_execution_mode_changes_the_fingerprint(self):
        """A privilege-bearing name must be inside the fingerprint's scope."""
        assert env_fingerprint(_connection_env()) != env_fingerprint(
            _connection_env(execution_mode="readonly")
        )

    def test_two_execution_modes_differ(self):
        assert env_fingerprint(_connection_env(execution_mode="readonly")) != env_fingerprint(
            _connection_env(execution_mode="writes")
        )

    def test_insertion_order_is_irrelevant(self):
        first = {"A": "1", "B": "2", "OSPREY_EXECUTION_MODE": "readonly"}
        second = {"OSPREY_EXECUTION_MODE": "readonly", "B": "2", "A": "1"}
        assert env_fingerprint(first) == env_fingerprint(second)

    def test_excluded_names_do_not_change_the_fingerprint(self):
        """The three per-connection names are outside the fingerprint's scope."""
        base = _connection_env(execution_mode="readonly")
        varied = dict(base)
        for name in POOL_FINGERPRINT_EXCLUDED_ENV:
            varied[name] = "something-else-entirely"
        assert env_fingerprint(varied) == env_fingerprint(base)

    def test_unlisted_new_name_is_covered_by_default(self):
        """Scope is a deny list: a name nobody anticipated still counts."""
        base = _connection_env()
        widened = {**base, "OSPREY_SOME_FUTURE_PRIVILEGE": "granted"}
        assert env_fingerprint(widened) != env_fingerprint(base)

    def test_name_value_boundary_cannot_be_smudged(self):
        """``{"AB": "C"}`` and ``{"A": "BC"}`` must not collide."""
        assert env_fingerprint({"AB": "C"}) != env_fingerprint({"A": "BC"})

    def test_does_not_expose_values(self):
        """The digest is opaque — a token cannot be read back out of it."""
        digest = env_fingerprint({"OSPREY_PANEL_TOKEN": "super-secret-token"})
        assert "super-secret-token" not in digest
        assert len(digest) == 64


# --------------------------------------------------------------------------- #
# The child's actual environment
# --------------------------------------------------------------------------- #


class TestAnEnvChangeReachesTheChild:
    def test_a_changed_marker_respawns_the_child(self, registry, report):
        """The whole point: after the marker changes the *child* runs readonly."""
        command = _reporting_command(report)

        first, reused = registry.get_or_create_session(
            "sess-1", command, 24, 80, extra_env=_connection_env(telemetry_id="sess-1")
        )
        assert reused is False
        assert _child_env_lines(report, 1) == ["<unset>"]

        # The deployment narrows this session's launch environment; the pool
        # must not hand back the warm child that never saw the marker.
        second, reused = registry.get_or_create_session(
            "sess-1",
            command,
            24,
            80,
            extra_env=_connection_env(execution_mode="readonly", telemetry_id="sess-1"),
        )

        assert reused is False
        assert second is not first
        # The child itself — not the store — reports the new posture.
        assert _child_env_lines(report, 2) == ["<unset>", "readonly"]
        assert second.is_alive

    def test_mismatch_terminates_the_stale_child(self, registry, report):
        """The replaced child is killed, not orphaned beside its replacement."""
        command = _reporting_command(report)
        stale, _ = registry.get_or_create_session(
            "sess-1", command, 24, 80, extra_env=_connection_env()
        )
        _child_env_lines(report, 1)
        assert stale.is_alive

        fresh, reused = registry.get_or_create_session(
            "sess-1", command, 24, 80, extra_env=_connection_env(execution_mode="readonly")
        )

        assert reused is False
        assert stale.is_alive is False
        assert registry.get_session("sess-1") is fresh
        assert fresh.is_alive

    def test_dropping_the_marker_also_reaches_the_child(self, registry, report):
        """Both directions: a readonly child is replaced for a writable one."""
        command = _reporting_command(report)
        registry.get_or_create_session(
            "sess-1", command, 24, 80, extra_env=_connection_env(execution_mode="readonly")
        )
        assert _child_env_lines(report, 1) == ["readonly"]

        _, reused = registry.get_or_create_session(
            "sess-1", command, 24, 80, extra_env=_connection_env()
        )

        assert reused is False
        assert _child_env_lines(report, 2) == ["readonly", "<unset>"]

    def test_two_pooled_sessions_hold_different_markers(self, registry, tmp_path):
        """The overlay is per session, and two live children can disagree."""
        sandboxed_report = tmp_path / "sandboxed.txt"
        writable_report = tmp_path / "writable.txt"

        sandboxed, _ = registry.get_or_create_session(
            "sess-sandbox",
            _reporting_command(sandboxed_report),
            24,
            80,
            extra_env=_connection_env(execution_mode="readonly", telemetry_id="sess-sandbox"),
        )
        writable, _ = registry.get_or_create_session(
            "sess-writes",
            _reporting_command(writable_report),
            24,
            80,
            extra_env=_connection_env(telemetry_id="sess-writes"),
        )

        assert _child_env_lines(sandboxed_report, 1) == ["readonly"]
        assert _child_env_lines(writable_report, 1) == ["<unset>"]
        assert sandboxed.is_alive and writable.is_alive

        # Reattaching either one leaves both children exactly as they were.
        again, reused = registry.get_or_create_session(
            "sess-sandbox",
            _reporting_command(sandboxed_report),
            24,
            80,
            extra_env=_connection_env(execution_mode="readonly", telemetry_id="sess-sandbox"),
        )
        assert reused is True
        assert again is sandboxed
        assert writable.is_alive
        assert sandboxed_report.read_text().split() == ["readonly"]


# --------------------------------------------------------------------------- #
# Liveness — a reconnect must not kill a running session
# --------------------------------------------------------------------------- #


class TestWarmReuse:
    def test_identical_env_keeps_the_warm_child(self, registry, report):
        command = _reporting_command(report)
        env = _connection_env(execution_mode="readonly", telemetry_id="sess-1")

        first, _ = registry.get_or_create_session("sess-1", command, 24, 80, extra_env=env)
        _child_env_lines(report, 1)

        second, reused = registry.get_or_create_session(
            "sess-1", command, 24, 80, extra_env=dict(env)
        )

        assert reused is True
        assert second is first
        # No second exec — the original child is still the one running.
        assert report.read_text().split() == ["readonly"]

    def test_per_connection_names_do_not_force_a_respawn(self, registry, report):
        """The scope decision, pinned.

        A reconnect to a live session arrives with a fresh
        ``OSPREY_TELEMETRY_SESSION_START`` (and ``switch_session`` reaches the
        pool with no telemetry names at all, while the spawn path passes them),
        so a fingerprint over the raw dict would kill a running agent on every
        reattach. Those three names are excluded; the child survives.
        """
        command = _reporting_command(report)
        first, _ = registry.get_or_create_session(
            "sess-1",
            command,
            24,
            80,
            extra_env=_connection_env(
                execution_mode="readonly",
                telemetry_id="sess-1",
                started_at="2026-08-23T00:00:00+00:00",
            ),
        )
        _child_env_lines(report, 1)

        # A later reconnect: new timestamp, session id now known, and the
        # switch_session shape (no telemetry names) — same posture.
        reattached, reused = registry.get_or_create_session(
            "sess-1",
            command,
            24,
            80,
            extra_env=_connection_env(execution_mode="readonly", session_id="sess-1"),
        )

        assert reused is True
        assert reattached is first
        assert report.read_text().split() == ["readonly"]

    def test_dead_child_respawns_even_with_a_matching_fingerprint(self, registry, tmp_path):
        """Liveness must not resurrect a corpse: a dead entry still respawns."""
        report = tmp_path / "child_env.txt"
        env = _connection_env(execution_mode="readonly")
        first, _ = registry.get_or_create_session(
            "sess-1", ["/bin/sh", "-c", "exit 0"], 24, 80, extra_env=env
        )
        deadline = time.monotonic() + 5
        while first.is_alive and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not first.is_alive

        second, reused = registry.get_or_create_session(
            "sess-1", _reporting_command(report), 24, 80, extra_env=dict(env)
        )
        assert reused is False
        assert _child_env_lines(report, 1) == ["readonly"]
        assert second.is_alive


# --------------------------------------------------------------------------- #
# Bookkeeping: rekey, terminate, and unrecorded entries
# --------------------------------------------------------------------------- #


class TestFingerprintBookkeeping:
    def test_rekey_carries_the_fingerprint(self, registry, report):
        """A renamed session keeps its posture and stays reusable under the new key."""
        command = _reporting_command(report)
        env = _connection_env(execution_mode="readonly", telemetry_id="temp-key")
        session, _ = registry.get_or_create_session("temp-key", command, 24, 80, extra_env=env)
        assert _child_env_lines(report, 1) == ["readonly"]

        registry.rekey_session("temp-key", "real-uuid")

        assert registry._env_fingerprints.get("real-uuid") == env_fingerprint(env)
        assert "temp-key" not in registry._env_fingerprints

        reattached, reused = registry.get_or_create_session(
            "real-uuid",
            command,
            24,
            80,
            extra_env=_connection_env(execution_mode="readonly", session_id="real-uuid"),
        )
        assert reused is True
        assert reattached is session
        assert report.read_text().split() == ["readonly"]

    def test_rekeyed_session_still_respawns_on_a_posture_change(self, registry, report):
        """The carried fingerprint is a real comparison, not a rubber stamp."""
        command = _reporting_command(report)
        registry.get_or_create_session(
            "temp-key", command, 24, 80, extra_env=_connection_env(execution_mode="readonly")
        )
        _child_env_lines(report, 1)
        registry.rekey_session("temp-key", "real-uuid")

        _, reused = registry.get_or_create_session(
            "real-uuid", command, 24, 80, extra_env=_connection_env()
        )

        assert reused is False
        assert _child_env_lines(report, 2) == ["readonly", "<unset>"]

    def test_terminate_forgets_the_fingerprint(self, registry, report):
        """Terminate, then respawn under a new overlay: no stale comparison."""
        command = _reporting_command(report)
        registry.get_or_create_session("sess-1", command, 24, 80, extra_env=_connection_env())
        _child_env_lines(report, 1)

        registry.terminate_session("sess-1")
        assert "sess-1" not in registry._env_fingerprints
        assert registry.get_session("sess-1") is None

        _, reused = registry.get_or_create_session(
            "sess-1", command, 24, 80, extra_env=_connection_env(execution_mode="readonly")
        )
        assert reused is False
        assert _child_env_lines(report, 2) == ["<unset>", "readonly"]

    def test_evicted_key_forgets_its_fingerprint(self, registry, tmp_path):
        """LRU eviction must not leave a fingerprint behind for the next tenant."""
        small = PtyRegistry(max_background=2)
        try:
            for name in ("a", "b"):
                small.get_or_create_session(
                    name,
                    _reporting_command(tmp_path / f"{name}.txt"),
                    24,
                    80,
                    extra_env=_connection_env(execution_mode="readonly"),
                )
            small.get_or_create_session(
                "c", _reporting_command(tmp_path / "c.txt"), 24, 80, extra_env=_connection_env()
            )
            assert "a" not in small._sessions
            assert "a" not in small._env_fingerprints
        finally:
            small.cleanup_all()

    def test_cleanup_all_clears_the_fingerprints(self, registry, tmp_path):
        registry.get_or_create_session(
            "sess-1",
            _reporting_command(tmp_path / "a.txt"),
            24,
            80,
            extra_env=_connection_env(execution_mode="readonly"),
        )
        registry.cleanup_all()
        assert registry._env_fingerprints == {}

    def test_unrecorded_entry_respawns_when_a_posture_is_requested(self):
        """An entry that never came through the spawn path proves nothing.

        Nothing in production inserts into ``_sessions`` without recording a
        fingerprint, so this is the belt-and-braces case: the only thing that
        may be assumed about such a child is the base environment (no overlay,
        no sandbox marker), and a caller asking for a sandbox therefore gets a
        respawn rather than a child of unproven posture.
        """
        registry = PtyRegistry(max_background=3)
        warm = MagicMock()
        warm.is_alive = True
        registry._sessions["sess-1"] = warm

        with patch.object(registry, "_spawn_session") as spawn:
            spawn.return_value = MagicMock(is_alive=True)
            session, reused = registry.get_or_create_session(
                "sess-1", ["cmd"], 24, 80, extra_env=_connection_env(execution_mode="readonly")
            )

        assert reused is False
        assert session is spawn.return_value
        warm.terminate.assert_called_once()

    def test_unrecorded_entry_is_reused_by_a_caller_with_no_overlay(self):
        """The other half of that assumption, stated explicitly."""
        registry = PtyRegistry(max_background=3)
        warm = MagicMock()
        warm.is_alive = True
        registry._sessions["sess-1"] = warm

        session, reused = registry.get_or_create_session("sess-1", ["cmd"], 24, 80)

        assert reused is True
        assert session is warm
        warm.terminate.assert_not_called()
