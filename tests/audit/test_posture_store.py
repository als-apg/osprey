"""Tests for the record-backed half of :func:`osprey.audit.posture.posture`.

``posture()`` used to be one line of environment: a child spawned with
``OSPREY_EXECUTION_MODE=readonly`` was sandboxed and everything else was not.
An operator narrows ONE control target, which no environment variable can
express — a deployment sandboxed on the live machine must still be able to
write to the virtual accelerator, and setting the variable would clamp both.

So the answer gains a second source, and these tests pin the seam between them:

* **The environment first.** ``OSPREY_EXECUTION_MODE == "readonly"`` is a
  sandbox and short-circuits: the record holds narrowings only, so reading it
  could not change that answer.
* **Otherwise the record, indexed by the target this process's writes are
  about.** The process learns that target from the ``OSPREY_CONTROL_TARGET``
  stamp when it carries one (the executor's sandbox subprocess), else from the
  controls server's state record.
* **The posture belongs to the deployment, not to a session.**
  ``OSPREY_POSTURE_SESSION`` is the audit session id and nothing else now: a
  process that carries none is narrowed by the same record as one that does.
* **Every failure degrades to the environment.** No record, a corrupt one, a
  target that cannot be resolved, an import that fails: each is "nothing
  narrowed here", never an exception out of a function three refusal paths
  call on every tool call.

The record can only ever narrow: an environment that already says sandbox
stays sandbox no matter what the record holds.
"""

from __future__ import annotations

import json
import os

import pytest

from osprey.audit import posture
from osprey_connectors import control_context, session_store
from tests._control_context_fixtures import write_control_context

pytestmark = pytest.mark.unit

SESSION_KEY = "11111111-2222-3333-4444-555555555555"


@pytest.fixture(autouse=True)
def agent_root(tmp_path, monkeypatch):
    """An agent-data root of our own, with every posture marker cleared.

    ``OSPREY_AGENT_DATA_ROOT`` is the one anchor both readers honour — the
    record's :func:`osprey_connectors.control_context.record_path` and the
    state file's
    :func:`osprey.mcp_server.control_system.target_state.state_dir` — so
    stamping it here is what puts the two files this module reads inside
    ``tmp_path`` instead of the developer's own deployment.

    Both caches are dropped on the way in AND on the way out: they are module
    globals keyed on file signatures, and two tests whose roots are both empty
    produce the same signature.
    """
    monkeypatch.setenv(posture.OSPREY_AGENT_DATA_ROOT, str(tmp_path))
    for marker in (
        posture.POSTURE_ENV_VAR,
        posture.POSTURE_SESSION_ENV_VAR,
        posture.CONTROL_TARGET_ENV_VAR,
    ):
        monkeypatch.delenv(marker, raising=False)
    session_store.invalidate_cache()
    posture.invalidate_session_target_cache()
    yield tmp_path
    session_store.invalidate_cache()
    posture.invalidate_session_target_cache()


def _state_dir(root):
    directory = root / session_store.STATE_DIR_NAME
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_state(root, target: str, *, owner_ppid: int | None = None, server_pid: int | None = None):
    """One controls-server state record naming *target*, owned by our parent.

    ``server_pid`` defaults to this process, which is trivially alive: a record
    whose owner is dead is residue and the resolver skips it.
    """
    pid = os.getpid() if server_pid is None else server_pid
    _state_dir(root)
    path = control_context.report_path_under(root, pid)
    path.write_text(
        json.dumps(
            {
                "target": target,
                "generation": 0,
                "server_pid": pid,
                "owner_ppid": os.getppid() if owner_ppid is None else owner_ppid,
                "targets": {},
                "children": [],
            }
        )
    )
    session_store.invalidate_cache()
    posture.invalidate_session_target_cache()
    return path


def write_record(root, posture_field, *, target: str = "live"):
    """The control-context record, ownerless, plus the cache drops these tests need.

    The payload itself is :func:`write_control_context` — one spelling of the
    record for every suite — and what is added here is the two caches this one
    reads through.
    """
    path = write_control_context(
        root, target=target, generation=0, posture=posture_field, owned_by=None
    )
    session_store.invalidate_cache()
    posture.invalidate_session_target_cache()
    return path


def in_session(monkeypatch, key: str = SESSION_KEY) -> None:
    monkeypatch.setenv(posture.POSTURE_SESSION_ENV_VAR, key)


# --------------------------------------------------------------------------
# The environment answer, and when it is the whole answer
# --------------------------------------------------------------------------


class TestEnvOnlyPaths:
    def test_nothing_narrowed_is_writes(self, agent_root):
        assert posture.posture() == posture.POSTURE_WRITES

    def test_a_readonly_marker_is_sandbox(self, agent_root, monkeypatch):
        monkeypatch.setenv(posture.POSTURE_ENV_VAR, posture.SANDBOX_MODE)
        assert posture.posture() == posture.POSTURE_SANDBOX

    @pytest.mark.parametrize("value", ["readwrite", "READONLY", "", "sandbox", "true"])
    def test_the_value_comparison_is_exact(self, agent_root, monkeypatch, value):
        """Only the exact ``readonly`` string sandboxes — unchanged by the record."""
        monkeypatch.setenv(posture.POSTURE_ENV_VAR, value)
        assert posture.posture() == posture.POSTURE_WRITES

    def test_an_env_sandbox_short_circuits_the_record(self, agent_root, monkeypatch):
        """Narrowing-only: the record can refuse writes, never grant them.

        Proven by sabotage rather than by an empty record: the lookup is
        replaced with something that raises, so a call that reached it would
        surface here instead of quietly returning the same answer.
        """

        def _explode() -> bool:  # pragma: no cover - must not run
            raise AssertionError("the record was read behind an env sandbox")

        monkeypatch.setattr(posture, "_target_is_sandboxed", _explode)
        monkeypatch.setenv(posture.POSTURE_ENV_VAR, posture.SANDBOX_MODE)
        write_state(agent_root, "live")
        write_record(agent_root, {"live": "sandbox"})

        assert posture.posture() == posture.POSTURE_SANDBOX


# --------------------------------------------------------------------------
# The record, indexed by this process's target
# --------------------------------------------------------------------------


class TestPerTargetLookup:
    def test_a_sandbox_entry_for_this_target_sandboxes(self, agent_root, monkeypatch):
        write_state(agent_root, "live")
        write_record(agent_root, {"live": "sandbox"})

        assert posture.posture() == posture.POSTURE_SANDBOX

    def test_a_sandbox_entry_for_another_target_does_not(self, agent_root, monkeypatch):
        """The whole feature: narrowing ``live`` leaves a session on ``va`` alone."""
        write_state(agent_root, "va")
        write_record(agent_root, {"live": "sandbox"})

        assert posture.posture() == posture.POSTURE_WRITES

    def test_a_bare_sandbox_narrows_this_target(self, agent_root, monkeypatch):
        """The pre-feature deployment-wide value still refuses, on every target."""
        write_state(agent_root, "standin")
        write_record(agent_root, "sandbox")

        assert posture.posture() == posture.POSTURE_SANDBOX

    def test_a_recorded_writes_entry_narrows_nothing(self, agent_root, monkeypatch):
        write_state(agent_root, "live")
        write_record(agent_root, {"live": "writes"})

        assert posture.posture() == posture.POSTURE_WRITES


# --------------------------------------------------------------------------
# The session key is an audit id, not an index
# --------------------------------------------------------------------------


class TestSessionKeyIsNotAnIndex:
    def test_a_narrowing_applies_without_a_session_key(self, agent_root, monkeypatch):
        """A dispatch worker and a CLI run answer from the same record.

        The posture is a property of the deployment now, so a process that
        belongs to no session is narrowed by it exactly like one that does.
        """
        write_state(agent_root, "live")
        write_record(agent_root, {"live": "sandbox"})

        assert posture.posture_session() is None
        assert posture.posture() == posture.POSTURE_SANDBOX

    def test_a_narrowing_applies_with_a_session_key_too(self, agent_root, monkeypatch):
        in_session(monkeypatch)
        write_state(agent_root, "live")
        write_record(agent_root, {"live": "sandbox"})

        assert posture.posture() == posture.POSTURE_SANDBOX

    def test_the_key_is_carried_verbatim_as_the_audit_session_id(self, agent_root, monkeypatch):
        in_session(monkeypatch)
        assert posture.posture_session() == SESSION_KEY

    def test_a_blank_key_is_no_key(self, agent_root, monkeypatch):
        monkeypatch.setenv(posture.POSTURE_SESSION_ENV_VAR, "   ")
        assert posture.posture_session() is None

    def test_the_key_does_not_narrow_anything_by_itself(self, agent_root, monkeypatch):
        in_session(monkeypatch)
        write_state(agent_root, "live")
        write_record(agent_root, {})

        assert posture.posture() == posture.POSTURE_WRITES


# --------------------------------------------------------------------------
# How the process learns its own target
# --------------------------------------------------------------------------


class TestTargetResolution:
    def test_the_control_target_stamp_wins_over_the_state_file(self, agent_root, monkeypatch):
        """Inside the executor's sandbox the run's own pin is the answer.

        The subprocess is stamped with the target its run was pinned to and the
        state file may already name another one — the target moved while the
        run was in flight. The stamp is what that run's writes are checked
        against, so it is what its posture is read for.
        """
        monkeypatch.setenv(posture.CONTROL_TARGET_ENV_VAR, "live")
        write_state(agent_root, "va")
        write_record(agent_root, {"live": "sandbox"})

        assert posture.posture() == posture.POSTURE_SANDBOX

    def test_the_stamp_also_wins_when_it_is_the_unnarrowed_one(self, agent_root, monkeypatch):
        monkeypatch.setenv(posture.CONTROL_TARGET_ENV_VAR, "va")
        write_state(agent_root, "live")
        write_record(agent_root, {"live": "sandbox"})

        assert posture.posture() == posture.POSTURE_WRITES

    def test_a_stamp_naming_an_unknown_target_falls_through(self, agent_root, monkeypatch):
        """A stamp is validated exactly as a record's target is.

        A value no reader knows can only index the posture to a key nothing
        writes, so answering with it would report "nothing narrowed" for a
        process that is actually on a narrowed machine.
        """
        monkeypatch.setenv(posture.CONTROL_TARGET_ENV_VAR, "somewhere-else")
        write_state(agent_root, "live")
        write_record(agent_root, {"live": "sandbox"})

        assert posture.posture() == posture.POSTURE_SANDBOX

    def test_an_unknown_stamp_with_no_state_record_is_no_answer(self, agent_root, monkeypatch):
        monkeypatch.setenv(posture.CONTROL_TARGET_ENV_VAR, "somewhere-else")
        write_record(agent_root, {"somewhere-else": "sandbox"})

        assert posture.posture() == posture.POSTURE_WRITES

    def test_a_blank_stamp_falls_through_to_the_state_file(self, agent_root, monkeypatch):
        monkeypatch.setenv(posture.CONTROL_TARGET_ENV_VAR, "  ")
        write_state(agent_root, "live")
        write_record(agent_root, {"live": "sandbox"})

        assert posture.posture() == posture.POSTURE_SANDBOX

    def test_a_state_record_owned_by_another_parent_is_not_ours(self, agent_root, monkeypatch):
        write_state(agent_root, "live", owner_ppid=os.getppid() + 100000)
        write_record(agent_root, {"live": "sandbox"})

        assert posture.posture() == posture.POSTURE_WRITES

    def test_a_state_record_whose_server_is_dead_is_residue(self, agent_root, monkeypatch):
        """A file left by a killed controls server names a target nobody is on."""
        write_state(agent_root, "live", server_pid=2**30)
        write_record(agent_root, {"live": "sandbox"})

        assert posture.posture() == posture.POSTURE_WRITES

    def test_two_state_records_under_one_parent_are_no_answer(self, agent_root, monkeypatch):
        """Ambiguous ownership resolves to nothing, exactly as it does elsewhere."""
        write_state(agent_root, "live")
        write_state(agent_root, "va", server_pid=os.getppid())
        write_record(agent_root, {"live": "sandbox", "va": "sandbox"})

        assert posture.posture() == posture.POSTURE_WRITES

    def test_an_unknown_target_name_is_no_answer(self, agent_root, monkeypatch):
        write_state(agent_root, "somewhere-else")
        write_record(agent_root, {"somewhere-else": "sandbox"})

        assert posture.posture() == posture.POSTURE_WRITES


# --------------------------------------------------------------------------
# Degradation: every failure is the environment answer, never an exception
# --------------------------------------------------------------------------


class TestDegradation:
    def test_no_record_at_all(self, agent_root, monkeypatch):
        write_state(agent_root, "live")

        assert posture.posture() == posture.POSTURE_WRITES

    def test_a_corrupt_record(self, agent_root, monkeypatch):
        write_state(agent_root, "live")
        path = control_context.record_path_under(agent_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json at all")
        session_store.invalidate_cache()

        assert posture.posture() == posture.POSTURE_WRITES

    @pytest.mark.skipif(
        hasattr(os, "geteuid") and os.geteuid() == 0,
        reason="root reads a mode-000 file, so the unreadable branch cannot be reached",
    )
    def test_an_unreadable_record(self, agent_root, monkeypatch):
        write_state(agent_root, "live")
        path = write_record(agent_root, {"live": "sandbox"})
        path.chmod(0o000)
        try:
            assert posture.posture() == posture.POSTURE_WRITES
        finally:
            path.chmod(0o600)

    def test_no_state_record_at_all(self, agent_root, monkeypatch):
        """A narrowed deployment whose target cannot be read is not clamped whole.

        This gate is the process-wide one: it refuses EVERY write tool in the
        server. Firing it on a target nobody could name would refuse writes to
        machines the operator never narrowed, so an unresolvable target answers
        the environment here. The fail-closed layer for one specific write is
        the connector's reference monitor, whose ``effective_writes`` takes the
        most restrictive narrowing when it cannot name its target.
        """
        write_record(agent_root, {"live": "sandbox"})

        assert posture.posture() == posture.POSTURE_WRITES

    def test_an_unresolvable_agent_data_root(self, agent_root, monkeypatch):
        monkeypatch.setenv(posture.OSPREY_AGENT_DATA_ROOT, str(agent_root / "nowhere"))

        assert posture.posture() == posture.POSTURE_WRITES

    def test_a_raising_target_resolver_degrades(self, agent_root, monkeypatch):
        """``posture()`` never raises: three refusal paths call it per tool call."""
        write_record(agent_root, {"live": "sandbox"})

        def _explode() -> str | None:
            raise RuntimeError("state directory is on fire")

        monkeypatch.setattr(posture, "session_control_target", _explode)

        assert posture.posture() == posture.POSTURE_WRITES

    def test_a_raising_target_resolver_keeps_an_env_sandbox(self, agent_root, monkeypatch):
        monkeypatch.setenv(posture.POSTURE_ENV_VAR, posture.SANDBOX_MODE)

        def _explode() -> str | None:  # pragma: no cover - short-circuited
            raise RuntimeError("state directory is on fire")

        monkeypatch.setattr(posture, "session_control_target", _explode)

        assert posture.posture() == posture.POSTURE_SANDBOX

    def test_a_raising_record_reader_degrades(self, agent_root, monkeypatch):
        write_state(agent_root, "live")
        write_record(agent_root, {"live": "sandbox"})

        def _boom(*_args, **_kwargs) -> str | None:
            raise RuntimeError("the record reader is on fire")

        monkeypatch.setattr(session_store, "target_posture", _boom)

        assert posture.posture() == posture.POSTURE_WRITES


# --------------------------------------------------------------------------
# Both caches follow their file's signature
# --------------------------------------------------------------------------


class TestSignatureCaches:
    def test_a_narrowing_lands_on_a_session_already_running(self, agent_root, monkeypatch):
        """The point of enforcing at write time: no respawn carries this."""
        write_state(agent_root, "live")
        write_record(agent_root, {})
        assert posture.posture() == posture.POSTURE_WRITES

        write_record(agent_root, {"live": "sandbox"})

        assert posture.posture() == posture.POSTURE_SANDBOX

    def test_lifting_a_narrowing_is_seen_too(self, agent_root, monkeypatch):
        write_state(agent_root, "live")
        write_record(agent_root, {"live": "sandbox"})
        assert posture.posture() == posture.POSTURE_SANDBOX

        write_record(agent_root, {})

        assert posture.posture() == posture.POSTURE_WRITES

    def test_a_switch_moves_which_entry_is_read(self, agent_root, monkeypatch):
        """The target cache re-resolves when the state file's signature moves."""
        write_state(agent_root, "va")
        write_record(agent_root, {"live": "sandbox"})
        assert posture.posture() == posture.POSTURE_WRITES

        pid = os.getpid()
        path = control_context.report_path_under(agent_root, pid)
        path.write_text(
            json.dumps(
                {
                    "target": "live",
                    "generation": 1,
                    "server_pid": pid,
                    "owner_ppid": os.getppid(),
                    "targets": {},
                    "children": [],
                }
            )
        )

        assert posture.posture() == posture.POSTURE_SANDBOX


# --------------------------------------------------------------------------
# The spellings this module restates
# --------------------------------------------------------------------------


class TestWireContract:
    def test_the_control_target_env_matches_the_runtime(self):
        """One spelling of the run's target stamp, restated in three places."""
        from osprey.mcp_server.python_executor import executor
        from osprey.runtime import ENV_CONTROL_TARGET

        assert posture.CONTROL_TARGET_ENV_VAR == ENV_CONTROL_TARGET
        assert posture.CONTROL_TARGET_ENV_VAR == executor.ENV_CONTROL_TARGET

    def test_the_agent_data_root_env_matches_the_record_reader(self):
        assert posture.OSPREY_AGENT_DATA_ROOT == session_store.AGENT_DATA_ROOT_ENV_VAR

    def test_the_posture_spellings_match_the_record_reader(self):
        assert posture.POSTURE_SANDBOX == session_store.POSTURE_SANDBOX
        assert posture.POSTURE_WRITES == session_store.POSTURE_WRITES

    def test_the_module_stays_a_leaf(self):
        """No top-level import of the MCP servers or the connector package.

        ``posture()`` is called by the audit middleware in every MCP server and
        by the executor's gates; both readers it now consults are imported
        inside the function precisely so that importing this module stays free.
        """
        import ast
        from pathlib import Path

        source = Path(posture.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported: list[str] = []
        for node in tree.body:  # top level only — a nested import is the point
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)

        assert not [name for name in imported if name.startswith("osprey_connectors")]
        assert not [name for name in imported if name.startswith("osprey.mcp_server")]
