"""Tests for the store-backed half of :func:`osprey.audit.posture.posture`.

``posture()`` used to be one line of environment: a session child spawned with
``OSPREY_EXECUTION_MODE=readonly`` was sandboxed and everything else was not.
The per-(session, target) posture narrows ONE control target for one session,
which no environment variable can express — a session sandboxed on the live
machine must still be able to write to the virtual accelerator, and setting the
variable would clamp both.

So the answer gains a second source, and these tests pin the seam between them:

* **No session key → the environment, untouched.** A dispatch worker, an
  ``agent_runner`` child, a CLI run: none of them belongs to a posture store
  key, and none of them may pay a file read to learn what it already knows.
  Pinned by making the store lookup *raise* and proving the answer arrives
  anyway.
* **A session key → the store, indexed by the SESSION's current target.** The
  process learns that target from the ``OSPREY_CONTROL_TARGET`` stamp when it
  carries one (the executor's sandbox subprocess), else from the controls
  server's state record whose ``owner_ppid`` is this process's parent — the
  rule ``executor._session_target_record`` and
  ``target_banner.resolve_session_target`` already use, because both this
  process and the controls server are children of one Claude Code.
* **Every failure degrades to the environment.** A missing store, a corrupt
  one, no readable state record, an import that fails: each is "nothing
  narrowed this session", never an exception out of a function three refusal
  paths call on every tool call.

The store can only ever narrow: an environment that already says sandbox stays
sandbox no matter what the store holds.
"""

from __future__ import annotations

import json
import os

import pytest

from osprey.audit import posture
from osprey_connectors import session_store

pytestmark = pytest.mark.unit

SESSION_KEY = "11111111-2222-3333-4444-555555555555"


@pytest.fixture(autouse=True)
def agent_root(tmp_path, monkeypatch):
    """An agent-data root of our own, with every posture marker cleared.

    ``OSPREY_AGENT_DATA_ROOT`` is the one anchor both readers honour — the
    store's :func:`osprey_connectors.session_store.state_dir` and the state
    file's :func:`osprey.mcp_server.control_system.target_state.state_dir` — so
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
    path = _state_dir(root) / f"target_state_{pid}.json"
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


def write_store(root, mapping):
    """The per-(session, target) store, as the web server persists it."""
    path = _state_dir(root) / session_store.STORE_FILENAME
    path.write_text(json.dumps(mapping))
    session_store.invalidate_cache()
    posture.invalidate_session_target_cache()
    return path


def in_session(monkeypatch, key: str = SESSION_KEY) -> None:
    monkeypatch.setenv(posture.POSTURE_SESSION_ENV_VAR, key)


# --------------------------------------------------------------------------
# No session key: the environment answer, and nothing else
# --------------------------------------------------------------------------


class TestEnvOnlyPaths:
    def test_no_session_key_and_no_marker_is_writes(self, agent_root):
        assert posture.posture() == posture.POSTURE_WRITES

    def test_no_session_key_and_a_readonly_marker_is_sandbox(self, agent_root, monkeypatch):
        monkeypatch.setenv(posture.POSTURE_ENV_VAR, posture.SANDBOX_MODE)
        assert posture.posture() == posture.POSTURE_SANDBOX

    @pytest.mark.parametrize("value", ["readwrite", "READONLY", "", "sandbox", "true"])
    def test_no_session_key_keeps_the_value_comparison(self, agent_root, monkeypatch, value):
        """Only the exact ``readonly`` string sandboxes — unchanged by the store."""
        monkeypatch.setenv(posture.POSTURE_ENV_VAR, value)
        assert posture.posture() == posture.POSTURE_WRITES

    def test_a_blank_session_key_is_no_session_key(self, agent_root, monkeypatch):
        monkeypatch.setenv(posture.POSTURE_SESSION_ENV_VAR, "   ")
        write_store(agent_root, {"": {"live": "sandbox"}})
        assert posture.posture() == posture.POSTURE_WRITES

    def test_the_store_is_not_consulted_without_a_session_key(self, agent_root, monkeypatch):
        """The dispatch / CLI path stays byte-for-byte what it was.

        Proven by sabotage rather than by mocking a file read: the lookup is
        replaced with something that raises, so a call that reached it would
        surface here instead of quietly degrading to the same answer the
        environment gives.
        """

        def _explode(session_key: str) -> bool:  # pragma: no cover - must not run
            raise AssertionError("the store was consulted without a session key")

        monkeypatch.setattr(posture, "_session_target_is_sandboxed", _explode)
        write_store(agent_root, {SESSION_KEY: {"live": "sandbox"}})
        write_state(agent_root, "live")

        assert posture.posture() == posture.POSTURE_WRITES

    def test_an_env_sandbox_stays_sandbox_with_an_empty_store(self, agent_root, monkeypatch):
        """Narrowing-only: the store can refuse writes, never grant them."""
        in_session(monkeypatch)
        monkeypatch.setenv(posture.POSTURE_ENV_VAR, posture.SANDBOX_MODE)
        write_state(agent_root, "live")
        write_store(agent_root, {})

        assert posture.posture() == posture.POSTURE_SANDBOX


# --------------------------------------------------------------------------
# A session key: the store, indexed by the session's target
# --------------------------------------------------------------------------


class TestPerTargetLookup:
    def test_a_sandbox_entry_for_the_session_target_sandboxes(self, agent_root, monkeypatch):
        in_session(monkeypatch)
        write_state(agent_root, "live")
        write_store(agent_root, {SESSION_KEY: {"live": "sandbox"}})

        assert posture.posture() == posture.POSTURE_SANDBOX

    def test_a_sandbox_entry_for_another_target_does_not(self, agent_root, monkeypatch):
        """The whole feature: narrowing ``live`` leaves the session's ``va`` alone."""
        in_session(monkeypatch)
        write_state(agent_root, "va")
        write_store(agent_root, {SESSION_KEY: {"live": "sandbox"}})

        assert posture.posture() == posture.POSTURE_WRITES

    def test_another_session_key_does_not_reach_this_one(self, agent_root, monkeypatch):
        in_session(monkeypatch)
        write_state(agent_root, "live")
        write_store(agent_root, {"another-session": {"live": "sandbox"}})

        assert posture.posture() == posture.POSTURE_WRITES

    def test_a_bare_legacy_sandbox_narrows_the_session_target(self, agent_root, monkeypatch):
        """The pre-feature session-wide value still refuses, on every target."""
        in_session(monkeypatch)
        write_state(agent_root, "standin")
        write_store(agent_root, {SESSION_KEY: "sandbox"})

        assert posture.posture() == posture.POSTURE_SANDBOX

    def test_a_stored_writes_entry_narrows_nothing(self, agent_root, monkeypatch):
        in_session(monkeypatch)
        write_state(agent_root, "live")
        write_store(agent_root, {SESSION_KEY: {"live": "writes"}})

        assert posture.posture() == posture.POSTURE_WRITES


# --------------------------------------------------------------------------
# How the process learns its own target
# --------------------------------------------------------------------------


class TestTargetResolution:
    def test_the_control_target_stamp_wins_over_the_state_file(self, agent_root, monkeypatch):
        """Inside the executor's sandbox the run's own pin is the answer.

        The subprocess is stamped with the target its run was pinned to and the
        state file may already name another one — the session switched while
        the run was in flight. The stamp is what that run's writes are checked
        against, so it is what its posture is read for.
        """
        in_session(monkeypatch)
        monkeypatch.setenv(posture.CONTROL_TARGET_ENV_VAR, "live")
        write_state(agent_root, "va")
        write_store(agent_root, {SESSION_KEY: {"live": "sandbox"}})

        assert posture.posture() == posture.POSTURE_SANDBOX

    def test_the_stamp_also_wins_when_it_is_the_unnarrowed_one(self, agent_root, monkeypatch):
        in_session(monkeypatch)
        monkeypatch.setenv(posture.CONTROL_TARGET_ENV_VAR, "va")
        write_state(agent_root, "live")
        write_store(agent_root, {SESSION_KEY: {"live": "sandbox"}})

        assert posture.posture() == posture.POSTURE_WRITES

    def test_a_stamp_naming_an_unknown_target_falls_through(self, agent_root, monkeypatch):
        """A stamp is validated exactly as a record's target is.

        A value no reader knows can only index the store to a key nothing
        writes, so answering with it would report "nothing narrowed" for a
        session that narrowed the target it is actually on.
        """
        in_session(monkeypatch)
        monkeypatch.setenv(posture.CONTROL_TARGET_ENV_VAR, "somewhere-else")
        write_state(agent_root, "live")
        write_store(agent_root, {SESSION_KEY: {"live": "sandbox"}})

        assert posture.posture() == posture.POSTURE_SANDBOX

    def test_an_unknown_stamp_with_no_record_is_no_answer(self, agent_root, monkeypatch):
        in_session(monkeypatch)
        monkeypatch.setenv(posture.CONTROL_TARGET_ENV_VAR, "somewhere-else")
        write_store(agent_root, {SESSION_KEY: {"somewhere-else": "sandbox"}})

        assert posture.posture() == posture.POSTURE_WRITES

    def test_a_blank_stamp_falls_through_to_the_state_file(self, agent_root, monkeypatch):
        in_session(monkeypatch)
        monkeypatch.setenv(posture.CONTROL_TARGET_ENV_VAR, "  ")
        write_state(agent_root, "live")
        write_store(agent_root, {SESSION_KEY: {"live": "sandbox"}})

        assert posture.posture() == posture.POSTURE_SANDBOX

    def test_a_record_owned_by_another_parent_is_not_ours(self, agent_root, monkeypatch):
        in_session(monkeypatch)
        write_state(agent_root, "live", owner_ppid=os.getppid() + 100000)
        write_store(agent_root, {SESSION_KEY: {"live": "sandbox"}})

        assert posture.posture() == posture.POSTURE_WRITES

    def test_a_record_whose_server_is_dead_is_residue(self, agent_root, monkeypatch):
        """A file left by a killed controls server names a target nobody is on."""
        in_session(monkeypatch)
        write_state(agent_root, "live", server_pid=2**30)
        write_store(agent_root, {SESSION_KEY: {"live": "sandbox"}})

        assert posture.posture() == posture.POSTURE_WRITES

    def test_two_records_under_one_parent_are_no_answer(self, agent_root, monkeypatch):
        """Ambiguous ownership resolves to nothing, exactly as it does elsewhere."""
        in_session(monkeypatch)
        write_state(agent_root, "live")
        write_state(agent_root, "va", server_pid=os.getppid())
        write_store(agent_root, {SESSION_KEY: {"live": "sandbox", "va": "sandbox"}})

        assert posture.posture() == posture.POSTURE_WRITES

    def test_an_unknown_target_name_is_no_answer(self, agent_root, monkeypatch):
        in_session(monkeypatch)
        write_state(agent_root, "somewhere-else")
        write_store(agent_root, {SESSION_KEY: {"somewhere-else": "sandbox"}})

        assert posture.posture() == posture.POSTURE_WRITES


# --------------------------------------------------------------------------
# Degradation: every failure is the environment answer, never an exception
# --------------------------------------------------------------------------


class TestDegradation:
    def test_no_store_at_all(self, agent_root, monkeypatch):
        in_session(monkeypatch)
        write_state(agent_root, "live")

        assert posture.posture() == posture.POSTURE_WRITES

    def test_a_corrupt_store(self, agent_root, monkeypatch):
        in_session(monkeypatch)
        write_state(agent_root, "live")
        path = _state_dir(agent_root) / session_store.STORE_FILENAME
        path.write_text("{not json at all")
        session_store.invalidate_cache()

        assert posture.posture() == posture.POSTURE_WRITES

    @pytest.mark.skipif(
        hasattr(os, "geteuid") and os.geteuid() == 0,
        reason="root reads a mode-000 file, so the unreadable branch cannot be reached",
    )
    def test_an_unreadable_store(self, agent_root, monkeypatch):
        in_session(monkeypatch)
        write_state(agent_root, "live")
        path = write_store(agent_root, {SESSION_KEY: {"live": "sandbox"}})
        path.chmod(0o000)
        try:
            assert posture.posture() == posture.POSTURE_WRITES
        finally:
            path.chmod(0o600)

    def test_no_state_record_at_all(self, agent_root, monkeypatch):
        """A narrowed session whose target cannot be read is not clamped whole.

        This gate is the process-wide one: it refuses EVERY write tool in the
        server. Firing it on a target nobody could name would refuse writes to
        machines the operator never narrowed, so an unresolvable target answers
        the environment here. The fail-closed layer for one specific write is
        the connector's reference monitor, whose ``effective_writes`` takes the
        most restrictive entry when it cannot name its target.
        """
        in_session(monkeypatch)
        write_store(agent_root, {SESSION_KEY: {"live": "sandbox"}})

        assert posture.posture() == posture.POSTURE_WRITES

    def test_an_unresolvable_agent_data_root(self, agent_root, monkeypatch):
        in_session(monkeypatch)
        monkeypatch.setenv(posture.OSPREY_AGENT_DATA_ROOT, str(agent_root / "nowhere"))

        assert posture.posture() == posture.POSTURE_WRITES

    def test_a_raising_target_resolver_degrades(self, agent_root, monkeypatch):
        """``posture()`` never raises: three refusal paths call it per tool call."""
        in_session(monkeypatch)
        write_store(agent_root, {SESSION_KEY: {"live": "sandbox"}})

        def _explode() -> str | None:
            raise RuntimeError("state directory is on fire")

        monkeypatch.setattr(posture, "session_control_target", _explode)

        assert posture.posture() == posture.POSTURE_WRITES

    def test_a_raising_target_resolver_keeps_an_env_sandbox(self, agent_root, monkeypatch):
        in_session(monkeypatch)
        monkeypatch.setenv(posture.POSTURE_ENV_VAR, posture.SANDBOX_MODE)

        def _explode() -> str | None:  # pragma: no cover - short-circuited
            raise RuntimeError("state directory is on fire")

        monkeypatch.setattr(posture, "session_control_target", _explode)

        assert posture.posture() == posture.POSTURE_SANDBOX


# --------------------------------------------------------------------------
# Both caches follow their file's signature
# --------------------------------------------------------------------------


class TestSignatureCaches:
    def test_a_narrowing_lands_on_a_session_already_running(self, agent_root, monkeypatch):
        """The point of enforcing at write time: no respawn carries this."""
        in_session(monkeypatch)
        write_state(agent_root, "live")
        write_store(agent_root, {})
        assert posture.posture() == posture.POSTURE_WRITES

        path = _state_dir(agent_root) / session_store.STORE_FILENAME
        path.write_text(json.dumps({SESSION_KEY: {"live": "sandbox"}}))

        assert posture.posture() == posture.POSTURE_SANDBOX

    def test_lifting_a_narrowing_is_seen_too(self, agent_root, monkeypatch):
        in_session(monkeypatch)
        write_state(agent_root, "live")
        write_store(agent_root, {SESSION_KEY: {"live": "sandbox"}})
        assert posture.posture() == posture.POSTURE_SANDBOX

        path = _state_dir(agent_root) / session_store.STORE_FILENAME
        path.write_text(json.dumps({}))

        assert posture.posture() == posture.POSTURE_WRITES

    def test_a_switch_moves_which_entry_is_read(self, agent_root, monkeypatch):
        """The target cache re-resolves when the state file's signature moves."""
        in_session(monkeypatch)
        write_state(agent_root, "va")
        write_store(agent_root, {SESSION_KEY: {"live": "sandbox"}})
        assert posture.posture() == posture.POSTURE_WRITES

        pid = os.getpid()
        path = _state_dir(agent_root) / f"target_state_{pid}.json"
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

    def test_the_agent_data_root_env_matches_the_store_reader(self):
        assert posture.OSPREY_AGENT_DATA_ROOT == session_store.AGENT_DATA_ROOT_ENV_VAR

    def test_the_posture_spellings_match_the_store_reader(self):
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
