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
  record's own ``target`` — the same file, read for both halves of the answer.
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

import os

import pytest

from osprey.audit import posture
from osprey_connectors import control_context, posture_store
from tests._control_context_fixtures import write_control_context

pytestmark = pytest.mark.unit

SESSION_KEY = "11111111-2222-3333-4444-555555555555"


@pytest.fixture(autouse=True)
def agent_root(tmp_path, monkeypatch):
    """An agent-data root of our own, with every posture marker cleared.

    ``OSPREY_AGENT_DATA_ROOT`` is the anchor
    :func:`osprey_connectors.control_context.record_path` honours, so stamping
    it here is what puts the record this module reads inside ``tmp_path``
    instead of the developer's own deployment. A test that skipped it would
    read whatever record this machine was last left pointed at.

    The record cache is dropped on the way in AND on the way out: it is a
    module global keyed on file signatures, and two tests whose roots are both
    empty produce the same signature.
    """
    monkeypatch.setenv(posture.OSPREY_AGENT_DATA_ROOT, str(tmp_path))
    for marker in (
        posture.POSTURE_ENV_VAR,
        posture.POSTURE_SESSION_ENV_VAR,
        posture.CONTROL_TARGET_ENV_VAR,
    ):
        monkeypatch.delenv(marker, raising=False)
    posture_store.invalidate_cache()
    yield tmp_path
    posture_store.invalidate_cache()


def write_record(root, posture_field, *, target: str = "live"):
    """The control-context record, ownerless, plus the cache drops these tests need.

    The payload itself is :func:`write_control_context` — one spelling of the
    record for every suite — and what is added here is the two caches this one
    reads through.
    """
    path = write_control_context(
        root, target=target, generation=0, posture=posture_field, owned_by=None
    )
    posture_store.invalidate_cache()
    return path


def rewrite_record(root, posture_field, *, target: str = "live"):
    """Replace the record the way its owner does, and touch NO cache.

    :func:`write_record` above drops the read cache, which is what a test
    staging a starting state wants. It is the wrong tool for the last class in
    this file, whose whole claim is that a rewrite is seen by a reader nobody
    told: the parse is keyed on ``(mtime_ns, size, inode)`` and the writer
    REPLACES the file, so the inode moves even for a same-size rewrite inside
    one filesystem clock tick. A second write that dropped the cache first
    would pass whether or not any of that were true.
    """
    record = control_context.ControlContext(
        target=target,
        generation=0,
        owner=None,
        posture=dict(posture_field),
        last_switch=None,
    )
    return control_context.write_record(record, path=control_context.record_path_under(root))


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
        write_record(agent_root, {"live": "sandbox"})

        assert posture.posture() == posture.POSTURE_SANDBOX


# --------------------------------------------------------------------------
# The record, indexed by this process's target
# --------------------------------------------------------------------------


class TestPerTargetLookup:
    def test_a_sandbox_entry_for_this_target_sandboxes(self, agent_root, monkeypatch):
        write_record(agent_root, {"live": "sandbox"})

        assert posture.posture() == posture.POSTURE_SANDBOX

    def test_a_sandbox_entry_for_another_target_does_not(self, agent_root, monkeypatch):
        """The whole feature: narrowing ``live`` leaves a deployment on ``va`` alone."""
        write_record(agent_root, {"live": "sandbox"}, target="va")

        assert posture.posture() == posture.POSTURE_WRITES

    def test_a_bare_sandbox_narrows_this_target(self, agent_root, monkeypatch):
        """The pre-feature deployment-wide value still refuses, on every target.

        Written against ``standin`` rather than the default so the claim is
        about the bare string and not about which target happens to be keyed.
        """
        write_record(agent_root, "sandbox", target="standin")

        assert posture.posture() == posture.POSTURE_SANDBOX

    def test_a_recorded_writes_entry_narrows_nothing(self, agent_root, monkeypatch):
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
        write_record(agent_root, {"live": "sandbox"})

        assert posture.posture_session() is None
        assert posture.posture() == posture.POSTURE_SANDBOX

    def test_a_narrowing_applies_with_a_session_key_too(self, agent_root, monkeypatch):
        in_session(monkeypatch)
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
        write_record(agent_root, {})

        assert posture.posture() == posture.POSTURE_WRITES


# --------------------------------------------------------------------------
# How the process learns its own target
# --------------------------------------------------------------------------


class TestTargetResolution:
    def test_the_control_target_stamp_wins_over_the_record(self, agent_root, monkeypatch):
        """Inside the executor's sandbox the run's own pin is the answer.

        The subprocess is stamped with the target its run was pinned to and the
        record may already name another one — the deployment moved while the
        run was in flight. The stamp is what that run's writes are checked
        against, so it is what its posture is read for. The record here names
        ``va``, so answering from it would report the run unnarrowed.
        """
        monkeypatch.setenv(posture.CONTROL_TARGET_ENV_VAR, "live")
        write_record(agent_root, {"live": "sandbox"}, target="va")

        assert posture.posture() == posture.POSTURE_SANDBOX

    def test_the_stamp_also_wins_when_it_is_the_unnarrowed_one(self, agent_root, monkeypatch):
        monkeypatch.setenv(posture.CONTROL_TARGET_ENV_VAR, "va")
        write_record(agent_root, {"live": "sandbox"})

        assert posture.posture() == posture.POSTURE_WRITES

    def test_a_stamp_naming_an_unknown_target_falls_through(self, agent_root, monkeypatch):
        """A stamp is validated exactly as a record's target is.

        A value no reader knows can only index the posture to a key nothing
        writes, so answering with it would report "nothing narrowed" for a
        process that is actually on a narrowed machine.
        """
        monkeypatch.setenv(posture.CONTROL_TARGET_ENV_VAR, "somewhere-else")
        write_record(agent_root, {"live": "sandbox"})

        assert posture.posture() == posture.POSTURE_SANDBOX

    def test_a_narrowing_keyed_to_an_unknown_name_reaches_nothing(self, agent_root, monkeypatch):
        """The other half: the dropped stamp cannot index its own entry either.

        The record's ``live`` is what the fall-through answers with, and the
        ``somewhere-else`` entry is a key nothing resolves to — so the process
        reads as unnarrowed rather than picking up an entry no target owns.
        """
        monkeypatch.setenv(posture.CONTROL_TARGET_ENV_VAR, "somewhere-else")
        write_record(agent_root, {"somewhere-else": "sandbox"})

        assert posture.posture() == posture.POSTURE_WRITES

    def test_a_blank_stamp_falls_through_to_the_record(self, agent_root, monkeypatch):
        monkeypatch.setenv(posture.CONTROL_TARGET_ENV_VAR, "  ")
        write_record(agent_root, {"live": "sandbox"})

        assert posture.posture() == posture.POSTURE_SANDBOX

    def test_a_record_naming_an_unknown_target_is_no_answer(self, agent_root, monkeypatch):
        """A record whose own ``target`` is unknown does not parse at all.

        Both halves of the answer come out of one file now, so a target name no
        reader knows costs the record its narrowings as well as its target —
        which is the fail-open direction this gate is required to take.
        """
        write_record(agent_root, {"somewhere-else": "sandbox"}, target="somewhere-else")

        assert posture.posture() == posture.POSTURE_WRITES


# --------------------------------------------------------------------------
# Degradation: every failure is the environment answer, never an exception
# --------------------------------------------------------------------------


class TestDegradation:
    def test_no_record_at_all(self, agent_root, monkeypatch):
        assert posture.posture() == posture.POSTURE_WRITES

    def test_a_corrupt_record(self, agent_root, monkeypatch):
        path = control_context.record_path_under(agent_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json at all")
        posture_store.invalidate_cache()

        assert posture.posture() == posture.POSTURE_WRITES

    @pytest.mark.skipif(
        hasattr(os, "geteuid") and os.geteuid() == 0,
        reason="root reads a mode-000 file, so the unreadable branch cannot be reached",
    )
    def test_an_unreadable_record(self, agent_root, monkeypatch):
        path = write_record(agent_root, {"live": "sandbox"})
        path.chmod(0o000)
        try:
            assert posture.posture() == posture.POSTURE_WRITES
        finally:
            path.chmod(0o600)

    def test_an_unresolvable_agent_data_root(self, agent_root, monkeypatch):
        monkeypatch.setenv(posture.OSPREY_AGENT_DATA_ROOT, str(agent_root / "nowhere"))

        assert posture.posture() == posture.POSTURE_WRITES

    def test_a_raising_target_resolver_degrades(self, agent_root, monkeypatch):
        """``posture()`` never raises: three refusal paths call it per tool call."""
        write_record(agent_root, {"live": "sandbox"})

        def _explode() -> str | None:
            raise RuntimeError("state directory is on fire")

        monkeypatch.setattr(posture, "recorded_control_target", _explode)

        assert posture.posture() == posture.POSTURE_WRITES

    def test_a_raising_target_resolver_keeps_an_env_sandbox(self, agent_root, monkeypatch):
        monkeypatch.setenv(posture.POSTURE_ENV_VAR, posture.SANDBOX_MODE)

        def _explode() -> str | None:  # pragma: no cover - short-circuited
            raise RuntimeError("state directory is on fire")

        monkeypatch.setattr(posture, "recorded_control_target", _explode)

        assert posture.posture() == posture.POSTURE_SANDBOX

    def test_a_raising_record_reader_degrades(self, agent_root, monkeypatch):
        write_record(agent_root, {"live": "sandbox"})

        def _boom(*_args, **_kwargs) -> str | None:
            raise RuntimeError("the record reader is on fire")

        monkeypatch.setattr(posture_store, "target_posture", _boom)

        assert posture.posture() == posture.POSTURE_WRITES


# --------------------------------------------------------------------------
# The reader follows the record's signature
# --------------------------------------------------------------------------


class TestSignatureCache:
    """A rewrite is seen by a reader nobody told.

    There is ONE cache left in this path — the record reader's — and every test
    below primes it with a first :func:`posture` call and then rewrites the
    record through :func:`rewrite_record`, which drops nothing. The assertion
    that follows is therefore about the signature key and not about the
    invalidation the fixtures happen to perform.
    """

    def test_a_narrowing_lands_on_a_session_already_running(self, agent_root, monkeypatch):
        """The point of enforcing at write time: no respawn carries this."""
        write_record(agent_root, {})
        assert posture.posture() == posture.POSTURE_WRITES

        rewrite_record(agent_root, {"live": "sandbox"})

        assert posture.posture() == posture.POSTURE_SANDBOX

    def test_lifting_a_narrowing_is_seen_too(self, agent_root, monkeypatch):
        write_record(agent_root, {"live": "sandbox"})
        assert posture.posture() == posture.POSTURE_SANDBOX

        rewrite_record(agent_root, {})

        assert posture.posture() == posture.POSTURE_WRITES

    def test_a_switch_moves_which_entry_is_read(self, agent_root, monkeypatch):
        """The target and the narrowings live in one file, so one write moves both."""
        write_record(agent_root, {"live": "sandbox"}, target="va")
        assert posture.posture() == posture.POSTURE_WRITES

        rewrite_record(agent_root, {"live": "sandbox"}, target="live")

        assert posture.posture() == posture.POSTURE_SANDBOX

    def test_a_same_size_rewrite_inside_one_clock_tick_is_still_seen(self, agent_root, monkeypatch):
        """The case the inode is in the key FOR.

        ``{"live": "sandbox"}`` and ``{"va": "sandbox"}`` are the same number of
        bytes, and two writes this close together can share an ``mtime_ns``. The
        deployment is on ``live``, so the first narrows this process and the
        second does not — an answer that could only come from a re-parse.
        """
        write_record(agent_root, {"live": "sandbox"})
        assert posture.posture() == posture.POSTURE_SANDBOX

        rewrite_record(agent_root, {"va": "sandbox"})

        assert posture.posture() == posture.POSTURE_WRITES


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
        assert posture.OSPREY_AGENT_DATA_ROOT == posture_store.AGENT_DATA_ROOT_ENV_VAR

    def test_the_posture_spellings_match_the_record_reader(self):
        assert posture.POSTURE_SANDBOX == posture_store.POSTURE_SANDBOX
        assert posture.POSTURE_WRITES == posture_store.POSTURE_WRITES

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
