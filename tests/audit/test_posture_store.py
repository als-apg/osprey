"""Tests for :func:`osprey.audit.posture.verdict` and the ledger posture over it.

The ledger's ``posture`` field used to be the environment plus the operator's
per-target narrowing. It is now the write verdict for the control target this
process's writes are about, in the ledger's two words: ``writes`` when the
verdict allows, ``sandbox`` when it refuses for any reason. These tests pin
that mapping and what feeds it:

* **The verdict's own order.** A read-only run, then the tier, then the target,
  owner, record and switch — :func:`osprey_connectors.posture_store.write_verdict`
  decides; this module only asks it about the right target.
* **The target this process's writes are about.** The ``OSPREY_CONTROL_TARGET``
  stamp when it names a known target (the executor's sandbox subprocess), else
  the record's own ``target``.
* **The record belongs to the deployment, not to a session.**
  ``OSPREY_POSTURE_SESSION`` is the audit session id and nothing else.
* **Never raises, never grants on a failure.** A config that cannot be read is
  ``tier_never``; a verdict that cannot be reached refuses; ``posture()`` reads
  ``sandbox`` for both.
"""

from __future__ import annotations

import json
import os

import pytest

from osprey.audit import posture
from osprey_connectors import control_context, posture_store
from osprey_connectors.safety_file import atomic_replace
from osprey_connectors_tests._harness import (
    isolate_verdict_environment,
    reset_verdict_state,
    write_control_context,
)

SESSION_KEY = "11111111-2222-3333-4444-555555555555"

#: A tier that allows writes, on a deployment whose live machine is real so
#: that ``live`` resolves.
SECTION = {"type": "epics", "writes": "allowed"}


@pytest.fixture(autouse=True)
def agent_root(tmp_path, monkeypatch):
    """An agent-data root of our own, a clean verdict environment, and the section.

    :func:`isolate_verdict_environment` clears every variable that moves a
    verdict (owner, tree, launch pin, read-only run), points
    ``OSPREY_AGENT_DATA_ROOT`` at ``tmp_path`` and pins the acting identity, so
    the record this module reads is the one the test wrote and not the
    developer's own. The section is patched at the audit package's one reader:
    the config loader is a process-wide singleton.
    """
    isolate_verdict_environment(monkeypatch, tmp_path)
    for marker in (posture.POSTURE_SESSION_ENV_VAR, posture.CONTROL_TARGET_ENV_VAR):
        monkeypatch.delenv(marker, raising=False)
    monkeypatch.setattr(posture, "control_system_section", lambda: dict(SECTION))
    yield tmp_path
    reset_verdict_state()


def write_record(root, posture_field=None, *, target: str = "live", switch=None):
    """The control-context record, ownerless, with the caches these tests read dropped.

    *switch* is ``{target: "on"}`` for the machines switched on; ``None`` keeps
    whatever switch is already on disk (none on a fresh root). *posture_field*
    is an old narrowing laid over the mirror verbatim, as a file written before
    the mirror would carry it.
    """
    path = write_control_context(root, target=target, generation=0, owned_by=None, switch=switch)
    if posture_field is not None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["posture"] = dict(posture_field)
        path.write_text(json.dumps(payload), encoding="utf-8")
    reset_verdict_state()
    return path


def rewrite_record(root, switch, *, target: str = "live"):
    """Replace the record with *switch*, atomically, and touch NO cache.

    The parse is keyed on ``(mtime_ns, size, inode)`` and the writer REPLACES
    the file, so a reader nobody told still sees the rewrite. The epoch on
    disk is kept, as the production writer keeps it.
    """
    path = control_context._record_path_under(root)
    epoch = json.loads(path.read_text(encoding="utf-8"))["switch_epoch"]
    record = control_context.ControlContext(
        target=target, generation=0, switch=dict(switch), switch_epoch=epoch
    )
    payload = json.dumps(record.to_payload(), indent=2).encode("utf-8")
    atomic_replace(path, payload, mode=0o600, sync="best_effort")
    return path


def in_session(monkeypatch, key: str = SESSION_KEY) -> None:
    monkeypatch.setenv(posture.POSTURE_SESSION_ENV_VAR, key)


def switched_on(root, target: str = "live", **kwargs):
    """A record pointed at *target* with *target*'s switch on."""
    return write_record(root, target=target, switch={target: "on"}, **kwargs)


# --------------------------------------------------------------------------
# The verdict, and the ledger's two words for it
# --------------------------------------------------------------------------


class TestVerdictMapsToTheWireValues:
    def test_a_switch_on_for_this_target_is_writes(self, agent_root):
        switched_on(agent_root)

        verdict = posture.verdict()
        assert verdict.allowed
        assert verdict.target == "live"
        assert posture.posture() == posture.POSTURE_WRITES

    def test_no_record_is_every_switch_off(self):
        verdict = posture.verdict()

        assert not verdict.allowed
        assert posture.posture() == posture.POSTURE_SANDBOX

    def test_a_record_with_the_switch_off_is_sandbox(self, agent_root):
        write_record(agent_root)

        assert posture.verdict().reason == "switch_off"
        assert posture.posture() == posture.POSTURE_SANDBOX

    def test_a_readonly_run_is_sandbox_whatever_the_switch(self, agent_root, monkeypatch):
        switched_on(agent_root)
        monkeypatch.setenv(posture.POSTURE_ENV_VAR, posture.SANDBOX_MODE)

        assert posture.verdict().reason == "readonly_run"
        assert posture.posture() == posture.POSTURE_SANDBOX

    @pytest.mark.parametrize("value", ["readwrite", "READONLY", "", "sandbox", "true"])
    def test_only_the_exact_readonly_value_is_a_readonly_run(self, agent_root, monkeypatch, value):
        switched_on(agent_root)
        monkeypatch.setenv(posture.POSTURE_ENV_VAR, value)

        assert posture.posture() == posture.POSTURE_WRITES

    def test_a_tier_that_allows_nothing_is_sandbox(self, agent_root, monkeypatch):
        """A demoted tier refuses before the record, so an ``on`` stays inert."""
        switched_on(agent_root)
        monkeypatch.setattr(posture, "control_system_section", lambda: {"type": "epics"})

        assert posture.verdict().reason == "tier_never"
        assert posture.posture() == posture.POSTURE_SANDBOX

    def test_a_switch_on_for_another_target_is_sandbox(self, agent_root):
        """The switch is per machine: ``va`` on says nothing about ``live``."""
        write_record(agent_root, target="live", switch={"va": "on"})

        assert posture.verdict().reason == "switch_off"
        assert posture.posture() == posture.POSTURE_SANDBOX

    def test_an_old_sandbox_narrowing_does_not_refuse_under_a_switch_that_is_on(self, agent_root):
        """The interim reading is retired: an old ``sandbox`` entry is inert."""
        switched_on(agent_root, posture_field={"live": "sandbox"})

        assert posture.verdict().allowed
        assert posture.posture() == posture.POSTURE_WRITES


class TestAnUnreadableConfigIsTierNever:
    def test_the_section_reader_answers_none_when_the_config_cannot_be_read(self, monkeypatch):
        from osprey_connectors import config as connector_config

        def _boom(*_args, **_kwargs):
            raise RuntimeError("config.yml is on fire")

        monkeypatch.undo()  # drop the autouse seam: this is the real reader
        monkeypatch.setattr(connector_config, "get_config_value", _boom)

        assert posture.control_system_section() is None

    def test_a_section_that_cannot_be_read_gives_tier_never(self, agent_root, monkeypatch):
        switched_on(agent_root)
        monkeypatch.setattr(posture, "control_system_section", lambda: None)

        assert posture.verdict().reason == "tier_never"
        assert posture.posture() == posture.POSTURE_SANDBOX

    def test_a_readonly_run_still_answers_first(self, monkeypatch):
        monkeypatch.setattr(posture, "control_system_section", lambda: None)
        monkeypatch.setenv(posture.POSTURE_ENV_VAR, posture.SANDBOX_MODE)

        assert posture.verdict().reason == "readonly_run"


# --------------------------------------------------------------------------
# The session key is an audit id, not an index
# --------------------------------------------------------------------------


class TestSessionKeyIsNotAnIndex:
    def test_the_switch_applies_without_a_session_key(self, agent_root):
        """A dispatch worker and a CLI run answer from the same record."""
        switched_on(agent_root)

        assert posture.posture_session() is None
        assert posture.posture() == posture.POSTURE_WRITES

    def test_the_switch_applies_with_a_session_key_too(self, agent_root, monkeypatch):
        in_session(monkeypatch)
        switched_on(agent_root)

        assert posture.posture() == posture.POSTURE_WRITES

    def test_the_key_is_carried_verbatim_as_the_audit_session_id(self, monkeypatch):
        in_session(monkeypatch)
        assert posture.posture_session() == SESSION_KEY

    def test_a_blank_key_is_no_key(self, monkeypatch):
        monkeypatch.setenv(posture.POSTURE_SESSION_ENV_VAR, "   ")
        assert posture.posture_session() is None

    def test_the_key_does_not_switch_anything_on_by_itself(self, agent_root, monkeypatch):
        in_session(monkeypatch)
        write_record(agent_root)

        assert posture.posture() == posture.POSTURE_SANDBOX


# --------------------------------------------------------------------------
# How the process learns its own target
# --------------------------------------------------------------------------


class TestTargetResolution:
    def test_the_control_target_stamp_wins_over_the_record(self, agent_root, monkeypatch):
        """Inside the executor's sandbox the run's own pin is the target asked about.

        The record names ``live`` with ``va`` switched on; the run was pinned to
        ``va``, so its writes are allowed although the deployment moved.
        """
        monkeypatch.setenv(posture.CONTROL_TARGET_ENV_VAR, "va")
        write_record(agent_root, target="live", switch={"va": "on"})

        assert posture.verdict().target == "va"
        assert posture.posture() == posture.POSTURE_WRITES

    def test_the_stamp_wins_when_its_machine_is_off(self, agent_root, monkeypatch):
        monkeypatch.setenv(posture.CONTROL_TARGET_ENV_VAR, "live")
        write_record(agent_root, target="va", switch={"va": "on"})

        assert posture.verdict().target == "live"
        assert posture.posture() == posture.POSTURE_SANDBOX

    def test_a_stamp_naming_an_unknown_target_falls_through(self, agent_root, monkeypatch):
        monkeypatch.setenv(posture.CONTROL_TARGET_ENV_VAR, "somewhere-else")
        switched_on(agent_root)

        assert posture.verdict().target == "live"
        assert posture.posture() == posture.POSTURE_WRITES

    def test_a_blank_stamp_falls_through_to_the_record(self, agent_root, monkeypatch):
        monkeypatch.setenv(posture.CONTROL_TARGET_ENV_VAR, "  ")
        switched_on(agent_root)

        assert posture.verdict().target == "live"

    def test_no_nameable_target_refuses(self, monkeypatch):
        """``None`` is passed through, and the verdict refuses what it cannot resolve."""
        monkeypatch.setattr(posture, "recorded_control_target", lambda: None)

        verdict = posture.verdict()
        assert not verdict.allowed
        assert verdict.reason == "target_unknown"
        assert posture.posture() == posture.POSTURE_SANDBOX


# --------------------------------------------------------------------------
# Never raises, never grants on a failure
# --------------------------------------------------------------------------


class TestDegradation:
    def test_a_corrupt_record_refuses(self, agent_root):
        path = control_context._record_path_under(agent_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json at all")
        reset_verdict_state()

        assert not posture.verdict().allowed
        assert posture.posture() == posture.POSTURE_SANDBOX

    @pytest.mark.skipif(
        hasattr(os, "geteuid") and os.geteuid() == 0,
        reason="root reads a mode-000 file, so the unreadable branch cannot be reached",
    )
    def test_an_unreadable_record_refuses(self, agent_root):
        path = switched_on(agent_root)
        path.chmod(0o000)
        try:
            reset_verdict_state()
            assert posture.posture() == posture.POSTURE_SANDBOX
        finally:
            path.chmod(0o600)

    def test_an_unresolvable_agent_data_root_refuses(self, agent_root, monkeypatch):
        switched_on(agent_root)
        monkeypatch.setenv(posture.OSPREY_AGENT_DATA_ROOT, str(agent_root / "nowhere"))
        reset_verdict_state()

        assert posture.posture() == posture.POSTURE_SANDBOX

    def test_a_raising_target_resolver_refuses_rather_than_raises(self, agent_root, monkeypatch):
        switched_on(agent_root)

        def _explode() -> str | None:
            raise RuntimeError("state directory is on fire")

        monkeypatch.setattr(posture, "recorded_control_target", _explode)

        verdict = posture.verdict()
        assert not verdict.allowed
        assert verdict.reason == "record_unavailable"
        assert posture.posture() == posture.POSTURE_SANDBOX

    def test_a_raising_write_verdict_refuses_rather_than_raises(self, agent_root, monkeypatch):
        switched_on(agent_root)

        def _boom(*_args, **_kwargs):
            raise RuntimeError("the verdict is on fire")

        monkeypatch.setattr(posture_store, "write_verdict", _boom)

        assert posture.verdict().reason == "record_unavailable"
        assert posture.posture() == posture.POSTURE_SANDBOX

    def test_a_raising_verdict_reads_sandbox(self, agent_root, monkeypatch):
        switched_on(agent_root)

        def _boom():
            raise RuntimeError("verdict() broke its contract")

        monkeypatch.setattr(posture, "verdict", _boom)

        assert posture.posture() == posture.POSTURE_SANDBOX


# --------------------------------------------------------------------------
# The reader follows the record's signature
# --------------------------------------------------------------------------


class TestSignatureCache:
    """A rewrite is seen by a reader nobody told.

    Each test primes the record reader with a first :func:`posture` call and then
    rewrites the record through :func:`rewrite_record`, which drops nothing, so
    the assertion is about the signature key and not about an invalidation.
    """

    def test_a_turn_off_lands_on_a_process_already_running(self, agent_root):
        switched_on(agent_root)
        assert posture.posture() == posture.POSTURE_WRITES

        rewrite_record(agent_root, {})

        assert posture.posture() == posture.POSTURE_SANDBOX

    def test_turning_it_on_is_seen_too(self, agent_root):
        write_record(agent_root, switch={})
        assert posture.posture() == posture.POSTURE_SANDBOX

        rewrite_record(agent_root, {"live": "on"})

        assert posture.posture() == posture.POSTURE_WRITES

    def test_a_same_size_rewrite_inside_one_clock_tick_is_still_seen(self, agent_root):
        """``{"va": "on"}`` and ``{"live": "on"}`` give records of the same size."""
        write_record(agent_root, switch={"va": "on"})
        assert posture.posture() == posture.POSTURE_SANDBOX

        rewrite_record(agent_root, {"va": "on"})  # prime the size of the rewrite
        size = control_context._record_path_under(agent_root).stat().st_size
        assert posture.posture() == posture.POSTURE_SANDBOX
        rewrite_record(agent_root, {"live": "on"})
        assert control_context._record_path_under(agent_root).stat().st_size == size

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

        # ``posture_env`` is the one exception: the read-only predicate's owner,
        # a standard-library leaf that loads nothing else of the package.
        assert not [
            name
            for name in imported
            if name.startswith("osprey_connectors") and name != "osprey_connectors.posture_env"
        ]
        assert not [name for name in imported if name.startswith("osprey.mcp_server")]
