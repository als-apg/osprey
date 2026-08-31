"""Tests for the per-session posture clamp in the executor's execution gates.

A web-terminal session switched to the sandbox posture spawns its child with
``OSPREY_EXECUTION_MODE=readonly``, and the MCP servers launched under that
session inherit it. The clamp is what makes the executor honour that posture:
an agent asking for ``execution_mode="readwrite"`` inside a sandboxed session
must be refused even though the *deployment* allows writes.

The semantics pinned here mirror ``osprey_connectors``' ``is_readonly_run``:
a **value** comparison, never a presence check. Any other value of the
variable — including ``"readwrite"`` and garbage — leaves both modes alone,
so the only thing that can sandbox a session is the sandbox posture itself.
"""

from __future__ import annotations

import json
import os

import pytest
from fastmcp.exceptions import ToolError

from osprey.audit import posture as posture_module
from osprey.mcp_server.python_executor.tools._execution_gates import enforce_posture_clamp

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _audit_zone(tmp_path, monkeypatch):
    """Redirect the audit zone into ``tmp_path`` for every test in this file.

    The clamp *records* its refusal, and ``writer.audit_dir`` resolves against
    the real project root — so without this a plain ``pytest`` run appends
    refusals that never happened to the deployment's own ledger, where an
    operator cannot tell them from the real ones.

    Autouse rather than requested by the two tests that fire the clamp today: a
    test added here later inherits the redirect instead of rediscovering the
    leak the hard way.
    """
    from osprey.audit import writer

    monkeypatch.setattr(writer, "audit_dir", lambda: tmp_path / "var" / "audit")
    return tmp_path / "var" / "audit"


def _envelope(exc_info) -> dict:
    """The structured error envelope ``make_error`` packed into the ToolError."""
    return json.loads(str(exc_info.value))


def test_readwrite_refused_under_sandbox_posture(monkeypatch):
    """The whole point: writes asked for inside a sandboxed session are refused."""
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")

    with pytest.raises(ToolError) as exc_info:
        enforce_posture_clamp("readwrite", tool="execute")

    envelope = _envelope(exc_info)
    assert envelope["error"] is True
    assert envelope["error_type"] == "safety_error"
    assert envelope["suggestions"]


def test_a_read_only_run_refusal_names_the_run_not_the_deployment_config(monkeypatch):
    """Two-vocabulary rule: nothing is wrong with the deployment config.

    Mirror of ``test_readonly_refusal_message_does_not_blame_deployment`` on the
    connector side — a refusal that mentions ``writes_enabled`` sends the
    operator to edit a config file that is not the gate.

    The environment source is a DEPLOYMENT-wide read-only run, so the remedy is
    the run's own switch. The header chip is named only to close the dead end:
    it already reads writes for this session and cannot lift the run.
    """
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")

    with pytest.raises(ToolError) as exc_info:
        enforce_posture_clamp("readwrite", tool="execute")

    envelope = _envelope(exc_info)
    text = envelope["error_message"] + " " + " ".join(envelope["suggestions"])
    assert "writes_enabled" not in text
    assert "readonly execution mode" in text.lower()
    assert "OSPREY_EXECUTION_MODE=readonly" in text
    # Named as the thing that does NOT lift it, never as the way out.
    assert "control-target chip in the header cannot lift" in text.lower()
    assert "switch the session to the writes posture" not in text.lower()


def test_readonly_passes_under_sandbox_posture(monkeypatch):
    """A readonly run is exactly what the sandbox posture permits."""
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")

    assert enforce_posture_clamp("readonly", tool="execute") is None


def test_no_mode_var_means_not_a_sandbox_run(monkeypatch):
    """Outside a postured session the variable is unset and the clamp is inert.

    Pins the same semantics as the connector test of this name: with no
    variable, the deployment gates alone decide, and *both* modes pass here.
    """
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)

    assert enforce_posture_clamp("readonly", tool="execute") is None
    assert enforce_posture_clamp("readwrite", tool="execute") is None


@pytest.mark.parametrize("value", ["readwrite", "READONLY", "", "sandbox", "true"])
def test_other_values_leave_both_modes_unchanged(monkeypatch, value):
    """Value comparison, not presence: only the exact ``readonly`` string clamps.

    ``readwrite`` is the writes posture and must not be mistaken for a sandbox;
    the remaining values are the ones a typo or a stale env would produce, and
    a presence check would sandbox the session on every one of them.
    """
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", value)

    assert enforce_posture_clamp("readonly", tool="execute") is None
    assert enforce_posture_clamp("readwrite", tool="execute") is None


# --------------------------------------------------------------------------
# The per-(session, target) posture: the clamp reads the store through posture()
# --------------------------------------------------------------------------
#
# A session narrows ONE control target from the header chip. Nothing respawns
# the executor for that, and nothing sets ``OSPREY_EXECUTION_MODE`` — setting it
# would sandbox every target at once. The clamp still has to refuse a readwrite
# run on a narrowed target, and still has to let one through on a target the
# operator left alone, which it does entirely through what ``posture.posture()``
# now answers.

SESSION_KEY = "11111111-2222-3333-4444-555555555555"


@pytest.fixture(autouse=True)
def _no_inherited_session(monkeypatch):
    """Clear the session markers so a developer's own shell cannot answer here.

    ``posture()`` consults the posture store only for a process carrying
    ``OSPREY_POSTURE_SESSION``; the tests above are about the environment alone
    and must not read a store because the terminal that started pytest happened
    to be a session child.
    """
    from osprey.audit import posture as posture_module
    from osprey_connectors import session_store

    for marker in (
        posture_module.POSTURE_SESSION_ENV_VAR,
        posture_module.CONTROL_TARGET_ENV_VAR,
        posture_module.OSPREY_AGENT_DATA_ROOT,
    ):
        monkeypatch.delenv(marker, raising=False)
    session_store.invalidate_cache()
    posture_module.invalidate_session_target_cache()
    yield
    session_store.invalidate_cache()
    posture_module.invalidate_session_target_cache()


@pytest.fixture
def session(tmp_path, monkeypatch):
    """A session child of a controls server, with a per-target posture store.

    Returns a handle whose ``.on(target)`` publishes the state record naming the
    session's target and whose ``.narrow(**entries)`` writes the store.
    """
    from osprey.audit import posture as posture_module
    from osprey_connectors import session_store

    root = tmp_path / "agent_data"
    directory = root / session_store.STATE_DIR_NAME
    directory.mkdir(parents=True)
    monkeypatch.setenv(posture_module.OSPREY_AGENT_DATA_ROOT, str(root))
    monkeypatch.setenv(posture_module.POSTURE_SESSION_ENV_VAR, SESSION_KEY)
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)

    class _Session:
        key = SESSION_KEY

        @staticmethod
        def on(target: str) -> None:
            pid = os.getpid()
            (directory / f"target_state_{pid}.json").write_text(
                json.dumps(
                    {
                        "target": target,
                        "generation": 0,
                        "server_pid": pid,
                        "owner_ppid": os.getppid(),
                        "targets": {},
                        "children": [],
                    }
                )
            )
            session_store.invalidate_cache()
            posture_module.invalidate_session_target_cache()

        @staticmethod
        def narrow(**entries: str) -> None:
            (directory / session_store.STORE_FILENAME).write_text(
                json.dumps({SESSION_KEY: entries})
            )
            session_store.invalidate_cache()
            posture_module.invalidate_session_target_cache()

    return _Session


def _executor_records(audit_root) -> list[dict]:
    """Every record on the executor surface, for the identity in force."""
    from osprey.audit.envelope import SURFACE_EXECUTOR
    from osprey.utils.identity import acting_identity

    path = audit_root / acting_identity() / f"{SURFACE_EXECUTOR}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_a_narrowed_session_target_refuses_a_readwrite_run(session, _audit_zone):
    """The narrowing lands on a session already running, with no respawn."""
    session.on("live")
    session.narrow(live="sandbox")

    with pytest.raises(ToolError) as exc_info:
        enforce_posture_clamp("readwrite", tool="execute")

    assert _envelope(exc_info)["error_type"] == "safety_error"


def test_a_narrowed_target_refusal_names_that_target(session, _audit_zone):
    """Per target, so the refusal says WHICH machine — and says nothing wider.

    This gate runs before every readwrite tool call, so "this session is in the
    sandbox posture" would read as a session-wide block to an operator whose
    session is working normally on every other target. The name has to be the
    one ``posture()`` clamped on, which is why both read the same resolver.
    """
    session.on("live")
    session.narrow(live="sandbox")

    with pytest.raises(ToolError) as exc_info:
        enforce_posture_clamp("readwrite", tool="execute")

    envelope = _envelope(exc_info)
    message = envelope["error_message"]
    assert "'live' control target" in message
    assert "for this session only" in message
    assert "This terminal session is in the sandbox posture" not in message
    assert "Turn writes back on for 'live' from the control-target chip in the header" in " ".join(
        envelope["suggestions"]
    )


def _clamp_fires(monkeypatch):
    """Pin ``posture()`` at sandbox so only the NAMING step is under test.

    The two cases below are about the window between the gate's two reads:
    ``posture()`` resolved a narrowed target and clamped, and the gate then
    resolves the target again to name it. Patching the resolver alone cannot
    stage that — ``posture()`` reads the very same function, so a resolver that
    answers ``None`` would simply stop the clamp firing at all.
    """
    monkeypatch.setattr(posture_module, "posture", lambda: posture_module.POSTURE_SANDBOX)


def test_a_target_lost_between_the_two_reads_names_no_machine(session, _audit_zone, monkeypatch):
    """The degraded cell: the clamp fired, but the target cannot be named now.

    A state file replaced between the two reads leaves the second answer
    ``None``. Inventing a machine name there would be worse than saying so: the
    store's rule with no resolvable target is that the MOST RESTRICTIVE entry
    decides, and which entry that was is not knowable here.
    """
    session.on("live")
    session.narrow(live="sandbox")
    _clamp_fires(monkeypatch)
    monkeypatch.setattr(posture_module, "session_control_target", lambda: None)

    with pytest.raises(ToolError) as exc_info:
        enforce_posture_clamp("readwrite", tool="execute")

    envelope = _envelope(exc_info)
    message = envelope["error_message"]
    assert "at least one control target" in message
    assert "most restrictive" in message
    assert "control-target chip in the header" in message
    assert "'live'" not in message
    assert "Turn writes back on from the control-target chip in the header" in " ".join(
        envelope["suggestions"]
    )


def test_a_raising_target_resolver_still_refuses(session, _audit_zone, monkeypatch):
    """Naming the target is a convenience; refusing is the contract.

    The resolver is documented never to raise, but this runs on the refusal
    path of every readwrite tool call — a surprise here must not turn a safety
    refusal into a 500.
    """

    def _explode() -> str | None:
        raise RuntimeError("state directory is on fire")

    session.on("live")
    session.narrow(live="sandbox")
    _clamp_fires(monkeypatch)
    monkeypatch.setattr(posture_module, "session_control_target", _explode)

    with pytest.raises(ToolError) as exc_info:
        enforce_posture_clamp("readwrite", tool="execute")

    assert "at least one control target" in _envelope(exc_info)["error_message"]


def test_the_refusal_files_reason_posture_and_posture_sandbox(session, _audit_zone):
    """The record an operator greps for: ``reason=posture``, ``posture=sandbox``.

    The posture field is not restated by the gate — it comes from
    ``posture.posture()``, which is exactly what the store now answers — so this
    also pins that a store-backed refusal is indistinguishable in the ledger
    from the session-wide one it replaces.
    """
    session.on("live")
    session.narrow(live="sandbox")

    with pytest.raises(ToolError):
        enforce_posture_clamp("readwrite", tool="execute")

    record = _executor_records(_audit_zone)[-1]
    assert record["reason"] == "posture"
    assert record["posture"] == "sandbox"
    assert record["decision"] == "refused"
    assert record["session"] == SESSION_KEY


def test_a_narrowing_on_another_target_leaves_this_session_alone(session, _audit_zone):
    """Read-only on the live machine must not stop work on the accelerator."""
    session.on("va")
    session.narrow(live="sandbox")

    assert enforce_posture_clamp("readwrite", tool="execute") is None
    assert _executor_records(_audit_zone) == []


def test_a_narrowed_target_still_permits_a_readonly_run(session, _audit_zone):
    """Reads are unaffected by the posture, per target as session-wide."""
    session.on("live")
    session.narrow(live="sandbox")

    assert enforce_posture_clamp("readonly", tool="execute") is None
    assert _executor_records(_audit_zone) == []


def test_an_unnarrowed_session_runs(session, _audit_zone):
    session.on("live")
    session.narrow()

    assert enforce_posture_clamp("readwrite", tool="execute") is None


def test_an_unresolvable_target_is_not_a_session_wide_clamp(session, _audit_zone):
    """No state record: the process cannot say which machine it is about.

    Refusing here would refuse writes on targets nobody narrowed. The
    fail-closed layer for one specific write is the connector's reference
    monitor, which takes the most restrictive entry when it cannot name a
    target; this gate answers the environment.
    """
    session.narrow(live="sandbox")

    assert enforce_posture_clamp("readwrite", tool="execute") is None
