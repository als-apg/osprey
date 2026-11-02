"""A dispatch job's control-system writes follow its owner, and a job with none is refused.

A dispatched job runs with ``OSPREY_DISPATCH_RUN=1`` (``sdk_runner`` stamps it)
and, when the dispatcher attributed the fire to a person, ``OSPREY_CONTROL_OWNER``.
The connector's reference monitor decides each write by the write verdict, so
under a bound control-context tree an owned job follows that owner's switch
and ``login_tiers`` rung, and a job that belongs to nobody — a cron fire — is
refused as ``owner_unscheduled``: no switch covers it.

Every case drives ``write_channel`` on a stub connector, through the real
monitor, against a real tree on disk. The last case builds the agent's
environment with the real ``run_dispatch`` and the sandbox child env from it,
so a sandboxed write in an owner-less job gets the same answer.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from osprey.agent_runner import ResultEvent
from osprey.mcp_server.dispatch_worker import sdk_runner
from osprey.mcp_server.sandbox_env import scrub_sandbox_child_env
from osprey_connectors import control_context, posture_store
from osprey_connectors.control_system._base import WriteOutcome
from osprey_connectors.refusals import refusal_phrase
from osprey_connectors.types import TARGET_LIVE
from osprey_connectors_tests._harness import (
    REFUSAL_BAN_LIST,
    VERDICT_ENV_VARS,
    write_control_context,
    write_payload,
)
from osprey_connectors_tests._harness._write_verdict_cases import case_environment
from osprey_connectors_tests._write_fakes import RecordingConnector, config_reader

__all__ = ["case_environment"]

CHANNEL = "SR:CORR:1:SP"

#: A deployment whose real machine allows writes, with a map naming two logins:
#: ``alice`` may write, ``bob`` was demoted. ``carol`` is not in it.
SECTION: dict[str, Any] = {
    "type": "epics",
    "writes": "allowed",
    "login_tiers": {"alice": "allowed", "bob": "never"},
}


@pytest.fixture(autouse=True)
def clean_environment(case_environment):
    """No stamped owner, tree, pin, dispatch mark or run mode."""


@pytest.fixture(autouse=True)
def deployment(monkeypatch):
    """Serve :data:`SECTION` through ``get_config_value``."""
    monkeypatch.setattr("osprey_connectors.config.get_config_value", config_reader(SECTION))


@pytest.fixture
def tree(tmp_path, monkeypatch) -> Path:
    """A provisioned control-context tree, bound for this process."""
    base = tmp_path / "shared" / posture_store.STATE_DIR_NAME
    base.mkdir(parents=True)
    (base / posture_store.CONTROL_TREE_MARKER_NAME).write_text("provisioned\n", encoding="utf-8")
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, str(base))
    posture_store.invalidate_cache()
    return base


def _set_switch(tree: Path, owner: str, state: str) -> None:
    """Write *owner*'s record in *tree* with the real machine's switch at *state*."""
    payload = control_context.ControlContext(
        target=TARGET_LIVE, generation=3, posture={}, switch_epoch=1
    ).to_payload()
    payload["switch"] = {TARGET_LIVE: state}
    write_payload(tree / owner / control_context.RECORD_FILENAME, payload)


def _dispatch_run(monkeypatch, owner: str | None) -> None:
    """This process is a dispatch job, attributed to *owner* (``None``: nobody)."""
    monkeypatch.setenv(posture_store.DISPATCH_RUN_ENV_VAR, "1")
    if owner is None:
        monkeypatch.delenv(posture_store.CONTROL_OWNER_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(posture_store.CONTROL_OWNER_ENV_VAR, owner)


def _connector() -> RecordingConnector:
    """A connector stamped with the real machine, as the factory stamps one."""
    connector = RecordingConnector()
    connector._connector_type = "epics"
    connector._control_target = TARGET_LIVE
    return connector


def _refused(result, reason: str) -> None:
    assert result.outcome is WriteOutcome.REFUSED
    assert result.refusal_reason == "WRITES_DISABLED"
    assert refusal_phrase(reason, "real") in result.error_message
    assert REFUSAL_BAN_LIST.search(result.error_message) is None


@pytest.mark.asyncio
async def test_a_cron_run_with_no_owner_is_refused_as_unscheduled(monkeypatch, tree):
    # Arrange: every login's switch is on, so no record is the reason.
    for login in ("alice", "bob", "carol"):
        _set_switch(tree, login, "asked")
    _dispatch_run(monkeypatch, None)
    connector = _connector()

    # Act
    result = await connector.write_channel(CHANNEL, 0.5)

    # Assert
    _refused(result, "owner_unscheduled")
    assert connector.writes == []


@pytest.mark.asyncio
async def test_a_cron_run_on_the_host_is_refused_as_unscheduled(monkeypatch, tmp_path):
    """No tree bound: the process account is not an owner for a dispatch job."""
    write_control_context(tmp_path / "data", owned_by=None, switch={TARGET_LIVE: "asked"})
    _dispatch_run(monkeypatch, None)
    connector = _connector()

    result = await connector.write_channel(CHANNEL, 0.5)

    _refused(result, "owner_unscheduled")
    assert connector.writes == []


@pytest.mark.asyncio
async def test_an_owned_run_with_the_switch_on_writes(monkeypatch, tree):
    _set_switch(tree, "alice", "asked")
    _dispatch_run(monkeypatch, "alice")
    connector = _connector()

    result = await connector.write_channel(CHANNEL, 0.5)

    assert result.outcome is WriteOutcome.CONFIRMED
    assert connector.writes == [(CHANNEL, 0.5)]


@pytest.mark.asyncio
async def test_an_owned_run_with_the_switch_off_is_refused(monkeypatch, tree):
    _set_switch(tree, "alice", "off")
    _dispatch_run(monkeypatch, "alice")
    connector = _connector()

    result = await connector.write_channel(CHANNEL, 0.5)

    _refused(result, "switch_off")
    assert connector.writes == []


@pytest.mark.asyncio
async def test_an_owner_the_map_does_not_name_is_refused(monkeypatch, tree):
    """``carol``'s switch is on, but the map does not say what she may do."""
    _set_switch(tree, "carol", "asked")
    _dispatch_run(monkeypatch, "carol")
    connector = _connector()

    result = await connector.write_channel(CHANNEL, 0.5)

    _refused(result, "owner_unknown")
    assert connector.writes == []


@pytest.mark.asyncio
async def test_a_demoted_owner_with_a_stale_on_is_refused(monkeypatch, tree):
    """``bob``'s ``"asked"`` predates his demotion; the tier rung comes first."""
    _set_switch(tree, "bob", "asked")
    _dispatch_run(monkeypatch, "bob")
    connector = _connector()

    result = await connector.write_channel(CHANNEL, 0.5)

    _refused(result, "tier_never")
    assert connector.writes == []
    # The stale "asked" stays on disk, inert.
    record = tree / "bob" / control_context.RECORD_FILENAME
    assert '"asked"' in record.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_turning_off_mid_run_refuses_the_next_write(monkeypatch, tree):
    _set_switch(tree, "alice", "asked")
    _dispatch_run(monkeypatch, "alice")
    connector = _connector()
    first = await connector.write_channel(CHANNEL, 0.5)
    assert first.outcome is WriteOutcome.CONFIRMED

    # Act: the owner turns the switch off while the job is still running.
    _set_switch(tree, "alice", "off")
    second = await connector.write_channel(CHANNEL, 0.6)

    # Assert: the same connector, no cache drop, refuses the next write.
    _refused(second, "switch_off")
    assert connector.writes == [(CHANNEL, 0.5)]


# --------------------------------------------------------------------------- #
# The sandbox child of a dispatch job
# --------------------------------------------------------------------------- #


async def _agent_env(monkeypatch, tmp_path, worker_env: dict[str, str], **kwargs) -> dict:
    """The environment ``run_dispatch`` hands the agent, over *worker_env*."""
    monkeypatch.setattr(
        "osprey.agent_runner.artifact_resolve.deployed_agent_data_root",
        lambda: tmp_path / "stub-agent-data",
    )
    monkeypatch.setattr(
        "osprey.agent_runner.clean_env.build_clean_env", lambda **kw: dict(worker_env)
    )
    monkeypatch.setattr(
        "osprey.agent_runner.sdk_context.build_system_prompt", lambda *a, **k: "system"
    )
    monkeypatch.setattr(
        "osprey.agent_runner.sdk_context.make_tool_allowlist",
        lambda tools, denied=(): lambda *a, **k: None,
    )
    monkeypatch.setattr("osprey_connectors.config.get_facility_timezone", lambda *a, **k: "UTC")
    captured: dict = {}

    async def fake_stream(project_dir, prompt, **kw):  # noqa: ARG001 - stream_query signature
        captured["env"] = kw["env"]
        yield ResultEvent(
            subtype="success",
            is_error=False,
            num_turns=1,
            duration_ms=0,
            session_id="s",
            total_cost_usd=0.0,
            usage=None,
            result=None,
            api_error_status=None,
        )

    monkeypatch.setattr(sdk_runner, "stream_query", fake_stream)
    await sdk_runner.run_dispatch("do it", ["Read"], event_queue=asyncio.Queue(), **kwargs)
    return captured["env"]


def _become(monkeypatch, child: dict[str, str]) -> None:
    """Make this process's osprey environment exactly *child*'s osprey names."""
    for name in VERDICT_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    for name, value in child.items():
        if name.startswith("OSPREY_"):
            monkeypatch.setenv(name, value)
    posture_store.invalidate_cache()


@pytest.mark.asyncio
async def test_a_sandboxed_write_in_an_owner_less_job_is_refused_as_unscheduled(
    monkeypatch, tmp_path, tree
):
    # Arrange: the worker runs with the tree bound; the fire carries no owner.
    for login in ("alice", "bob", "carol"):
        _set_switch(tree, login, "asked")
    worker_env = {posture_store.CONTROL_CONTEXT_TREE_ENV_VAR: str(tree), "PATH": "/usr/bin"}
    agent_env = await _agent_env(monkeypatch, tmp_path, worker_env, run_id="run-1")

    # Act: the executor builds its child's environment from the agent's.
    child = scrub_sandbox_child_env(agent_env)
    _become(monkeypatch, child)
    connector = _connector()
    result = await connector.write_channel(CHANNEL, 0.5)

    # Assert: the child still knows it is a dispatch job, and refuses.
    assert agent_env[posture_store.DISPATCH_RUN_ENV_VAR] == "1"
    assert child[posture_store.DISPATCH_RUN_ENV_VAR] == "1"
    assert posture_store.CONTROL_OWNER_ENV_VAR not in child
    _refused(result, "owner_unscheduled")
    assert connector.writes == []


@pytest.mark.asyncio
async def test_a_sandboxed_write_in_an_owned_job_follows_the_owner(monkeypatch, tmp_path, tree):
    """The control for the case above: the same child env, with an owner, writes."""
    _set_switch(tree, "alice", "asked")
    worker_env = {posture_store.CONTROL_CONTEXT_TREE_ENV_VAR: str(tree), "PATH": "/usr/bin"}
    agent_env = await _agent_env(monkeypatch, tmp_path, worker_env, run_id="run-2", owner="alice")

    child = scrub_sandbox_child_env(agent_env)
    _become(monkeypatch, child)
    connector = _connector()
    result = await connector.write_channel(CHANNEL, 0.5)

    assert child[posture_store.DISPATCH_RUN_ENV_VAR] == "1"
    assert child[posture_store.CONTROL_OWNER_ENV_VAR] == "alice"
    assert result.outcome is WriteOutcome.CONFIRMED
    assert connector.writes == [(CHANNEL, 0.5)]
