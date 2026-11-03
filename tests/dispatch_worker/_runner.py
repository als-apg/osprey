"""Helpers for the suites that drive ``sdk_runner.run_dispatch`` without a project."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from osprey.agent_runner import ResultEvent


def stub_osprey_helpers(
    monkeypatch: pytest.MonkeyPatch,
    *,
    agent_data: Path | None = None,
    open_allowlist: bool = False,
) -> None:
    """Stub the deferred OSPREY helpers so ``run_dispatch`` runs without a project.

    Args:
        monkeypatch: The test's monkeypatch.
        agent_data: When given, the runner's agent-data root (where it creates
            the agent's Claude state directory).
        open_allowlist: Stub the tool allowlist open, for suites that drive
            the full client setup.
    """
    if agent_data is not None:
        monkeypatch.setattr(
            "osprey.agent_runner.artifact_resolve.deployed_agent_data_root", lambda: agent_data
        )
    if open_allowlist:
        monkeypatch.setattr(
            "osprey.agent_runner.sdk_context.make_tool_allowlist",
            lambda tools, denied=(): lambda *a, **k: None,
        )
    monkeypatch.setattr("osprey.agent_runner.clean_env.build_clean_env", lambda **kw: {})
    monkeypatch.setattr(
        "osprey.agent_runner.sdk_context.build_system_prompt", lambda *a, **k: "system"
    )
    monkeypatch.setattr("osprey_connectors.config.get_facility_timezone", lambda *a, **k: "UTC")


def result_event(**overrides: Any) -> ResultEvent:
    """A successful result record; any field can be overridden."""
    fields: dict[str, Any] = {
        "subtype": "success",
        "is_error": False,
        "num_turns": 1,
        "duration_ms": 0,
        "session_id": "s",
        "total_cost_usd": 0.1,
        "usage": None,
        "result": None,
        "api_error_status": None,
    }
    fields.update(overrides)
    return ResultEvent(**fields)


async def drain(queue: asyncio.Queue) -> list[dict]:
    """Every event already on ``queue``, in order, without waiting for more."""
    events = []
    while not queue.empty():
        events.append(await queue.get())
    return events
