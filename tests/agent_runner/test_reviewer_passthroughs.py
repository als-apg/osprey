"""``tools``, ``cwd`` and ``env_overrides`` on the agent-options builder and ``run_query``.

A reviewer runs with no tools, in a directory of its own, with a few variables
set over the project's routed environment. Each keyword touches only what it
names: the provider, the model, the proxy and the launcher keep resolving from
``project_dir``, and a keyword left at ``None`` yields today's options.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from claude_agent_sdk import ClaudeAgentOptions

from osprey.agent_runner import launcher, primitives
from osprey.agent_runner.runner import run_query

ROUTED_ENV = {"CLAUDECODE": "", "ANTHROPIC_AUTH_TOKEN": "routed-token", "SHARED": "routed"}
MODEL = "main-model"
PROXY_PORT = 43210
PROXIED_SPEC = SimpleNamespace(
    needs_proxy=True,
    upstream_base_url="https://upstream.example/v1",
    auth_env_var="ANTHROPIC_AUTH_TOKEN",
    provider="proxied",
)


@pytest.fixture(autouse=True)
def seen(monkeypatch: pytest.MonkeyPatch) -> dict[str, list]:
    """Stub the routing seams and record the directory and env each one is handed."""
    calls: dict[str, list] = {"dirs": [], "proxy": []}

    def recording(value):
        def seam(project_dir, *_a, **_k):
            calls["dirs"].append(project_dir)
            return value() if callable(value) else value

        return seam

    real_cli_path = launcher.project_agent_cli_path
    real_thinking = launcher.project_thinking_display

    def cli_path(project_dir, *a, **k):
        calls["dirs"].append(project_dir)
        return real_cli_path(project_dir, *a, **k)

    def thinking(project_dir, *a, **k):
        calls["dirs"].append(project_dir)
        return real_thinking(project_dir, *a, **k)

    def start_proxy(spec, env):
        calls["proxy"].append((spec, dict(env)))
        return PROXY_PORT

    monkeypatch.setattr(primitives.launcher, "project_agent_cli_path", cli_path)
    monkeypatch.setattr(primitives.launcher, "project_thinking_display", thinking)
    monkeypatch.setattr(primitives, "sdk_env", recording(lambda: dict(ROUTED_ENV)))
    monkeypatch.setattr(primitives, "resolve_default_model", recording(MODEL))
    monkeypatch.setattr(primitives, "_resolve_project_spec", recording(None))
    monkeypatch.setattr(primitives, "start_proxy_for", start_proxy)
    return calls


def _differing_fields(left: ClaudeAgentOptions, right: ClaudeAgentOptions) -> set[str]:
    return {
        f.name
        for f in fields(ClaudeAgentOptions)
        if getattr(left, f.name) != getattr(right, f.name)
    }


# ---------------------------------------------------------------------------
# All three left at None
# ---------------------------------------------------------------------------


def test_all_three_at_none_build_the_same_options_as_leaving_them_out(tmp_path: Path):
    today = primitives.build_agent_options(tmp_path, disallowed_tools=["Edit"])

    passed = primitives.build_agent_options(
        tmp_path, disallowed_tools=["Edit"], tools=None, cwd=None, env_overrides=None
    )

    assert _differing_fields(today, passed) == set()
    assert passed.tools is None
    assert passed.cwd == str(tmp_path)
    assert passed.env == {**ROUTED_ENV, **primitives.TASK_TOOLS_ENV}


def test_all_three_at_none_keep_a_supplied_env_verbatim(tmp_path: Path):
    supplied = {"ANTHROPIC_BASE_URL": "http://127.0.0.1:9"}

    options = primitives.build_agent_options(
        tmp_path, disallowed_tools=[], env=supplied, tools=None, cwd=None, env_overrides=None
    )

    assert options.env == {**supplied, **primitives.TASK_TOOLS_ENV}
    assert options.model is None


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------


def test_an_empty_tools_list_stays_an_empty_toolset(tmp_path: Path):
    baseline = primitives.build_agent_options(tmp_path, disallowed_tools=[])

    options = primitives.build_agent_options(tmp_path, disallowed_tools=[], tools=[])

    assert options.tools == []
    assert _differing_fields(baseline, options) == {"tools"}


def test_named_tools_are_forwarded_as_a_list(tmp_path: Path):
    options = primitives.build_agent_options(tmp_path, disallowed_tools=[], tools=("Read", "Grep"))

    assert options.tools == ["Read", "Grep"]


def test_an_empty_tools_tuple_stays_an_empty_toolset(tmp_path: Path):
    options = primitives.build_agent_options(tmp_path, disallowed_tools=[], tools=())

    assert options.tools == []


# ---------------------------------------------------------------------------
# cwd
# ---------------------------------------------------------------------------


def test_cwd_changes_only_the_working_directory(tmp_path: Path):
    project_dir = tmp_path / "project"
    elsewhere = tmp_path / "elsewhere"
    baseline = primitives.build_agent_options(project_dir, disallowed_tools=[])

    options = primitives.build_agent_options(project_dir, disallowed_tools=[], cwd=elsewhere)

    assert options.cwd == str(elsewhere)
    assert _differing_fields(baseline, options) == {"cwd"}


def test_cwd_leaves_every_resolution_on_the_project(tmp_path: Path, seen):
    project_dir = tmp_path / "project"
    elsewhere = tmp_path / "elsewhere"

    options = primitives.build_agent_options(project_dir, disallowed_tools=[], cwd=elsewhere)

    # Launcher twice, then provider env, model and spec.
    assert seen["dirs"] == [project_dir] * 5
    assert options.mcp_servers == str(project_dir / primitives.RENDERED_MCP_CONFIG)


# ---------------------------------------------------------------------------
# env_overrides
# ---------------------------------------------------------------------------


def test_every_override_is_present_empty_strings_included(tmp_path: Path):
    overrides = {"SHARED": "", "ADDED_EMPTY": "", "ADDED": "value"}

    options = primitives.build_agent_options(tmp_path, disallowed_tools=[], env_overrides=overrides)

    assert options.env["SHARED"] == ""
    assert options.env["ADDED_EMPTY"] == ""
    assert options.env["ADDED"] == "value"
    assert options.env["ANTHROPIC_AUTH_TOKEN"] == ROUTED_ENV["ANTHROPIC_AUTH_TOKEN"]


def test_an_override_wins_over_the_task_tools_switch(tmp_path: Path):
    switch = next(iter(primitives.TASK_TOOLS_ENV))

    options = primitives.build_agent_options(
        tmp_path, disallowed_tools=[], env_overrides={switch: "0"}
    )

    assert options.env[switch] == "0"


def test_overrides_change_only_the_env(tmp_path: Path):
    baseline = primitives.build_agent_options(tmp_path, disallowed_tools=[])

    options = primitives.build_agent_options(
        tmp_path, disallowed_tools=[], env_overrides={"ADDED": "value"}
    )

    assert _differing_fields(baseline, options) == {"env"}
    assert options.model == MODEL
    assert options.env == {**baseline.env, "ADDED": "value"}


def test_an_empty_overrides_mapping_changes_nothing(tmp_path: Path):
    baseline = primitives.build_agent_options(tmp_path, disallowed_tools=[])

    options = primitives.build_agent_options(tmp_path, disallowed_tools=[], env_overrides={})

    assert _differing_fields(baseline, options) == set()


def test_overrides_do_not_reach_the_proxy_start(
    tmp_path: Path, seen, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(primitives, "_resolve_project_spec", lambda *_a, **_k: PROXIED_SPEC)
    baseline = primitives.build_agent_options(tmp_path, disallowed_tools=[])

    options = primitives.build_agent_options(
        tmp_path,
        disallowed_tools=[],
        env_overrides={"ANTHROPIC_AUTH_TOKEN": "", "ADDED": "value"},
    )

    assert seen["proxy"][0] == seen["proxy"][1] == (PROXIED_SPEC, ROUTED_ENV)
    assert options.env["ANTHROPIC_BASE_URL"] == f"http://127.0.0.1:{PROXY_PORT}"
    assert options.env["ANTHROPIC_BASE_URL"] == baseline.env["ANTHROPIC_BASE_URL"]
    assert options.env["ANTHROPIC_AUTH_TOKEN"] == ""
    assert options.model == baseline.model


def test_overrides_apply_over_a_supplied_env_without_mutating_either(tmp_path: Path):
    supplied = {"SHARED": "supplied"}
    overrides = MappingProxyType({"SHARED": "", "ADDED": "value"})

    options = primitives.build_agent_options(
        tmp_path, disallowed_tools=[], env=supplied, env_overrides=overrides
    )

    assert options.env["SHARED"] == ""
    assert options.env["ADDED"] == "value"
    assert options.model is None
    assert supplied == {"SHARED": "supplied"}
    assert dict(overrides) == {"SHARED": "", "ADDED": "value"}


# ---------------------------------------------------------------------------
# run_query
# ---------------------------------------------------------------------------


async def _capture_run_query_options(project_dir: Path, **keywords) -> ClaudeAgentOptions:
    captured: list[ClaudeAgentOptions] = []

    async def no_messages():
        return
        yield

    client = MagicMock()
    client.query = AsyncMock(return_value=None)
    client.receive_response = MagicMock(return_value=no_messages())
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=client)
    context.__aexit__ = AsyncMock(return_value=False)

    def capture(options: ClaudeAgentOptions) -> MagicMock:
        captured.append(options)
        return context

    with (
        patch("osprey.agent_runner.runner.ClaudeSDKClient", side_effect=capture),
        patch("osprey.agent_runner.primitives.await_mcp_ready", new=AsyncMock(return_value=[])),
        patch("osprey.agent_runner.primitives.expected_mcp_servers", return_value=set()),
    ):
        await run_query(project_dir, "query", disallowed_tools=[], **keywords)

    assert len(captured) == 1
    return captured[0]


@pytest.mark.asyncio
async def test_run_query_forwards_all_three(tmp_path: Path):
    elsewhere = tmp_path / "elsewhere"

    options = await _capture_run_query_options(
        tmp_path, tools=[], cwd=elsewhere, env_overrides={"ADDED": ""}
    )

    assert options.tools == []
    assert options.cwd == str(elsewhere)
    assert options.env["ADDED"] == ""
    assert options.model == MODEL


@pytest.mark.asyncio
async def test_run_query_leaves_all_three_at_none_by_default(tmp_path: Path):
    options = await _capture_run_query_options(tmp_path)

    assert (
        _differing_fields(options, primitives.build_agent_options(tmp_path, disallowed_tools=[]))
        == set()
    )
