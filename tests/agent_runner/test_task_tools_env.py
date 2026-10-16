"""Every Claude Code process OSPREY starts is offered the task-list tools.

The CLI offers ``TaskCreate``, ``TaskGet``, ``TaskList`` and ``TaskUpdate`` by
default only to the models it recognises by name; ``CLAUDE_CODE_ENABLE_TODO_TOOLS``
offers them on every model. Each launch path composes its own environment, so
each is checked here: the shared base environment (the terminal's PTY child,
``osprey chat``), the SDK operator and chat sessions, and every set of SDK
options, whoever routed the run. The dispatch worker's run is checked beside
its other environment tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.agent_runner import primitives
from osprey.agent_runner.clean_env import build_base_child_env, build_clean_env
from osprey.interfaces.web_terminal.operator_session import build_operator_child_env
from osprey.interfaces.web_terminal.pty_manager import build_pty_env

SWITCH = "CLAUDE_CODE_ENABLE_TODO_TOOLS"


@pytest.fixture(params=[None, "0"], ids=["unset", "off-in-the-parent"])
def parent_env(request, monkeypatch) -> None:
    """The parent process either does not name the switch or turns it off."""
    if request.param is None:
        monkeypatch.delenv(SWITCH, raising=False)
    else:
        monkeypatch.setenv(SWITCH, request.param)


@pytest.mark.usefixtures("parent_env")
def test_the_base_child_env_turns_the_task_list_tools_on():
    assert build_base_child_env()[SWITCH] == "1"


@pytest.mark.usefixtures("parent_env")
def test_the_terminal_child_env_turns_the_task_list_tools_on():
    assert build_pty_env()[SWITCH] == "1"


@pytest.mark.usefixtures("parent_env")
def test_the_sdk_child_env_turns_the_task_list_tools_on(tmp_path: Path):
    assert build_clean_env(project_cwd=str(tmp_path))[SWITCH] == "1"


@pytest.mark.usefixtures("parent_env")
def test_the_operator_and_chat_session_env_turns_the_task_list_tools_on(tmp_path: Path):
    assert build_operator_child_env(str(tmp_path))[SWITCH] == "1"


@pytest.mark.parametrize(
    "supplied",
    [{}, {"ANTHROPIC_BASE_URL": "http://127.0.0.1:9"}, {SWITCH: "0"}],
    ids=["empty", "routed-by-the-caller", "off-in-the-supplied-env"],
)
def test_sdk_options_turn_the_task_list_tools_on_for_a_supplied_env(
    tmp_path: Path, supplied: dict[str, str]
):
    options = primitives.build_agent_options(tmp_path, disallowed_tools=[], env=supplied)

    assert options.env[SWITCH] == "1"
    assert SWITCH not in supplied or supplied[SWITCH] == "0", "the caller's dict is not mutated"


def test_sdk_options_turn_the_task_list_tools_on_when_the_builder_routes_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(primitives, "sdk_env", lambda *_a, **_k: {"CLAUDECODE": ""})
    monkeypatch.setattr(primitives, "resolve_default_model", lambda *_a, **_k: "main-model")
    monkeypatch.setattr(primitives, "_resolve_project_spec", lambda *_a, **_k: None)

    options = primitives.build_agent_options(tmp_path, disallowed_tools=[])

    assert options.env[SWITCH] == "1"
