# ruff: noqa: F811 - the launcher tests' fixtures are imported, then requested by name
"""The turn-state hook's CLI branch: the transcript pointer a host launch keeps.

``osprey chat`` and ``osprey query`` have no web terminal and no hub ledger, so
"this session" follows a ``/clear`` through a pointer file instead:
``<agent-data root>/cli-transcripts/<OSPREY_CLI_LAUNCH>``, holding the current
transcript id. The hook moves it on ``SessionStart`` (never a compaction) and on
``UserPromptSubmit``; ``osprey.stores.artifact_scope.current_scope`` reads it.

Pinned here: which payloads move the pointer and which do not, that a web
process never writes one, the seven-day prune of other launches' files, that a
nested launch keeps its own pointer, that each launcher exports a fresh launch
id, and — end to end, with the hook run as Claude Code runs it — that the id the
hook writes is the transcript ``current_scope`` resolves.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import osprey.templates.claude_code.claude.hooks.osprey_turn_state as turn_state
from osprey.build.claude_code_telemetry import HOST_SESSION_PREFIX
from osprey.cli.chat_cmd import chat
from osprey.cli.query_cmd import query
from osprey.stores.artifact_scope import CLI_POINTER_DIR, current_scope
from osprey_connectors.workspace import DEFAULT_AGENT_DATA_BASE_DIR
from tests.cli._scoped_subprocess import patch_subprocess
from tests.cli.test_host_launch_record import (  # noqa: F401 - fixtures requested by name
    _config,
    _deploy,
    _launch_stubs,
    _record_port,
    _restore_process_state,
    fake_ensure,
    real_record,
    runner,
)
from tests.fixtures.lifecycle_repo import (  # noqa: F401 - fixtures requested by name
    lifecycle_repo,
    lifecycle_repo_factory,
)

from .conftest import HOOKS_DIR

LAUNCH = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
OTHER_LAUNCH = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
TERMINAL_SESSION = f"{HOST_SESSION_PREFIX}cccccccc-cccc-4ccc-8ccc-cccccccccccc"
FIRST = "11111111-1111-4111-8111-111111111111"
CLEARED = "22222222-2222-4222-8222-222222222222"
DAY = 24 * 60 * 60


@pytest.fixture
def root(tmp_path):
    """The pointer root the in-process hook writes under (the conftest aims it here)."""
    return tmp_path


def _pointer(root: Path, launch: str = LAUNCH) -> Path:
    return root / CLI_POINTER_DIR / launch


def _cli_env(monkeypatch, *, launch=LAUNCH, terminal=TERMINAL_SESSION):
    monkeypatch.delenv("OSPREY_SESSION_ID", raising=False)
    monkeypatch.delenv("OSPREY_WEB_PORT", raising=False)
    monkeypatch.setenv("OSPREY_TERMINAL_SESSION", terminal)
    monkeypatch.setenv("OSPREY_CLI_LAUNCH", launch)


def _run(monkeypatch, event, session_id=CLEARED, **extra):
    payload = {"hook_event_name": event, "session_id": session_id, **extra}
    monkeypatch.setattr(turn_state.sys, "stdin", io.StringIO(json.dumps(payload)))
    return turn_state.main()


def _no_urlopen(monkeypatch):
    calls = []
    monkeypatch.setattr(
        turn_state.urllib.request, "urlopen", lambda *a, **k: calls.append(a) or None
    )
    return calls


# ---------------------------------------------------------------------------
# Which payloads move the pointer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source", ["startup", "resume", "clear"])
def test_a_session_start_writes_the_payload_transcript(monkeypatch, root, source, capsys):
    _cli_env(monkeypatch)
    posts = _no_urlopen(monkeypatch)

    assert _run(monkeypatch, "SessionStart", source=source) == 0

    assert _pointer(root).read_text() == f"{CLEARED}\n"
    assert posts == [], "a host launch has no web terminal to report to"
    assert capsys.readouterr().out == ""


def test_a_clear_rewrites_an_existing_pointer(monkeypatch, root):
    _cli_env(monkeypatch)
    _run(monkeypatch, "SessionStart", session_id=FIRST, source="startup")
    assert _pointer(root).read_text().strip() == FIRST

    _run(monkeypatch, "SessionStart", session_id=CLEARED, source="clear")

    assert _pointer(root).read_text().strip() == CLEARED
    leftovers = [p.name for p in (root / CLI_POINTER_DIR).iterdir() if p.name != LAUNCH]
    assert leftovers == [], "the atomic write leaves no temporary file behind"


def test_a_prompt_submit_writes_the_pointer(monkeypatch, root):
    _cli_env(monkeypatch)

    _run(monkeypatch, "UserPromptSubmit")

    assert _pointer(root).read_text().strip() == CLEARED


def test_a_compaction_does_not_move_the_pointer(monkeypatch, root):
    _cli_env(monkeypatch)
    _run(monkeypatch, "SessionStart", session_id=FIRST, source="startup")

    assert _run(monkeypatch, "SessionStart", session_id=CLEARED, source="compact") == 0

    assert _pointer(root).read_text().strip() == FIRST


@pytest.mark.parametrize("event", ["Stop", "StopFailure", "PreToolUse"])
def test_other_events_write_nothing(monkeypatch, root, event):
    _cli_env(monkeypatch)

    _run(monkeypatch, event)

    assert not (root / CLI_POINTER_DIR).exists()


@pytest.mark.parametrize("session_id", ["", None, 7, "two\nlines"])
def test_a_payload_without_a_usable_transcript_writes_nothing(monkeypatch, root, session_id):
    _cli_env(monkeypatch)

    assert _run(monkeypatch, "UserPromptSubmit", session_id=session_id) == 0

    assert not _pointer(root).exists()


# ---------------------------------------------------------------------------
# Which processes write one
# ---------------------------------------------------------------------------


def test_a_web_process_writes_no_pointer(monkeypatch, root):
    _cli_env(monkeypatch)
    monkeypatch.setenv("OSPREY_SESSION_ID", "33333333-3333-4333-8333-333333333333")
    monkeypatch.setenv("OSPREY_WEB_PORT", "9")
    posts = _no_urlopen(monkeypatch)

    assert _run(monkeypatch, "SessionStart", source="clear") == 0

    assert not (root / CLI_POINTER_DIR).exists()
    assert len(posts) == 1, "the web branch still reports the turn edge"


@pytest.mark.parametrize(
    "terminal", ["", "web-cccccccc-cccc-4ccc-8ccc-cccccccccccc", "cccccccc-chat-"]
)
def test_a_process_without_a_host_terminal_session_writes_nothing(monkeypatch, root, terminal):
    _cli_env(monkeypatch, terminal=terminal)

    _run(monkeypatch, "UserPromptSubmit")

    assert not (root / CLI_POINTER_DIR).exists()


def test_a_process_without_a_launch_id_writes_nothing(monkeypatch, root):
    _cli_env(monkeypatch)
    monkeypatch.delenv("OSPREY_CLI_LAUNCH")

    _run(monkeypatch, "UserPromptSubmit")

    assert not (root / CLI_POINTER_DIR).exists()


@pytest.mark.parametrize("launch", [".", "..", "../escape", "a/b", "a\\b", "nul\0byte"])
def test_an_unsafe_launch_name_writes_nothing(monkeypatch, root, launch):
    _cli_env(monkeypatch)
    # monkeypatch.setenv refuses a NUL byte; the hook must refuse it too.
    monkeypatch.setattr(turn_state, "_cli_launch", lambda: launch)

    assert _run(monkeypatch, "UserPromptSubmit") == 0

    assert not (root / CLI_POINTER_DIR).exists()
    assert not (root / "escape").exists()


def test_an_unwritable_root_fails_open(monkeypatch, tmp_path, capsys):
    _cli_env(monkeypatch)
    blocker = tmp_path / "file-not-dir"
    blocker.write_text("x")
    monkeypatch.setattr(turn_state, "_pointer_root", lambda _payload=None: str(blocker))

    assert _run(monkeypatch, "UserPromptSubmit") == 0
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# Pruning and nesting
# ---------------------------------------------------------------------------


def _age(path: Path, seconds: float) -> None:
    then = time.time() - seconds
    os.utime(path, (then, then))


def test_a_write_prunes_only_other_launches_past_the_bound(monkeypatch, root):
    directory = root / CLI_POINTER_DIR
    directory.mkdir()
    stale_other = directory / OTHER_LAUNCH
    young_other = directory / "dddddddd-dddd-4ddd-8ddd-dddddddddddd"
    own = directory / LAUNCH
    for path in (stale_other, young_other, own):
        path.write_text(f"{FIRST}\n")
    _age(stale_other, 8 * DAY)
    _age(young_other, 6 * DAY)
    _age(own, 30 * DAY)
    _cli_env(monkeypatch)

    _run(monkeypatch, "UserPromptSubmit")

    assert not stale_other.exists()
    assert young_other.read_text().strip() == FIRST
    assert own.read_text().strip() == CLEARED


def test_the_age_bound_is_seven_days():
    assert turn_state.POINTER_MAX_AGE_S == 7 * DAY


def test_a_nested_launch_leaves_the_outer_pointer_untouched(monkeypatch, root):
    # The outer `osprey chat` launch and the one started from its agent's Bash
    # share the inherited terminal session, but not the launch id.
    _cli_env(monkeypatch, launch=LAUNCH)
    _run(monkeypatch, "SessionStart", session_id=FIRST, source="startup")

    _cli_env(monkeypatch, launch=OTHER_LAUNCH)
    _run(monkeypatch, "SessionStart", session_id=CLEARED, source="startup")

    assert _pointer(root, LAUNCH).read_text().strip() == FIRST
    assert _pointer(root, OTHER_LAUNCH).read_text().strip() == CLEARED


def test_the_hook_constants_match_the_reader():
    assert turn_state.CLI_POINTER_DIR == CLI_POINTER_DIR
    assert turn_state.HOST_SESSION_PREFIX == HOST_SESSION_PREFIX


# ---------------------------------------------------------------------------
# End to end: the hook as Claude Code runs it, read back by current_scope
# ---------------------------------------------------------------------------


def _run_hook_process(repo: Path, env_extra: dict[str, str], payload: dict) -> str:
    env = {
        name: os.environ[name] for name in ("PATH", "PYTHONPATH", "TMPDIR") if name in os.environ
    }
    env.update({"HOME": str(repo.parent / "home"), "CLAUDE_PROJECT_DIR": str(repo), **env_extra})
    result = subprocess.run(
        [sys.executable, str(HOOKS_DIR / "osprey_turn_state.py")],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        cwd=repo,
        env=env,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    return result.stderr


def test_end_to_end_the_hook_writes_and_current_scope_reads_the_new_id(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    env = {"OSPREY_TERMINAL_SESSION": TERMINAL_SESSION, "OSPREY_CLI_LAUNCH": LAUNCH}
    agent_data = repo / DEFAULT_AGENT_DATA_BASE_DIR

    _run_hook_process(
        repo, env, {"hook_event_name": "SessionStart", "source": "startup", "session_id": FIRST}
    )
    assert current_scope(env, root=agent_data).transcript == FIRST

    _run_hook_process(
        repo, env, {"hook_event_name": "SessionStart", "source": "clear", "session_id": CLEARED}
    )

    scope = current_scope({**env, "CLAUDE_CODE_SESSION_ID": FIRST}, root=agent_data)
    assert scope.kind == "cli"
    assert scope.transcript == CLEARED


# ---------------------------------------------------------------------------
# The launchers export a fresh launch id
# ---------------------------------------------------------------------------

INHERITED = "inherited-launch"


def _assert_fresh(launch: str) -> None:
    assert launch != INHERITED
    assert uuid.UUID(launch).version == 4


@pytest.mark.usefixtures("fake_ensure")
def test_chat_exports_a_fresh_launch_id_even_when_one_is_inherited(
    runner, lifecycle_repo, monkeypatch
):
    repo = _deploy(lifecycle_repo, _config(lifecycle_repo), credentials=False)
    monkeypatch.setenv("OSPREY_CLI_LAUNCH", INHERITED)
    launched: list[dict] = []

    def fake_run(_argv, *_args, env=None, **_kwargs):
        launched.append(dict(env or {}))
        return SimpleNamespace(returncode=0)

    with patch_subprocess("osprey.cli.chat_cmd", side_effect=fake_run):
        first = runner.invoke(chat, ["--repo", str(repo)])
        second = runner.invoke(chat, ["--repo", str(repo)])

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    ids = [env["OSPREY_CLI_LAUNCH"] for env in launched]
    for launch in ids:
        _assert_fresh(launch)
    assert ids[0] != ids[1]


@pytest.mark.usefixtures("fake_ensure")
def test_query_exports_a_fresh_launch_id_even_when_one_is_inherited(
    runner, lifecycle_repo, monkeypatch
):
    repo = _deploy(lifecycle_repo, _config(lifecycle_repo), credentials=False)
    monkeypatch.setenv("OSPREY_CLI_LAUNCH", INHERITED)
    seen: dict = {}

    async def _run_query(*_args, **_kw):
        seen["launch"] = os.environ.get("OSPREY_CLI_LAUNCH")
        return MagicMock(text_blocks=["ok"], is_error=False, mcp_servers=[], tool_traces=[])

    with (
        patch("osprey.cli.query_cmd.run_query", new=AsyncMock(side_effect=_run_query)),
        patch("osprey.cli.query_cmd.read_only_disallowed_tools", return_value=[]),
        patch("osprey.cli.query_cmd.expected_mcp_servers", return_value=set()),
        patch("osprey.cli.query_cmd.evaluate_verdict", return_value=0),
    ):
        result = runner.invoke(query, ["--repo", str(repo), "hello"])

    assert result.exit_code == 0, result.output
    _assert_fresh(seen["launch"])
