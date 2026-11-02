"""The agent cannot reach what decides how its own code is reviewed.

Three files and one config family steer the review: the reviewer's prompt, the
facility's ``profile.yml``, the rendered ``build/config.yml`` and the
``code_review.*`` keys in it. An agent that could change any of them could turn
off, or talk its way past, the check on its own code. Every route an agent has to
those files ends in an existing guard, and this module proves each one against a
real scratch deployment:

================================  ==========================================
Route                             Guard
================================  ==========================================
``execute`` / ``execute_file``    path policy, in readonly and in readwrite
``Write`` / ``NotebookEdit``      the memory-guard hook
``setup_patch`` of a key          the protected-key table
``setup_patch`` of a file         the patchable-file whitelist
a prompt under agent data         ``resolve_prompt``
================================  ==========================================

An unprotected route is a failure here, never an expected one, so nothing in this
module is marked ``xfail`` or ``skip``. Each group also shows one write the same
guard lets through, so a guard that refused everything could not pass.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from osprey.audit import posture
from osprey.cli.profile_conventions import (
    is_protected_key,
    is_protected_key_path,
    protected_view,
)
from osprey.mcp_server.python_executor.executor import ExecutionResult
from osprey.mcp_server.workspace.tools import setup as setup_tools
from osprey.services.code_review.settings import (
    DEFAULT_PROMPT,
    ReviewConfigError,
    ReviewSettings,
    resolve_prompt,
)
from osprey_connectors_tests._write_fakes import ARMED_SECTION, ARMED_TARGET, arm_switch
from tests.hooks.conftest import HOOKS_DIR, _curated_env
from tests.mcp_server.conftest import assert_raises_error, get_tool_fn

SETUP_MOD = "osprey.mcp_server.workspace.tools.setup"

#: The rendered config, which both the running agent and ``setup_patch`` read.
RENDERED_CONFIG = "build/config.yml"

#: Every file that steers the review, as a deployment-relative path.
REVIEW_FILES = (DEFAULT_PROMPT, "profile.yml", RENDERED_CONFIG)

#: Literal writes, the only spelling the static path policy can judge.
WRITE_SPELLINGS = {
    "open": "open({path!r}, 'w')\n",
    "pathlib": "from pathlib import Path\nPath({path!r}).write_text('rewritten')\n",
}

#: What the path policy's refusal says, matched on its own words.
SAYS_PATH = "location the deployment protects"

#: Every shipped key, the family itself and keys nothing reads yet: the family is
#: closed to the agent whatever it grows into.
CODE_REVIEW_KEYS = (
    "code_review",
    "code_review.enabled",
    "code_review.level",
    "code_review.prompt",
    "code_review.modes",
    "code_review.time_limit",
    "code_review.not_a_key_yet",
    "code_review.not_a_key_yet.below",
)

#: Keys that share a word with the family and must stay open: a matcher that
#: protected everything would otherwise satisfy this module.
NEAR_MISSES = ("code_reviewer.enabled", "code_reviews", "services.code_review.enabled")

_FILE_ROUTE = pytest.mark.parametrize("relpath", REVIEW_FILES)
_MODE = pytest.mark.parametrize("execution_mode", ["readonly", "readwrite"])


@pytest.fixture
def deployment(tmp_path, monkeypatch) -> Path:
    """A scratch deployment holding the prompt, the profile and the render."""
    import osprey.mcp_server.python_executor.executor as executor

    root = (tmp_path / "repo").resolve()
    contents = {
        DEFAULT_PROMPT: "You review code.\n",
        "profile.yml": "name: scratch\n",
        RENDERED_CONFIG: yaml.dump(
            {
                "code_review": {"enabled": True, "level": "block"},
                "ui": {"theme": "light"},
            }
        ),
    }
    for relpath, text in contents.items():
        path = root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(executor, "_resolve_project_root", lambda: root)
    monkeypatch.chdir(root)
    return root


def _bytes_of(root: Path) -> dict[str, bytes]:
    return {relpath: (root / relpath).read_bytes() for relpath in REVIEW_FILES}


# ---------------------------------------------------------------------------
# execute and execute_file: the path policy
# ---------------------------------------------------------------------------


@pytest.fixture
def armed_deployment(deployment, control_context_root, monkeypatch):
    """The deployment with writes allowed, so path policy is the gate that answers."""
    from osprey_connectors import posture_store

    monkeypatch.delenv("OSPREY_CONTROL_TARGET", raising=False)
    monkeypatch.setattr(posture, "control_system_section", lambda: dict(ARMED_SECTION))
    arm_switch(monkeypatch, control_context_root, ARMED_TARGET, stamp=False)
    yield deployment
    posture_store.invalidate_cache()


@pytest.fixture
def launch():
    """The subprocess backend, replaced; a refusal must leave it unawaited."""
    result = ExecutionResult(
        success=True,
        stdout="",
        stderr="",
        figures=[],
        execution_method_used="subprocess",
        execution_time_seconds=0.1,
    )
    mock = AsyncMock(return_value=result)
    with patch("osprey.mcp_server.python_executor.executor.execute_code", mock):
        yield mock


async def _execute(tool_name: str, root: Path, code: str, execution_mode: str):
    """Run *code* through one executor tool; ``execute_file`` reads it from a script."""
    if tool_name == "execute":
        from osprey.mcp_server.python_executor.tools.python_execute import execute

        return await get_tool_fn(execute)(
            code=code,
            description="protection probe",
            execution_mode=execution_mode,
            save_output=False,
        )

    from osprey.mcp_server.python_executor.tools.python_execute_file import execute_file

    script = root / "probe.py"
    script.write_text(code, encoding="utf-8")
    return await get_tool_fn(execute_file)(
        file_path=str(script),
        description="protection probe",
        execution_mode=execution_mode,
        save_output=False,
    )


_TOOL = pytest.mark.parametrize("tool_name", ["execute", "execute_file"])


@_TOOL
@_MODE
@_FILE_ROUTE
@pytest.mark.parametrize("spelling", sorted(WRITE_SPELLINGS))
async def test_execute_cannot_write_a_review_file(
    tool_name, execution_mode, relpath, spelling, armed_deployment, launch
):
    code = WRITE_SPELLINGS[spelling].format(path=relpath)
    before = _bytes_of(armed_deployment)

    with assert_raises_error(error_type="safety_error") as ctx:
        await _execute(tool_name, armed_deployment, code, execution_mode)

    assert SAYS_PATH in ctx["envelope"]["error_message"]
    launch.assert_not_awaited()
    assert _bytes_of(armed_deployment) == before


@_MODE
@_FILE_ROUTE
async def test_execute_cannot_write_a_review_file_by_absolute_path(
    execution_mode, relpath, armed_deployment, launch
):
    code = WRITE_SPELLINGS["open"].format(path=str(armed_deployment / relpath))

    with assert_raises_error(error_type="safety_error") as ctx:
        await _execute("execute", armed_deployment, code, execution_mode)

    assert SAYS_PATH in ctx["envelope"]["error_message"]
    launch.assert_not_awaited()


@_TOOL
@_MODE
async def test_execute_may_still_write_elsewhere(
    tool_name, execution_mode, armed_deployment, launch
):
    code = WRITE_SPELLINGS["open"].format(path="notes.txt")

    await _execute(tool_name, armed_deployment, code, execution_mode)

    launch.assert_awaited_once()


# ---------------------------------------------------------------------------
# Write and NotebookEdit: the memory-guard hook
# ---------------------------------------------------------------------------


@pytest.fixture
def run_guard(deployment, tmp_path, monkeypatch):
    """Run the shipped memory-guard hook the way Claude Code does, in a subprocess.

    The hook resolves its allowed areas from ``$HOME`` and the project dir, so both
    point into this test's scratch tree.
    """
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(deployment))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.delenv("OSPREY_CONTROL_OWNER", raising=False)

    def run(tool_name: str, tool_input: dict, **payload_extra) -> dict:
        payload = {
            "tool_name": tool_name,
            "tool_input": tool_input,
            "cwd": str(deployment),
            **payload_extra,
        }
        result = subprocess.run(
            [sys.executable, str(HOOKS_DIR / "osprey_memory_guard.py")],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            env=_curated_env(),
            cwd=str(deployment),
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout.strip().splitlines()[-1])["hookSpecificOutput"]

    return run


#: How each tool names its target and what else its input carries.
_GUARDED_TOOL_INPUT = {
    "Write": lambda path: {"file_path": path, "content": "rewritten"},
    "NotebookEdit": lambda path: {"notebook_path": path, "new_source": "rewritten"},
}
_GUARDED_TOOL = pytest.mark.parametrize("tool_name", sorted(_GUARDED_TOOL_INPUT))


@_GUARDED_TOOL
@_FILE_ROUTE
@pytest.mark.parametrize("spelled", ["absolute", "relative"])
def test_agent_tool_cannot_write_a_review_file(tool_name, relpath, spelled, deployment, run_guard):
    target = str(deployment / relpath) if spelled == "absolute" else relpath
    before = _bytes_of(deployment)

    decision = run_guard(tool_name, _GUARDED_TOOL_INPUT[tool_name](target))

    assert decision["permissionDecision"] == "deny"
    assert _bytes_of(deployment) == before


@_FILE_ROUTE
def test_planning_does_not_open_a_review_file_to_write(relpath, deployment, run_guard):
    decision = run_guard(
        "Write", _GUARDED_TOOL_INPUT["Write"](str(deployment / relpath)), permission_mode="plan"
    )

    assert decision["permissionDecision"] == "deny"


@_GUARDED_TOOL
def test_agent_tool_may_still_write_where_the_guard_says(tool_name, deployment, run_guard):
    """The hook's own refusal names an area; a write there goes through."""
    refusal = run_guard(tool_name, _GUARDED_TOOL_INPUT[tool_name](str(deployment / DEFAULT_PROMPT)))
    allowed = re.search(r"Allowed director(?:y|ies):\s*(\S+)/", refusal["permissionDecisionReason"])
    assert allowed, refusal["permissionDecisionReason"]
    name = "note.md" if tool_name == "Write" else "note.ipynb"

    decision = run_guard(tool_name, _GUARDED_TOOL_INPUT[tool_name](f"{allowed[1]}/{name}"))

    assert decision["permissionDecision"] == "allow"


# ---------------------------------------------------------------------------
# setup_patch: the protected-key table and the patchable-file whitelist
# ---------------------------------------------------------------------------


@pytest.fixture
def patch_tool(deployment):
    """``setup_patch`` aimed at the deployment's rendered config."""
    with (
        patch(f"{SETUP_MOD}.resolve_config_path", return_value=deployment / RENDERED_CONFIG),
        patch(f"{SETUP_MOD}.notify_agent_activity_async"),
    ):
        yield get_tool_fn(setup_tools.setup_patch)


@pytest.mark.parametrize("key", CODE_REVIEW_KEYS)
@pytest.mark.parametrize("value", ["false", "{enabled: false}"], ids=["scalar", "block"])
async def test_setup_patch_refuses_every_code_review_key(key, value, deployment, patch_tool):
    before = _bytes_of(deployment)

    with assert_raises_error(error_type="protected_key") as ctx:
        await patch_tool(file="config.yml", key_path=key, value=value)

    assert key in ctx["envelope"]["error_message"]
    assert _bytes_of(deployment) == before


@pytest.mark.parametrize("key", NEAR_MISSES)
async def test_setup_patch_still_writes_a_near_miss(key, deployment, patch_tool):
    await patch_tool(file="config.yml", key_path=key, value="sentinel")

    rendered = yaml.safe_load((deployment / RENDERED_CONFIG).read_text(encoding="utf-8"))
    node = rendered
    for segment in key.split("."):
        node = node[segment]
    assert node == "sentinel"
    assert rendered["code_review"] == {"enabled": True, "level": "block"}


@pytest.mark.parametrize(
    "file",
    [*REVIEW_FILES, f"../{DEFAULT_PROMPT}", "./config.yml"],
)
async def test_setup_patch_refuses_any_file_but_the_patchable_ones(file, deployment, patch_tool):
    before = _bytes_of(deployment)

    with assert_raises_error(error_type="validation_error") as ctx:
        await patch_tool(file=file, key_path="anything", value="rewritten")

    assert "not patchable" in ctx["envelope"]["error_message"]
    assert _bytes_of(deployment) == before


@pytest.mark.parametrize("key", CODE_REVIEW_KEYS)
def test_table_protects_the_whole_family(key):
    assert is_protected_key("config.yml", key)
    assert is_protected_key_path("config.yml", tuple(key.split(".")))


@pytest.mark.parametrize("key", NEAR_MISSES)
def test_table_leaves_near_misses_open(key):
    assert not is_protected_key("config.yml", key)
    assert not is_protected_key_path("config.yml", tuple(key.split(".")))


def test_table_family_reaches_the_protected_view():
    """The web terminal's config routes judge a rewrite through this view."""
    view = protected_view(
        "config.yml", {"code_review": {"enabled": True}, "ui": {"theme": "light"}}
    )

    assert any(path[0] == "code_review" for path in view)
    assert not any(path[0] == "ui" for path in view)


# ---------------------------------------------------------------------------
# A prompt under the agent's own data root: resolve_prompt
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "base_dir",
    [
        lambda root: "data",
        lambda root: "data/code_review",
        lambda root: str(root / "data" / "code_review"),
    ],
    ids=["over-all-data", "over-the-prompt", "absolute"],
)
def test_prompt_under_the_agent_data_root_is_refused(base_dir, deployment):
    config = {"agent_data": {"base_dir": base_dir(deployment)}}

    with pytest.raises(ReviewConfigError, match="agent"):
        resolve_prompt(ReviewSettings(enabled=True), deployment, config)


def test_prompt_outside_the_agent_data_root_is_read(deployment):
    text = resolve_prompt(ReviewSettings(enabled=True), deployment, {})

    assert text == "You review code.\n"
