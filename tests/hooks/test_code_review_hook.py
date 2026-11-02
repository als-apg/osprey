"""The notebook surface of the code review: ``osprey_code_review.py``.

``main`` runs in-process with the reviewer stubbed at ``review.run_query``, the single seam. The
runs with ``osprey`` blocked are subprocesses, since nothing else can block the import.
"""

from __future__ import annotations

import ast
import asyncio
import io
import json
import subprocess
import sys
import textwrap
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from osprey.agent_runner.primitives import SDKWorkflowResult
from osprey.services.code_review import notebook as notebook_module
from osprey.services.code_review import review as review_module
from osprey.services.code_review import settings as settings_module
from osprey.services.code_review.contract import SURFACE_NOTEBOOK, ReviewFailure, ReviewRequest
from osprey.services.code_review.settings import DEFAULT_PROMPT, HOOK_OVERHEAD_S
from osprey_connectors.workspace import DEFAULT_AGENT_DATA_BASE_DIR
from tests.hooks.conftest import HOOKS_DIR, hook_audit_records, import_hook

SECRET_REASON = "SECRET_REASON_calls_caput"
NEW_SOURCE = "SECRET_CODE = caput('SR:C01:Set', 7)"
DISK_SOURCE = "on_disk = 1"
DISK_OUTPUT = "OUTPUT_THE_REVIEWER_MUST_NOT_SEE"

REPLIES = {
    "pass": {"verdict": "pass", "reasons": []},
    "concern": {"verdict": "concern", "reasons": [SECRET_REASON]},
    "block": {"verdict": "block", "reasons": [SECRET_REASON]},
}


def _reply(text: str) -> SDKWorkflowResult:
    return SDKWorkflowResult(
        text_blocks=[text],
        result=SimpleNamespace(subtype="success", is_error=False),  # type: ignore[arg-type]
    )


class StubRunner:
    """Stands in for ``run_query``: records each call, then answers."""

    def __init__(self, text: str, delay: float = 0.0) -> None:
        self.text = text
        self.delay = delay
        self.calls: list[tuple[tuple, dict]] = []

    async def __call__(self, *args, **kwargs) -> SDKWorkflowResult:
        self.calls.append((args, kwargs))
        if self.delay:
            await asyncio.sleep(self.delay)
        return _reply(self.text)


class Env:
    """One deployment on disk, and the hook wired to observe it."""

    def __init__(self, root: Path, hook, monkeypatch, capsys) -> None:
        self.root = root
        self.hook = hook
        self.monkeypatch = monkeypatch
        self.capsys = capsys
        self.audits: list[dict] = []
        self.notices: list[tuple[str, str, str]] = []
        self.real_notify = hook.notify_activity
        monkeypatch.setattr(hook, "emit_audit", self._emit_audit)
        monkeypatch.setattr(hook, "notify_activity", lambda *args: self.notices.append(args))

    def _emit_audit(self, name, _hook_input, decision, subject, reason, detail=None):
        self.audits.append(
            {"name": name, "decision": decision, "subject": subject, "reason": reason}
            | {"detail": detail}
        )

    def config(self, section=None, *, raw: str | None = None) -> None:
        text = raw if raw is not None else yaml.safe_dump({"code_review": section})
        (self.root / "config.yml").write_text(text, encoding="utf-8")

    def stub(self, verdict: str = "pass", *, text: str | None = None, delay: float = 0.0):
        runner = StubRunner(json.dumps(REPLIES[verdict]) if text is None else text, delay)
        self.monkeypatch.setattr(review_module, "run_query", runner)
        return runner

    def notebook(self, tree: str = "notebooks", content: str | None = None) -> Path:
        path = self.root / DEFAULT_AGENT_DATA_BASE_DIR / tree / "x.ipynb"
        path.parent.mkdir(parents=True, exist_ok=True)
        if content is None:
            cell = {
                "id": "c1",
                "cell_type": "code",
                "source": DISK_SOURCE,
                "outputs": [{"text": DISK_OUTPUT}],
            }
            content = json.dumps({"cells": [cell]})
        path.write_text(content, encoding="utf-8")
        return path

    def run(self, path, *, argv=(), tool="NotebookEdit", **tool_input):
        """Run ``main``; return the parsed output, or ``None`` when stdout stayed empty."""
        payload = {
            "tool_name": tool,
            "tool_use_id": "toolu_1",
            "tool_input": {
                "notebook_path": str(path),
                "new_source": NEW_SOURCE,
                "cell_id": "c1",
                "edit_mode": "replace",
                "cell_type": "code",
            }
            | tool_input,
        }
        self.monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
        self.monkeypatch.setattr(sys, "argv", ["osprey_code_review.py", *argv])
        with pytest.raises(SystemExit) as exit_info:
            self.hook.main()
        assert exit_info.value.code == 0
        stdout = self.capsys.readouterr().out
        return json.loads(stdout)["hookSpecificOutput"] if stdout.strip() else None

    def assert_nothing_happened(self, output, runner=None) -> None:
        assert output is None
        assert self.audits == []
        assert self.notices == []
        if runner is not None:
            assert runner.calls == []


@pytest.fixture
def env(tmp_path, monkeypatch, capsys) -> Env:
    root = tmp_path / "deployment"
    prompt = root / DEFAULT_PROMPT
    prompt.parent.mkdir(parents=True)
    prompt.write_text("Review this code.", encoding="utf-8")
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(root))
    monkeypatch.delenv("OSPREY_CONFIG", raising=False)
    monkeypatch.delenv("OSPREY_HOOK_DEBUG", raising=False)
    monkeypatch.chdir(tmp_path)
    return Env(root, import_hook("osprey_code_review"), monkeypatch, capsys)


def _is_deny(output) -> bool:
    return output is not None and output.get("permissionDecision") == "deny"


# -- Off ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "config",
    [{"enabled": False}, {"enabled": False, "level": "warn"}, {}],
    ids=["disabled", "disabled-with-a-bad-level", "enabled-left-out"],
)
def test_a_section_that_is_off_does_nothing(env, config):
    env.config(config)
    runner = env.stub("block")

    env.assert_nothing_happened(env.run(env.notebook()), runner)


def test_no_section_does_nothing(env):
    env.config(raw="other: 1\n")
    runner = env.stub("block")

    env.assert_nothing_happened(env.run(env.notebook()), runner)


def test_no_config_file_does_nothing(env):
    runner = env.stub("block")

    env.assert_nothing_happened(env.run(env.notebook()), runner)


# -- What is reviewed --------------------------------------------------------


@pytest.mark.parametrize("tree", ["artifacts", "notebooks"])
def test_both_trees_are_reviewed(env, tree):
    env.config({"enabled": True, "level": "block"})
    env.stub("block")

    output = env.run(env.notebook(tree))

    assert _is_deny(output)
    assert env.audits[0]["subject"] == f"{tree}/x.ipynb"


@pytest.mark.parametrize(
    "path",
    ["elsewhere/x.ipynb", f"{DEFAULT_AGENT_DATA_BASE_DIR}/other/x.ipynb"],
    ids=["outside-agent-data", "another-agent-data-subdir"],
)
def test_a_write_outside_the_trees_gets_no_opinion(env, path):
    env.config({"enabled": True, "level": "block"})
    runner = env.stub("block")

    env.assert_nothing_happened(env.run(env.root / path), runner)


def test_a_traversal_out_of_a_tree_gets_no_opinion(env):
    env.config({"enabled": True, "level": "block"})
    runner = env.stub("block")
    path = f"{env.notebook()}/../../../../outside.ipynb"

    env.assert_nothing_happened(env.run(path), runner)


def test_another_tool_gets_no_opinion(env):
    env.config({"enabled": True, "level": "block"})
    runner = env.stub("block")

    env.assert_nothing_happened(env.run(env.notebook(), tool="Write"), runner)


def test_a_call_naming_no_path_gets_no_opinion(env):
    env.config({"enabled": True, "level": "block"})
    runner = env.stub("block")

    env.assert_nothing_happened(env.run("", notebook_path=""), runner)


@pytest.mark.parametrize(
    "tool_input",
    [{"edit_mode": "delete"}, {"cell_type": "markdown"}],
    ids=["delete", "markdown"],
)
def test_an_edit_writing_no_code_is_not_reviewed(env, tool_input):
    env.config({"enabled": True, "level": "block"})
    runner = env.stub("block")

    env.assert_nothing_happened(env.run(env.notebook(), **tool_input), runner)


def test_the_reviewer_gets_the_request_and_no_tools(env):
    env.config({"enabled": True, "level": "block"})
    runner = env.stub("pass")

    env.run(env.notebook())

    ((args, kwargs),) = runner.calls
    render_dir, message = args
    assert render_dir == env.root
    assert NEW_SOURCE in message
    assert DISK_SOURCE in message
    assert DISK_OUTPUT not in message
    assert "id=c1 index=0 edit_mode=replace" in message
    assert "## Surface\n\nnotebook" in message
    assert kwargs["system_prompt"] == "Review this code."
    assert kwargs["tools"] == []
    assert kwargs["max_turns"] == 1


# -- The level table ---------------------------------------------------------

_TABLE = [
    ("pass", "flag", "allowed", None),
    ("pass", "block", "allowed", None),
    ("concern", "flag", "allowed", "note"),
    ("concern", "block", "allowed", "note"),
    ("block", "flag", "allowed", "note"),
    ("block", "block", "refused", "deny"),
]


@pytest.mark.parametrize(("verdict", "level", "decision", "effect"), _TABLE)
def test_the_level_table(env, verdict, level, decision, effect):
    env.config({"enabled": True, "level": level})
    env.stub(verdict)
    notebook = env.notebook()
    before = notebook.read_bytes()

    output = env.run(notebook)

    assert notebook.read_bytes() == before
    if effect is None:
        assert output is None
    elif effect == "note":
        assert "permissionDecision" not in output
        assert output["hookEventName"] == "PreToolUse"
        assert SECRET_REASON in output["additionalContext"]
    else:
        assert _is_deny(output)
        assert output["hookEventName"] == "PreToolUse"
        assert SECRET_REASON in output["permissionDecisionReason"]

    (record,) = env.audits
    assert record == {
        "name": "code-review",
        "decision": decision,
        "subject": "notebooks/x.ipynb",
        "reason": "code_review",
        "detail": f"review={verdict} level={level}",
    }
    assert len(env.notices) == (0 if verdict == "pass" else 1)
    for tool, kind, _detail in env.notices:
        assert (tool, kind) == ("NotebookEdit", "review")


@pytest.mark.parametrize(("level", "denied"), [("flag", False), ("block", True)])
def test_a_failure_at_each_level(env, level, denied):
    env.config({"enabled": True, "level": level})
    env.stub(text="no json here")

    output = env.run(env.notebook())

    assert _is_deny(output) is denied
    (record,) = env.audits
    assert record["detail"] == f"review=failure level={level} kind=unparseable"
    assert record["decision"] == ("refused" if denied else "allowed")
    assert len(env.notices) == 1
    assert env.notices[0][1] == "review"
    if denied:
        assert "do not retry" in output["permissionDecisionReason"]
    else:
        assert "proceeded unreviewed" in output["additionalContext"]


def test_records_and_notices_hold_no_reasons_and_no_code(env):
    env.config({"enabled": True, "level": "block"})
    env.stub("block")

    env.run(env.notebook())

    written = json.dumps(env.audits) + json.dumps(env.notices)
    assert SECRET_REASON not in written
    assert NEW_SOURCE not in written


def test_a_level_change_reaches_the_next_edit_without_a_rebuild(env):
    env.stub("block")
    notebook = env.notebook()

    env.config({"enabled": True, "level": "block"})
    assert _is_deny(env.run(notebook))

    env.config({"enabled": True, "level": "flag"})
    assert not _is_deny(env.run(notebook))

    env.config({"enabled": False})
    env.audits.clear()
    assert env.run(notebook) is None
    assert env.audits == []


# -- The deadline ------------------------------------------------------------


def test_an_overrun_of_the_time_limit_denies_at_block(env):
    env.config({"enabled": True, "level": "block", "time_limit": 0.05})
    env.stub("pass", delay=5)

    output = env.run(env.notebook())

    assert _is_deny(output)
    assert env.audits[0]["detail"] == "review=failure level=block kind=timeout"


def test_an_unparseable_reply_denies_at_block(env):
    env.config({"enabled": True, "level": "block"})
    env.stub(text="looks fine to me")

    assert _is_deny(env.run(env.notebook()))
    assert env.audits[0]["detail"] == "review=failure level=block kind=unparseable"


def test_the_deadline_caps_the_review(env):
    env.config({"enabled": True, "level": "block"})
    env.stub("pass", delay=5)

    argv = ("--deadline", str(HOOK_OVERHEAD_S + 0.05))
    assert _is_deny(env.run(env.notebook(), argv=argv))
    assert env.audits[0]["detail"] == "review=failure level=block kind=timeout"


def test_the_deadline_never_extends_the_time_limit(env):
    env.config({"enabled": True, "level": "block", "time_limit": 0.05})
    env.stub("pass", delay=5)

    assert _is_deny(env.run(env.notebook(), argv=("--deadline", "600")))


@pytest.mark.parametrize("raw", ["soon", "0", "-5", "nan"])
def test_a_malformed_deadline_is_the_default(env, raw):
    env.config({"enabled": True, "level": "block"})
    env.stub("pass")

    assert env.run(env.notebook(), argv=("--deadline", raw)) is None
    assert env.audits[0]["detail"] == "review=pass level=block"


def test_the_default_deadline_is_the_rendered_timeout(env):
    assert env.hook._parse_deadline([]) == 90.0


# -- Config that cannot be used ----------------------------------------------


def test_unparseable_yaml_denies_with_no_level(env):
    env.config(raw="code_review: [unclosed\n")
    runner = env.stub("pass")

    assert _is_deny(env.run(env.notebook()))
    assert runner.calls == []
    (record,) = env.audits
    assert record["decision"] == "refused"
    assert record["detail"] == "review=failure kind=config"
    assert len(env.notices) == 1


@pytest.mark.parametrize("raw", ["just some text\n", "- a\n- b\n"], ids=["scalar", "list"])
def test_yaml_that_is_no_mapping_denies(env, raw):
    env.config(raw=raw)
    runner = env.stub("pass")

    assert _is_deny(env.run(env.notebook()))
    assert runner.calls == []
    (record,) = env.audits
    assert record["decision"] == "refused"
    assert record["detail"].startswith("review=failure kind=")


@pytest.mark.parametrize(
    "section",
    [{"enabled": True, "level": "warn"}, {"enabled": "yes"}, {"enabled": 1}, "on"],
    ids=["bad-level", "string-enabled", "int-enabled", "not-a-mapping"],
)
def test_a_malformed_section_denies_with_no_level(env, section):
    env.config(section)
    runner = env.stub("pass")

    assert _is_deny(env.run(env.notebook()))
    assert runner.calls == []
    (record,) = env.audits
    assert record["detail"] == "review=failure kind=config"


# -- What goes wrong while reviewing -----------------------------------------


@pytest.mark.parametrize(("level", "denied"), [("flag", False), ("block", True)])
def test_a_corrupt_notebook_is_reviewed_with_an_empty_context(env, level, denied):
    env.config({"enabled": True, "level": level})
    runner = env.stub("block")

    output = env.run(env.notebook(content="{ not json"))

    assert _is_deny(output) is denied
    ((args, _kwargs),) = runner.calls
    assert "## Notebook" not in args[1]
    assert NEW_SOURCE in args[1]
    assert env.audits[0]["detail"] == f"review=block level={level}"


@pytest.mark.parametrize(("level", "denied"), [("flag", False), ("block", True)])
def test_a_request_that_raises_is_a_runner_failure(env, level, denied, monkeypatch):
    env.config({"enabled": True, "level": level})
    runner = env.stub("pass")

    def explode(*_args, **_kwargs):
        raise RuntimeError("the request broke")

    monkeypatch.setattr(notebook_module, "notebook_review_request", explode)

    output = env.run(env.notebook())

    assert _is_deny(output) is denied
    assert runner.calls == []
    (record,) = env.audits
    assert record["detail"] == f"review=failure level={level} kind=runner"
    assert len(env.notices) == 1


def test_an_exception_before_the_settings_are_read_is_unavailable(env, monkeypatch):
    env.config({"enabled": True, "level": "flag"})
    runner = env.stub("pass")

    def explode(_path):
        raise RuntimeError("the reader broke")

    monkeypatch.setattr(settings_module, "read_review_config", explode)

    output = env.run(env.notebook())

    assert _is_deny(output)
    assert "the reader broke" in output["permissionDecisionReason"]
    assert runner.calls == []
    (record,) = env.audits
    assert record["detail"] == "review=failure kind=unavailable"
    assert record["decision"] == "refused"
    assert len(env.notices) == 1


# -- The hook's own run ------------------------------------------------------


def test_a_run_from_outside_the_render_still_reviews(env, tmp_path, monkeypatch):
    env.config({"enabled": True, "level": "block"})
    runner = env.stub("block")
    notebook = env.notebook()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "config.yml").write_text("code_review: {enabled: false}\n", encoding="utf-8")
    monkeypatch.chdir(elsewhere)

    assert _is_deny(env.run(notebook))
    assert runner.calls[0][0][0] == env.root


def test_the_unavailable_wording_matches_the_core(env):
    """The import-failure path restates the core's wording, because it has no core to ask."""
    env.notebook()
    output = env.hook._refuse_unavailable({}, "notebooks/x.ipynb", "boom")
    request = ReviewRequest(SURFACE_NOTEBOOK, "", "notebooks/x.ipynb")
    outcome = review_module.decide(ReviewFailure("unavailable", "boom"), None)
    fields = review_module.audit_fields(outcome, request)

    (record,) = env.audits
    assert record["decision"] == fields.decision
    assert record["reason"] == fields.reason
    assert record["detail"] == fields.detail
    assert env.notices[0][2] == review_module.notice_text(outcome, request)
    reason = output["hookSpecificOutput"]["permissionDecisionReason"]
    assert reason == review_module.agent_message(outcome)


def test_a_long_notice_is_posted_truncated(env, monkeypatch):
    posted = []
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout: posted.append((request, timeout)) or SimpleNamespace(close=int),
    )

    env.real_notify("NotebookEdit", "review", "x" * 5000)

    ((request, timeout),) = posted
    assert timeout == 1
    body = json.loads(request.data)
    assert body["tool"] == "NotebookEdit"
    assert body["target"]["kind"] == "review"
    assert len(body["target"]["detail"]) == 1024
    assert body["target"]["detail"].endswith("…")


def test_a_terminal_that_is_down_costs_only_the_notice(env, monkeypatch):
    def refuse(*_args, **_kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)

    env.real_notify("NotebookEdit", "review", "detail")


def test_a_notice_of_the_exact_bound_is_not_truncated(env):
    request = env.hook._activity_request("NotebookEdit", "review", "x" * 1024)

    assert json.loads(request.data)["target"]["detail"] == "x" * 1024


def test_the_bearer_is_sent_only_when_the_token_holds_a_value(env, monkeypatch):
    monkeypatch.setenv("OSPREY_PANEL_TOKEN", "   ")
    assert env.hook._activity_request("t", "review", "d").get_header("Authorization") is None

    monkeypatch.setenv("OSPREY_PANEL_TOKEN", "abc")
    assert env.hook._activity_request("t", "review", "d").get_header("Authorization") == (
        "Bearer abc"
    )


def test_the_hook_file_imports_nothing_from_the_mcp_server():
    tree = ast.parse((HOOKS_DIR / "osprey_code_review.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    assert not [name for name in imported if name.startswith("osprey.mcp_server")]


# -- With osprey blocked -----------------------------------------------------


def _run_blocked(root: Path, notebook: Path) -> subprocess.CompletedProcess:
    script = textwrap.dedent(f"""
        import sys
        sys.modules["osprey"] = None
        sys.path.insert(0, {str(HOOKS_DIR)!r})
        import osprey_code_review
        osprey_code_review.main()
    """)
    payload = {
        "tool_name": "NotebookEdit",
        "tool_use_id": "toolu_1",
        "tool_input": {"notebook_path": str(notebook), "new_source": NEW_SOURCE},
    }
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        input=json.dumps(payload),
        cwd=root,
        env={"PATH": "/usr/bin:/bin", "CLAUDE_PROJECT_DIR": str(root), "OSPREY_WEB_PORT": "1"},
    )


@pytest.mark.parametrize(
    "section",
    [{"enabled": True, "level": "flag"}, {"enabled": False}],
    ids=["enabled-at-flag", "disabled"],
)
def test_a_run_with_osprey_blocked_denies(env, section):
    env.config(section)

    result = _run_blocked(env.root, env.notebook())

    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)["hookSpecificOutput"]
    assert output["hookEventName"] == "PreToolUse"
    assert output["permissionDecision"] == "deny"
    assert "kind unavailable" in output["permissionDecisionReason"]
    (record,) = hook_audit_records(env.root, "hook_code_review")
    assert record["decision"] == "refused"
    assert record["reason"] == "code_review"
    assert record["detail"] == "review=failure kind=unavailable"
    assert record["subject"] == "notebooks/x.ipynb"


def test_a_run_with_osprey_blocked_leaves_other_trees_alone(env):
    env.config({"enabled": True})
    outside = env.root / "elsewhere" / "x.ipynb"

    result = _run_blocked(env.root, outside)

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert hook_audit_records(env.root, "hook_code_review") == []
