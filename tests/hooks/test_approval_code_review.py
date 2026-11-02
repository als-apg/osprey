"""The code review ahead of the approval prompt: ``osprey_approval.py`` on ``execute`` and ``execute_file``.

``main`` runs in-process with the reviewer stubbed at ``review.review_code``, the seam the
executor suite stubs, so the stub applies the real level table. The deployment is the one the
Approval preview suite builds: writes armed on the simulator, every executor call put to a
person. What is asserted is what the operator reads on the prompt, what the executor reads back
from the ask stamp, and what the Approval preview says.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

import nbformat
import pytest
import yaml

from osprey.services.code_review import review as review_module
from osprey.services.code_review.contract import SURFACE_EXECUTE, ReviewFailure, ReviewVerdict
from osprey.services.code_review.review import AUDIT_REASON, agent_message, decide
from osprey.services.code_review.stamp import (
    REVIEWED_DIGEST_KEY,
    REVIEWED_MESSAGE_KEY,
    REVIEWED_VERDICT_KEY,
)
from osprey_connectors_tests._harness import (
    pin_identity,
    state_dir_under,
    write_control_context,
)
from tests.hooks.conftest import import_hook

EXECUTE = "mcp__python__execute"
EXECUTE_FILE = "mcp__python__execute_file"
CODE = "x = 1\nprint(x)\n"
WRITING_CODE = "import epics\nepics.caput('SR:C01:Set', 7)\n"
DESCRIPTION = "print one"
REASON = "SECRET_REASON_the_stub_objects"
REVISION = "print('revised')\n"

BLOCK = ReviewVerdict("block", (REASON,), REVISION)
CONCERN = ReviewVerdict("concern", (REASON,))
PASS = ReviewVerdict("pass", ())
FAILURE = ReviewFailure("timeout", "no reply within 60 seconds")

HOOK_CONFIG = {"server_prefixes": ["mcp__python__"], "approval_prefixes": ["mcp__python__"]}

#: The budget the render gives the two reviewed executor rules: the prompt's own five seconds
#: plus the default review deadline.
BUDGET = ("--budget", "95")


def _digest(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


class ReviewStub:
    """Stands in for ``review_code``: records its inputs, then applies the real level table."""

    def __init__(self) -> None:
        self.result: ReviewVerdict | ReviewFailure = PASS
        self.calls: list[dict] = []

    async def __call__(self, request, settings, *, repo_root, render_dir):
        self.calls.append(
            {
                "request": request,
                "settings": settings,
                "repo_root": repo_root,
                "render_dir": render_dir,
            }
        )
        return decide(self.result, settings.level)


class Deployment:
    """One deployment on disk, and the hook wired to run against it in-process."""

    def __init__(self, root: Path, hook, monkeypatch, capsys) -> None:
        self.root = root
        self.hook = hook
        self.monkeypatch = monkeypatch
        self.capsys = capsys
        self.config_path = root / "config.yml"
        self.state_dir = state_dir_under(root / "agent_data")
        self.audits: list[dict] = []
        self.notices: list[tuple] = []
        self.stub = ReviewStub()
        monkeypatch.setattr(review_module, "review_code", self.stub)
        monkeypatch.setattr(hook, "emit_audit", self._emit_audit)
        monkeypatch.setattr(
            "osprey.mcp_server.http.notify_agent_activity",
            lambda *args, **kwargs: self.notices.append((args, kwargs)),
        )

    def _emit_audit(self, name, _hook_input, decision, subject, reason, detail=None, **_):
        self.audits.append(
            {"name": name, "decision": decision, "subject": subject, "reason": reason}
            | {"detail": detail}
        )

    def configure(self, review: dict | None, *, policy: str = "always", modes=None) -> None:
        section = dict(review or {})
        if modes is not None:
            section["modes"] = list(modes)
        config = {
            "approval": {"enabled": True, "default_policy": policy},
            "control_system": {"writes": "allowed"},
        }
        if review is not None:
            config["code_review"] = section
        self.config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    def run(self, tool, tool_input, *, argv=BUDGET, tool_use_id="toolu_1", event="PreToolUse"):
        """Run ``main``; return the hook's output, or ``None`` when stdout stayed empty."""
        payload = {
            "hook_event_name": event,
            "tool_name": tool,
            "tool_use_id": tool_use_id,
            "session_id": "conv-1",
            "permission_mode": "default",
            "tool_input": tool_input,
            "cwd": str(self.root),
        }
        if event != "PreToolUse":
            payload["tool_response"] = {"content": "ok"}
        self.monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
        self.monkeypatch.setattr(sys, "argv", ["osprey_approval.py", *argv])
        with pytest.raises(SystemExit) as exit_info:
            self.hook.main()
        assert exit_info.value.code == 0
        stdout = self.capsys.readouterr().out
        return json.loads(stdout)["hookSpecificOutput"] if stdout.strip() else None

    def stamp(self, tool_use_id="toolu_1") -> dict | None:
        path = self.state_dir / f"approval_ask_{tool_use_id}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def previews(self) -> list[tuple[dict, nbformat.NotebookNode]]:
        found = []
        for index in sorted(self.root.rglob("artifacts.json")):
            for entry in json.loads(index.read_text(encoding="utf-8"))["entries"]:
                if entry["title"] == "Approval preview":
                    notebook = nbformat.read(index.parent / entry["filename"], as_version=4)
                    found.append((entry, notebook))
        return found

    def review_audits(self) -> list[dict]:
        return [audit for audit in self.audits if audit["reason"] == AUDIT_REASON]


@pytest.fixture
def deployment(tmp_path, monkeypatch, capsys) -> Deployment:
    pin_identity(monkeypatch)
    root = tmp_path / "repo"
    agent_data = root / "agent_data"
    state_dir_under(agent_data).mkdir(parents=True)
    # The write verdict asks for a switch-on record; without one every write
    # defers as switch_off before the prompt this suite reads.
    write_control_context(agent_data, target="va", switch={"va": "on"})
    hook_config = tmp_path / "hook_config.json"
    hook_config.write_text(json.dumps(HOOK_CONFIG), encoding="utf-8")
    monkeypatch.setenv("OSPREY_AGENT_DATA_ROOT", str(agent_data))
    monkeypatch.setenv("OSPREY_CONFIG", str(root / "config.yml"))
    monkeypatch.setenv("OSPREY_HOOK_CONFIG", str(hook_config))
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(root))
    monkeypatch.delenv("OSPREY_HOOK_DEBUG", raising=False)
    monkeypatch.delenv("OSPREY_DISPATCH_RUN", raising=False)
    monkeypatch.chdir(root)
    return Deployment(root, import_hook("osprey_approval"), monkeypatch, capsys)


def _execute(code: str = CODE, mode: str = "readwrite") -> dict:
    return {"code": code, "execution_mode": mode, "description": DESCRIPTION}


def _reason(output) -> str:
    return output["permissionDecisionReason"]


def _assert_asked_unreviewed(deployment: Deployment, output) -> None:
    assert output["permissionDecision"] == "ask"
    assert "Code review" not in _reason(output)
    assert deployment.stub.calls == []
    assert deployment.review_audits() == []
    assert deployment.notices == []
    stamp = deployment.stamp()
    assert stamp is not None
    assert REVIEWED_DIGEST_KEY not in stamp


# ---------------------------------------------------------------------------
# Off, and not reviewed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "review", [None, {"enabled": False}, {}], ids=["no-section", "disabled", "enabled-left-out"]
)
def test_a_feature_that_is_off_asks_as_before(deployment, review):
    deployment.configure(review)
    deployment.stub.result = BLOCK

    _assert_asked_unreviewed(deployment, deployment.run(EXECUTE, _execute()))


def test_by_default_a_readonly_run_is_asked_unreviewed(deployment):
    """A readonly run with a write pattern is asked under a selective policy, but not reviewed."""
    deployment.configure({"enabled": True, "level": "block"}, policy="selective")
    deployment.stub.result = BLOCK

    output = deployment.run(EXECUTE, _execute(WRITING_CODE, "readonly"))

    assert "Code contains control system write patterns." in _reason(output)
    _assert_asked_unreviewed(deployment, output)


def test_a_readonly_run_is_reviewed_when_modes_name_it(deployment):
    deployment.configure(
        {"enabled": True, "level": "block"}, policy="selective", modes=["readonly", "readwrite"]
    )
    deployment.stub.result = BLOCK

    output = deployment.run(EXECUTE, _execute(WRITING_CODE, "readonly"))

    assert output["permissionDecision"] == "deny"
    (call,) = deployment.stub.calls
    assert call["request"].mode == "readonly"


def test_a_render_with_no_room_leaves_the_review_to_the_executor(deployment):
    """A render from before the review passes no ``--budget``; the five seconds it then has are
    the prompt's own, so nothing is reviewed here and nothing is stamped."""
    deployment.configure({"enabled": True, "level": "block"})
    deployment.stub.result = BLOCK

    _assert_asked_unreviewed(deployment, deployment.run(EXECUTE, _execute(), argv=()))


# ---------------------------------------------------------------------------
# The level table, as the operator and the executor see it
# ---------------------------------------------------------------------------


def test_a_block_in_blocking_mode_denies_before_anyone_is_asked(deployment):
    deployment.configure({"enabled": True, "level": "block"})
    deployment.stub.result = BLOCK

    output = deployment.run(EXECUTE, _execute())

    assert output["permissionDecision"] == "deny"
    assert _reason(output) == agent_message(decide(BLOCK, "block"))
    assert REASON in _reason(output)
    assert REVISION in _reason(output)
    assert "revise and resubmit" in _reason(output)
    assert deployment.stamp() is None, "a denied call was never asked, so it has no stamp"
    assert deployment.previews() == [], "nothing was put in front of the operator"
    (audit,) = deployment.review_audits()
    assert audit["decision"] == "refused"
    assert audit["subject"] == EXECUTE
    assert audit["detail"] == "review=block level=block mode=readwrite"
    assert REASON not in json.dumps(deployment.audits)
    ((tool, kind), kwargs), *_ = deployment.notices
    assert (tool, kind) == ("execute", "review")
    assert kwargs["detail"] == f"Code review on execute: block, refused: {EXECUTE}"
    assert REASON not in kwargs["detail"]


def test_a_concern_goes_onto_the_prompt_and_into_the_stamp(deployment):
    deployment.configure({"enabled": True, "level": "block"})
    deployment.stub.result = CONCERN

    output = deployment.run(EXECUTE, _execute())

    assert output["permissionDecision"] == "ask"
    reason = _reason(output)
    assert "Code review: concern" in reason
    assert f"  - {REASON}" in reason
    assert reason.index("Code review: concern") < reason.index("Approval preview:")

    stamp = deployment.stamp()
    assert stamp[REVIEWED_DIGEST_KEY] == _digest(CODE)
    assert stamp[REVIEWED_VERDICT_KEY] == "concern"
    assert stamp[REVIEWED_MESSAGE_KEY] == agent_message(decide(CONCERN, "block"))
    assert stamp["approved_target"] == "va", "the review rides beside the binding, not instead"

    ((_entry, notebook),) = deployment.previews()
    header = notebook.cells[0].source
    assert "**Status:** Pending approval" in header
    assert "**Code review**" in header
    assert f"Code review: concern\n  - {REASON}" in header
    assert notebook.cells[1].source == CODE

    (audit,) = deployment.review_audits()
    assert audit["decision"] == "allowed"
    assert audit["detail"] == "review=concern level=block mode=readwrite"
    ((tool, kind), kwargs), *_ = deployment.notices
    assert (tool, kind) == ("execute", "review")
    assert kwargs["detail"] == f"Code review on execute: concern, proceeded: {EXECUTE}"


def test_a_pass_stamps_the_digest_and_has_nothing_to_tell_the_agent(deployment):
    deployment.configure({"enabled": True, "level": "block"})
    deployment.stub.result = PASS

    output = deployment.run(EXECUTE, _execute())

    assert output["permissionDecision"] == "ask"
    assert "Code review: pass" in _reason(output)
    stamp = deployment.stamp()
    assert stamp[REVIEWED_DIGEST_KEY] == _digest(CODE)
    assert stamp[REVIEWED_VERDICT_KEY] == "pass"
    assert REVIEWED_MESSAGE_KEY not in stamp
    (audit,) = deployment.review_audits()
    assert audit["decision"] == "allowed"
    assert audit["detail"] == "review=pass level=block mode=readwrite"
    assert deployment.notices == [], "a pass sends no notice"


def test_a_block_in_advisory_mode_asks_with_the_note(deployment):
    deployment.configure({"enabled": True, "level": "flag"})
    deployment.stub.result = BLOCK

    output = deployment.run(EXECUTE, _execute())

    assert output["permissionDecision"] == "ask"
    reason = _reason(output)
    assert "Code review: block (advisory mode: the code may still run)" in reason
    assert f"  - {REASON}" in reason
    assert "The reviewer suggested a revision, which the agent receives." in reason
    assert REVISION not in reason, "the revision is the agent's, not the prompt's"
    stamp = deployment.stamp()
    assert stamp[REVIEWED_VERDICT_KEY] == "block"
    assert REVISION in stamp[REVIEWED_MESSAGE_KEY]
    assert stamp[REVIEWED_MESSAGE_KEY] == agent_message(decide(BLOCK, "flag"))


def test_a_failure_in_blocking_mode_denies_and_says_not_to_retry(deployment):
    deployment.configure({"enabled": True, "level": "block"})
    deployment.stub.result = FAILURE

    output = deployment.run(EXECUTE, _execute())

    assert output["permissionDecision"] == "deny"
    assert "kind timeout" in _reason(output)
    assert "do not retry" in _reason(output)
    (audit,) = deployment.review_audits()
    assert audit["decision"] == "refused"
    assert audit["detail"] == "review=failure level=block kind=timeout mode=readwrite"


def test_a_failure_in_advisory_mode_asks_unstamped(deployment):
    """No verdict was reached, so the executor's own review stays in place."""
    deployment.configure({"enabled": True, "level": "flag"})
    deployment.stub.result = FAILURE

    output = deployment.run(EXECUTE, _execute())

    assert output["permissionDecision"] == "ask"
    assert "Code review: did not run (timeout)" in _reason(output)
    stamp = deployment.stamp()
    assert stamp is not None
    assert REVIEWED_DIGEST_KEY not in stamp
    (audit,) = deployment.review_audits()
    assert audit["decision"] == "allowed"
    assert audit["detail"] == "review=failure level=flag kind=timeout mode=readwrite"


def test_a_reviewer_that_raises_leaves_the_ask_as_it_was(deployment):
    deployment.configure({"enabled": True, "level": "block"})

    async def boom(*_args, **_kwargs):
        raise RuntimeError("the stub broke")

    deployment.monkeypatch.setattr(review_module, "review_code", boom)

    output = deployment.run(EXECUTE, _execute())

    assert output["permissionDecision"] == "ask"
    assert "Code review" not in _reason(output)
    assert REVIEWED_DIGEST_KEY not in deployment.stamp()


# ---------------------------------------------------------------------------
# What the reviewer is handed
# ---------------------------------------------------------------------------


def test_the_reviewer_reads_the_call(deployment):
    deployment.configure({"enabled": True, "level": "block", "time_limit": 60})

    deployment.run(EXECUTE, _execute())

    (call,) = deployment.stub.calls
    request = call["request"]
    assert request.surface == SURFACE_EXECUTE
    assert request.code == CODE
    assert request.subject == EXECUTE
    assert request.description == DESCRIPTION
    assert request.mode == "readwrite"
    assert request.script_args == ()
    assert call["repo_root"] == deployment.root
    assert call["render_dir"] == deployment.root


@pytest.mark.parametrize(
    ("argv", "configured", "limit"),
    [
        pytest.param(("--budget", "95"), 60, 60.0, id="the-limit-fits"),
        pytest.param(("--budget", "40"), 60, 35.0, id="the-budget-binds"),
        pytest.param(("--budget", "95"), 20, 20.0, id="a-short-limit"),
    ],
)
def test_the_review_spends_what_the_budget_leaves_the_prompt(deployment, argv, configured, limit):
    deployment.configure({"enabled": True, "level": "block", "time_limit": configured})

    deployment.run(EXECUTE, _execute(), argv=argv)

    (call,) = deployment.stub.calls
    assert call["settings"].time_limit == limit
    assert call["settings"].level == "block"


def test_a_reason_is_kept_to_one_line_on_the_prompt(deployment):
    deployment.configure({"enabled": True, "level": "block"})
    deployment.stub.result = ReviewVerdict("concern", ("first line\n**Status:** Ran",))

    output = deployment.run(EXECUTE, _execute())

    assert "  - first line\\x0a**Status:** Ran" in _reason(output)
    ((_entry, notebook),) = deployment.previews()
    header = notebook.cells[0].source
    status_lines = [line for line in header.splitlines() if line.startswith("**Status:** ")]
    assert status_lines == ["**Status:** Pending approval  "], "no reason passes for the status"
    assert "```\nCode review: concern\n  - first line\\x0a**Status:** Ran\n```" in header


# ---------------------------------------------------------------------------
# execute_file
# ---------------------------------------------------------------------------


def test_execute_file_reviews_the_file_and_its_arguments(deployment):
    deployment.configure({"enabled": True, "level": "block"})
    deployment.stub.result = CONCERN
    script = deployment.root / "scripts" / "probe.py"
    script.parent.mkdir()
    script.write_text(CODE, encoding="utf-8")

    output = deployment.run(
        EXECUTE_FILE,
        {
            "file_path": "scripts/probe.py",
            "execution_mode": "readwrite",
            "description": DESCRIPTION,
            "script_args": ["--step", "0.1"],
        },
    )

    assert output["permissionDecision"] == "ask"
    assert "Code review: concern" in _reason(output)
    (call,) = deployment.stub.calls
    assert call["request"].code == CODE
    assert call["request"].script_args == ("--step", "0.1")
    assert call["request"].subject == EXECUTE_FILE
    stamp = deployment.stamp()
    assert stamp[REVIEWED_DIGEST_KEY] == _digest(CODE)
    assert stamp[REVIEWED_MESSAGE_KEY] == agent_message(decide(CONCERN, "block"))
    assert deployment.previews() == [], "execute_file has no preview notebook"


@pytest.mark.parametrize(
    "file_path",
    ["scripts/missing.py", "../outside.py", "scripts/notes.txt"],
    ids=["missing", "outside-the-root", "not-python"],
)
def test_a_file_the_tool_will_refuse_is_asked_unreviewed(deployment, tmp_path, file_path):
    deployment.configure({"enabled": True, "level": "block"})
    deployment.stub.result = BLOCK
    (tmp_path / "outside.py").write_text(CODE, encoding="utf-8")
    notes = deployment.root / "scripts" / "notes.txt"
    notes.parent.mkdir()
    notes.write_text(CODE, encoding="utf-8")

    output = deployment.run(EXECUTE_FILE, {"file_path": file_path, "execution_mode": "readwrite"})

    _assert_asked_unreviewed(deployment, output)


# ---------------------------------------------------------------------------
# The preview keeps the review once the call is settled
# ---------------------------------------------------------------------------


def test_settling_the_preview_keeps_what_the_review_said(deployment):
    deployment.configure({"enabled": True, "level": "block"})
    deployment.stub.result = CONCERN
    deployment.run(EXECUTE, _execute(), tool_use_id="toolu_settle")

    deployment.run(EXECUTE, _execute(), tool_use_id="toolu_settle", event="PostToolUse")

    ((entry, notebook),) = deployment.previews()
    header = notebook.cells[0].source
    assert "**Status:** Ran after operator approval" in header
    assert f"Code review: concern\n  - {REASON}" in header
    assert entry["description"] == "Code ran after operator approval (mode: readwrite)"
