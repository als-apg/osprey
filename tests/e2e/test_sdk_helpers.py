"""Unit tests for tests/e2e/sdk_helpers.py pure-logic helpers.

No LLM, no build, no DB — fast logic checks. Excluded from the model-capability
matrix (scripts/benchmark/matrix_e2e_config.json) since it measures nothing
about the model under test.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from osprey.agent_runner.primitives import SDKWorkflowResult, ToolTrace
from osprey.agent_runner.project_paths import claude_project_dir
from tests.e2e.sdk_helpers import (
    _DEFAULT_ARIEL_DB_URI,
    HOOK_ATTACHMENT_PREFIX,
    TRANSCRIPT_SUBDIR,
    HookEvent,
    _bind_approval_policy,
    _override_ariel_db_uri,
    dump_agent_transcript,
    hook_attachments,
)

_PER_CELL_URI = "postgresql://ariel:ariel@localhost:5432/ariel_gpt-oss-20b_seed1"


def _write_config(tmp_path, uri: str = _DEFAULT_ARIEL_DB_URI):
    """Write a minimal rendered config.yml with an ARIEL DB uri line."""
    (tmp_path / "config.yml").write_text(
        f"ariel:\n  database:\n    uri: {uri}\n",
        encoding="utf-8",
    )
    return tmp_path / "config.yml"


@pytest.mark.unit
def test_override_rewrites_uri_when_env_set(tmp_path, monkeypatch):
    """With OSPREY_ARIEL_DB_URI set, the rendered config points at the per-cell DB."""
    config_path = _write_config(tmp_path)
    monkeypatch.setenv("OSPREY_ARIEL_DB_URI", _PER_CELL_URI)

    _override_ariel_db_uri(tmp_path)

    text = config_path.read_text(encoding="utf-8")
    assert _PER_CELL_URI in text
    # The bare default must not stand alone as the configured uri.
    assert f"uri: {_DEFAULT_ARIEL_DB_URI}\n" not in text


@pytest.mark.unit
def test_override_noop_when_env_unset(tmp_path, monkeypatch):
    """No override env → config is left untouched (shared default DB)."""
    config_path = _write_config(tmp_path)
    monkeypatch.delenv("OSPREY_ARIEL_DB_URI", raising=False)

    _override_ariel_db_uri(tmp_path)

    assert f"uri: {_DEFAULT_ARIEL_DB_URI}\n" in config_path.read_text(encoding="utf-8")


@pytest.mark.unit
def test_override_noop_when_env_equals_default(tmp_path, monkeypatch):
    """Override that equals the default is a no-op (no needless rewrite)."""
    config_path = _write_config(tmp_path)
    monkeypatch.setenv("OSPREY_ARIEL_DB_URI", _DEFAULT_ARIEL_DB_URI)

    _override_ariel_db_uri(tmp_path)

    assert f"uri: {_DEFAULT_ARIEL_DB_URI}\n" in config_path.read_text(encoding="utf-8")


@pytest.mark.unit
def test_override_noop_for_non_ariel_project(tmp_path, monkeypatch):
    """A project without the default ARIEL URI (e.g. the hello_world preset,
    ``ariel: {enabled: false}``) is left untouched — there is no real ARIEL DB
    to redirect, so the override must not fail the build.
    """
    (tmp_path / "config.yml").write_text(
        "ariel: {enabled: false}\nmodel: haiku\n", encoding="utf-8"
    )
    monkeypatch.setenv("OSPREY_ARIEL_DB_URI", _PER_CELL_URI)

    # Must not raise.
    _override_ariel_db_uri(tmp_path)

    text = (tmp_path / "config.yml").read_text(encoding="utf-8")
    assert _PER_CELL_URI not in text
    assert "ariel: {enabled: false}" in text


# ---------------------------------------------------------------------------
# The agent-transcript dump. Its whole reason to exist is that the judge sees
# tool results previewed at 300 characters, so the property worth pinning is
# that the artifact is NOT truncated — a dump that inherited the same limit
# would look like coverage while answering nothing.
# ---------------------------------------------------------------------------


def _result_with_a_long_tool_result(size: int = 5000):
    """One trace whose result is far longer than the judge's 300-char preview."""
    return SDKWorkflowResult(
        text_blocks=["I measured the ring.", "The response is linear."],
        tool_traces=[
            ToolTrace(
                name="mcp__bluesky__get_run_data",
                input={"run_uid": "abc123"},
                result="x" * size,
                tool_use_id="tu_1",
            )
        ],
    )


@pytest.mark.unit
def test_transcript_dump_is_inert_when_unarmed(tmp_path, monkeypatch):
    """No OSPREY_CI_DIAG_DIR — every local run — writes nothing at all."""
    monkeypatch.delenv("OSPREY_CI_DIAG_DIR", raising=False)
    monkeypatch.chdir(tmp_path)

    # ``render`` is passed to prove the transcript read is downstream of the
    # arming check: an unarmed run must not go near the session directory.
    assert (
        dump_agent_transcript("orbit_response", _result_with_a_long_tool_result(), render=tmp_path)
        is None
    )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.unit
def test_transcript_dump_keeps_tool_results_whole(tmp_path, monkeypatch):
    """The payload must carry the full tool result, not a preview.

    This is the defect the dump exists to close: `_to_workflow_result` cuts
    every result to 300 characters before the judge sees it, and a
    `get_run_data` body spends that budget on `run_uid` and `columns`.
    """
    monkeypatch.setenv("OSPREY_CI_DIAG_DIR", str(tmp_path))

    target = dump_agent_transcript("orbit_response", _result_with_a_long_tool_result())

    assert target is not None
    assert target == tmp_path / TRANSCRIPT_SUBDIR / "orbit_response.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert len(payload["tool_traces"][0]["result"]) == 5000
    assert payload["response"] == "I measured the ring.\nThe response is linear."
    # No render, no hook stdout: the six plan-stack dump sites that call this
    # without one must keep exactly the payload they had.
    assert "hook_attachments" not in payload


@pytest.mark.unit
def test_transcript_dump_sanitises_the_name(tmp_path, monkeypatch):
    """Parametrised ids carry `[]`, and paths carry `/`; neither may escape
    the diagnostics directory or produce an unopenable filename."""
    monkeypatch.setenv("OSPREY_CI_DIAG_DIR", str(tmp_path))

    target = dump_agent_transcript("../grid[two_axis]", SDKWorkflowResult(), render=tmp_path)

    assert target is not None
    assert target.parent == tmp_path / TRANSCRIPT_SUBDIR
    assert target.name == ".._grid_two_axis_.json"
    # A result with no session id resolves to no transcript, and the key is
    # still written — an empty list says "looked, found none", which a missing
    # key cannot.
    assert json.loads(target.read_text(encoding="utf-8"))["hook_attachments"] == []


@pytest.mark.unit
def test_transcript_dump_never_raises(tmp_path, monkeypatch):
    """A diagnostic that fails the test it observes would mask the failure it
    exists to explain. Here the target directory is occupied by a file, so the
    mkdir cannot succeed."""
    blocker = tmp_path / "diag"
    blocker.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv("OSPREY_CI_DIAG_DIR", str(blocker))

    assert dump_agent_transcript("orbit_response", SDKWorkflowResult(), render=blocker) is None


# ---------------------------------------------------------------------------
# Hook stdout read back out of the session transcript. Synthetic JSONL: the
# layout is Claude Code's, not ours, so these pin what the reader tolerates
# rather than what the CLI happens to write.
# ---------------------------------------------------------------------------


def _session_result(session_id: str) -> SDKWorkflowResult:
    """A workflow result carrying nothing but the session id the reader needs."""
    return SDKWorkflowResult(result=SimpleNamespace(session_id=session_id))  # type: ignore[arg-type]


def _write_transcript(tmp_path, monkeypatch, session_id: str, records: list) -> Path:
    """Plant a session transcript where `claude_project_dir` will look for it.

    Returns the render directory to hand the reader. `CLAUDE_CONFIG_DIR` is
    redirected so the test never reads — or creates — anything under the
    developer's real ~/.claude.
    """
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    render = (tmp_path / "render").resolve()
    render.mkdir(parents=True, exist_ok=True)
    transcript = claude_project_dir(render) / f"{session_id}.jsonl"
    transcript.parent.mkdir(parents=True, exist_ok=True)
    transcript.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )
    return render


@pytest.mark.unit
def test_hook_attachments_reads_every_hook_prefixed_attachment(tmp_path, monkeypatch):
    """Prefix match, not an allow-list of one.

    `hook_success` is the only spelling verified against CLI 2.1.241, so a hook
    that failed or timed out is recorded under a type this test cannot name.
    Matching the prefix is what keeps that hook's stdout in the CI artifact —
    and a hook that died mid-approval is exactly the run someone needs to read.
    """
    render = _write_transcript(
        tmp_path,
        monkeypatch,
        "sess-1",
        [
            {"type": "user", "message": {"role": "user", "content": "hi"}},
            {"attachment": {"type": "selected_lines_in_ide", "stdout": "not a hook"}},
            {
                "attachment": {
                    "type": "hook_success",
                    "hookName": "PreToolUse:mcp__controls__channel_write",
                    "hookEvent": "PreToolUse",
                    "toolUseID": "tu_ok",
                    "stdout": '{"hookSpecificOutput": {"permissionDecision": "ask"}}',
                }
            },
            {
                "attachment": {
                    "type": "hook_error",
                    "hookName": "PreToolUse:mcp__controls__channel_write",
                    "hookEvent": "PreToolUse",
                    "toolUseID": "tu_broken",
                    "stdout": "Traceback (most recent call last):",
                }
            },
        ],
    )

    found = hook_attachments(_session_result("sess-1"), render)

    assert [a["toolUseID"] for a in found] == ["tu_ok", "tu_broken"]
    assert all(a["type"].startswith(HOOK_ATTACHMENT_PREFIX) for a in found)
    # The dicts come back raw — nothing is projected away, because a reader
    # chasing an unexplained refusal does not know in advance which key holds
    # the answer.
    assert found[1]["stdout"] == "Traceback (most recent call last):"
    assert found[1]["hookName"] == "PreToolUse:mcp__controls__channel_write"


@pytest.mark.unit
def test_hook_attachments_survives_a_malformed_transcript(tmp_path, monkeypatch):
    """A truncated write or a bare JSON scalar must not raise out of a
    diagnostic path — the run being diagnosed is the one that already failed."""
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    render = (tmp_path / "render").resolve()
    render.mkdir(parents=True, exist_ok=True)
    transcript = claude_project_dir(render) / "sess-2.jsonl"
    transcript.parent.mkdir(parents=True, exist_ok=True)
    transcript.write_text(
        "\n".join(
            [
                "{not json at all",
                '"a bare string"',
                "[1, 2, 3]",
                json.dumps({"attachment": "not a dict"}),
                json.dumps({"attachment": {"type": "hook_success", "toolUseID": "tu_1"}}),
            ]
        ),
        encoding="utf-8",
    )

    assert [a["toolUseID"] for a in hook_attachments(_session_result("sess-2"), render)] == ["tu_1"]


@pytest.mark.unit
def test_hook_attachments_empty_without_a_session_or_transcript(tmp_path, monkeypatch):
    """No session id and no transcript both mean "nothing to read", not an error.

    Callers assert on what they expected to find; a bare empty list is a
    clearer failure than an exception raised from an observer.
    """
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    render = (tmp_path / "render").resolve()
    render.mkdir(parents=True, exist_ok=True)

    assert hook_attachments(SDKWorkflowResult(), render) == []
    assert hook_attachments(_session_result("never-written"), render) == []


@pytest.mark.unit
def test_transcript_dump_carries_the_hook_stdout_when_given_a_render(tmp_path, monkeypatch):
    """The artifact reason this exists: tool traces alone cannot say why a call
    was asked about, because the approval wording only ever reaches the
    transcript."""
    render = _write_transcript(
        tmp_path,
        monkeypatch,
        "sess-3",
        [
            {
                "attachment": {
                    "type": "hook_success",
                    "hookName": "PreToolUse:mcp__controls__channel_write",
                    "hookEvent": "PreToolUse",
                    "toolUseID": "tu_1",
                    "stdout": '{"hookSpecificOutput": {"permissionDecisionReason": "live machine"}}',
                }
            }
        ],
    )
    monkeypatch.setenv("OSPREY_CI_DIAG_DIR", str(tmp_path / "diag"))

    target = dump_agent_transcript("switch_run", _session_result("sess-3"), render=render)

    assert target is not None
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert [a["toolUseID"] for a in payload["hook_attachments"]] == ["tu_1"]
    assert "live machine" in payload["hook_attachments"][0]["stdout"]


# ---------------------------------------------------------------------------
# Approval-policy arity. A policy written before the permission context existed
# takes two arguments and must keep working untouched; a policy that wants to
# see why the hook asked takes a third. The binder decides which from the
# signature, so a `TypeError` raised inside a three-argument policy surfaces as
# itself instead of being mistaken for the older shape.
# ---------------------------------------------------------------------------


class _FakeContext:
    """Stand-in for the SDK's permission context — no SDK install required."""

    def __init__(self, decision_reason: str | None = None):
        self.decision_reason = decision_reason


@pytest.mark.unit
def test_two_arg_policy_is_called_with_two_arguments():
    """The pre-existing form is dispatched unchanged — the context is dropped."""
    seen: list[tuple] = []

    def policy(tool_name, tool_input):
        seen.append((tool_name, tool_input))
        return True

    bound = _bind_approval_policy(policy)

    assert bound("Bash", {"command": "ls"}, _FakeContext("hook said ask")) is True
    assert seen == [("Bash", {"command": "ls"})]


@pytest.mark.unit
def test_three_arg_policy_receives_the_context():
    """The widened form gets the context object itself, not a copy of a field."""
    seen: list[tuple] = []
    context = _FakeContext("write outside the render")

    def policy(tool_name, tool_input, ctx):
        seen.append((tool_name, tool_input, ctx))
        return False

    bound = _bind_approval_policy(policy)

    assert bound("Write", {"file_path": "/etc/hosts"}, context) is False
    assert seen == [("Write", {"file_path": "/etc/hosts"}, context)]
    assert seen[0][2].decision_reason == "write outside the render"


@pytest.mark.unit
def test_var_positional_policy_receives_the_context():
    """A ``*args`` policy can take the context, so it is given the context."""
    seen: list[tuple] = []

    def policy(*args):
        seen.append(args)
        return True

    _bind_approval_policy(policy)("Read", {}, _FakeContext("why"))

    assert len(seen[0]) == 3


@pytest.mark.unit
def test_binder_does_not_swallow_type_errors_from_the_policy():
    """A TypeError raised *inside* a policy is a bug, not an arity signal."""

    def policy(tool_name, tool_input, ctx):
        raise TypeError("boom inside the policy")

    with pytest.raises(TypeError, match="boom inside the policy"):
        _bind_approval_policy(policy)("Bash", {}, _FakeContext())


@pytest.mark.unit
def test_hook_event_decision_reason_defaults_and_accepts():
    """The new field is optional (existing constructions keep working) and
    carries the hook's own wording when the callback records it."""
    without = HookEvent(tool_name="Bash", tool_input={}, decision="allow")
    assert without.decision_reason is None
    assert without.reason is None

    with_reason = HookEvent(
        tool_name="Write",
        tool_input={"file_path": "notes.md"},
        decision="deny",
        reason="custom_policy",
        decision_reason="target is outside the render",
    )
    assert with_reason.reason == "custom_policy"
    assert with_reason.decision_reason == "target is outside the render"
