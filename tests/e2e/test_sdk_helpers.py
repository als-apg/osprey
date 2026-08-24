"""Unit tests for tests/e2e/sdk_helpers.py pure-logic helpers.

No LLM, no build, no DB — fast logic checks. Excluded from the model-capability
matrix (scripts/benchmark/matrix_e2e_config.json) since it measures nothing
about the model under test.
"""

from __future__ import annotations

import json

import pytest

from osprey.agent_runner.primitives import SDKWorkflowResult, ToolTrace
from tests.e2e.sdk_helpers import (
    _DEFAULT_ARIEL_DB_URI,
    TRANSCRIPT_SUBDIR,
    HookEvent,
    _bind_approval_policy,
    _override_ariel_db_uri,
    dump_agent_transcript,
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

    assert dump_agent_transcript("orbit_response", _result_with_a_long_tool_result()) is None
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


@pytest.mark.unit
def test_transcript_dump_sanitises_the_name(tmp_path, monkeypatch):
    """Parametrised ids carry `[]`, and paths carry `/`; neither may escape
    the diagnostics directory or produce an unopenable filename."""
    monkeypatch.setenv("OSPREY_CI_DIAG_DIR", str(tmp_path))

    target = dump_agent_transcript("../grid[two_axis]", SDKWorkflowResult())

    assert target is not None
    assert target.parent == tmp_path / TRANSCRIPT_SUBDIR
    assert target.name == ".._grid_two_axis_.json"


@pytest.mark.unit
def test_transcript_dump_never_raises(tmp_path, monkeypatch):
    """A diagnostic that fails the test it observes would mask the failure it
    exists to explain. Here the target directory is occupied by a file, so the
    mkdir cannot succeed."""
    blocker = tmp_path / "diag"
    blocker.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv("OSPREY_CI_DIAG_DIR", str(blocker))

    assert dump_agent_transcript("orbit_response", SDKWorkflowResult()) is None


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
