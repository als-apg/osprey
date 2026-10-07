"""The approval ask can rewrite the call it gates: ``updatedInput`` beside ``"ask"``.

A guarded tool binds the call a human approved to the call that runs by a field
the approval hook writes into the tool's input. The hook returns a PreToolUse
envelope with ``permissionDecision: "ask"`` and an ``updatedInput`` mapping that
is the agent's input plus the binding keys, overwriting any value the agent
supplied for them. That carrier holds only if the rewritten input — not the
agent's original — is what reaches the tool on both approval paths:

* the interactive Claude Code CLI prompt, approved by a person;
* an agent-SDK session whose ``can_use_tool`` approver returns a bare
  ``PermissionResultAllow()`` with no ``updated_input`` — the shape of the
  dispatch worker's backstop, ``osprey.agent_runner.tool_policy.make_backstop``
  (wired by ``osprey.mcp_server.dispatch_worker.sdk_runner`` as its
  ``can_use_tool``), and of ``osprey.agent_runner.sdk_context.make_tool_allowlist``.

Both paths were proven live against Claude Code CLI 2.1.267 (the scaffolding
default for ``claude_code.cli_version``) and claude-agent-sdk 0.2.136: a
PreToolUse command hook answering ``"ask"`` with an added key in
``updatedInput`` reached ``mcp__python__execute`` with that key, after a prompt
approval and after a bare-allow approver alike. Those runs need a model and a
person, so they live outside the suite; this module pins what they rest on:

* the envelope uses only keys the pinned SDK declares for a PreToolUse hook,
  and ``"ask"`` is one of its decisions;
* every envelope the shipped approval hook builds stays inside that key set;
* a bare ``PermissionResultAllow()`` from either OSPREY approver sends the CLI
  back the input the CLI asked about — the hook-rewritten one — unchanged;
* the versions the proof holds for, so a bump re-proves before it lands.
"""

from __future__ import annotations

import ast
import json
import typing
from pathlib import Path
from typing import Any

import claude_agent_sdk
import pytest
from claude_agent_sdk import PermissionResultAllow
from claude_agent_sdk._internal.query import Query
from claude_agent_sdk.types import PreToolUseHookSpecificOutput

from osprey.agent_runner.sdk_context import make_tool_allowlist
from osprey.agent_runner.tool_policy import make_backstop
from osprey.cli.templates.scaffolding import _DEFAULT_CLAUDE_CLI_VERSION
from tests.hooks.test_hook_docstring_frontmatter import HOOKS_DIR

GUARDED_TOOL = "mcp__python__execute"

#: The binding keys the approval hook adds to a guarded tool's input.
BINDING_KEYS = ("approved_journal_sha256", "approved_target", "approved_view_sha256")

#: The versions the live proof of both approval paths ran against.
PROVEN_CLI_VERSION = "2.1.267"
PROVEN_SDK_VERSION = "0.2.136"


def ask_with_updated_input(tool_input: dict[str, Any], binding: dict[str, str]) -> dict:
    """The ask envelope that carries *binding* into the gated call's input."""
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "ask",
            "permissionDecisionReason": "approval required",
            "updatedInput": {**tool_input, **binding},
        }
    }


def _declared_pretooluse_keys() -> dict[str, Any]:
    return typing.get_type_hints(PreToolUseHookSpecificOutput, include_extras=False)


def _literal_values(annotation: Any) -> set[str]:
    """Every string a ``Literal`` (possibly under ``NotRequired``) admits."""
    values: set[str] = set()
    for arg in typing.get_args(annotation) or ():
        if isinstance(arg, str):
            values.add(arg)
        else:
            values |= _literal_values(arg)
    return values


class _RecordingTransport:
    """A transport that keeps what the SDK writes back to the CLI."""

    def __init__(self) -> None:
        self.written: list[dict] = []

    async def connect(self) -> None:  # pragma: no cover - never connected
        pass

    async def write(self, data: str) -> None:
        self.written.append(json.loads(data))

    def read_messages(self):  # pragma: no cover - never read
        raise NotImplementedError

    async def close(self) -> None:  # pragma: no cover - never closed
        pass

    def is_ready(self) -> bool:  # pragma: no cover - never asked
        return True

    async def end_input(self) -> None:  # pragma: no cover - never ended
        pass


async def _answer_permission_request(approver, tool_input: dict[str, Any]) -> dict:
    """What the SDK sends the CLI for one ``can_use_tool`` request about *tool_input*."""
    transport = _RecordingTransport()
    query = Query(transport=transport, is_streaming_mode=True, can_use_tool=approver)
    await query._handle_control_request(
        {
            "type": "control_request",
            "request_id": "req-1",
            "request": {
                "subtype": "can_use_tool",
                "tool_name": GUARDED_TOOL,
                "input": tool_input,
                "permission_suggestions": None,
                "blocked_path": None,
            },
        }
    )
    (reply,) = transport.written
    assert reply["response"]["subtype"] == "success", reply
    return reply["response"]["response"]


# --- the envelope ------------------------------------------------------------------


def test_the_envelope_uses_only_keys_the_sdk_declares_for_pretooluse() -> None:
    declared = _declared_pretooluse_keys()
    envelope = ask_with_updated_input({"code": "print(1)"}, {"approved_journal_sha256": "none"})

    assert set(envelope["hookSpecificOutput"]) <= set(declared)
    assert {"permissionDecision", "updatedInput"} <= set(declared)


def test_ask_is_a_pretooluse_decision() -> None:
    assert "ask" in _literal_values(_declared_pretooluse_keys()["permissionDecision"])


def test_updated_input_is_the_agents_input_plus_the_binding() -> None:
    agent_input = {"code": "print(1)", "approved_journal_sha256": "agent-chosen"}
    binding = {
        "approved_journal_sha256": "none",
        "approved_target": "demo",
        "approved_view_sha256": "0" * 64,
    }

    updated = ask_with_updated_input(agent_input, binding)["hookSpecificOutput"]["updatedInput"]

    assert updated == {"code": "print(1)", **binding}
    assert agent_input["approved_journal_sha256"] == "agent-chosen"
    assert set(BINDING_KEYS) <= set(updated)


def _hook_specific_output_keys(source: str) -> list[set[str]]:
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values, strict=False):
            if (
                isinstance(key, ast.Constant)
                and key.value == "hookSpecificOutput"
                and isinstance(value, ast.Dict)
            ):
                found.append(
                    {
                        k.value
                        for k in value.keys
                        if isinstance(k, ast.Constant) and isinstance(k.value, str)
                    }
                )
    return found


def test_every_approval_hook_envelope_stays_inside_the_declared_keys() -> None:
    source = (Path(HOOKS_DIR) / "osprey_approval.py").read_text(encoding="utf-8")
    envelopes = _hook_specific_output_keys(source)
    declared = set(_declared_pretooluse_keys())

    assert envelopes, "osprey_approval.py builds no hookSpecificOutput mapping"
    for keys in envelopes:
        assert keys <= declared, f"undeclared PreToolUse keys: {sorted(keys - declared)}"


# --- the bare-allow approval path ---------------------------------------------------


def test_a_bare_allow_leaves_updated_input_unset() -> None:
    assert PermissionResultAllow().updated_input is None


@pytest.mark.parametrize(
    "approver",
    [
        pytest.param(make_backstop([GUARDED_TOOL], {}), id="dispatch-backstop"),
        pytest.param(make_tool_allowlist([GUARDED_TOOL]), id="sdk-context-allowlist"),
    ],
)
async def test_a_bare_allow_returns_the_hook_rewritten_input_unchanged(approver) -> None:
    rewritten = ask_with_updated_input(
        {"code": "print(1)"},
        {"approved_journal_sha256": "none", "approved_target": "demo"},
    )["hookSpecificOutput"]["updatedInput"]

    response = await _answer_permission_request(approver, dict(rewritten))

    assert response == {"behavior": "allow", "updatedInput": rewritten}


# --- the versions the proof holds for -----------------------------------------------


def test_the_proof_names_the_pinned_cli() -> None:
    assert _DEFAULT_CLAUDE_CLI_VERSION == PROVEN_CLI_VERSION, (
        "the default Claude Code CLI moved; re-prove that updatedInput reaches the "
        "tool after an interactive approval, then update PROVEN_CLI_VERSION"
    )


def test_the_proof_names_the_pinned_sdk() -> None:
    assert claude_agent_sdk.__version__ == PROVEN_SDK_VERSION, (
        "claude-agent-sdk moved; re-prove that updatedInput reaches the tool after a "
        "bare PermissionResultAllow(), then update PROVEN_SDK_VERSION"
    )
