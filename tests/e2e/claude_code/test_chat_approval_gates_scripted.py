"""The chat child's approval gates, proved on the Claude Code the project pins.

The chat launches its child with the project's own hooks, no hooks of
its own and no bypass, and puts every tool ask the child raises in front of the
operator as an approval card. These scripted runs drive that real launch
(:class:`~osprey.interfaces.web_terminal.operator_session.OperatorSession`)
against a real render, with the scripted model (:mod:`tests.e2e.scripted_model`)
making the calls, and prove six things:

(a) a write the limits hook denies is not executed, although the approval hook
    asks about it, and the operator is never shown it;
(b) a write the writes-check hook denies (writes off for the deployment) is not
    executed, the approval hook defers to that deny rather than asking, and the
    operator is never shown it;
(c) a write only the approval hook asks about is put to the operator, with no
    "Allow for this session" offered, and an allow lets it run;
(d) an operator's decline blocks the call.
(e) an approved write the tool then refuses is put to the operator, is allowed, and is
    refused in the tool; the model is still told the operator approved it.
(f) a session allow rule for the tool, installed as "Allow for this session" would,
    does not skip the hook's ask: the CLI ranks the ask above the rule, and suggests
    no such rule for a hook ask in the first place.
(h) the write switch's position decides a governed call: at ``asked`` a write is carded on
    a human's record; at ``allowed`` it runs with no card on a ``write_switch`` record the
    server accepts, with an audit entry naming the switch and the position; moving back to
    ``asked`` makes the very next write prompt; code naming the posture store is carded at
    ``allowed`` too; and a readonly run still refuses the write at ``allowed``.

In (a) and (b) the operator approves everything, so an approval the hooks did
not earn would run the write, and the test would see it executed.

The approval hook tells the model, in the request that carries an approved call's result,
that the operator approved it (APPROVAL_NOTE). That note reaches the request after (c) and
(e), and no request at all after (a), (b) or (d).

With code review on, the scripted model answers the reviewer too, and two more runs prove
the review comes first: (j) a passed execute is carded with its verdict, runs on an allow and
is reviewed once, the executor taking the review from the ask stamp; (k) a blocked execute is
denied with its reasons and never carded. A third proves the review with no prompt: (l) with the
switch at ``allowed`` the executor's review refuses a blocked run.

Each execute that (c), (d) and (e) add also settles the review notebook its approval left:
it reads "ran" in (c), "not run" in (d), and "failed" in (e).

Each gate's own decision for each call is read from the agent record's hook debug
lines, which record every decision a hook makes, denials included. The transcript is not the
witness: it keeps a hook's output when the hook asks or allows, and nothing when
the hook denies.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import nbformat
import pytest
import yaml
from claude_agent_sdk.types import PermissionRuleValue, PermissionUpdate

from osprey.agent_runner.primitives import await_mcp_ready, expected_mcp_servers
from osprey.agent_runner.project_paths import CLAUDE_CONFIG_DIR_ENV
from osprey.interfaces.web_terminal.operator_session import OperatorSession
from osprey.stores.artifact_store import ArtifactStore
from osprey_connectors import control_context
from osprey_connectors.identity import acting_identity
from osprey_connectors.workspace import resolve_shared_data_root
from tests.e2e import sdk_helpers
from tests.e2e.claude_code._chat_turn import (
    TASK,
    Answer,
    calls_and_results,
    carries_result,
    drive_turn,
    record_conversations,
)
from tests.e2e.scripted_model import ScriptedCall
from tests.e2e.sdk_helpers import render_dir

pytestmark = [pytest.mark.model_free, pytest.mark.harness_benchmark]

WRITE = "mcp__controls__channel_write"
READ = "mcp__controls__channel_read"
#: Bounded writable setpoint in the safety limits fixture: [0, 100].
SETPOINT = "SR:DIAG:TEMP:01:TEMPERATURE:SP"

#: The reason the limits hook gives for an over-limit write.
LIMITS_REASON = "CHANNEL LIMITS VIOLATION"
#: The reason the writes-check hook gives under a deployment-wide readonly run.
WRITES_OFF_REASON = "WRITES OFF"
#: What the operator's decline says back to the model.
OPERATOR_DECLINED = "Declined by the scripted operator."

#: What the approval hook tells the model about a call a person approved at its prompt.
APPROVAL_NOTE = "The operator approved this call at the approval prompt before it ran."

EXECUTE = "mcp__python__execute"
#: What the scenarios' execute calls run. Readwrite mode, so the approval hook asks about it.
EXECUTE_CODE = "x = 1 + 1\nprint(x)\n"
#: The description the python tool requires on every call.
EXECUTE_DESCRIPTION = "Scripted proof: approval review notebook"
REVIEW_TITLE = "Approval preview"
REVIEW_NOT_RUN = "Not run: declined or interrupted at the prompt"


def write(channel: str, value: float) -> ScriptedCall:
    return ScriptedCall(WRITE, {"operations": [{"channel": channel, "value": value}]})


def read(channel: str) -> ScriptedCall:
    return ScriptedCall(READ, {"channels": [channel], "include_metadata": False})


def unfinished_write() -> ScriptedCall:
    """A write with no ``operations``, the tool's required argument: the tool refuses it."""
    return ScriptedCall(WRITE, {})


def execute_readwrite() -> ScriptedCall:
    """An execute in readwrite mode: the approval hook asks about it, and the code runs."""
    return ScriptedCall(
        EXECUTE,
        {"code": EXECUTE_CODE, "description": EXECUTE_DESCRIPTION, "execution_mode": "readwrite"},
    )


def failing_execute() -> ScriptedCall:
    """An approved execute whose code raises, so the tool reports the run as failed."""
    code = 'raise RuntimeError("refused by the scripted proof")'
    return ScriptedCall(
        EXECUTE,
        {"code": code, "description": EXECUTE_DESCRIPTION, "execution_mode": "readwrite"},
    )


def _review_notebooks(repo) -> dict[str, tuple[str, str]]:
    """Every review notebook the render's artifact store holds, as ``id -> (status, description)``.

    The store is the one the approval hook writes, under the root the render's own config
    resolves to.
    """
    config = yaml.safe_load((render_dir(repo) / "config.yml").read_text(encoding="utf-8"))
    store = ArtifactStore(workspace_root=resolve_shared_data_root(config), auto_launch=False)
    reviews = {}
    for entry in store.list_entries():
        if entry.title != REVIEW_TITLE:
            continue
        notebook = nbformat.read(store.get_file_path(entry.id), as_version=4)
        (line,) = [
            line
            for line in notebook.cells[0].source.splitlines()
            if line.startswith("**Status:** ")
        ]
        reviews[entry.id] = (line.removeprefix("**Status:** ").rstrip(), entry.description)
    return reviews


def _new_review(repo, before: dict[str, tuple[str, str]]) -> tuple[str, str]:
    """The ``(status, description)`` of the one review notebook added since *before*."""
    added = [value for key, value in _review_notebooks(repo).items() if key not in before]
    assert len(added) == 1, f"expected one new review notebook, saw {len(added)}"
    return added[0]


def approves_everything(_card: dict[str, Any]) -> Answer:
    return Answer(allow=True)


def declines_everything(_card: dict[str, Any]) -> Answer:
    return Answer(allow=False, message=OPERATOR_DECLINED)


@dataclass
class ChatRun:
    """What one chat turn produced, from the frames the chat itself received.

    Attributes:
        calls: Every ``tool_use`` frame, in order.
        results: The ``tool_result`` frame of each call, by tool-use id.
        cards: Every approval card the operator was shown.
        gates: Each hook's own decision for each call, by tool-use id, as the hook
            logged it: ``(hook, status)`` in the order the hooks ran.
        conversations: The message list of every request the endpoint received, in order.
    """

    calls: list[dict[str, Any]] = field(default_factory=list)
    results: dict[str, dict[str, Any]] = field(default_factory=dict)
    cards: list[dict[str, Any]] = field(default_factory=list)
    gates: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    conversations: list[list[dict[str, Any]]] = field(default_factory=list)

    def only_call(self, tool: str) -> dict[str, Any]:
        """The one call of *tool*; the scenarios make exactly one each."""
        calls = [call for call in self.calls if call["tool_name_raw"] == tool]
        assert len(calls) == 1, f"expected one {tool} call, saw {len(calls)}"
        return calls[0]

    def was_carded(self, call: dict[str, Any]) -> bool:
        return any(card["tool_use_id"] == call["tool_use_id"] for card in self.cards)

    def statuses(self, call: dict[str, Any]) -> list[tuple[str, str]]:
        """Every ``(hook, status)`` logged for *call*, in the order the hooks ran."""
        return self.gates.get(call["tool_use_id"], [])

    def describe(self, call: dict[str, Any]) -> str:
        """The facts a failing assertion needs, in one line."""
        result = self.results.get(call["tool_use_id"], {})
        return (
            f"gates={self.statuses(call)} carded={self.was_carded(call)} "
            f"result={result.get('is_error')}:{str(result.get('preview'))[:160]!r}"
        )

    def note_reached_model(self, call: dict[str, Any]) -> bool:
        """Whether the approval note is in the first request that carries *call*'s result.

        The note travels with the result, so it must sit after the result in the request
        the result first arrives in.
        """
        for messages in self.conversations:
            for index, message in enumerate(messages):
                if carries_result(message, call["tool_use_id"]):
                    return APPROVAL_NOTE in json.dumps(messages[index:])
        return False

    def note_reached_any_request(self) -> bool:
        """Whether the approval note is in any request the endpoint received."""
        return any(APPROVAL_NOTE in json.dumps(messages) for messages in self.conversations)


async def drive_chat(repo, scripted_model, turns, *, operator, env=None) -> tuple[Any, ChatRun]:
    """Run *turns* through the chat's own launch on *repo*, as *operator*.

    The turn is :func:`~tests.e2e.claude_code._chat_turn.drive_turn`'s: the render is
    trusted, hook debugging is on, *env* is on top, and the operator answers each
    approval card. The scripted model's every request is kept whole.
    """
    model = scripted_model(turns)
    conversations = record_conversations(model)
    frames, cards, gates = await drive_turn(repo, TASK, operator, env=env)
    model.assert_script_consumed()
    calls, results = calls_and_results(frames)
    run = ChatRun(
        calls=calls, results=results, cards=cards, gates=gates, conversations=conversations
    )
    return model, run


def _rendered_ask_list(repo) -> list[str]:
    """The ``permissions.ask`` entries the render's settings carry."""
    settings = json.loads((render_dir(repo) / ".claude" / "settings.json").read_text("utf-8"))
    return list(settings.get("permissions", {}).get("ask", []))


# ---------------------------------------------------------------------------
# Precondition of every proof here: the render is trusted in a Claude Code state
# the harness owns, never in the developer's own.
# ---------------------------------------------------------------------------


def test_the_proofs_run_under_the_harness_own_claude_state(tmp_path_factory):
    """The e2e conftest points ``CLAUDE_CONFIG_DIR`` into the temp tree for a
    model-free test and names the same directory to the helpers that seed trust."""
    assert sdk_helpers.harness_config_dir is not None
    assert os.environ[CLAUDE_CONFIG_DIR_ENV] == str(sdk_helpers.harness_config_dir)
    assert sdk_helpers.harness_config_dir.is_relative_to(tmp_path_factory.getbasetemp())


# ---------------------------------------------------------------------------
# (a) the limits hook denies; the approval hook asks; the operator approves everything
# ---------------------------------------------------------------------------


async def test_an_over_limit_write_is_not_executed_though_approval_asks(
    safety_project, scripted_model
):
    """(a): the limits hook denies, the approval hook asks, and the write does not run."""
    model, run = await drive_chat(
        safety_project,
        scripted_model,
        [[read(SETPOINT)], [write(SETPOINT, 999.0)], [read(SETPOINT)]],
        operator=approves_everything,
    )
    assert WRITE in model.offered_tools
    call = run.only_call(WRITE)
    facts = run.describe(call)
    print(f"chat-gates (a) {facts}")

    # The conflict is real: the approval hook asked, and the limits hook denied, for this call.
    assert ("approval", "ask") in run.statuses(call), facts
    assert ("limits", "deny") in run.statuses(call), facts

    # Deny beats ask: the callback is never consulted for a call a hook denied.
    assert not run.was_carded(call), f"the operator was shown an over-limit write: {facts}"

    # Not executed: the refusal is the limits hook's, before the tool ran.
    refused = run.results[call["tool_use_id"]]
    assert refused["is_error"], facts
    assert refused["preview"].startswith(f"PreToolUse:{WRITE} hook error"), facts
    assert LIMITS_REASON in refused["preview"], facts
    assert not run.note_reached_any_request(), "the model was told a refused call was approved"


# ---------------------------------------------------------------------------
# (b) writes-check denies (writes off); the approval hook defers; the operator approves everything
# ---------------------------------------------------------------------------


async def test_a_write_with_writes_off_is_not_executed_and_approval_defers(
    safety_project, scripted_model
):
    """(b): a deployment-wide readonly run denies the write at writes-check. On an
    ``allowed`` render the write's rule carries both hooks, so the render pulls its
    static ``permissions.ask`` and denies nothing: the hook chain is the whole boundary,
    and the approval hook defers on a write the verdict refuses, so no operator is
    offered an approval for a write that cannot run."""
    assert WRITE not in _rendered_ask_list(safety_project), (
        "the render still carries a static ask for the write; an ask there would put "
        "the refused write in front of the operator"
    )
    model, run = await drive_chat(
        safety_project,
        scripted_model,
        [[read(SETPOINT)], [write(SETPOINT, 75.0)], [read(SETPOINT)]],
        operator=approves_everything,
        env={"OSPREY_EXECUTION_MODE": "readonly"},
    )
    assert WRITE in model.offered_tools
    call = run.only_call(WRITE)
    facts = run.describe(call)
    print(f"chat-gates (b) {facts}")

    assert ("approval", "defer") in run.statuses(call), facts
    assert ("approval", "ask") not in run.statuses(call), facts
    assert ("writes-check", "deny") in run.statuses(call), facts

    assert not run.was_carded(call), f"the operator was shown a write with writes off: {facts}"

    refused = run.results[call["tool_use_id"]]
    assert refused["is_error"], facts
    assert refused["preview"].startswith(f"PreToolUse:{WRITE} hook error"), facts
    assert WRITES_OFF_REASON in refused["preview"], facts
    assert not run.note_reached_any_request(), "the model was told a refused call was approved"


# ---------------------------------------------------------------------------
# (c) an approval-only ask reaches the operator; an allow runs the write
# ---------------------------------------------------------------------------


async def test_an_approval_only_write_is_carded_and_an_allow_runs_it(
    safety_project, scripted_model
):
    """(c): in limits, writes enabled. The card reaches the operator, and the allow runs the
    write; the scenarios above cannot be passing on a chat that never asks. The execute that
    follows is an approval too, and its review notebook reads that it ran."""
    before = _review_notebooks(safety_project)
    _model, run = await drive_chat(
        safety_project,
        scripted_model,
        [[read(SETPOINT)], [write(SETPOINT, 75.0)], [execute_readwrite()], [read(SETPOINT)]],
        operator=approves_everything,
    )
    call = run.only_call(WRITE)
    facts = run.describe(call)
    print(f"chat-gates (c) {facts}")

    assert run.was_carded(call), f"the approval-only write never reached the operator: {facts}"
    assert ("approval", "ask") in run.statuses(call), facts
    (card,) = [c for c in run.cards if c["tool_use_id"] == call["tool_use_id"]]
    assert card["can_allow_session"] is False, "a write was offered 'Allow for this session'"
    executed = run.results[call["tool_use_id"]]
    assert not executed["is_error"], f"an allowed write was refused: {facts}"
    assert run.note_reached_model(call), f"the model was not told the operator approved it: {facts}"
    execute = run.only_call(EXECUTE)
    print(f"chat-gates (c) execute {run.describe(execute)}")
    assert run.was_carded(execute), (
        f"the execute never reached the operator: {run.describe(execute)}"
    )
    assert not run.results[execute["tool_use_id"]]["is_error"], run.describe(execute)
    assert _new_review(safety_project, before) == (
        "Ran after operator approval",
        "Code ran after operator approval (mode: readwrite)",
    )


# ---------------------------------------------------------------------------
# (d) the operator's decline blocks the call
# ---------------------------------------------------------------------------


async def test_an_operator_decline_blocks_an_approval_only_write(safety_project, scripted_model):
    """(d): the same write, declined on its card: the call is blocked, and says so. The
    execute declined on its card settles its review as not run."""
    before = _review_notebooks(safety_project)
    _model, run = await drive_chat(
        safety_project,
        scripted_model,
        [[read(SETPOINT)], [write(SETPOINT, 75.0)], [execute_readwrite()], [read(SETPOINT)]],
        operator=declines_everything,
    )
    call = run.only_call(WRITE)
    facts = run.describe(call)
    print(f"chat-gates (d) {facts}")

    assert run.was_carded(call), f"the decline was never offered: {facts}"
    blocked = run.results[call["tool_use_id"]]
    assert blocked["is_error"], facts
    assert OPERATOR_DECLINED in blocked["preview"], facts
    assert not run.note_reached_any_request(), "the model was told a declined call was approved"
    execute = run.only_call(EXECUTE)
    print(f"chat-gates (d) execute {run.describe(execute)}")
    assert run.was_carded(execute), (
        f"the execute never reached the operator: {run.describe(execute)}"
    )
    assert _new_review(safety_project, before) == (
        REVIEW_NOT_RUN,
        "Code not run: declined or interrupted at the prompt (mode: readwrite)",
    )


# ---------------------------------------------------------------------------
# (e) an approved write the tool refuses is still put to the operator, and the model is told
# ---------------------------------------------------------------------------


async def test_an_approved_write_that_fails_in_the_tool_still_tells_the_model(
    safety_project, scripted_model
):
    """(e): the operator allows a write the tool then refuses. The model is still told the
    operator approved it, because the approval came before the refusal. The execute the tool
    refuses settles its review as failed."""
    before = _review_notebooks(safety_project)
    _model, run = await drive_chat(
        safety_project,
        scripted_model,
        [[read(SETPOINT)], [unfinished_write()], [failing_execute()], [read(SETPOINT)]],
        operator=approves_everything,
    )
    call = run.only_call(WRITE)
    facts = run.describe(call)
    print(f"chat-gates (e) {facts}")

    assert run.was_carded(call), f"the approval-only write never reached the operator: {facts}"
    assert ("approval", "ask") in run.statuses(call), facts
    failed = run.results[call["tool_use_id"]]
    assert failed["is_error"], f"the write was not refused in the tool: {facts}"
    assert run.note_reached_model(call), f"the model was not told the operator approved it: {facts}"
    execute = run.only_call(EXECUTE)
    print(f"chat-gates (e) execute {run.describe(execute)}")
    assert run.was_carded(execute), (
        f"the execute never reached the operator: {run.describe(execute)}"
    )
    assert run.results[execute["tool_use_id"]]["is_error"], run.describe(execute)
    assert _new_review(safety_project, before) == (
        "Failed after operator approval",
        "Code failed after operator approval (mode: readwrite)",
    )


# ---------------------------------------------------------------------------
# (f) a session allow rule does not skip the hook's ask
# ---------------------------------------------------------------------------


def _session_rule(tool_name: str) -> PermissionUpdate:
    """The rule "Allow for this session" would install for *tool_name*, were it offered."""
    return PermissionUpdate(
        type="addRules",
        rules=[PermissionRuleValue(tool_name=tool_name)],
        behavior="allow",
        destination="session",
    )


async def test_a_session_allow_rule_does_not_skip_the_hooks_ask(
    safety_project, scripted_model, monkeypatch
):
    """(f): the chat withholds "Allow for this session" under an approval prefix because a
    session rule that skipped the hook's ask would run every later call unasked. This run
    installs such a rule anyway, as the button would, and reads how the CLI ranks it: the
    hook still asks, and every write is still carded. The rule mechanism itself is proved
    on the CLI's own ask first: a Read outside the render is carded once and allowed for
    the session, with the rule the CLI suggests (``Read(//<repo>/**)``), and a Read of a
    second file beside it runs with no card.

    The lever is the server's rule filter, replaced to hand back the CLI's suggestions
    narrowed to the session, or a rule for the asked tool when the CLI suggests none.
    """
    from osprey.interfaces.web_terminal import operator_session

    suggested: dict[str, list[Any]] = {}

    def hand_back_a_session_rule(tool_name, suggestions, _surface):
        suggested.setdefault(tool_name, []).extend(suggestions)
        kept = [
            replace(s, destination="session")
            for s in suggestions
            if getattr(s, "type", None) == "addRules" and getattr(s, "behavior", None) == "allow"
        ]
        return kept or [_session_rule(tool_name)]

    monkeypatch.setattr(operator_session, "_standing_rules", hand_back_a_session_rule)

    first_file = Path(safety_project) / "outside-the-render-a.txt"
    second_file = Path(safety_project) / "outside-the-render-b.txt"
    first_file.write_text("the first file beside the render\n")
    second_file.write_text("the second file beside the render\n")
    read_first = ScriptedCall("Read", {"file_path": str(first_file)})
    read_second = ScriptedCall("Read", {"file_path": str(second_file)})
    first_of: dict[str, bool] = {}

    def allows_each_tool_for_the_session(card: dict[str, Any]) -> Answer:
        tool = card["tool_name_raw"]
        if first_of.setdefault(tool, True):
            first_of[tool] = False
            return Answer(allow=True, for_session=True)
        return Answer(allow=True)

    _model, run = await drive_chat(
        safety_project,
        scripted_model,
        [[read_first], [read_second], [write(SETPOINT, 10.0)], [write(SETPOINT, 20.0)]],
        operator=allows_each_tool_for_the_session,
    )
    reads = [c for c in run.calls if c["tool_name_raw"] == "Read"]
    writes = [c for c in run.calls if c["tool_name_raw"] == WRITE]
    assert len(reads) == 2 and len(writes) == 2, [c["tool_name_raw"] for c in run.calls]
    print(f"chat-gates (f) reads={[run.describe(c) for c in reads]}")
    print(f"chat-gates (f) writes={[run.describe(c) for c in writes]} suggested={suggested}")

    # The CLI's own ask: the suggested rule takes, and the second file is read unasked.
    assert run.was_carded(reads[0]), run.describe(reads[0])
    assert not run.was_carded(reads[1]), f"the session rule did not take: {run.describe(reads[1])}"
    second_read = run.results[reads[1]["tool_use_id"]]
    assert not second_read["is_error"], run.describe(reads[1])
    assert "second file" in second_read["preview"], run.describe(reads[1])

    # The hook's ask: the CLI suggests no rule for it, and one installed anyway does not
    # rank above it. Every write is asked about, and carded, and runs only when allowed.
    assert suggested.get(WRITE, []) == [], f"the CLI suggested a rule for a hook ask: {suggested}"
    for call in writes:
        assert ("approval", "ask") in run.statuses(call), run.describe(call)
        assert run.was_carded(call), f"a session rule skipped the hook's ask: {run.describe(call)}"
        assert not run.results[call["tool_use_id"]]["is_error"], run.describe(call)
    assert len(run.cards) == 3, [c["tool_name_raw"] for c in run.cards]


# ---------------------------------------------------------------------------
# (g) an approved call the operator interrupts: how its review notebook settles
# ---------------------------------------------------------------------------


def sleeping_execute() -> ScriptedCall:
    """An approved execute that runs long enough for the operator to stop the turn under it."""
    code = "import time\ntime.sleep(30)\nprint('slept')\n"
    return ScriptedCall(
        EXECUTE,
        {"code": code, "description": EXECUTE_DESCRIPTION, "execution_mode": "readwrite"},
    )


async def _turn(session: OperatorSession, prompt: str, operator, *, interrupt_after_s=None):
    """Run one turn on *session*, answering cards as *operator*; optionally Stop it.

    With *interrupt_after_s* set, the turn is interrupted that long after its first card
    is answered, as the operator's Stop button would.
    """
    frames: list[dict[str, Any]] = []
    stopper: asyncio.Task[None] | None = None
    token = session.acquire_turn()
    try:
        async for frame in session.run_turn(prompt, token, timeout_s=120.0):
            if frame.get("type") == "heartbeat":
                continue
            frames.append(frame)
            if frame.get("type") == "permission_request":
                answer = operator(frame)
                await session.resolve_permission(
                    frame["request_id"], allow=answer.allow, message=answer.message
                )
                if interrupt_after_s is not None and stopper is None:

                    async def stop_later() -> None:
                        await asyncio.sleep(interrupt_after_s)
                        await session.interrupt()

                    stopper = asyncio.ensure_future(stop_later())
    finally:
        if stopper is not None:
            stopper.cancel()
    return frames


async def test_an_interrupted_approved_execute_is_settled_not_run_at_the_next_stop(
    safety_project, scripted_model
):
    """(g): the operator approves an execute, then stops the turn while it runs. No outcome
    event reaches the hook, so the review stays as the ask left it until the conversation
    next stops; that Stop settles it as not run. The next turn ends normally and does
    exactly that.
    """
    before = _review_notebooks(safety_project)
    model = scripted_model([[sleeping_execute()], [read(SETPOINT)]])
    render = render_dir(safety_project)
    sdk_helpers.trust_render(safety_project)
    child_env = {**sdk_helpers.sdk_env(render), "OSPREY_HOOK_DEBUG": "1"}
    session = OperatorSession(cwd=str(render), env=child_env)
    try:
        await session.start()
        await await_mcp_ready(session._agent._client, expected_mcp_servers(render))
        first = await _turn(session, TASK, approves_everything, interrupt_after_s=3.0)
        after_interrupt = _new_review(safety_project, before)
        print(f"chat-gates (g) after the interrupt: {after_interrupt}")
        second = await _turn(session, "Carry on.", declines_everything)
    finally:
        await session.stop()
    model.assert_script_consumed()

    (execute,) = [f for f in first if f.get("type") == "tool_use"]
    assert execute["tool_name_raw"] == EXECUTE
    result = next(f for f in first if f.get("type") == "tool_result")
    assert result["is_error"], f"the interrupted execute reported a run: {result}"
    assert "slept" not in str(result.get("preview")), result
    assert not any(f.get("type") == "permission_request" for f in second)

    # Until the conversation stops again the review still reads as the ask left it.
    assert after_interrupt[0].startswith("Pending"), after_interrupt
    # The next turn's Stop settles it: the call never reported an outcome, so not run.
    assert _new_review(safety_project, before) == (
        REVIEW_NOT_RUN,
        "Code not run: declined or interrupted at the prompt (mode: readwrite)",
    )


# ---------------------------------------------------------------------------
# (h) the switch's position decides: `asked` prompts, `allowed` runs on a write_switch record
# ---------------------------------------------------------------------------

#: The approval hook's audit surface (``hook_`` plus the hook's own name).
APPROVAL_SURFACE = "hook_approval"


@contextlib.contextmanager
def switch_at(repo, position: sdk_helpers.SwitchPosition):
    """Move this process's switch for *repo*'s baseline target to *position*, then back.

    The module's deployment is armed at ``asked`` (the fixture's narrowest arming word),
    so every proof above runs there; this puts it back whatever the proof inside did.
    """
    sdk_helpers.set_switch_in_record(repo, owner=acting_identity(), position=position)
    try:
        yield
    finally:
        sdk_helpers.set_switch_in_record(repo, owner=acting_identity(), position="asked")


def _approval_gates(run: ChatRun, call: dict[str, Any]) -> list[Any]:
    """The approval hook's own records for *call*, each with its logged ``detail``."""
    return [gate for gate in run.statuses(call) if gate[0] == "approval"]


def store_naming_execute() -> ScriptedCall:
    """A read-only execute whose code spells the posture store's file name in a string."""
    code = f'print("{control_context.RECORD_FILENAME}")\n'
    return ScriptedCall(
        EXECUTE,
        {"code": code, "description": EXECUTE_DESCRIPTION, "execution_mode": "readonly"},
    )


async def test_at_asked_a_write_prompts_and_its_human_record_is_accepted(
    safety_project, scripted_model
):
    """(h1): at ``asked`` the write is carded, the stamp the card waits on names a human
    and the position, and the server accepts that record: the write runs."""
    stamps: list[dict[str, Any] | None] = []

    def reads_the_stamp_and_approves(card: dict[str, Any]) -> Answer:
        stamps.append(sdk_helpers.approval_trail(safety_project, card["tool_use_id"]).stamp)
        return Answer(allow=True)

    started = sdk_helpers.record_now()
    with switch_at(safety_project, "asked"):
        _model, run = await drive_chat(
            safety_project,
            scripted_model,
            [[write(SETPOINT, 40.0)], [read(SETPOINT)]],
            operator=reads_the_stamp_and_approves,
        )
    call = run.only_call(WRITE)
    facts = run.describe(call)
    trail = sdk_helpers.approval_trail(safety_project, call["tool_use_id"], since=started)
    print(f"chat-gates (h1) {facts} stamp={stamps} entries={trail.entries}")

    assert run.was_carded(call), f"a write at asked was not put to the operator: {facts}"
    (gate,) = [g for g in _approval_gates(run, call) if g == ("approval", "ask")]
    assert "position=asked" in gate.detail, gate
    (stamp,) = stamps
    assert stamp is not None, "the card was shown with no ask stamp on disk"
    assert stamp["approver"] == "human" and stamp["position"] == "asked", stamp
    assert not run.results[call["tool_use_id"]]["is_error"], f"the server refused it: {facts}"
    assert any(
        "approver=human" in str(line.get("detail"))
        for line in trail.entries
        if not str(line.get("surface")).startswith("hook_")
    ), f"no server audit entry accepted the human record: {trail.entries}"


async def test_at_allowed_a_write_runs_unprompted_on_a_write_switch_record(
    safety_project, scripted_model
):
    """(h2): at ``allowed`` the same write runs with no prompt and no card. The hook files a
    ``write_switch`` record, the server accepts it, and the hook's audit entry names the
    switch and the position."""
    started = sdk_helpers.record_now()
    with switch_at(safety_project, "allowed"):
        _model, run = await drive_chat(
            safety_project,
            scripted_model,
            [[write(SETPOINT, 45.0)], [read(SETPOINT)]],
            operator=declines_everything,
        )
    call = run.only_call(WRITE)
    facts = run.describe(call)
    trail = sdk_helpers.approval_trail(safety_project, call["tool_use_id"], since=started)
    print(f"chat-gates (h2) {facts} entries={trail.entries}")

    assert not run.cards, f"a write at allowed reached the operator: {facts}"
    assert ("approval", "ask") not in run.statuses(call), facts
    (gate,) = [g for g in _approval_gates(run, call) if g == ("approval", "allow")]
    assert "position=allowed" in gate.detail, gate
    assert not run.results[call["tool_use_id"]]["is_error"], f"the server refused it: {facts}"

    switch_lines = [
        line
        for line in trail.lines(APPROVAL_SURFACE)
        if line.get("decision") == "approved" and line.get("reason") == "switch"
    ]
    assert len(switch_lines) == 1, f"no single switch approval on the record: {trail.entries}"
    assert switch_lines[0].get("detail") == "approver=write_switch position=allowed", switch_lines
    assert any(
        "approver=write_switch" in str(line.get("detail"))
        for line in trail.entries
        if not str(line.get("surface")).startswith("hook_")
    ), f"no server audit entry accepted the write_switch record: {trail.entries}"


async def test_moving_back_to_asked_makes_the_very_next_write_prompt(
    safety_project, scripted_model
):
    """(h3): the first write runs unprompted at ``allowed``; the switch moves back to
    ``asked`` before the model answers with the second, and the second is carded."""
    model = scripted_model([[write(SETPOINT, 50.0)], [write(SETPOINT, 55.0)], [read(SETPOINT)]])
    conversations = record_conversations(model)
    answer = model.answer
    moved: list[bool] = []

    def move_back_after_the_first_result(body: dict[str, Any], credential: str | None):
        messages = [m for m in body.get("messages") or [] if isinstance(m, dict)]
        if not moved and any(
            isinstance(block, dict) and block.get("type") == "tool_result"
            for m in messages
            if m.get("role") == "user" and isinstance(m.get("content"), list)
            for block in m["content"]
        ):
            sdk_helpers.set_switch_in_record(
                safety_project, owner=acting_identity(), position="asked"
            )
            moved.append(True)
        return answer(body, credential)

    model.answer = move_back_after_the_first_result
    with switch_at(safety_project, "allowed"):
        frames, cards, gates = await drive_turn(safety_project, TASK, approves_everything)
    model.assert_script_consumed()
    calls, results = calls_and_results(frames)
    run = ChatRun(calls=calls, results=results, cards=cards, gates=gates)
    run.conversations = conversations
    first, second = [c for c in run.calls if c["tool_name_raw"] == WRITE]
    print(f"chat-gates (h3) first={run.describe(first)} second={run.describe(second)}")

    assert moved, "the switch was never moved back"
    assert not run.was_carded(first), run.describe(first)
    assert ("approval", "allow") in run.statuses(first), run.describe(first)
    assert not run.results[first["tool_use_id"]]["is_error"], run.describe(first)
    assert run.was_carded(second), f"the write after the move was not put: {run.describe(second)}"
    (gate,) = [g for g in _approval_gates(run, second) if g == ("approval", "ask")]
    assert "position=asked" in gate.detail, gate
    assert not run.results[second["tool_use_id"]]["is_error"], run.describe(second)


async def test_at_allowed_a_store_naming_execute_still_prompts(safety_project, scripted_model):
    """(h4): code that names the posture store is put to a person in every position, so at
    ``allowed`` it is still carded; the operator's decline blocks it."""
    with switch_at(safety_project, "allowed"):
        _model, run = await drive_chat(
            safety_project,
            scripted_model,
            [[store_naming_execute()], [read(SETPOINT)]],
            operator=declines_everything,
        )
    execute = run.only_call(EXECUTE)
    facts = run.describe(execute)
    print(f"chat-gates (h4) {facts}")

    assert run.was_carded(execute), f"a store-naming execute ran unprompted at allowed: {facts}"
    (gate,) = [g for g in _approval_gates(run, execute) if g == ("approval", "ask")]
    assert "position=asked" in gate.detail, gate
    blocked = run.results[execute["tool_use_id"]]
    assert blocked["is_error"] and OPERATOR_DECLINED in blocked["preview"], facts


async def test_at_allowed_a_readonly_run_still_refuses_the_write(safety_project, scripted_model):
    """(h5): ``OSPREY_EXECUTION_MODE=readonly`` refuses the write at writes-check whatever
    the switch says; the approval hook defers and nobody is shown it."""
    with switch_at(safety_project, "allowed"):
        _model, run = await drive_chat(
            safety_project,
            scripted_model,
            [[write(SETPOINT, 60.0)], [read(SETPOINT)]],
            operator=approves_everything,
            env={"OSPREY_EXECUTION_MODE": "readonly"},
        )
    call = run.only_call(WRITE)
    facts = run.describe(call)
    print(f"chat-gates (h5) {facts}")

    assert ("writes-check", "deny") in run.statuses(call), facts
    assert ("approval", "defer") in run.statuses(call), facts
    assert ("approval", "allow") not in run.statuses(call), facts
    assert not run.was_carded(call), facts
    refused = run.results[call["tool_use_id"]]
    assert refused["is_error"] and WRITES_OFF_REASON in refused["preview"], facts


# ---------------------------------------------------------------------------
# (j)/(k) code review on: the review comes before the prompt, and the executor trusts it
# ---------------------------------------------------------------------------

#: What the scripted reviewer blocks: code that names this call. Nothing else is read.
BLOCKED_CALL = "write_channel"
#: The reason the scripted reviewer gives for a block.
BLOCK_REASON = "Calls write_channel under a description that says it only reads."
BLOCKED_CODE = (
    'from osprey.runtime import write_channel as wc\nwc("SR:MAG:HCM:01:CURRENT:SP", 1.0)\n'
)


def scripted_reviewer(message: str) -> str:
    """Block code that names :data:`BLOCKED_CALL`; pass everything else."""
    if BLOCKED_CALL in message:
        return json.dumps({"verdict": "block", "reasons": [BLOCK_REASON]})
    return json.dumps({"verdict": "pass", "reasons": []})


def _review_header(repo, before: dict[str, tuple[str, str]]) -> str:
    """The header cell of the one review notebook added since *before*."""
    config = yaml.safe_load((render_dir(repo) / "config.yml").read_text(encoding="utf-8"))
    store = ArtifactStore(workspace_root=resolve_shared_data_root(config), auto_launch=False)
    (entry,) = [e for e in store.list_entries() if e.title == REVIEW_TITLE and e.id not in before]
    return nbformat.read(store.get_file_path(entry.id), as_version=4).cells[0].source


async def _drive_reviewed(repo, scripted_model, turns) -> tuple[Any, ChatRun]:
    """Run *turns* as :func:`drive_chat` does, the scripted reviewer answering each review."""
    model = scripted_model(turns, scripted_reviewer)
    frames, cards, gates = await drive_turn(repo, TASK, approves_everything)
    model.assert_script_consumed()
    calls, results = calls_and_results(frames)
    return model, ChatRun(calls=calls, results=results, cards=cards, gates=gates)


async def test_a_reviewed_execute_shows_its_verdict_and_is_not_reviewed_twice(
    safety_project_code_review, scripted_model
):
    """(j): the reviewer passes the execute before the prompt. The card and the review
    notebook carry the verdict, the allow runs the code, and the reviewer is asked once:
    the executor takes the review the ask stamp carries instead of reading the code again.
    """
    repo = safety_project_code_review
    before = _review_notebooks(repo)
    model, run = await _drive_reviewed(
        repo, scripted_model, [[execute_readwrite()], [read(SETPOINT)]]
    )
    execute = run.only_call(EXECUTE)
    facts = run.describe(execute)
    print(f"chat-gates (j) {facts} reviews={len(model.reviews)}")

    assert len(model.reviews) == 1, f"the code was reviewed {len(model.reviews)} times: {facts}"
    assert EXECUTE_CODE in model.reviews[0]
    assert run.was_carded(execute), f"the reviewed execute never reached the operator: {facts}"
    (card,) = [c for c in run.cards if c["tool_use_id"] == execute["tool_use_id"]]
    assert "Code review: pass" in (card["reason"] or ""), card["reason"]
    assert not run.results[execute["tool_use_id"]]["is_error"], facts
    assert "Code review" in _review_header(repo, before)
    assert _new_review(repo, before)[0] == "Ran after operator approval"


async def test_a_blocked_execute_is_denied_before_the_operator_is_asked(
    safety_project_code_review, scripted_model
):
    """(k): the reviewer blocks the execute. The approval hook denies it with the reasons,
    the operator is never shown it although they would approve anything, and it never runs.
    """
    repo = safety_project_code_review
    before = _review_notebooks(repo)
    blocked = ScriptedCall(
        EXECUTE,
        {"code": BLOCKED_CODE, "description": "Read the corrector.", "execution_mode": "readwrite"},
    )
    model, run = await _drive_reviewed(repo, scripted_model, [[blocked], [read(SETPOINT)]])
    execute = run.only_call(EXECUTE)
    facts = run.describe(execute)
    print(f"chat-gates (k) {facts} reviews={len(model.reviews)}")

    assert len(model.reviews) == 1, f"the code was reviewed {len(model.reviews)} times: {facts}"
    assert ("approval", "refused") in run.statuses(execute), facts
    assert not run.was_carded(execute), f"the operator was shown a blocked execute: {facts}"
    refused = run.results[execute["tool_use_id"]]
    assert refused["is_error"], facts
    assert refused["preview"].startswith(f"PreToolUse:{EXECUTE} hook error"), facts
    assert BLOCK_REASON in refused["preview"], facts
    assert _review_notebooks(repo) == before, "a blocked execute left a review notebook"


async def test_at_allowed_the_executor_reviews_what_no_prompt_saw(
    safety_project_code_review, scripted_model
):
    """(l): at ``allowed`` the hook asks nobody and reviews nothing, so the executor's review is
    the only one. A block there refuses the run in the tool, past the hook that let it through.
    """
    repo = safety_project_code_review
    blocked = ScriptedCall(
        EXECUTE,
        {"code": BLOCKED_CODE, "description": "Read the corrector.", "execution_mode": "readwrite"},
    )
    with switch_at(repo, "allowed"):
        model, run = await _drive_reviewed(repo, scripted_model, [[blocked], [read(SETPOINT)]])
    execute = run.only_call(EXECUTE)
    facts = run.describe(execute)
    print(f"chat-gates (l) {facts} reviews={len(model.reviews)}")

    assert len(model.reviews) == 1, f"the code was reviewed {len(model.reviews)} times: {facts}"
    assert not run.cards, f"a run at allowed reached the operator: {facts}"
    assert ("approval", "allow") in run.statuses(execute), facts
    refused = run.results[execute["tool_use_id"]]
    assert refused["is_error"], facts
    assert not refused["preview"].startswith("PreToolUse:"), facts
    assert BLOCK_REASON in refused["preview"], facts
