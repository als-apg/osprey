"""The chat's plan-mode round trip, proved on the Claude Code the project pins.

The chat child enters plan mode on its own: the model calls ``EnterPlanMode``, which
the CLI runs without an operator card. In plan mode the model writes its plan to the
plan file the CLI names, then calls ``ExitPlanMode``. That call is the card the
operator sees ("Ready to start on this plan?"), and it carries the plan the CLI reads
back from the file. These scripted runs drive that real launch
(:class:`~osprey.interfaces.web_terminal.operator_session.OperatorSession`) against a
real render, with the scripted model (:mod:`tests.e2e.scripted_model`) making the
calls, and prove:

(a) Approve: an ``ExitPlanMode`` with no plan on file is refused without a card. Once the
    plan is written, the card carries it. Approve leaves plan mode and the model is told so.
    A plan-file write after that is refused, because the hook allows the plan file only
    while the session is planning.
(b) Keep planning: the operator's decline reaches the model as an error and the session
    stays in plan mode. A file beside the plan file is refused, the revised plan is
    accepted, and the next ``ExitPlanMode`` puts a second card in front of the operator.

Every hook decision is read from the agent record's hook debug lines, and every card the
operator was shown is recorded, so a stray card or a landed write cannot pass unseen.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from tests.e2e import sdk_helpers
from tests.e2e.claude_code._chat_turn import Answer, calls_and_results, carries_result, drive_turn
from tests.e2e.scripted_model import ScriptedCall, ScriptedModel

pytestmark = [pytest.mark.model_free, pytest.mark.harness_benchmark]

TASK = "Plan the change, then carry it out once the plan is approved."
ENTER = "EnterPlanMode"
EXIT = "ExitPlanMode"
WRITE = "Write"

#: Placeholders in the scripted inputs. The CLI picks the plan file's name, and the
#: plans directory is per test, so neither can be written into a script up front.
PLAN_FILE = "<<plan-file>>"
PLANS_DIR = "<<plans-dir>>"

PLAN_TEXT = "# Plan\n\n1. Read the setpoint.\n2. Report it.\n"
PLAN_REVISED = "# Plan\n\n1. Read the setpoint.\n2. Report it twice.\n"

#: What the CLI tells the model when ExitPlanMode has no plan to approve.
NO_PLAN = "There is no plan to approve yet."
#: What the CLI tells the model when ExitPlanMode is called outside plan mode.
NOT_PLANNING = "You are not in plan mode."
#: What the operator's decline says back to the model (the chat's own default).
KEEP_PLANNING = "The operator wants to keep planning."
#: The header of the memory guard's refusal of a write.
WRITE_REFUSED = "WRITE DENIED"


def _write(path: str, content: str) -> ScriptedCall:
    return ScriptedCall(WRITE, {"file_path": path, "content": content})


def _fill(value: Any, plan_file: str | None, plans_dir: Path) -> Any:
    if not isinstance(value, str):
        return value
    if PLAN_FILE in value:
        assert plan_file is not None, "the plan file path has not reached the conversation yet"
        value = value.replace(PLAN_FILE, plan_file)
    return value.replace(PLANS_DIR, str(plans_dir))


def _fill_placeholders(model: ScriptedModel, plans_dir: Path) -> list[list[dict[str, Any]]]:
    """Fill the scripted inputs' placeholders, and keep the message list of every request.

    The CLI names the plan file in the plan-mode reminder it sends with the request after
    ``EnterPlanMode``. The path is read from there, so the write lands in the file the CLI
    will read back, not in a guess.
    """
    answer = model.answer
    conversations: list[list[dict[str, Any]]] = []
    plan_file = re.compile(re.escape(str(plans_dir)) + r"/[^/\s\"'`()<>]+\.md")

    def answer_with_paths(body, credential):
        messages = [m for m in body.get("messages") or [] if isinstance(m, dict)]
        conversations.append(messages)
        blocks, stop = answer(body, credential)
        found = plan_file.search(json.dumps(messages))
        path = found.group(0) if found else None
        for block in blocks:
            if block.get("type") == "tool_use":
                block["input"] = {k: _fill(v, path, plans_dir) for k, v in block["input"].items()}
        return blocks, stop

    model.answer = answer_with_paths
    return conversations


@dataclass
class PlanRun:
    """What one chat turn produced, from the frames the chat itself received."""

    calls: list[dict[str, Any]] = field(default_factory=list)
    results: dict[str, dict[str, Any]] = field(default_factory=dict)
    cards: list[dict[str, Any]] = field(default_factory=list)
    gates: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    conversations: list[list[dict[str, Any]]] = field(default_factory=list)

    def result(self, call: dict[str, Any]) -> dict[str, Any]:
        """The ``tool_result`` frame of *call*; a call with none fails the test here."""
        assert call["tool_use_id"] in self.results, f"no result for {self.describe(call)}"
        return self.results[call["tool_use_id"]]

    def cards_for(self, call: dict[str, Any]) -> list[dict[str, Any]]:
        return [card for card in self.cards if card["tool_use_id"] == call["tool_use_id"]]

    def statuses(self, call: dict[str, Any]) -> list[tuple[str, str]]:
        """Every ``(hook, status)`` logged for *call*, in the order the hooks ran."""
        return self.gates.get(call["tool_use_id"], [])

    def reached_model(self, call: dict[str, Any]) -> bool:
        """Whether the result of *call* went back to the model."""
        return any(
            carries_result(message, call["tool_use_id"])
            for messages in self.conversations
            for message in messages
        )

    def describe(self, call: dict[str, Any]) -> str:
        """The facts a failing assertion needs, in one line."""
        result = self.results.get(call["tool_use_id"], {})
        return (
            f"{call['tool_name_raw']} gates={self.statuses(call)} "
            f"cards={len(self.cards_for(call))} "
            f"result={result.get('is_error')}:{str(result.get('preview'))[:160]!r}"
        )


def plans_path() -> Path:
    """Where the CLI keeps plans: under the Claude Code state the harness owns for this test.

    That state is where a deployed web terminal keeps the chat's: the render is trusted
    in it, as the container entrypoint trusts a deployment's render, so the project's
    allow rules apply. Plans land in it, not in the developer's home.
    """
    assert sdk_helpers.harness_config_dir is not None
    return sdk_helpers.harness_config_dir / "plans"


async def drive_plan(repo, scripted_model, turns, *, operator) -> tuple[ScriptedModel, PlanRun]:
    """Run *turns* through the chat's own launch on *repo*, as *operator*."""
    model = scripted_model(turns)
    conversations = _fill_placeholders(model, plans_path())
    frames, cards, gates = await drive_turn(repo, TASK, operator, await_controls=False)
    model.assert_script_consumed()
    calls, results = calls_and_results(frames)
    run = PlanRun(
        calls=calls, results=results, cards=cards, gates=gates, conversations=conversations
    )
    return model, run


async def test_an_approved_plan_leaves_plan_mode_and_the_agent_continues(
    safety_project, scripted_model
):
    """(a): the plan is refused until written, carded with its text, approved, and left."""
    model, run = await drive_plan(
        safety_project,
        scripted_model,
        [
            [ScriptedCall(ENTER, {})],
            [ScriptedCall(EXIT, {})],
            [_write(PLAN_FILE, PLAN_TEXT)],
            [ScriptedCall(EXIT, {})],
            [_write(PLAN_FILE, PLAN_REVISED)],
            [ScriptedCall(EXIT, {})],
        ],
        operator=lambda card: Answer(allow=True),
    )
    assert {ENTER, EXIT} <= set(model.offered_tools), "the chat is not offered plan mode"
    assert len(run.calls) == 6, [c["tool_name_raw"] for c in run.calls]
    enter, early, planned, approve, late_write, late_exit = run.calls

    assert not run.result(enter)["is_error"], run.describe(enter)
    assert not run.cards_for(enter), f"entering plan mode asked the operator: {run.describe(enter)}"

    assert run.result(early)["is_error"], run.describe(early)
    assert NO_PLAN in run.result(early)["preview"], run.describe(early)
    assert not run.cards_for(early), f"an empty plan was carded: {run.describe(early)}"

    assert ("memory-guard", "allow") in run.statuses(planned), run.describe(planned)
    assert not run.result(planned)["is_error"], run.describe(planned)
    plan_file = next(plans_path().glob("*.md"))
    assert plan_file.read_text(encoding="utf-8") == PLAN_TEXT

    (card,) = run.cards_for(approve)
    assert card["kind"] == "plan", card
    assert card["input"]["plan"] == PLAN_TEXT, "the operator was not shown the plan"
    assert not run.result(approve)["is_error"], run.describe(approve)
    assert "approved" in run.result(approve)["preview"], run.describe(approve)
    assert run.reached_model(approve), "the model was not told the plan was approved"

    assert ("memory-guard", "deny") in run.statuses(late_write), run.describe(late_write)
    assert WRITE_REFUSED in run.result(late_write)["preview"], run.describe(late_write)
    assert plan_file.read_text(encoding="utf-8") == PLAN_TEXT, "a write after approval landed"

    assert run.result(late_exit)["is_error"], run.describe(late_exit)
    assert NOT_PLANNING in run.result(late_exit)["preview"], run.describe(late_exit)
    assert not run.cards_for(late_exit), f"a card after approval: {run.describe(late_exit)}"

    assert len(run.cards) == 1, f"stray operator cards: {[c['tool_name_raw'] for c in run.cards]}"


async def test_keep_planning_keeps_the_session_in_plan_mode(safety_project, scripted_model):
    """(b): the decline reaches the model, a neighbour is refused, and planning goes on."""
    answers = iter([Answer(allow=False), Answer(allow=True)])
    model, run = await drive_plan(
        safety_project,
        scripted_model,
        [
            [ScriptedCall(ENTER, {})],
            [_write(PLAN_FILE, PLAN_TEXT)],
            [ScriptedCall(EXIT, {})],
            [_write(f"{PLANS_DIR}/sub/extra.md", "not the plan")],
            [_write(PLAN_FILE, PLAN_REVISED)],
            [ScriptedCall(EXIT, {})],
            [ScriptedCall(EXIT, {})],
        ],
        operator=lambda card: next(answers),
    )
    assert len(run.calls) == 7, [c["tool_name_raw"] for c in run.calls]
    enter, planned, declined, neighbour, revised, approve, late_exit = run.calls

    assert not run.result(enter)["is_error"], run.describe(enter)
    assert not run.cards_for(enter), f"entering plan mode asked the operator: {run.describe(enter)}"

    (first_card,) = run.cards_for(declined)
    assert first_card["input"]["plan"] == PLAN_TEXT, "the operator was not shown the plan"
    assert run.result(declined)["is_error"], run.describe(declined)
    assert KEEP_PLANNING in run.result(declined)["preview"], run.describe(declined)
    assert run.reached_model(declined), "the model was not told the operator kept it planning"

    assert ("memory-guard", "deny") in run.statuses(neighbour), run.describe(neighbour)
    assert not (plans_path() / "sub").exists(), "a nested plan write landed"

    assert ("memory-guard", "allow") in run.statuses(revised), run.describe(revised)
    plan_file = next(plans_path().glob("*.md"))
    assert plan_file.read_text(encoding="utf-8") == PLAN_REVISED

    (second_card,) = run.cards_for(approve)
    assert second_card["input"]["plan"] == PLAN_REVISED, "the revised plan was not carded"
    assert not run.result(approve)["is_error"], run.describe(approve)

    assert run.result(late_exit)["is_error"], run.describe(late_exit)
    assert NOT_PLANNING in run.result(late_exit)["preview"], run.describe(late_exit)
    assert len(run.cards) == 2, f"stray operator cards: {[c['tool_name_raw'] for c in run.cards]}"
    assert planned["tool_name_raw"] == WRITE
