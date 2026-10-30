"""One turn through the Simple view's own launch, for the scripted chat proofs.

The approval-gates, plan-mode and attachment-read proofs each put a scripted model
behind the real chat child (:class:`~osprey.interfaces.web_terminal.operator_session.OperatorSession`)
and read what the chat itself received: the frames, the approval cards the operator
was shown, and each hook's own decision from the agent record's hook debug lines. This module is
that harness, written once.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from osprey.agent_runner.primitives import await_mcp_ready, expected_mcp_servers
from osprey.interfaces.web_terminal.operator_session import OperatorSession
from tests.e2e import sdk_helpers
from tests.e2e.scripted_model import ScriptedModel
from tests.e2e.sdk_helpers import render_dir

TASK = "Carry out the scripted steps."

#: The ``source`` every hook debug line carries on the record.
HOOK_SOURCE = "hook"


@dataclass(frozen=True)
class Answer:
    """What the operator does with one approval card."""

    allow: bool
    message: str | None = None
    #: Allow the tool for the rest of the session too, where the card offers it.
    for_session: bool = False


class Gate(tuple):
    """One hook decision for one call: equal to ``(hook, status)``, with its ``detail``.

    It compares, hashes and unpacks as the ``(hook, status)`` pair, so a proof that
    asks ``("approval", "ask") in statuses`` reads it unchanged; :attr:`detail` is the
    free-form text the hook logged beside the decision (``position=allowed tool=...``),
    ``""`` when it logged none.
    """

    detail: str

    def __new__(cls, hook: str, status: str, detail: str = "") -> Gate:
        gate = super().__new__(cls, (hook, status))
        gate.detail = detail
        return gate

    @property
    def hook(self) -> str:
        return self[0]

    @property
    def status(self) -> str:
        return self[1]

    def __repr__(self) -> str:
        return f"({self[0]!r}, {self[1]!r}, detail={self.detail!r})"


def hook_debug_since(repo: Path, start: datetime) -> list[dict[str, Any]]:
    """Every hook debug line the agent record holds from *start* on, oldest first.

    Read from the record through :class:`~osprey.agent_record.client.RecordClient`
    with the host login, held by :func:`~osprey.agent_record.host.ensure_host_record`
    (which starts the host-local record when none runs), and drained, so a line a
    hook wrote just before the read is counted. A read the record could not answer
    whole fails the proof rather than report missing decisions as none.
    """
    from osprey.agent_record.client import RecordClient
    from osprey.agent_record.contract import KIND_DEBUG, Drain
    from osprey.agent_record.host import ensure_host_record

    config = yaml.safe_load((render_dir(repo) / "config.yml").read_text(encoding="utf-8"))
    with ensure_host_record(repo, config) as record:
        client = RecordClient(record.url, record.login, record.token)
        answer = client.entries(KIND_DEBUG, since=start, source=HOOK_SOURCE, wait=Drain())
    assert answer.available, f"the agent record is unavailable: {answer.unavailable}"
    assert answer.complete, f"the hook debug read is incomplete: {answer.reason}"
    assert not answer.more, "the hook debug lines of one turn overflowed a page"
    bodies = [
        item["body"]
        for item in answer.data or []
        if isinstance(item, dict) and isinstance(item.get("body"), dict)
    ]
    bodies.reverse()  # the record answers newest first
    return bodies


def gates_since(repo: Path, start: datetime) -> dict[str, list[Gate]]:
    """Every hook decision the agent record holds from *start* on.

    The hooks log each decision they make, denials included, when hook debugging is on
    (the child runs with it on), as a hook ``debug`` entry on the record. The decisions
    are keyed by tool-use id, as :class:`Gate` -- ``(hook, status)`` with the line's
    ``detail`` -- in the order the hooks ran.
    """
    gates: dict[str, list[Gate]] = {}
    for record in hook_debug_since(repo, start):
        gates.setdefault(str(record.get("tool_use_id")), []).append(
            Gate(
                str(record.get("hook")), str(record.get("status")), str(record.get("detail") or "")
            )
        )
    return gates


def carries_result(message: dict[str, Any], tool_use_id: str) -> bool:
    """Whether *message* is a user turn that carries the result of *tool_use_id*."""
    content = message.get("content")
    if message.get("role") != "user" or not isinstance(content, list):
        return False
    return any(
        isinstance(block, dict)
        and block.get("type") == "tool_result"
        and block.get("tool_use_id") == tool_use_id
        for block in content
    )


def record_conversations(model: ScriptedModel) -> list[list[dict[str, Any]]]:
    """Keep the message list of every request *model* receives, in order.

    The endpoint records only the tool results a request carries. Context the harness
    adds beside them (the approval note) is in the conversation, so it is kept whole,
    as the endpoint received it.
    """
    conversations: list[list[dict[str, Any]]] = []
    answer = model.answer

    def keep_and_answer(body: dict[str, Any], credential: str | None):
        conversations.append([m for m in body.get("messages") or [] if isinstance(m, dict)])
        return answer(body, credential)

    model.answer = keep_and_answer
    return conversations


def calls_and_results(
    frames: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Every ``tool_use`` frame, in order, and the ``tool_result`` frame of each by tool-use id."""
    calls = [frame for frame in frames if frame.get("type") == "tool_use"]
    results = {
        frame["tool_use_id"]: frame for frame in frames if frame.get("type") == "tool_result"
    }
    return calls, results


async def drive_turn(
    repo: Path,
    prompt: str,
    operator: Callable[[dict[str, Any]], Answer],
    *,
    env: dict[str, str] | None = None,
    await_controls: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[Gate]]]:
    """Run one turn of *prompt* through the chat's own launch on *repo*, as *operator*.

    The child is launched by :class:`OperatorSession` with the render as its working
    directory, the e2e environment (the scripted credential and endpoint), hook
    debugging on, and *env* on top. The render is trusted first, as a deployment's
    entrypoint trusts it, so the project's allow rules apply. The operator answers
    each approval card the chat raises.

    The chat child does not wait for its MCP servers before the first prompt, so with
    *await_controls* this run does: a call the scripted model makes before the controls
    server is connected would be an unknown tool, and the scenario would not be the one
    named.

    Returns ``(frames, cards, gates)``: every non-heartbeat frame the chat produced, the
    approval cards among them, and each hook's decisions for this turn (:func:`gates_since`).
    """
    render = render_dir(repo)
    sdk_helpers.trust_render(repo)
    child_env = {**sdk_helpers.sdk_env(render), "OSPREY_HOOK_DEBUG": "1", **(env or {})}
    start = datetime.now(UTC)
    session = OperatorSession(cwd=str(render), env=child_env)
    frames: list[dict[str, Any]] = []
    cards: list[dict[str, Any]] = []
    try:
        await session.start()
        if await_controls:
            # The client is the session's own (a private attribute, as the session tests read).
            servers = await await_mcp_ready(session._agent._client, expected_mcp_servers(render))
            status = {server["name"]: server.get("status") for server in servers}
            assert status.get("controls") == "connected", f"controls did not connect: {status}"
        token = session.acquire_turn()
        async for frame in session.run_turn(prompt, token, timeout_s=120.0):
            if frame.get("type") == "heartbeat":
                continue
            frames.append(frame)
            if frame.get("type") == "permission_request":
                cards.append(frame)
                answer = operator(frame)
                resolved = await session.resolve_permission(
                    frame["request_id"],
                    allow=answer.allow,
                    message=answer.message,
                    for_session=answer.for_session,
                )
                assert resolved, "the card was not waiting when the operator answered"
    finally:
        await session.stop()
    return frames, cards, gates_since(repo, start)
