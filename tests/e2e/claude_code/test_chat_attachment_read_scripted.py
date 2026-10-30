"""The chat's agent can read a file the operator attached, on the Claude Code the project pins.

The composer stores an attachment through ``POST /api/chat/{chat_id}/attachments``, and
the message it sends carries the stored path in the block the agent is told to read with
its Read tool. The render allows the agent to read exactly the uploads directory, by its
absolute path (``Read(//<agent-data root>/uploads/**)``): the agent runs in the render,
and the uploads sit under the agent-data root beside it, where a rule relative to the
working directory would not reach. This run stores a real PNG through that route into
the render's agent-data root, sends the prompt the composer builds (the typed text, then
the block), and has the scripted model open the path the route returned. It proves, on
the CLI the project pins:

(a) the prompt the agent receives carries the block, with the stored path in it;
(b) the Read of the stored PNG succeeds, and the operator is shown no card for it;
(c) no hook refuses the call; each hook's decision is read from the record's hook debug lines;
(d) the rule is no wider than the uploads directory: a file beside it, under the same
    agent-data root, still raises the CLI's own path card, and a decline refuses it.

The workspace is trusted before the child starts, as the container entrypoint trusts a
deployed render, so the project's allow rules apply. The run does not widen any rule.

The scripted model spends no tokens. The child's launch, the recorded requests and the
hook debug read are the scripted chat proofs' shared harness (``_chat_turn``).
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.routes import chat_attachments
from osprey.interfaces.web_terminal.routes.chat_attachments import (
    BATCH_HEADER,
    NAME_HEADER,
    router,
)
from osprey_connectors.workspace import resolve_shared_data_root
from tests.e2e.claude_code._chat_turn import Answer, drive_turn, record_conversations
from tests.e2e.scripted_model import ScriptedCall
from tests.e2e.sdk_helpers import render_dir

pytestmark = [pytest.mark.model_free, pytest.mark.harness_benchmark]

CHAT = "aaaaaaaa-1111-2222-3333-444444444444"
BATCH = "batch-0001-aaaa"
#: A valid one-pixel PNG, so the file is an image the CLI can open.
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)
#: What the operator types. The composer appends the block to it and sends the whole.
TYPED = "What does this panel show?"
#: The block the composer appends, in the format chat-attach.js writes.
BLOCK_HEAD = "\n\nAttached files (read them with the Read tool):\n"
#: The reason the CLI gives for a path outside its allowed working directories.
PATH_REASON = "Path is outside allowed working directories"


def _agent_data_root(repo: Path) -> Path:
    """The agent-data root the render's own config resolves to."""
    config = yaml.safe_load((render_dir(repo) / "config.yml").read_text(encoding="utf-8"))
    return Path(resolve_shared_data_root(config))


def _store_png(root: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Store the PNG through the attachment route, as the composer does, into *root*."""
    monkeypatch.setattr(chat_attachments, "resolve_agent_data_root", lambda app=None: str(root))
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        response = client.post(
            f"/api/chat/{CHAT}/attachments",
            content=PNG,
            headers={NAME_HEADER: "panel.png", BATCH_HEADER: BATCH},
        )
    assert response.status_code == 201, response.text
    return response.json()


def _prompt(stored: dict[str, object]) -> str:
    """The prompt the composer sends for one attached file: the typed text, then the block."""
    return f"{TYPED}{BLOCK_HEAD}- {stored['path']} ({stored['type']}, {stored['size']} B)"


async def _chat_turn(repo: Path, scripted_model, prompt: str, turns) -> dict:
    """Send *prompt* through the chat's own launch on *repo*, with *turns* scripted.

    The operator declines every permission card, so an unexpected ask fails the run
    where it is raised.
    """
    model = scripted_model(turns)
    conversations = record_conversations(model)
    frames, cards, gates = await drive_turn(
        repo, prompt, lambda _card: Answer(allow=False, message="scripted: no card is expected")
    )
    model.assert_script_consumed()
    return {"frames": frames, "cards": cards, "gates": gates, "conversations": conversations}


def _the_read(facts: dict) -> tuple[dict, dict]:
    """The one Read call of the turn and its result."""
    (call,) = [f for f in facts["frames"] if f.get("type") == "tool_use"]
    assert call["tool_name_raw"] == "Read"
    result = next(
        f
        for f in facts["frames"]
        if f.get("type") == "tool_result" and f.get("tool_use_id") == call["tool_use_id"]
    )
    return call, result


async def test_the_prompt_carries_the_block_and_the_agent_reads_the_stored_png(
    safety_project, scripted_model, monkeypatch
):
    """(a), (b) and (c): the block reaches the agent, its Read succeeds with no card, and
    no hook refuses it."""
    root = _agent_data_root(safety_project)
    stored = _store_png(root, monkeypatch)
    stored_path = str(stored["path"])
    assert Path(stored_path).is_relative_to(root / "uploads" / CHAT)

    facts = await _chat_turn(
        safety_project,
        scripted_model,
        _prompt(stored),
        [[ScriptedCall("Read", {"file_path": stored_path})]],
    )

    # (a) The request the endpoint received carries the typed text, the block and the path.
    prompt_seen = json.dumps(facts["conversations"][0])
    assert TYPED in prompt_seen, "the typed text did not reach the agent"
    assert "Attached files (read them with the Read tool):" in prompt_seen
    assert stored_path in prompt_seen, "the agent was not given the stored path"

    # (b) The rendered rule covers the upload: no card, and the read succeeds.
    assert not facts["cards"], f"the upload's Read raised a card: {facts['cards']}"
    call, result = _the_read(facts)
    assert call["input"]["file_path"] == stored_path
    assert not result.get("is_error"), f"the Read of the stored PNG failed: {result}"

    # (c) No hook refused the call. The hook debug lines record every decision, denials included.
    statuses = facts["gates"].get(call["tool_use_id"], [])
    print(f"attachment read: gates={statuses}")
    assert not [status for _hook, status in statuses if status == "deny"], statuses


async def test_a_file_beside_the_uploads_directory_still_raises_the_path_card(
    safety_project, scripted_model
):
    """(d): the rule is the uploads directory and nothing wider.

    A PNG written beside ``uploads/``, under the same agent-data root, is outside the
    rule and outside the working directory, so the CLI raises its own path card for the
    read; the operator declines it, and the read is refused.
    """
    root = _agent_data_root(safety_project)
    neighbour = root / "beside-the-uploads.png"
    neighbour.write_bytes(PNG)
    stored = {"path": str(neighbour), "type": "image/png", "size": len(PNG)}

    facts = await _chat_turn(
        safety_project,
        scripted_model,
        _prompt(stored),
        [[ScriptedCall("Read", {"file_path": str(neighbour)})]],
    )

    (card,) = facts["cards"]
    assert card["tool_name_raw"] == "Read"
    assert card["input"]["file_path"] == str(neighbour)
    assert card["reason"] == PATH_REASON, card
    _call, result = _the_read(facts)
    assert result.get("is_error"), f"a read beside the uploads directory ran: {result}"
