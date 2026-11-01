"""A session key resolves to the conversation its live chat child is on (#1483).

``/clear`` sent as a message in the Simple view moves the chat child to a new
transcript without a restart. The expert view's child reports such a move
through the agent-turn hook; the chat child runs with none, so until now the
key kept pointing at the old transcript: a replay served the old turns and
the next child resumed them. The chat route now records the id the child
reports after every turn, and the stream carries the boundary.

Driven through a live server with the SDK faked at the operator-session seam,
so the assertion is at the server boundary: the map after the turn, and what
the next child is started with.
"""

from __future__ import annotations

import json
import uuid

import httpx
import pytest

from osprey.interfaces.web_terminal import transcript_map
from tests.interfaces.web_terminal._chat_server import PLANS, live_chat_server, seed_transcript

pytestmark = pytest.mark.timeout(60)

KEY = str(uuid.uuid4())
MOVED_TO = str(uuid.uuid4())


def _turn(base_url: str, prompt: str) -> list[dict]:
    response = httpx.post(
        f"{base_url}/api/chat",
        params={"stream": "false"},
        json={"prompt": prompt, "chat_id": KEY},
        timeout=30,
    )
    assert response.status_code == 200, response.text
    return response.json()["events"]


def test_a_cleared_chat_moves_its_key_and_the_next_child_resumes_the_new_transcript(tmp_path):
    with live_chat_server(tmp_path) as (base_url, app):
        PLANS["hello"] = [("text", "hi"), ("result", False, KEY)]
        PLANS["/clear"] = [("result", False, MOVED_TO)]
        PLANS["after"] = [("text", "fresh"), ("result", False, MOVED_TO)]

        first = _turn(base_url, "hello")
        assert transcript_map.get(app, KEY) == KEY, "a child on its own key moves nothing"
        assert [e["type"] for e in first if e["type"] == "session_reset"] == ["session_reset"], (
            "a fresh conversation carries the route's own marker, once"
        )

        cleared = _turn(base_url, "/clear")
        # The turn's own ``turn_start`` aside, the move is the marker, then the
        # result that named the new transcript -- and no text.
        assert [e["type"] for e in cleared if e["type"] != "turn_start"] == [
            "session_reset",
            "result",
        ], cleared
        assert transcript_map.get(app, KEY) == MOVED_TO
        store = transcript_map.store_path()
        assert store is not None and json.loads(store.read_text(encoding="utf-8")) == {
            KEY: MOVED_TO
        }

        # A turn that stays on the new transcript is not a move.
        after = _turn(base_url, "after")
        assert "session_reset" not in [e["type"] for e in after]

        # The next child under the key continues the conversation the operator
        # is in, not the one before /clear. A child resumes only a transcript
        # that is on disk; a real child left one under the new id, so one is
        # seeded here (and one under the key, which is what used to be resumed).
        seed_transcript(tmp_path / "project", KEY, [("user", "hello"), ("assistant", "hi")])
        seed_transcript(tmp_path / "project", MOVED_TO, [("user", "after"), ("assistant", "fresh")])
        assert httpx.post(f"{base_url}/__test__/evict-all", timeout=10).status_code == 200
        _turn(base_url, "after")
        state = httpx.get(f"{base_url}/__test__/chat-state", timeout=10).json()
        assert [o["resume"] for o in state["options"]] == [MOVED_TO]
