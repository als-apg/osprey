"""Unit tests for the scripted model endpoint in tests/e2e/scripted_model.py.

HTTP-level: a plain client posts Messages requests and reads what comes back.
No Claude Code, no render, no model.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from tests.e2e.scripted_model import FINAL_TEXT, ScriptedCall, ScriptedModel

pytestmark = pytest.mark.model_free

GLOB = ScriptedCall("Glob", {"pattern": "*.yml"})
READ = ScriptedCall("Read", {"file_path": "config.yml"})

# An ignored system proxy must not be asked to reach loopback.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


@pytest.fixture
def serve():
    """Start endpoints for one test and stop them afterwards."""
    started: list[ScriptedModel] = []

    def _serve(script):
        model = ScriptedModel(script)
        model.start()
        started.append(model)
        return model

    yield _serve
    for model in started:
        model.stop()


def _body(*, assistant_turns: int = 0, tools: bool = True, stream: bool = True) -> dict:
    """An agent request that has already received ``assistant_turns`` turns."""
    messages: list[dict] = [{"role": "user", "content": "go"}]
    for n in range(assistant_turns):
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": f"t{n}", "name": "Glob", "input": {}}],
            }
        )
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": f"t{n}", "content": f"result {n}"}
                ],
            }
        )
    body: dict = {"model": "m", "stream": stream, "messages": messages}
    if tools:
        body["tools"] = [{"name": "Glob"}, {"name": "Read"}]
    return body


def _post(
    model: ScriptedModel, body, *, path: str = "/v1/messages", token: str = "tok"
) -> tuple[int, str]:
    data = body if isinstance(body, bytes) else json.dumps(body).encode()
    request = urllib.request.Request(
        model.base_url + path,
        data=data,
        headers={"content-type": "application/json", "authorization": f"Bearer {token}"},
        method="POST",
    )
    try:
        with _OPENER.open(request, timeout=10) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()


def _streamed(sse: str) -> tuple[list[dict], str | None]:
    """The content blocks and stop reason an event stream adds up to."""
    blocks: dict[int, dict] = {}
    stop = None
    for line in sse.splitlines():
        if not line.startswith("data: "):
            continue
        event = json.loads(line[len("data: ") :])
        if event["type"] == "content_block_start":
            blocks[event["index"]] = dict(event["content_block"])
        elif event["type"] == "content_block_delta":
            delta = event["delta"]
            if delta["type"] == "input_json_delta":
                blocks[event["index"]]["input"] = json.loads(delta["partial_json"])
            else:
                blocks[event["index"]]["text"] += delta["text"]
        elif event["type"] == "message_delta":
            stop = event["delta"]["stop_reason"]
    return [blocks[i] for i in sorted(blocks)], stop


def test_first_request_streams_the_first_turn(serve):
    model = serve([[GLOB]])
    status, sse = _post(model, _body())
    blocks, stop = _streamed(sse)
    assert status == 200
    assert stop == "tool_use"
    assert [(b["type"], b["name"], b["input"]) for b in blocks] == [
        ("tool_use", "Glob", {"pattern": "*.yml"})
    ]


def test_a_turn_with_two_calls_streams_two_blocks_with_distinct_ids(serve):
    model = serve([[GLOB, READ]])
    blocks, _ = _streamed(_post(model, _body())[1])
    assert [b["name"] for b in blocks] == ["Glob", "Read"]
    assert len({b["id"] for b in blocks}) == 2


def test_the_turn_follows_the_conversation_not_the_request_count(serve):
    """A retried request gets the turn it asked for again."""
    model = serve([[GLOB], [READ]])
    first, _ = _streamed(_post(model, _body())[1])
    retried, _ = _streamed(_post(model, _body())[1])
    second, _ = _streamed(_post(model, _body(assistant_turns=1))[1])
    assert first == retried
    assert [b["name"] for b in second] == ["Read"]


def test_an_exhausted_script_ends_the_run_with_text(serve):
    model = serve([[GLOB]])
    blocks, stop = _streamed(_post(model, _body(assistant_turns=1))[1])
    assert stop == "end_turn"
    assert blocks == [{"type": "text", "text": FINAL_TEXT}]


def test_a_request_without_tools_gets_text_and_consumes_no_turn(serve):
    model = serve([[GLOB]])
    blocks, stop = _streamed(_post(model, _body(tools=False))[1])
    assert (stop, blocks[0]["type"]) == ("end_turn", "text")
    assert model.requests[0].turn_index is None
    after, _ = _streamed(_post(model, _body())[1])
    assert [b["name"] for b in after] == ["Glob"]


def test_a_non_streaming_request_gets_one_json_message(serve):
    model = serve([[GLOB]])
    status, text = _post(model, _body(stream=False))
    message = json.loads(text)
    assert status == 200
    assert message["stop_reason"] == "tool_use"
    assert message["content"][0]["name"] == "Glob"


def test_requests_are_recorded_with_offer_results_and_token(serve):
    model = serve([[GLOB], [READ]])
    _post(model, _body(), token="alpha")
    _post(model, _body(assistant_turns=1), token="alpha")
    assert model.offered_tools == ["Glob", "Read"]
    assert [r.turn_index for r in model.requests] == [0, 1]
    assert [r.bearer_token for r in model.requests] == ["alpha", "alpha"]
    assert model.requests[1].tool_results == [
        {"type": "tool_result", "tool_use_id": "t0", "content": "result 0"}
    ]


def test_offered_tools_fails_when_the_agent_never_asked(serve):
    model = serve([[GLOB]])
    with pytest.raises(AssertionError, match="never asked the scripted model"):
        _ = model.offered_tools


def test_an_undelivered_turn_fails_the_consumption_check(serve):
    model = serve([[GLOB], [READ]])
    _post(model, _body())
    with pytest.raises(AssertionError, match="turn 1"):
        model.assert_script_consumed()
    _post(model, _body(assistant_turns=1))
    with pytest.raises(AssertionError, match="never returned"):
        model.assert_script_consumed()
    _post(model, _body(assistant_turns=2))
    model.assert_script_consumed()


def test_token_counting_is_answered(serve):
    model = serve([[GLOB]])
    status, text = _post(model, _body(), path="/v1/messages/count_tokens")
    assert status == 200
    assert json.loads(text)["input_tokens"] > 0
    assert model.requests == []


def test_an_unknown_route_is_refused_and_recorded(serve):
    model = serve([[GLOB]])
    status, _ = _post(model, {}, path="/v1/complete")
    assert status == 404
    assert model.unexpected == ["POST /v1/complete"]


def test_a_malformed_body_is_a_400_not_a_turn(serve):
    model = serve([[GLOB]])
    status, _ = _post(model, b"{not json")
    assert status == 400
    assert model.requests == []


def test_an_empty_turn_is_rejected_when_the_script_is_built():
    with pytest.raises(ValueError, match="turn 1"):
        ScriptedModel([[GLOB], []])


def test_an_empty_script_ends_at_once(serve):
    model = serve([])
    blocks, stop = _streamed(_post(model, _body())[1])
    assert (stop, blocks[0]["text"]) == ("end_turn", FINAL_TEXT)
    model.assert_script_consumed()


def test_the_fixture_points_the_run_at_the_endpoint_with_the_dummy_token(
    monkeypatch, scripted_model
):
    import os

    from tests.e2e.scripted_model import SCRIPTED_AUTH_TOKEN

    monkeypatch.setenv("ALS_APG_API_KEY", "a-real-looking-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "another-real-looking-key")
    model = scripted_model([[GLOB]])
    assert os.environ["OSPREY_E2E_PROXY_BASE_URL"] == model.base_url
    assert os.environ["ALS_APG_API_KEY"] == SCRIPTED_AUTH_TOKEN
    assert "ANTHROPIC_API_KEY" not in os.environ


def test_the_fixture_overrides_a_session_proxy(monkeypatch, scripted_model):
    import os

    monkeypatch.setenv("OSPREY_E2E_PROXY_BASE_URL", "http://127.0.0.1:9")
    model = scripted_model([[GLOB]])
    assert os.environ["OSPREY_E2E_PROXY_BASE_URL"] == model.base_url
