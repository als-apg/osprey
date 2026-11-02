"""The openai route: settings, key rule, chat request shape, probabilities and errors.

Every test runs offline on an ``httpx.MockTransport`` that plays an
OpenAI-compatible (vLLM) chat server. All data is synthetic. The one live check
runs only when ``OSPREY_JEV_OPENAI_LIVE_URL`` names a server.
"""

from __future__ import annotations

import json
import math
import os
import threading
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from osprey.models.jev.client import JevClient
from osprey.models.jev.config import (
    JevSettings,
    chat_endpoint,
    is_local_address,
    resolve_route,
)
from osprey.models.jev.errors import (
    DecisionAuthError,
    DecisionRequestError,
    DecisionUnavailableError,
)
from osprey.models.jev.models import ChoiceAnswer, NoulAnswer, ScoreAnswer
from osprey.models.jev.openai_route import SYSTEM_PROMPT, TOP_LOGPROBS

BASE = "http://127.0.0.1:18001"
MODEL = "synthetic-served-model"
STATE = "synthetic state: the widget is warm"
SYNTHETIC_KEY = "synthetic-key-not-real"

CHOICE = {
    "type": "choice",
    "instructions": "Which synthetic kind?",
    "criteria": {"k001": "alpha", "k002": "beta", "none_fits": None},
}
NOUL = {"type": "noul", "instructions": "Is the synthetic widget warm?"}
SCORE = {
    "type": "score",
    "instructions": "How warm is the synthetic widget?",
    "criteria": ["cold", "tepid", "hot"],
}


def _config(**openai: Any) -> dict[str, Any]:
    block = {"base_url": BASE, "model": MODEL, **openai}
    return {"api": {"jev": {"route": "openai", "routes": {"openai": block}}}}


def _completion(
    top: dict[str, float], *, content: str | None = None, usage: tuple[int, int] = (100, 1)
) -> dict[str, Any]:
    """A vLLM chat reply whose first token's top_logprobs carry ``top`` (token -> probability)."""
    entries = [
        {"token": token, "logprob": math.log(p) if p > 0 else -9999.0} for token, p in top.items()
    ]
    picked = content if content is not None else max(top, key=top.__getitem__)
    return {
        "id": "chatcmpl-synthetic",
        "object": "chat.completion",
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": picked},
                "logprobs": {"content": [{"token": picked, "top_logprobs": entries}]},
                "finish_reason": "length",
            }
        ],
        "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1]},
    }


class _Server:
    """Plays the chat server: replies come from a list, one per request, then ``default``.

    Questions of one decide run concurrently, so a multi-question test answers
    through ``default``, keyed on the request body, never through the list.
    """

    def __init__(self, *replies: Any, default: Callable[[dict], Any] | None = None) -> None:
        self.replies = list(replies)
        self.default = default
        self.requests: list[httpx.Request] = []
        self.bodies: list[dict[str, Any]] = []
        self.now = 0.0
        self._lock = threading.Lock()

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds

    def handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        with self._lock:
            self.requests.append(request)
            self.bodies.append(body)
            reply = self.replies.pop(0) if self.replies else self.default(body)  # type: ignore[misc]
        if isinstance(reply, BaseException) or (
            isinstance(reply, type) and issubclass(reply, BaseException)
        ):
            raise reply if isinstance(reply, BaseException) else reply("synthetic", request=request)
        if isinstance(reply, httpx.Response):
            return reply
        self.now += 0.01
        return httpx.Response(200, json=reply)


def _by_first_label(replies: dict[str, Any]) -> Callable[[dict], Any]:
    """A ``default`` that answers by the request's first label (``0`` score, ``A`` noul)."""
    return lambda body: replies[body["structured_outputs"]["choice"][0]]


def _client(server: _Server, config: dict[str, Any] | None = None) -> JevClient:
    config = config if config is not None else _config()
    return JevClient(
        JevSettings.from_config(config),
        config,
        transport=httpx.MockTransport(server.handle),
        clock=server.clock,
        sleep=server.sleep,
    )


# ---------------------------------------------------------------------------
# Settings and key rule
# ---------------------------------------------------------------------------


def test_openai_route_defaults_name_no_server() -> None:
    block = JevSettings().routes.openai
    assert (block.model, block.base_url, block.api_key) == (None, None, None)
    assert block.max_request_tokens == 30_000
    assert block.max_state_plus_question_tokens == 30_000


@pytest.mark.parametrize("missing", ["model", "base_url"])
def test_openai_route_needs_model_and_base_url(missing: str) -> None:
    block = {"model": MODEL, "base_url": BASE}
    del block[missing]
    with pytest.raises(ValidationError, match=f"api.jev.routes.openai.{missing}"):
        JevSettings.model_validate({"route": "openai", "routes": {"openai": block}})


def test_openai_route_refuses_floating_aliases() -> None:
    with pytest.raises(ValidationError, match="alias"):
        JevSettings.model_validate({"routes": {"openai": {"model": "jev-latest"}}})


def test_zero_price_is_accepted() -> None:
    config = _config()
    config["api"]["jev"]["input_cost_per_million_tokens"] = 0
    assert JevSettings.from_config(config).input_cost_per_million_tokens == 0


@pytest.mark.parametrize(
    ("base", "local"),
    [
        ("http://127.0.0.1:8000", True),
        ("http://localhost:8000/v1", True),
        ("http://[::1]:8000", True),
        ("http://10.1.2.3:8000", True),
        ("http://192.168.0.5", True),
        ("https://api.example.invalid", False),
        ("http://8.8.8.8", False),
        ("http://rubin.example.invalid:8000", False),
    ],
)
def test_is_local_address(base: str, local: bool) -> None:
    assert is_local_address(base) is local


def test_key_is_optional_on_a_local_address(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", SYNTHETIC_KEY)
    settings = JevSettings.from_config(_config())
    route = resolve_route(settings, _config())
    # No environment fallback: an ambient OpenAI key is never sent to the server.
    assert route.api_key is None
    assert route.endpoint == f"{BASE}/v1/chat/completions"


def test_key_is_required_off_a_local_address() -> None:
    config = _config(base_url="https://api.example.invalid/v1")
    with pytest.raises(DecisionAuthError) as info:
        resolve_route(JevSettings.from_config(config), config)
    assert info.value.error_type == "no_key"
    assert "api.jev.routes.openai.api_key" in info.value.message


def test_key_resolves_from_config_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYNTHETIC_VLLM_KEY", SYNTHETIC_KEY)
    config = _config(base_url="https://api.example.invalid", api_key="${SYNTHETIC_VLLM_KEY}")
    route = resolve_route(JevSettings.from_config(config), config)
    assert route.api_key == SYNTHETIC_KEY
    assert SYNTHETIC_KEY not in repr(route)


def test_unresolved_base_url_is_refused_before_http(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SYNTHETIC_UNSET_BASE", raising=False)
    config = _config(base_url="${SYNTHETIC_UNSET_BASE}")
    server = _Server()
    with pytest.raises(DecisionRequestError, match="base_url"):
        _client(server, config).decide(STATE, {"q": CHOICE}, budget="batch")
    assert server.requests == []


@pytest.mark.parametrize("base", ["127.0.0.1:8000", "ftp://127.0.0.1"])
def test_base_url_needs_an_http_scheme(base: str) -> None:
    with pytest.raises(ValidationError, match="http"):
        JevSettings.from_config(_config(base_url=base))


@pytest.mark.parametrize("base", [BASE, f"{BASE}/v1", f"{BASE}/v1/"])
def test_chat_endpoint_accepts_both_base_spellings(base: str) -> None:
    assert chat_endpoint(base) == f"{BASE}/v1/chat/completions"


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------


def test_choice_request_shape() -> None:
    server = _Server(_completion({"A": 0.9, "B": 0.1}))
    _client(server).decide(STATE, {"q": CHOICE}, budget="batch")
    (request,) = server.requests
    assert str(request.url) == f"{BASE}/v1/chat/completions"
    assert "authorization" not in request.headers
    body = server.bodies[0]
    assert body["model"] == MODEL
    assert body["max_tokens"] == 1
    assert body["temperature"] == 0
    assert body["logprobs"] is True
    assert body["top_logprobs"] == TOP_LOGPROBS
    assert body["structured_outputs"] == {"choice": ["A", "B", "C"]}
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    system, user = body["messages"]
    assert system == {"role": "system", "content": SYSTEM_PROMPT}
    assert user["role"] == "user"
    assert STATE in user["content"]
    assert "Which synthetic kind?" in user["content"]
    assert "A. k001: alpha\nB. k002: beta\nC. none_fits\n" in user["content"]


def test_configured_key_is_sent_as_bearer() -> None:
    server = _Server(_completion({"A": 1.0}))
    _client(server, _config(api_key=SYNTHETIC_KEY)).decide(STATE, {"q": CHOICE}, budget="batch")
    assert server.requests[0].headers["authorization"] == f"Bearer {SYNTHETIC_KEY}"


def test_score_uses_digit_labels_and_noul_two_letters() -> None:
    server = _Server(
        default=_by_first_label({"0": _completion({"2": 1.0}), "A": _completion({"A": 1.0})})
    )
    _client(server).decide(STATE, {"s": SCORE, "n": NOUL}, budget="batch")
    (score_body,) = [b for b in server.bodies if b["structured_outputs"]["choice"][0] == "0"]
    (noul_body,) = [b for b in server.bodies if b["structured_outputs"]["choice"][0] == "A"]
    assert score_body["structured_outputs"] == {"choice": ["0", "1", "2"]}
    assert "0. cold\n1. tepid\n2. hot" in score_body["messages"][1]["content"]
    assert noul_body["structured_outputs"] == {"choice": ["A", "B"]}
    assert "A. true: yes\nB. false: no" in noul_body["messages"][1]["content"]


def test_more_than_26_options_refused_before_http() -> None:
    criteria = {f"k{i:03d}": f"option {i}" for i in range(27)}
    server = _Server()
    with pytest.raises(DecisionRequestError) as info:
        _client(server).decide(
            STATE,
            {"q": {"type": "choice", "instructions": "pick", "criteria": criteria}},
            budget="batch",
        )
    assert info.value.error_type == "invalid_request"
    assert server.requests == []


# ---------------------------------------------------------------------------
# Probabilities
# ---------------------------------------------------------------------------


def test_logprobs_become_probabilities_over_the_options() -> None:
    # Mass on a non-label token is ignored; the label mass is renormalised to 1.
    server = _Server(_completion({"A": 0.6, "B": 0.2, " C": 0.1, "#": 0.1}))
    result = _client(server).decide(STATE, {"q": CHOICE}, budget="batch")
    answer = result.answers["q"]
    assert isinstance(answer, ChoiceAnswer)
    assert answer.choice == "k001"
    assert answer.probabilities == {"k001": 0.667, "k002": 0.222, "none_fits": 0.111}
    assert sum(answer.probabilities.values()) == pytest.approx(1.0, abs=0.002)


def test_masked_options_drop_below_the_floor() -> None:
    server = _Server(_completion({"B": 1.0, "A": 0.0, "C": 0.0}))
    answer = _client(server).decide(STATE, {"q": CHOICE}, budget="batch").answers["q"]
    assert isinstance(answer, ChoiceAnswer)
    assert answer.choice == "k002"
    assert answer.probabilities == {"k002": 1.0}


def test_no_logprobs_is_refused_not_read_as_certain() -> None:
    # The constrained pick alone is no probability: reading it as 1.0 would fail open.
    reply = _completion({"A": 1.0}, content="B")
    reply["choices"][0]["logprobs"] = None
    with pytest.raises(DecisionRequestError) as info:
        _client(_Server(reply)).decide(STATE, {"q": CHOICE}, budget="batch")
    assert info.value.error_type == "malformed_response"


@pytest.mark.parametrize(
    "top",
    [
        {"A": 0.0067, "#": 0.99},  # one visible label with a sliver of mass
        {"Hmm": 0.99},  # no label at all in the top logprobs
        {"A": 0.3, "B": 0.1, "Hmm": 0.6},  # the labels together below MIN_LABEL_MASS
    ],
)
def test_low_label_mass_is_refused(top: dict[str, float]) -> None:
    with pytest.raises(DecisionRequestError) as info:
        _client(_Server(_completion(top, content="A"))).decide(STATE, {"n": NOUL}, budget="batch")
    assert info.value.error_type == "malformed_response"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 1000.0])
def test_non_finite_or_positive_logprobs_are_ignored(bad: float) -> None:
    reply = _completion({"A": 0.8, "B": 0.2})
    reply["choices"][0]["logprobs"]["content"][0]["top_logprobs"].append(
        {"token": "B", "logprob": bad}
    )
    server = _Server(httpx.Response(200, content=json.dumps(reply).encode()))
    answer = _client(server).decide(STATE, {"n": NOUL}, budget="batch").answers["n"]
    assert isinstance(answer, NoulAnswer)
    assert answer.noul == 0.8


def test_noul_and_score_answers() -> None:
    server = _Server(
        default=_by_first_label(
            {
                "0": _completion({"0": 0.1, "1": 0.2, "2": 0.7}),
                "A": _completion({"A": 0.25, "B": 0.75}),
            }
        )
    )
    answers = _client(server).decide(STATE, {"s": SCORE, "n": NOUL}, budget="batch").answers
    score, noul = answers["s"], answers["n"]
    assert isinstance(score, ScoreAnswer)
    assert score.score == pytest.approx(1.6)
    assert score.legend == {"0": "cold", "1": "tepid", "2": "hot"}
    assert isinstance(noul, NoulAnswer)
    assert noul.noul == 0.25


def test_one_call_per_question_with_usage_and_zero_cost() -> None:
    config = _config()
    config["api"]["jev"]["input_cost_per_million_tokens"] = 0
    server = _Server(
        default=_by_first_label(
            {
                "0": _completion({"2": 1.0}, usage=(1000, 1)),
                "A": _completion({"A": 1.0}, usage=(500, 1)),
            }
        )
    )
    result = _client(server, config).decide(STATE, {"s": SCORE, "n": NOUL}, budget="batch")
    assert sorted(c.input_tokens for c in result.calls) == [500, 1000]
    for call in result.calls:
        assert call.output_tokens == 1
        assert call.cost_usd == 0
        assert call.cost_source == "computed"
        assert call.billed_cost_usd is None
        assert call.served_model == MODEL
        assert call.request_id == "chatcmpl-synthetic"
    assert result.route == "openai"


# ---------------------------------------------------------------------------
# Errors and retries
# ---------------------------------------------------------------------------


def test_reply_with_no_label_is_malformed() -> None:
    reply = _completion({"#": 1.0}, content="maybe")
    with pytest.raises(DecisionRequestError) as info:
        _client(_Server(reply)).decide(STATE, {"q": CHOICE}, budget="batch")
    assert info.value.error_type == "malformed_response"


def test_undecodable_reply_is_malformed() -> None:
    server = _Server(httpx.Response(200, content=b"not json"))
    with pytest.raises(DecisionRequestError) as info:
        _client(server).decide(STATE, {"q": CHOICE}, budget="batch")
    assert info.value.error_type == "malformed_response"


def test_400_is_a_request_error_without_retry() -> None:
    server = _Server(httpx.Response(400, json={"object": "error", "message": "bad", "code": 400}))
    with pytest.raises(DecisionRequestError) as info:
        _client(server).decide(STATE, {"q": CHOICE}, budget="batch")
    assert info.value.status == 400
    assert len(server.requests) == 1


def test_401_is_an_auth_error() -> None:
    server = _Server(httpx.Response(401, json={"error": {"message": "no", "code": 401}}))
    with pytest.raises(DecisionAuthError):
        _client(server).decide(STATE, {"q": CHOICE}, budget="batch")


def test_5xx_is_retried_for_that_question_only() -> None:
    failed: list[bool] = []

    def reply(body: dict) -> Any:
        if body["structured_outputs"]["choice"][0] == "0":
            return _completion({"2": 1.0})
        if not failed:
            failed.append(True)
            return httpx.Response(503)
        return _completion({"A": 1.0})

    server = _Server(default=reply)
    result = _client(server).decide(STATE, {"s": SCORE, "n": NOUL}, budget="batch")
    assert set(result.answers) == {"s", "n"}
    assert sorted(c.attempts for c in result.calls) == [1, 2]
    assert len(server.requests) == 3


def test_timeout_is_unavailable_after_retries() -> None:
    server = _Server(default=lambda body: httpx.ReadTimeout("synthetic"))
    with pytest.raises(DecisionUnavailableError) as info:
        _client(server).decide(STATE, {"q": CHOICE}, budget="interactive")
    assert info.value.error_type == "timeout"
    assert len(server.requests) == 2  # interactive: one retry


def test_connection_failure_is_transport() -> None:
    server = _Server(default=lambda body: httpx.ConnectError("synthetic"))
    with pytest.raises(DecisionUnavailableError) as info:
        _client(server).decide(STATE, {"q": CHOICE}, budget="interactive")
    assert info.value.error_type == "transport"


# ---------------------------------------------------------------------------
# Live (opt-in)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("OSPREY_JEV_OPENAI_LIVE_URL"),
    reason="set OSPREY_JEV_OPENAI_LIVE_URL (and OSPREY_JEV_OPENAI_LIVE_MODEL) to a vLLM server",
)
def test_live_openai_route_answers_a_choice() -> None:
    config = _config(
        base_url=os.environ["OSPREY_JEV_OPENAI_LIVE_URL"],
        model=os.environ.get("OSPREY_JEV_OPENAI_LIVE_MODEL", "qwen3-32b"),
    )
    question = {
        "type": "choice",
        "instructions": "Is the operator's command safe to run on the accelerator?",
        "criteria": {"risky": "it deletes the machine's safety interlock config", "safe": None},
    }
    with JevClient(JevSettings.from_config(config), config) as client:
        result = client.decide("command: rm -rf /etc/interlocks", {"q": question}, budget="batch")
    answer = result.answers["q"]
    assert isinstance(answer, ChoiceAnswer)
    assert answer.probabilities
    assert sum(answer.probabilities.values()) == pytest.approx(1.0, abs=0.01)
    assert result.calls[0].input_tokens > 0
