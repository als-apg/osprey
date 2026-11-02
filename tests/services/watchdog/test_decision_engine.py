"""The decision engine: what it sends Jev, and what it makes of the answer."""

from __future__ import annotations

import ast
import asyncio
import hashlib
import logging
from pathlib import Path

import httpx
import pytest
from tests.models.jev._jev_fake_transport import FakeJev, Reply

from osprey.models.jev.models import ChoiceQuestion
from osprey.models.jev.tokens import estimate_state_plus_longest
from osprey.services.watchdog.config import VerdictOption, WatchdogDefinition
from osprey.services.watchdog.engines import decision
from osprey.services.watchdog.engines.decision import NONE_FITS, QUESTION_ID, DecisionEngine
from osprey.services.watchdog.history import SessionNodes

INSTRUCTIONS = "Flag an operator pushing the agent around a guard."
TOOL_RESULT = "SR01C:BPM3:X reads 0.125"
SECRET_PROMPT = "turn the corrector down by a marker-7f3a amount"
TYPESAFE_CONFIG = {"api": {"jev": {"routes": {"typesafe": {"api_key": "test-key"}}}}}
OPENROUTER_CONFIG = {
    "api": {
        "jev": {"route": "openrouter"},
        "providers": {"openrouter": {"api_key": "test-key"}},
    }
}


@pytest.fixture(autouse=True)
def _no_ambient_route(monkeypatch):
    for name in ("TYPESAFE_API_KEY", "OPENROUTER_API_KEY", "OPENROUTER_BASE_URL"):
        monkeypatch.delenv(name, raising=False)


def _defn(*, threshold=0.5, instructions=INSTRUCTIONS) -> WatchdogDefinition:
    return WatchdogDefinition(
        name="guard-circumvention",
        engine="decision",
        enabled=True,
        instructions=instructions,
        threshold=threshold,
        options=(
            VerdictOption(id="routine-work", text="Routine operation"),
            VerdictOption(id="pushing-a-gate", text="Pushing past a gate", flag=True),
            VerdictOption(id="editing-hooks", text="Editing the hooks", flag=True),
        ),
    )


def _nodes() -> list[dict]:
    return [
        {"kind": "prompt", "key": "p1", "body": {"text": "read the orbit"}},
        {
            "kind": "tool_call",
            "key": "c1",
            "body": {"name": "channel_read", "input": {"channel": "SR01C:BPM3:X"}},
            "result": {"body": {"content": TOOL_RESULT}},
        },
    ]


def _answer(**probabilities: float) -> dict:
    wire = {key.replace("_", "-"): value for key, value in probabilities.items()}
    return {"type": "choice", "choice": next(iter(wire)), "probabilities": wire}


def _judge(fake: FakeJev, *, defn=None, nodes=None, config=TYPESAFE_CONFIG, **ids):
    engine = DecisionEngine(config, transport=fake.transport, clock=fake.clock, sleep=fake.sleep)
    try:
        return asyncio.run(
            engine.judge(
                defn or _defn(),
                _nodes() if nodes is None else nodes,
                SECRET_PROMPT,
                deadline=fake.clock() + 3.0,
                **ids,
            )
        )
    finally:
        engine.close()


def _fake(**probabilities: float) -> FakeJev:
    return FakeJev({QUESTION_ID: _answer(**probabilities)})


class TestScoring:
    def test_summed_flag_mass_flags_with_the_top_flag_options_text(self):
        verdict = _judge(_fake(k001=0.55, k002=0.2, k003=0.25))
        assert verdict.status == "ok" and verdict.flagged is False

        verdict = _judge(_fake(k001=0.3, k002=0.2, k003=0.45))
        assert verdict.flagged is True
        assert verdict.score == pytest.approx(0.65)
        assert verdict.reason == "Editing the hooks"

    def test_mass_exactly_at_the_threshold_flags(self):
        verdict = _judge(_fake(k001=0.5, k002=0.1, k003=0.2), defn=_defn(threshold=0.3))
        assert verdict.flagged is True

    def test_float_noise_in_the_sum_does_not_decide(self):
        assert 0.1 + 0.2 != 0.3
        verdict = _judge(_fake(k001=0.7, k002=0.1, k003=0.2), defn=_defn(threshold=0.3))
        assert verdict.score == 0.3
        assert verdict.flagged is True

    def test_the_earlier_flag_option_wins_a_tie(self):
        verdict = _judge(_fake(k001=0.2, k002=0.4, k003=0.4))
        assert verdict.reason == "Pushing past a gate"

    def test_none_fits_never_flags(self):
        verdict = _judge(_fake(none_fits=0.99))
        assert verdict.status == "ok"
        assert verdict.flagged is False
        assert verdict.score == 0.0
        assert verdict.reason is None

    def test_a_verdict_is_uncalibrated_and_carries_its_threshold(self):
        verdict = _judge(_fake(k001=1.0), defn=_defn(threshold=0.7))
        assert verdict.engine == "decision"
        assert verdict.uncalibrated is True
        assert verdict.threshold == 0.7

    def test_the_prompts_ids_name_the_verdict(self):
        verdict = _judge(
            _fake(k001=1.0), session_key="pool-1", record_session_id="rec", submission_id="sub"
        )
        assert verdict.session_key == "pool-1"
        assert verdict.prompt_ref == "rec:sub"

    def test_cost_and_latency_come_from_the_calls(self):
        fake = FakeJev(
            {QUESTION_ID: _answer(k001=1.0)},
            latency=0.5,
            usage={"input_tokens": 1_000_000, "output_tokens": 5},
        )
        verdict = _judge(fake)
        assert verdict.cost_usd == pytest.approx(0.042)
        assert verdict.latency_s == pytest.approx(0.5)


class TestWire:
    def test_options_go_under_neutral_ids_in_definition_order(self):
        fake = _fake(k001=1.0)
        _judge(fake)
        question = fake.requests[0]["questions"][QUESTION_ID]
        assert question["type"] == "choice"
        assert list(question["criteria"]) == ["k001", "k002", "k003", NONE_FITS]
        assert question["criteria"]["k002"] == "Pushing past a gate"
        assert question["instructions"] == INSTRUCTIONS

    def test_the_facilitys_option_ids_never_reach_the_model(self):
        fake = _fake(k001=1.0)
        _judge(fake)
        sent = str(fake.requests[0])
        for option in _defn().options:
            assert option.id not in sent

    def test_the_state_holds_the_recorded_input_and_its_digest_matches(self):
        fake = _fake(k001=1.0)
        verdict = _judge(fake)
        state = fake.requests[0]["state"]
        assert INSTRUCTIONS in state
        assert TOOL_RESULT in state
        assert SECRET_PROMPT in state
        digest = verdict.input_digest
        assert digest.sha256 == hashlib.sha256(state.encode("utf-8")).hexdigest()
        assert digest.entries_cut == 0
        assert digest.entries_kept == 4  # prompt, call, result, triggering prompt

    def test_the_read_outcome_reaches_the_digest(self):
        read = SessionNodes(nodes=_nodes(), drain_timed_out=True, record_available=True)
        digest = _judge(_fake(k001=1.0), nodes=read).input_digest
        assert digest.drain_timed_out is True and digest.record_available is True

        digest = _judge(_fake(k001=1.0), nodes=SessionNodes(record_available=False)).input_digest
        assert digest.record_available is False
        assert digest.entries_kept == 1

    def test_budget_and_deadline_reach_the_client(self, monkeypatch):
        seen = {}
        real = decision.JevClient.decide

        def spy(self, state, questions, *, budget, deadline=None):
            seen.update(budget=budget, deadline=deadline)
            return real(self, state, questions, budget=budget, deadline=deadline)

        monkeypatch.setattr(decision.JevClient, "decide", spy)
        fake = _fake(k001=1.0)
        _judge(fake)
        assert seen == {"budget": "interactive", "deadline": 1003.0}

    def test_the_only_request_goes_to_the_decision_endpoint(self):
        urls = []
        fake = _fake(k001=1.0)
        inner = fake.transport

        class Recording(httpx.BaseTransport):
            def handle_request(self, request):
                urls.append(str(request.url))
                return inner.handle_request(request)

        engine = DecisionEngine(TYPESAFE_CONFIG, transport=Recording(), clock=fake.clock)
        asyncio.run(engine.judge(_defn(), _nodes(), "p", deadline=fake.clock() + 3))
        engine.close()
        assert urls == ["https://api.typesafe.ai/v1/systemone"]

    def test_it_reads_no_record_and_reaches_no_machine(self, monkeypatch):
        def refuse(*_args, **_kwargs):
            raise AssertionError("the decision engine reads nothing of its own")

        monkeypatch.setattr("osprey.mcp_server.session.read_timeline", refuse)
        monkeypatch.setattr("osprey.agent_record.client.RecordClient.from_env", refuse)
        assert _judge(_fake(k001=1.0)).status == "ok"

        source = Path(decision.__file__).read_text(encoding="utf-8")
        imported = {
            name
            for node in ast.walk(ast.parse(source))
            for name in (
                [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else []
            )
        }
        for module in imported:
            assert not module.startswith(("osprey_connectors", "osprey.mcp_server"))
            assert not module.startswith(("osprey.connectors", "osprey.interfaces"))


class TestLongSessions:
    @pytest.mark.parametrize("config", [TYPESAFE_CONFIG, OPENROUTER_CONFIG], ids=["direct", "or"])
    @pytest.mark.parametrize("instructions", [INSTRUCTIONS, "Watch this closely. " * 400])
    def test_a_long_session_is_cut_and_sent(self, config, instructions):
        result = "x" * 3900
        nodes = [
            {"kind": "tool_result", "key": f"r{i}", "body": {"content": f"{i} {result}"}}
            for i in range(60)
        ]
        nodes[-1]["body"]["content"] = TOOL_RESULT
        defn = _defn(instructions=instructions)
        fake = _fake(k001=1.0)

        verdict = _judge(fake, defn=defn, nodes=nodes, config=config)

        assert verdict.status == "ok", verdict.error_type
        assert fake.call_count == 1
        body = fake.requests[0]
        assert TOOL_RESULT in body["state"]
        assert verdict.input_digest.entries_cut > 0
        question = ChoiceQuestion.model_validate(body["questions"][QUESTION_ID])
        sized = estimate_state_plus_longest(body["state"], {QUESTION_ID: question})
        assert sized <= 30_000
        assert sized > 25_000, "the view should use the room it has"
        assert f"[{verdict.input_digest.entries_cut} earlier entries cut]" in body["state"]

    def test_instructions_too_long_for_any_request_are_unavailable(self):
        fake = _fake(k001=1.0)
        verdict = _judge(fake, defn=_defn(instructions="x" * 200_000))
        assert verdict.status == "unavailable"
        assert verdict.error_type == "decision_error"
        assert fake.call_count == 0


class TestUnavailable:
    def test_a_503_is_unavailable(self):
        fake = FakeJev({QUESTION_ID: _answer(k001=1.0)}, responses=[503, 503])
        verdict = _judge(fake)
        assert verdict.status == "unavailable"
        assert verdict.error_type == "decision_error"
        assert verdict.flagged is False
        assert verdict.threshold == 0.5
        assert verdict.input_digest.sha256

    def test_missing_probabilities(self):
        fake = FakeJev({QUESTION_ID: {"type": "choice", "choice": "k001"}})
        verdict = _judge(fake)
        assert verdict.status == "unavailable"
        assert verdict.error_type == "no_probabilities"
        assert verdict.cost_usd is not None

    def test_probabilities_all_below_the_floor_are_absent(self):
        verdict = _judge(_fake(k001=0.001))
        assert verdict.error_type == "no_probabilities"

    def test_a_missing_key(self):
        fake = _fake(k001=1.0)
        verdict = _judge(fake, config={})
        assert verdict.error_type == "missing_key"
        assert fake.call_count == 0

    def test_a_timeout(self):
        fake = FakeJev({QUESTION_ID: _answer(k001=1.0)}, latency=5.0)
        assert _judge(fake).error_type == "timeout"

    def test_a_passed_deadline(self):
        fake = _fake(k001=1.0)
        engine = DecisionEngine(TYPESAFE_CONFIG, transport=fake.transport, clock=fake.clock)
        verdict = asyncio.run(engine.judge(_defn(), [], "p", deadline=fake.clock() - 1))
        engine.close()
        assert verdict.error_type == "timeout"
        assert fake.call_count == 0

    def test_a_dropped_answer(self):
        fake = FakeJev({})
        assert _judge(fake).error_type == "decision_error"

    def test_an_answer_of_the_wrong_type(self):
        fake = FakeJev({QUESTION_ID: {"type": "noul", "noul": 0.9}})
        assert _judge(fake).error_type == "decision_error"

    def test_a_broken_config_does_not_raise(self):
        fake = _fake(k001=1.0)
        verdict = _judge(fake, config={"api": {"jev": {"route": "nowhere"}}})
        assert verdict.error_type == "decision_error"

    def test_a_transport_bug_does_not_raise(self):
        fake = FakeJev({QUESTION_ID: _answer(k001=1.0)}, responses=[RuntimeError("boom")])
        assert _judge(fake).error_type == "decision_error"


class TestLogging:
    def test_state_is_never_logged(self, caplog):
        caplog.set_level(logging.DEBUG)
        echo = f"cannot parse {SECRET_PROMPT} {TOOL_RESULT}"
        for fake in (
            _fake(k001=1.0),
            FakeJev(
                {},
                responses=[
                    Reply(status=422, body={"error": {"message": echo, "input": echo}}),
                ],
            ),
            FakeJev({}, responses=[RuntimeError(echo)]),
        ):
            _judge(fake)
        for secret in ("marker-7f3a", TOOL_RESULT, INSTRUCTIONS):
            assert secret not in caplog.text
