"""The scripted Jev test double: answers, per-attempt scripts, rules, clock and records.

All data here is synthetic.
"""

from __future__ import annotations

import asyncio
import threading
import time

import httpx
import pytest
from tests.models.jev._jev_fake_transport import (
    SYNTHETIC_REQUEST_ID_PREFIX,
    FakeJev,
    Reply,
    body_tokens,
    tokens_over,
)

from osprey.models.jev.errors import DecisionRequestError, error_from_response
from osprey.models.jev.models import parse_response

MODEL = "typesafe/jev-1.13"
SERVED = "typesafe/jev-1.13-20260917"
URL = "https://synthetic.invalid/v1/systemone"

NOUL = {"type": "noul", "instructions": "Is the synthetic widget warm?"}
CHOICE = {
    "type": "choice",
    "instructions": "Which synthetic kind?",
    "criteria": {"o1": "alpha", "o2": "beta"},
}


def _body(**questions: dict) -> dict:
    return {"model": MODEL, "state": "synthetic state", "questions": questions or {"q1": NOUL}}


def _post(fake: FakeJev, body: dict | None = None, *, read: float | None = None) -> httpx.Response:
    timeout = httpx.Timeout(5.0, read=read) if read is not None else httpx.Timeout(None)
    with httpx.Client(transport=fake.transport, timeout=timeout) as client:
        return client.post(URL, json=body if body is not None else _body())


def test_fixed_answer_parses_through_the_real_normaliser():
    fake = FakeJev({"q1": {"type": "noul", "noul": 0.91234}}, model=SERVED)
    resp = _post(fake)
    assert resp.status_code == 200
    parsed = parse_response(resp.json(), ["q1"])
    assert parsed.model == SERVED
    assert parsed.answers["q1"].noul == pytest.approx(0.912)


def test_callable_answer_sees_question_and_body():
    seen = []

    def answer(question, body):
        seen.append((question["type"], body["state"]))
        return {"type": "choice", "choice": "o2", "probabilities": {"o1": 0.2, "o2": 0.8}}

    fake = FakeJev({"pick": answer})
    resp = _post(fake, _body(pick=CHOICE))
    assert resp.json()["answers"]["pick"]["choice"] == "o2"
    assert seen == [("choice", "synthetic state")]


def test_unscripted_question_is_left_out_of_the_reply():
    fake = FakeJev({"q1": {"type": "noul", "noul": 0.5}})
    resp = _post(fake, _body(q1=NOUL, q2=NOUL))
    assert set(resp.json()["answers"]) == {"q1"}


def test_default_model_echoes_the_request_and_usage_is_synthetic():
    fake = FakeJev({"q1": {"type": "noul", "noul": 0.1}})
    body = _body()
    payload = _post(fake, body).json()
    assert payload["model"] == MODEL
    assert payload["usage"] == {"input_tokens": body_tokens(body), "output_tokens": 5}


def test_chosen_usage_fixed_and_callable():
    fake = FakeJev(usage={"input_tokens": 476, "output_tokens": 70, "cost": 0.00002})
    assert _post(fake).json()["usage"]["cost"] == 0.00002
    fake = FakeJev(usage=lambda body: {"input_tokens": len(body["questions"]), "output_tokens": 1})
    assert _post(fake, _body(a=NOUL, b=NOUL)).json()["usage"]["input_tokens"] == 2


def test_response_sequence_is_consumed_per_attempt_then_answers_normally():
    fake = FakeJev(
        {"q1": {"type": "noul", "noul": 0.7}},
        responses=[429, Reply(status=503, body={"detail": "busy"}), None],
    )
    statuses = [_post(fake).status_code for _ in range(4)]
    assert statuses == [429, 503, 200, 200]
    assert fake.call_count == 4


def test_scripted_exception_class_and_instance_are_raised():
    fake = FakeJev(responses=[httpx.ConnectError, RuntimeError("synthetic boom")])
    with pytest.raises(httpx.ConnectError):
        _post(fake)
    with pytest.raises(RuntimeError, match="synthetic boom"):
        _post(fake)
    assert _post(fake).status_code == 200
    assert fake.call_count == 3


def test_request_id_header_is_synthetic_and_overridable():
    fake = FakeJev(responses=[None, Reply(status=500, headers={"x-typesafe-request-id": "r-9"})])
    assert _post(fake).headers["x-typesafe-request-id"] == f"{SYNTHETIC_REQUEST_ID_PREFIX}1"
    assert _post(fake).headers["x-typesafe-request-id"] == "r-9"


def test_raw_bytes_body_is_sent_verbatim():
    fake = FakeJev(responses=[Reply(status=502, body=b"<html>gateway</html>")])
    assert _post(fake).content == b"<html>gateway</html>"


def test_reject_if_predicate_maps_to_max_tokens_exceeded():
    body_err = {"detail": {"error_type": "max_tokens_exceeded"}}
    fake = FakeJev({"q1": {"type": "noul", "noul": 0.5}})
    fake.reject_if(tokens_over(40), status=400, body=body_err)

    small = {
        "model": MODEL,
        "state": "s",
        "questions": {"q1": {"type": "noul", "instructions": "x"}},
    }
    assert body_tokens(small) <= 40
    assert _post(fake, small).status_code == 200

    big = _body(q1={"type": "noul", "instructions": "y" * 400})
    resp = _post(fake, big)
    assert resp.status_code == 400
    err = error_from_response(resp.status_code, resp.headers, resp.content)
    assert isinstance(err, DecisionRequestError)
    assert err.error_type == "max_tokens_exceeded"


def test_scripted_attempt_wins_over_rules():
    fake = FakeJev(responses=[None, 503]).reject_if(lambda body: True, status=400)
    assert _post(fake).status_code == 400
    assert _post(fake).status_code == 503


def test_sleep_advances_clock_without_real_sleeping():
    fake = FakeJev()
    start = fake.clock()
    t0 = time.monotonic()
    fake.sleep(3600.0)
    assert time.monotonic() - t0 < 1.0
    assert fake.clock() == start + 3600.0
    with pytest.raises(ValueError):
        fake.sleep(-1.0)


def test_latency_under_timeout_advances_clock_by_latency():
    fake = FakeJev(latency=1.5)
    start = fake.clock()
    assert _post(fake, read=2.0).status_code == 200
    assert fake.clock() == start + 1.5


def test_latency_over_timeout_raises_read_timeout_and_advances_by_timeout():
    fake = FakeJev(latency=5.0)
    start = fake.clock()
    with pytest.raises(httpx.ReadTimeout):
        _post(fake, read=2.0)
    assert fake.clock() == start + 2.0
    assert fake.call_count == 1


def test_latency_without_read_timeout_advances_fully():
    fake = FakeJev(latency=lambda body: 7.0)
    start = fake.clock()
    assert _post(fake).status_code == 200
    assert fake.clock() == start + 7.0


def test_per_reply_latency_overrides_the_default():
    fake = FakeJev(latency=0.1, responses=[Reply(latency=9.0), None])
    start = fake.clock()
    with pytest.raises(httpx.ReadTimeout):
        _post(fake, read=3.0)
    assert fake.clock() == start + 3.0
    _post(fake, read=3.0)
    assert fake.clock() == pytest.approx(start + 3.1)


def test_records_every_request_body():
    fake = FakeJev()
    _post(fake, _body(a=NOUL))
    _post(fake, _body(b=CHOICE))
    assert [list(b["questions"]) for b in fake.requests] == [["a"], ["b"]]
    assert fake.requests[1]["questions"]["b"] == CHOICE


def test_peak_in_flight_counts_overlapping_requests():
    fake = FakeJev(dwell=0.2)
    threads = [threading.Thread(target=_post, args=(fake,)) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert fake.call_count == 3
    assert fake.peak_in_flight == 3
    assert fake.in_flight == 0


def test_sequential_requests_peak_at_one():
    fake = FakeJev()
    for _ in range(3):
        _post(fake)
    assert fake.peak_in_flight == 1


def test_unscriptable_entry_is_refused():
    fake = FakeJev(responses=["nonsense"])
    with pytest.raises(TypeError):
        _post(fake)


async def _apost(fake: FakeJev) -> httpx.Response:
    async with httpx.AsyncClient(transport=fake.transport) as client:
        return await client.post(URL, json=_body())


def test_async_client_is_served_too():
    fake = FakeJev({"q1": {"type": "noul", "noul": 0.3}})
    resp = asyncio.run(_apost(fake))
    assert resp.json()["answers"]["q1"]["noul"] == 0.3
