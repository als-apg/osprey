"""DispatchClient: the fire -> handshake -> worker-poll sequence and the uniform
terminal result contract
``{status, text_output, run_id, error, artifacts, input_artifacts, failure_class,
num_tool_calls, error_code}``.

Every path is driven through the DI seams (``client=`` MockTransport,
``monotonic=``, ``sleep=``) so nothing sleeps or hits the network. The
``failure_class`` / ``num_tool_calls`` fields are the retry-policy inputs: a
worker terminal body passes them through verbatim (absent -> ``None`` meaning
UNKNOWN, never ``0``), and every bridge-local error result defaults them to
``"infrastructure"`` / ``None``.

Sections below correspond to the three concerns this suite covers: the sequence
and result contract, the structured ``error_code`` surface, and the one-shot
:meth:`DispatchClient.status` probe. A final section covers the ``trust_env``
wiring, which is configuration-driven rather than hardcoded.
"""

import httpx
import pytest

from osprey.bridges.core.config import CoreConfig
from osprey.bridges.core.dispatch_client import (
    DispatchClient,
    DispatchPipelineError,
    _error_result,
)

CFG = CoreConfig(
    dispatcher_url="http://disp:10010",
    worker_url="http://work:9190",
    event_dispatcher_token="disp-token",
    dispatch_worker_token="work-token",
    trigger="unit-question",
    poll_interval=1.0,
    poll_budget=300.0,
)

CONTRACT_KEYS = {
    "status",
    "text_output",
    "run_id",
    "error",
    "artifacts",
    "input_artifacts",
    "failure_class",
    "num_tool_calls",
    "error_code",
}


def _client(handler, *, monotonic=None, sleep=None):
    """A DispatchClient wired to a MockTransport handler with frozen time.

    ``monotonic`` defaults to a constant clock (deadlines never trip); pass a
    callable (e.g. an iterator's ``next``) to force budget exhaustion. ``sleep``
    is a no-op recorder by default.
    """
    transport = httpx.MockTransport(handler)
    return DispatchClient(
        CFG,
        client=httpx.Client(transport=transport),
        monotonic=(monotonic or (lambda: 0.0)),
        sleep=(sleep or (lambda _s: None)),
    )


def _seq_monotonic(values):
    """A monotonic() that yields successive values then sticks on the last."""
    it = iter(values)
    last = [0.0]

    def _m():
        try:
            last[0] = next(it)
        except StopIteration:
            pass
        return last[0]

    return _m


# ==========================================================================
# Sequence + uniform terminal result contract
# ==========================================================================


# --------------------------------------------------------------------------
# _error_result: the single-source error branch of the contract
# --------------------------------------------------------------------------
def test_error_result_has_full_contract_shape():
    r = _error_result("R1", "boom")
    assert set(r) == CONTRACT_KEYS


def test_error_result_defaults_infrastructure_and_unknown_count():
    r = _error_result("R1", "boom")
    assert r["status"] == "error"
    assert r["text_output"] == ""
    assert r["run_id"] == "R1"
    assert r["error"] == "boom"
    assert r["artifacts"] == []
    assert r["failure_class"] == "infrastructure"
    assert r["num_tool_calls"] is None


def test_error_result_run_id_may_be_none():
    assert _error_result(None, "no run")["run_id"] is None


def test_error_result_failure_class_overridable():
    r = _error_result("R1", "boom", failure_class="policy")
    assert r["failure_class"] == "policy"
    assert r["num_tool_calls"] is None


# --------------------------------------------------------------------------
# poll_worker: terminal-body passthrough of the classification fields
# --------------------------------------------------------------------------
def _worker_handler(body, *, status_code=200):
    def handler(request):
        assert request.headers["Authorization"] == "Bearer work-token"
        return httpx.Response(status_code, json=body)

    return handler


def test_poll_worker_completed_passes_classification_through():
    body = {
        "status": "completed",
        "text_output": "done",
        "error": None,
        "artifacts": ["a1"],
        "failure_class": None,
        "num_tool_calls": 7,
    }
    r = _client(_worker_handler(body)).poll_worker("R1")
    assert set(r) == CONTRACT_KEYS
    assert r["status"] == "completed"
    assert r["text_output"] == "done"
    assert r["run_id"] == "R1"
    assert r["artifacts"] == ["a1"]
    assert r["failure_class"] is None
    assert r["num_tool_calls"] == 7


def test_poll_worker_error_status_carries_failure_class():
    body = {
        "status": "error",
        "text_output": "",
        "error": "agent blew up",
        "failure_class": "agent",
        "num_tool_calls": 3,
    }
    r = _client(_worker_handler(body)).poll_worker("R2")
    assert r["status"] == "error"
    assert r["error"] == "agent blew up"
    assert r["failure_class"] == "agent"
    assert r["num_tool_calls"] == 3


def test_poll_worker_absent_fields_are_none_not_zero():
    # Older worker predating the fields: keys omitted entirely.
    body = {"status": "completed", "text_output": "hi"}
    r = _client(_worker_handler(body)).poll_worker("R3")
    assert r["failure_class"] is None
    assert r["num_tool_calls"] is None


def test_poll_worker_explicit_zero_tool_calls_preserved():
    # A present 0 is a real count (agent ran no tools), NOT unknown.
    body = {"status": "completed", "text_output": "hi", "num_tool_calls": 0}
    r = _client(_worker_handler(body)).poll_worker("R4")
    assert r["num_tool_calls"] == 0


def test_poll_worker_null_artifacts_degrade_to_empty_list():
    body = {"status": "completed", "text_output": "hi", "artifacts": None}
    r = _client(_worker_handler(body)).poll_worker("R5")
    assert r["artifacts"] == []


@pytest.mark.parametrize(
    "body_extra, expected",
    [
        # present list -> passed through verbatim
        (
            {"input_artifacts": [{"entry_id": "in1", "filename": "data.csv", "mime": "text/csv"}]},
            [{"entry_id": "in1", "filename": "data.csv", "mime": "text/csv"}],
        ),
        ({"input_artifacts": []}, []),  # explicit empty
        ({"input_artifacts": None}, []),  # explicit null -> []
        ({}, []),  # older worker: key absent -> []
    ],
)
def test_poll_worker_input_artifacts_passthrough(body_extra, expected):
    body = {"status": "completed", "text_output": "hi", **body_extra}
    r = _client(_worker_handler(body)).poll_worker("R9")
    assert r["input_artifacts"] == expected


def test_error_result_carries_empty_input_artifacts():
    assert _error_result("R1", "boom")["input_artifacts"] == []


# --------------------------------------------------------------------------
# poll_worker: non-terminal polling, budget exhaustion, HTTP errors
# --------------------------------------------------------------------------
def test_poll_worker_polls_past_pending_until_terminal():
    responses = iter(
        [
            httpx.Response(200, json={"status": "running"}),
            httpx.Response(404),  # brief post-register race, tolerated
            httpx.Response(200, json={"status": "completed", "text_output": "ok"}),
        ]
    )

    def handler(request):
        return next(responses)

    r = _client(handler).poll_worker("R6")
    assert r["status"] == "completed"
    assert r["text_output"] == "ok"


def test_poll_worker_budget_exhaustion_returns_infrastructure_error():
    # Worker never terminal; monotonic crosses the deadline (0 + 300).
    body = {"status": "running"}
    client = _client(
        _worker_handler(body),
        monotonic=_seq_monotonic([0.0, 301.0]),
    )
    r = client.poll_worker("R7")
    assert set(r) == CONTRACT_KEYS
    assert r["status"] == "error"
    assert r["run_id"] == "R7"
    assert r["failure_class"] == "infrastructure"
    assert r["num_tool_calls"] is None
    assert "timed out" in r["error"]


def test_poll_worker_non_404_http_error_raises():
    with pytest.raises(DispatchPipelineError):
        _client(_worker_handler({}, status_code=500)).poll_worker("R8")


# --------------------------------------------------------------------------
# fire + wait_for_run_id: the handshake steps
# --------------------------------------------------------------------------
def test_fire_returns_dispatch_id():
    def handler(request):
        assert request.method == "POST"
        assert str(request.url) == "http://disp:10010/webhook/unit-question"
        assert request.headers["Authorization"] == "Bearer disp-token"
        return httpx.Response(202, json={"dispatch_id": "D1"})

    assert _client(handler).fire("hello") == "D1"


def test_fire_missing_dispatch_id_raises():
    def handler(request):
        return httpx.Response(202, json={})

    with pytest.raises(DispatchPipelineError):
        _client(handler).fire("hello")


def test_fire_non_2xx_raises():
    def handler(request):
        return httpx.Response(500, text="nope")

    with pytest.raises(DispatchPipelineError):
        _client(handler).fire("hello")


def test_wait_for_run_id_returns_run_id_on_completed():
    def handler(request):
        return httpx.Response(200, json={"status": "completed", "result": {"run_id": "R9"}})

    assert _client(handler).wait_for_run_id("D1", deadline=100.0) == "R9"


def test_wait_for_run_id_dispatcher_error_raises():
    def handler(request):
        return httpx.Response(200, json={"status": "error", "error": "dispatch failed"})

    with pytest.raises(DispatchPipelineError):
        _client(handler).wait_for_run_id("D1", deadline=100.0)


# --------------------------------------------------------------------------
# run(): full sequence, on_run_id hook, error conversion
# --------------------------------------------------------------------------
def _full_handler(*, worker_body):
    def handler(request):
        path = request.url.path
        if path == "/webhook/unit-question":
            return httpx.Response(202, json={"dispatch_id": "D1"})
        if path == "/dispatch/D1":
            return httpx.Response(200, json={"status": "completed", "result": {"run_id": "R1"}})
        if path == "/dispatch/R1":
            return httpx.Response(200, json=worker_body)
        return httpx.Response(404)

    return handler


def test_run_full_sequence_returns_terminal_contract():
    body = {"status": "completed", "text_output": "final", "num_tool_calls": 5}
    seen = []
    r = _client(_full_handler(worker_body=body)).run("q", on_run_id=seen.append)
    assert set(r) == CONTRACT_KEYS
    assert r["status"] == "completed"
    assert r["text_output"] == "final"
    assert r["num_tool_calls"] == 5
    assert r["failure_class"] is None
    # on_run_id fires with the handshake run_id before the worker poll.
    assert seen == ["R1"]


def test_run_handshake_dispatch_error_returns_infrastructure_result():
    def handler(request):
        if request.url.path == "/webhook/unit-question":
            return httpx.Response(500, text="down")
        return httpx.Response(404)

    r = _client(handler).run("q")
    assert set(r) == CONTRACT_KEYS
    assert r["status"] == "error"
    assert r["run_id"] is None  # never got a run_id
    assert r["failure_class"] == "infrastructure"
    assert r["num_tool_calls"] is None


def test_run_network_error_during_handshake_converted_to_result():
    def handler(request):
        raise httpx.ConnectError("no route")

    r = _client(handler).run("q")
    assert r["status"] == "error"
    assert r["run_id"] is None
    assert r["failure_class"] == "infrastructure"


def test_run_worker_error_after_run_id_keeps_run_id():
    def handler(request):
        path = request.url.path
        if path == "/webhook/unit-question":
            return httpx.Response(202, json={"dispatch_id": "D1"})
        if path == "/dispatch/D1":
            return httpx.Response(200, json={"status": "completed", "result": {"run_id": "R1"}})
        # worker poll fails hard (non-404)
        return httpx.Response(500, text="worker down")

    seen = []
    r = _client(handler).run("q", on_run_id=seen.append)
    assert r["status"] == "error"
    assert r["run_id"] == "R1"  # preserved so a restart can reconcile
    assert r["failure_class"] == "infrastructure"
    assert seen == ["R1"]


# ==========================================================================
# error_code surface on the dispatch result contract
#
# When the dispatcher pool result is status="error" with a structured error_code
# (e.g. an input_files rejection), DispatchClient lifts that code onto the uniform
# error result so osprey.bridges.core.retry.is_retryable can classify it. Every
# result dict carries an error_code key (None when there is none).
# ==========================================================================

# --- _error_result / DispatchPipelineError carry the field -------------------------


def test_error_result_error_code_default_none():
    assert _error_result("R1", "boom")["error_code"] is None


def test_error_result_error_code_settable():
    r = _error_result(None, "rejected", error_code="input_files_cap_exceeded")
    assert r["error_code"] == "input_files_cap_exceeded"


def test_dispatch_error_carries_error_code_attribute():
    assert (
        DispatchPipelineError("nope", error_code="input_files_invalid").error_code
        == "input_files_invalid"
    )


def test_dispatch_error_default_error_code_none():
    assert DispatchPipelineError("plain").error_code is None


# --- wait_for_run_id propagates the dispatcher's code onto the exception ----


def test_wait_for_run_id_error_code_propagates_to_exception():
    def handler(request):
        return httpx.Response(
            200, json={"status": "error", "error": "bad", "error_code": "input_files_invalid"}
        )

    with pytest.raises(DispatchPipelineError) as exc_info:
        _client(handler).wait_for_run_id("D1", deadline=100.0)
    assert exc_info.value.error_code == "input_files_invalid"


def test_wait_for_run_id_error_without_code_has_none():
    def handler(request):
        return httpx.Response(200, json={"status": "error", "error": "generic"})

    with pytest.raises(DispatchPipelineError) as exc_info:
        _client(handler).wait_for_run_id("D1", deadline=100.0)
    assert exc_info.value.error_code is None


# --- run() lifts the dispatcher error_code onto the result -----------------


def test_run_surfaces_dispatcher_error_code_on_result():
    # Dispatcher pool result is status="error" carrying an input_files error_code;
    # run() must lift it onto the returned error result (not swallow it).
    def handler(request):
        path = request.url.path
        if path == "/webhook/unit-question":
            return httpx.Response(202, json={"dispatch_id": "D1"})
        if path == "/dispatch/D1":
            return httpx.Response(
                200,
                json={
                    "status": "error",
                    "error": "too many files",
                    "error_code": "input_files_cap_exceeded",
                },
            )
        return httpx.Response(404)

    r = _client(handler).run("q")
    assert r["status"] == "error"
    assert r["run_id"] is None  # rejected at the handshake, never got a run_id
    assert r["error_code"] == "input_files_cap_exceeded"


def test_run_ordinary_failure_has_none_error_code():
    # A plain transport/handshake failure carries no structured error_code.
    def handler(request):
        if request.url.path == "/webhook/unit-question":
            return httpx.Response(500, text="down")
        return httpx.Response(404)

    assert _client(handler).run("q")["error_code"] is None


# --- poll_worker keeps the contract uniform --------------------------------


def test_poll_worker_terminal_body_error_code_passed_through():
    body = {"status": "error", "error": "x", "error_code": "input_files_invalid"}

    def handler(request):
        return httpx.Response(200, json=body)

    assert _client(handler).poll_worker("R1")["error_code"] == "input_files_invalid"


def test_poll_worker_terminal_body_without_error_code_is_none():
    def handler(request):
        return httpx.Response(200, json={"status": "completed", "text_output": "ok"})

    assert _client(handler).poll_worker("R1")["error_code"] is None


# ==========================================================================
# DispatchClient.status: a single non-polling GET /dispatch/{run_id} used to
# reconcile a persisted run_id on restart.
#
# Unlike poll_worker it never sleeps, never loops, and returns the worker's raw
# body verbatim (possibly non-terminal). 404 is a signal -> None; any other
# non-200 raises DispatchPipelineError.
# ==========================================================================

STATUS_CFG = CoreConfig(
    worker_url="http://work:9190",
    dispatch_worker_token="work-token",
    poll_budget=300.0,
)


def _status_client(handler):
    return DispatchClient(STATUS_CFG, client=httpx.Client(transport=httpx.MockTransport(handler)))


def _status_handler(response):
    """Return a handler that asserts the GET shape and yields ``response``."""

    def handler(request):
        assert request.method == "GET"
        assert str(request.url) == "http://work:9190/dispatch/R1"
        assert request.headers["Authorization"] == "Bearer work-token"
        return response

    return handler


def test_status_returns_terminal_body_verbatim():
    body = {
        "status": "completed",
        "text_output": "done",
        "artifacts": ["a1"],
        "failure_class": None,
        "num_tool_calls": 4,
    }
    r = _status_client(_status_handler(httpx.Response(200, json=body))).status("R1")
    assert r == body


def test_status_returns_non_terminal_body():
    # The whole point: a live run reports pending and status() hands it back
    # rather than blocking or synthesizing a result.
    body = {"status": "pending"}
    assert _status_client(_status_handler(httpx.Response(200, json=body))).status("R1") == body


def test_status_404_returns_none():
    assert _status_client(_status_handler(httpx.Response(404))).status("R1") is None


def test_status_500_raises_dispatch_error():
    with pytest.raises(DispatchPipelineError):
        _status_client(_status_handler(httpx.Response(500, text="worker down"))).status("R1")


def test_status_does_not_sleep_or_poll():
    # A single request is issued: the handler that raises on a 2nd call proves
    # there is no polling loop.
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] > 1:
            raise AssertionError("status() must issue exactly one request")
        return httpx.Response(200, json={"status": "running"})

    def boom(_s):
        raise AssertionError("status() must not sleep")

    client = DispatchClient(
        STATUS_CFG,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=boom,
    )
    assert client.status("R1") == {"status": "running"}
    assert calls["n"] == 1


# ==========================================================================
# trust_env is configuration, not a per-module constant
#
# The default client honors ``cfg.trust_env`` (default False) rather than
# hardcoding httpx's proxy-trusting default, so a bridge's proxy behavior is one
# setting shared by every core module instead of differing silently per module.
# ==========================================================================


@pytest.mark.parametrize("trust_env", [False, True])
def test_default_client_honors_cfg_trust_env(trust_env):
    cfg = CoreConfig(poll_budget=300.0, trust_env=trust_env)
    client = DispatchClient(cfg)
    try:
        assert client._client.trust_env is trust_env
    finally:
        client._client.close()


def test_injected_client_is_used_as_given():
    # An injected client is never rebuilt, so cfg.trust_env does not apply to it.
    injected = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(404)))
    client = DispatchClient(CoreConfig(poll_budget=300.0, trust_env=True), client=injected)
    assert client._client is injected
    injected.close()
