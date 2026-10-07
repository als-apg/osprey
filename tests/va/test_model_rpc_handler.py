"""The model-RPC handler answers every call exactly once, off the model's thread.

The handler is driven here through fakes: an operation that records how it
was answered, a surface that records what was asked of it, a driver, and an
enqueue that only records the items it is handed. The test plays the run
loop: it runs an item's jobs, then its completion, by hand.
"""

from __future__ import annotations

import importlib
import json
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

import pytest

pytest.importorskip("p4p")

from osprey.services.virtual_accelerator.serving import model_rpc_handler
from osprey.services.virtual_accelerator.serving.model_rpc import (
    ERR_NOT_READY,
    ERR_TIMEOUT,
    REQUEST_TYPE,
    VERBS,
    ModelRpcError,
    RpcRequest,
    build_request,
    error_reply,
)
from osprey.services.virtual_accelerator.serving.model_rpc_handler import (
    DEFAULT_PVA_PORT,
    DISPATCH,
    MODEL_WRITE_VERBS,
    RpcCall,
    RpcFrontDoor,
    pva_endpoint,
)

#: Long enough that the timer never fires in a test that does not wait for it.
_NEVER = 60.0


class FakeOp:
    """A server operation that records every completion."""

    def __init__(self, request: Any = None, *, fail_first_done: bool = False) -> None:
        self._request = build_request("info") if request is None else request
        self._fail_first_done = fail_first_done
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def value(self) -> Any:
        return self._request

    def done(self, value: Any = None, error: str | None = None) -> None:
        with self._lock:
            self.calls.append({"value": value, "error": error})
            if self._fail_first_done:
                self._fail_first_done = False
                raise RuntimeError("the reply could not be delivered")


class FakeSurface:
    """A model surface that records what it is asked and answers each verb by name."""

    def __init__(self, raises: dict[str, Exception] | None = None) -> None:
        self.raises = raises or {}
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def _verb(self, verb: str, *args: Any) -> dict[str, str]:
        self.calls.append((verb, args))
        if verb in self.raises:
            raise self.raises[verb]
        return {"verb": verb}

    def info(self) -> dict[str, str]:
        return self._verb("info")

    def get(self, names: Any) -> dict[str, str]:
        return self._verb("get", names)

    def diff(self, get_param: Callable[[str], Any]) -> dict[str, str]:
        return self._verb("diff", get_param)

    def status(self) -> dict[str, str]:
        return self._verb("status")

    def set(self, values: Any, token: str | None) -> dict[str, str]:
        return self._verb("set", values, token)

    def reset(self, token: str | None) -> dict[str, str]:
        return self._verb("reset", token)

    def record_cycle(self, ms: float) -> None:
        self.calls.append(("record_cycle", (ms,)))

    def record_queue_depth(self, n: int) -> None:
        self.calls.append(("record_queue_depth", (n,)))

    def record_refusal(self, text: str) -> None:
        self.calls.append(("record_refusal", (text,)))

    def recorded(self, name: str) -> list[tuple[Any, ...]]:
        return [args for verb, args in self.calls if verb == name]


class FakeDriver:
    def getParam(self, _address: str) -> float:
        return 0.0


class Loop:
    """An enqueue that only records, and the hand that runs what it recorded."""

    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def enqueue(
        self,
        values: dict[str, Any],
        done: Callable[[str | None], None] | None = None,
        reset: bool = False,
        *,
        jobs: Any = (),
    ) -> None:
        self.items.append({"values": values, "done": done, "reset": reset, "jobs": list(jobs)})

    def run(self, error: str | None = None) -> None:
        """Run every recorded item as the run loop would: jobs, then completion."""
        while self.items:
            item = self.items.pop(0)
            for job in item["jobs"]:
                try:
                    job()
                except Exception:  # the run loop logs a failed job and moves on
                    pass
            if item["done"] is not None:
                try:
                    item["done"](error)
                except Exception:  # the run loop logs a failed completion
                    pass


def _door(
    surface: FakeSurface,
    loop: Loop,
    *,
    driver: Any = None,
    queue_depth: Callable[[], int] = lambda: 0,
    timeout_s: float = _NEVER,
) -> RpcFrontDoor:
    the_driver = FakeDriver() if driver is None else driver
    return RpcFrontDoor(
        surface,
        enqueue=loop.enqueue,
        driver=lambda: the_driver,
        queue_depth=queue_depth,
        timeout_s=timeout_s,
    )


def _request(verb: str) -> Any:
    if verb == "set":
        return build_request("set", values={"x": 1.0}, token="secret")
    if verb == "reset":
        return build_request("reset", token="secret")
    if verb == "get":
        return build_request("get", names=["x"])
    return build_request(verb)


def _document(reply: Any) -> Any:
    return json.loads(reply["value"])


def _wait_for(condition: Callable[[], bool], within: float = 5.0) -> None:
    deadline = time.monotonic() + within
    while not condition():
        assert time.monotonic() < deadline, "timed out waiting"
        time.sleep(0.005)


def test_a_second_completion_is_dropped() -> None:
    """p4p completes an operation once; a second completion is an error its
    client never sees, so the call drops it."""
    op = FakeOp()
    call = RpcCall(op)

    assert call.complete(error_reply("first")) is True
    assert call.complete(error_reply("second")) is False
    assert len(op.calls) == 1
    assert _document(op.calls[0]["value"]) == {"ok": False, "error": "first"}


def _race(call: RpcCall, barrier: threading.Barrier, won: list[bool], text: str) -> None:
    barrier.wait()
    won.append(call.complete(error_reply(text)))


def test_racing_completions_answer_once() -> None:
    """The job and the timer race to answer; exactly one of them does."""
    for _ in range(200):
        op = FakeOp()
        call = RpcCall(op)
        barrier = threading.Barrier(2)
        won: list[bool] = []
        racers = [
            threading.Thread(target=_race, args=(call, barrier, won, text))
            for text in ("job", "timer")
        ]
        for racer in racers:
            racer.start()
        for racer in racers:
            racer.join()

        assert len(op.calls) == 1
        assert sorted(won) == [False, True]


def test_a_call_the_loop_never_runs_is_answered_with_the_timeout() -> None:
    """A client told nothing waits out its own timeout knowing only that the
    server did not answer; the contract's own message says why."""
    op = FakeOp()
    loop = Loop()
    _door(FakeSurface(), loop, timeout_s=0.05)(None, op)

    timers = [each for each in threading.enumerate() if isinstance(each, threading.Timer)]
    assert timers
    # Daemon, so a call still being waited on never holds up a shutdown.
    assert all(timer.daemon for timer in timers)

    _wait_for(lambda: bool(op.calls))
    assert len(op.calls) == 1
    assert _document(op.calls[0]["value"]) == {"ok": False, "error": ERR_TIMEOUT}


def test_a_late_job_after_the_timeout_answers_nothing() -> None:
    op = FakeOp()
    loop = Loop()
    _door(FakeSurface(), loop, timeout_s=0.05)(None, op)
    _wait_for(lambda: bool(op.calls))

    loop.run()

    assert len(op.calls) == 1
    assert _document(op.calls[0]["value"]) == {"ok": False, "error": ERR_TIMEOUT}


def test_a_malformed_request_is_refused_without_the_run_loop() -> None:
    """A malformed request never reaches the model, so it never needs the
    loop's thread -- and would otherwise occupy a cycle to be told so."""
    bad_verb = REQUEST_TYPE.wrap("model_rpc", kws={"verb": "launch"})
    no_query: dict[str, Any] = {}

    for op in (FakeOp(request=bad_verb), FakeOp(request=no_query)):
        loop = Loop()
        surface = FakeSurface()
        _door(surface, loop)(None, op)

        assert len(op.calls) == 1
        assert op.calls[0]["value"] is None
        assert op.calls[0]["error"]
        assert loop.items == []
        assert surface.calls == []


def test_a_call_before_the_driver_exists_is_refused() -> None:
    """The PVA server is listening from the moment it is created, which is
    before the driver a served value is read through exists."""
    op = FakeOp()
    loop = Loop()
    door = RpcFrontDoor(
        FakeSurface(), enqueue=loop.enqueue, driver=lambda: None, queue_depth=lambda: 0
    )

    door(None, op)

    assert op.calls == [{"value": None, "error": ERR_NOT_READY}]
    assert loop.items == []


@pytest.mark.parametrize("verb", VERBS)
def test_taking_a_call_never_touches_the_surface(verb: str) -> None:
    """A call is taken on a p4p worker thread, which is no more allowed to
    touch the model than the Channel Access server thread is. The verb is
    dispatched by a job instead, and the job runs on the run loop."""
    surface = FakeSurface()
    loop = Loop()
    _door(surface, loop)(None, FakeOp(request=_request(verb)))

    assert surface.calls == []
    # Empty values, so the item carrying the job runs no model pass of its
    # own: the job is the whole of what a read costs the loop.
    assert all(item["values"] == {} for item in loop.items)
    if verb in MODEL_WRITE_VERBS:
        assert len(loop.items) == 2
        assert len(loop.items[0]["jobs"]) == 1
        assert loop.items[0]["done"] is None
        assert loop.items[1]["jobs"] == []
        assert loop.items[1]["done"] is not None
    else:
        assert len(loop.items) == 1
        assert len(loop.items[0]["jobs"]) == 1
        assert loop.items[0]["done"] is None
    loop.run()


@pytest.mark.parametrize("verb", VERBS)
def test_a_surface_that_raises_still_gets_an_answer(verb: str) -> None:
    """The run loop logs a job that raises and moves on. So a job that
    returned without answering would cost its client the whole timeout, and
    tell it nothing when the timeout expired."""
    refused = FakeSurface(raises={verb: ModelRpcError("refused by the surface")})
    op = FakeOp(request=_request(verb))
    loop = Loop()
    _door(refused, loop)(None, op)
    loop.run()

    assert _document(op.calls[-1]["value"]) == {"ok": False, "error": "refused by the surface"}
    # The surface records the refusals it raises itself.
    assert refused.recorded("record_refusal") == []
    assert len(refused.recorded("record_cycle")) == 1

    broken = FakeSurface(raises={verb: RuntimeError("wiring")})
    op = FakeOp(request=_request(verb))
    loop = Loop()
    _door(broken, loop)(None, op)
    loop.run()

    error = _document(op.calls[-1]["value"])["error"]
    assert verb in error
    assert "wiring" in error
    # A write that failed some other way is a refused write too; a failed
    # read is no write at all.
    expected = [(error,)] if verb in MODEL_WRITE_VERBS else []
    assert broken.recorded("record_refusal") == expected
    assert len(broken.recorded("record_cycle")) == 1

    answered = FakeSurface()
    op = FakeOp(request=_request(verb))
    loop = Loop()
    _door(answered, loop)(None, op)
    loop.run()

    assert _document(op.calls[-1]["value"]) == {"ok": True, "result": {"verb": verb}}
    assert len(answered.recorded("record_cycle")) == 1


@pytest.mark.parametrize("verb", ["info", "set"])
def test_the_queue_depth_is_read_by_the_job(verb: str) -> None:
    """``status`` reports the run loop's own queue, which only the loop's
    thread reads."""
    reads: list[int] = []

    def queue_depth() -> int:
        reads.append(7)
        return 7

    surface = FakeSurface()
    loop = Loop()
    _door(surface, loop, queue_depth=queue_depth)(None, FakeOp(request=_request(verb)))

    assert reads == []
    loop.run()
    assert reads == [7]
    assert surface.recorded("record_queue_depth") == [(7,)]


def test_a_write_answers_with_its_pass() -> None:
    """A write replies only after a publishing pass has run, with the pass's
    error if it failed."""
    surface = FakeSurface()
    op = FakeOp(request=_request("set"))
    loop = Loop()
    _door(surface, loop)(None, op)
    loop.run(error=None)
    assert op.calls == [{"value": op.calls[0]["value"], "error": None}]
    assert _document(op.calls[0]["value"]) == {"ok": True, "result": {"verb": "set"}}

    surface = FakeSurface()
    op = FakeOp(request=_request("set"))
    loop = Loop()
    _door(surface, loop)(None, op)
    loop.run(error="boom")
    assert len(op.calls) == 1
    assert _document(op.calls[0]["value"]) == {"ok": False, "error": "boom"}
    assert surface.recorded("record_refusal") == [("boom",)]

    surface = FakeSurface()
    op = FakeOp(request=_request("reset"))
    loop = Loop()
    _door(surface, loop)(None, op)
    loop.items[0]["jobs"] = []  # the job that would have kept a reply never ran
    loop.run(error=None)
    assert len(op.calls) == 1
    assert _document(op.calls[0]["value"]) == {
        "ok": False,
        "error": "the model surface failed on reset",
    }


#: The arguments each verb hands the surface method of its name.
_SURFACE_ARGUMENTS: dict[str, Callable[[RpcRequest, FakeDriver], tuple[Any, ...]]] = {
    "info": lambda request, driver: (),
    "get": lambda request, driver: (request.names,),
    # The served value beside the model's truth: only the driver knows the
    # first of the two.
    "diff": lambda request, driver: (driver.getParam,),
    "status": lambda request, driver: (),
    "set": lambda request, driver: (request.values, request.token),
    "reset": lambda request, driver: (request.token,),
}


@pytest.mark.parametrize("verb", VERBS)
def test_the_verb_table_answers_every_verb_the_contract_admits(verb: str) -> None:
    """A verb the contract admits and the table does not would be parsed,
    enqueued, and answered with a refusal for no reason a client can act on."""
    assert set(DISPATCH) == set(VERBS)
    request = RpcRequest(verb=verb, names=("x",), values={"x": 1.0}, token="secret")
    surface = FakeSurface()
    driver = FakeDriver()

    assert DISPATCH[verb](request, surface, driver) == {"verb": verb}
    assert surface.calls == [(verb, _SURFACE_ARGUMENTS[verb](request, driver))]


def test_the_reported_endpoint_is_the_port_the_server_binds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read from the variable the pvAccess server itself binds from, so the
    address a client is told to use cannot drift from the one in use."""
    monkeypatch.setattr(model_rpc_handler.socket, "gethostname", lambda: "va-host")

    monkeypatch.setenv("EPICS_PVAS_SERVER_PORT", "5999")
    assert pva_endpoint() == "va-host:5999"

    monkeypatch.setenv("EPICS_PVAS_SERVER_PORT", "  ")
    assert pva_endpoint() == f"va-host:{DEFAULT_PVA_PORT}"

    monkeypatch.delenv("EPICS_PVAS_SERVER_PORT")
    assert pva_endpoint() == f"va-host:{DEFAULT_PVA_PORT}"


def test_the_handler_imports_no_server_library(monkeypatch: pytest.MonkeyPatch) -> None:
    """The handler runs on any host p4p installs on, with neither Channel
    Access server nor serving runner behind it."""
    for blocked in ("pcaspy", "lume_pva_apg"):
        for name in [n for n in sys.modules if n == blocked or n.startswith(f"{blocked}.")]:
            monkeypatch.setitem(sys.modules, name, None)
        monkeypatch.setitem(sys.modules, blocked, None)
    monkeypatch.delitem(sys.modules, model_rpc_handler.__name__)

    imported = importlib.import_module(model_rpc_handler.__name__)

    assert imported.RpcFrontDoor.__name__ == "RpcFrontDoor"
