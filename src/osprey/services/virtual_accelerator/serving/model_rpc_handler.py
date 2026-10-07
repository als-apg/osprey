"""The model-RPC protocol layer: take a call, hand the run loop the work, answer once.

A call is parsed, refused early when it can be, enqueued onto the run loop,
and answered exactly once, with a timeout on every path. :class:`RpcFrontDoor`
is the callable the RPC channel is served through; :data:`DISPATCH` maps each
verb to what it asks of the model surface.

This module imports p4p and no Channel Access server, so it is importable,
and its behaviour testable, on any host p4p installs on.

A call is taken on a p4p worker thread, and the model is not thread-safe: the
model is touched only by jobs the run loop runs, never on the thread that
received the call.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from collections.abc import Callable, Mapping
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from p4p import Value

from osprey.services.virtual_accelerator.serving.model_rpc import (
    ERR_NOT_READY,
    ERR_TIMEOUT,
    RPC_TIMEOUT_S,
    ModelRpcError,
    error_reply,
    ok_reply,
    parse_request,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from p4p.server import ServerOperation

    from osprey.services.virtual_accelerator.serving.model_rpc import RpcRequest
    from osprey.services.virtual_accelerator.serving.model_surface import ModelSurface

#: The pvAccess port a model RPC client reaches this server on when the
#: environment names none. p4p's own default, and therefore the port the
#: server binds in that case too.
DEFAULT_PVA_PORT = "5075"

#: The verbs that write. Only a refused *write* is what ``status`` reports as
#: the last refusal, so a refused read is answered and not recorded.
MODEL_WRITE_VERBS = frozenset({"set", "reset"})


def instance_name() -> str:
    """This server's instance name: the host it answers as.

    Deployed, that is the container's hostname, which compose takes from the
    service key -- ``virtual-accelerator`` for the instance the model surface
    belongs to, and its own key for a second machine standing beside it. So a
    client holding a reply can tell which server produced it without either
    server being configured with a name for itself.
    """
    return socket.gethostname()


def pva_endpoint() -> str:
    """Where a client reaches this server's model RPC: host and pvAccess port.

    The port is read from the same environment variable the pvAccess server
    binds from, so this reports the address in use rather than a second,
    separately configured copy of it that could disagree with it.
    """
    port = os.environ.get("EPICS_PVAS_SERVER_PORT", "").strip() or DEFAULT_PVA_PORT
    return f"{instance_name()}:{port}"


class RpcCall:
    """One model RPC operation, answered exactly once.

    Two threads race to answer: the run loop, once the job it was handed has
    dispatched the verb, and the timer that stops waiting for the run loop.
    p4p completes an operation once -- a second completion is an error its
    client never sees -- so the first reply here wins and the other is
    dropped.
    """

    def __init__(self, op: ServerOperation) -> None:
        self._op = op
        self._lock = threading.Lock()
        self._answered = False

    def complete(self, reply: Value) -> bool:
        """Answer with ``reply``; ``False`` if the call was already answered."""
        with self._lock:
            if self._answered:
                return False
            self._answered = True
        self._op.done(reply)
        return True


#: What each verb the contract admits asks of the model surface, given the
#: request, the surface and the driver that serves the control system's side.
DISPATCH: Mapping[str, Callable[[RpcRequest, ModelSurface, Any], Any]] = MappingProxyType(
    {
        "info": lambda request, surface, driver: surface.info(),
        "get": lambda request, surface, driver: surface.get(request.names),
        # What the control system serves for an address, read from the driver
        # that serves it -- the same driver whose existence was checked before
        # the job was enqueued.
        "diff": lambda request, surface, driver: surface.diff(driver.getParam),
        "status": lambda request, surface, driver: surface.status(),
        "set": lambda request, surface, driver: surface.set(request.values, request.token),
        "reset": lambda request, surface, driver: surface.reset(request.token),
    }
)


def surface_reply(
    surface: ModelSurface, request: RpcRequest, driver: Any, queue_depth: int
) -> Value:
    """Run ``request``'s verb on ``surface`` and return the reply; never raises.

    Called on the run loop's thread, the one place a model RPC reaches the
    model. ``queue_depth`` is the run loop's queue as it stood, which
    ``status`` reports. A refused write is recorded by the surface that
    refused it -- see ``ModelSurface._refusal`` -- so only a write that
    failed some other way is recorded here, and a refused read never is.
    """
    started = time.monotonic()
    try:
        surface.record_queue_depth(queue_depth)
        reply: Value = ok_reply(DISPATCH[request.verb](request, surface, driver))
    except ModelRpcError as exc:
        reply = error_reply(str(exc))
    except Exception as exc:  # the client is owed an answer, whatever failed
        text = f"the model surface failed on {request.verb}: {str(exc) or type(exc).__name__}"
        if request.verb in MODEL_WRITE_VERBS:
            surface.record_refusal(text)
        reply = error_reply(text)
    finally:
        surface.record_cycle((time.monotonic() - started) * 1000.0)
    return reply


class RpcFrontDoor:
    """The callable the model RPC channel is served through.

    Args:
        surface: the model surface the verbs are answered over.
        enqueue: the run loop's ``_enqueue``; the only way work reaches the
            model.
        driver: returns the driver a served value is read through, or
            ``None`` while the server is still starting.
        queue_depth: returns the run loop's queue depth; called only from a
            job, on the run loop's thread.
        timeout_s: how long a call waits for the run loop before it is
            answered with the timeout.
    """

    def __init__(
        self,
        surface: ModelSurface,
        *,
        enqueue: Callable[..., None],
        driver: Callable[[], Any | None],
        queue_depth: Callable[[], int],
        timeout_s: float = RPC_TIMEOUT_S,
    ) -> None:
        self._surface = surface
        self._enqueue = enqueue
        self._driver = driver
        self._queue_depth = queue_depth
        self._timeout_s = timeout_s

    def __call__(self, _channel: Any, op: ServerOperation) -> None:
        """Take one model RPC call and hand the run loop the work.

        A read verb is one jobs-only item that replies in its job. A write
        verb is two items: a job that dispatches and keeps the reply, then
        an empty item whose publishing pass runs before its completion sends
        the kept reply -- or the pass's error. The call is answered by
        whichever gets there first, the run loop or the timeout.
        """
        try:
            request = parse_request(op.value())
        except ModelRpcError as exc:
            op.done(error=str(exc))
            return
        driver = self._driver()
        if driver is None:
            op.done(error=ERR_NOT_READY)
            return

        call = RpcCall(op)
        timeout = threading.Timer(self._timeout_s, call.complete, (error_reply(ERR_TIMEOUT),))
        timeout.daemon = True
        timeout.start()
        if request.verb not in MODEL_WRITE_VERBS:
            self._enqueue({}, jobs=[partial(self._reply_now, request, driver, call, timeout)])
            return
        kept: list[Value] = []
        self._enqueue({}, jobs=[partial(self._keep_reply, request, driver, kept)])
        self._enqueue({}, done=partial(self._send_kept, request, call, timeout, kept))

    def _reply_now(
        self, request: RpcRequest, driver: Any, call: RpcCall, timeout: threading.Timer
    ) -> None:
        """Dispatch a read verb and answer the call, on the run loop's thread."""
        call.complete(surface_reply(self._surface, request, driver, self._queue_depth()))
        timeout.cancel()

    def _keep_reply(self, request: RpcRequest, driver: Any, kept: list[Value]) -> None:
        """Dispatch a write verb and keep its reply for the pass after it."""
        kept.append(surface_reply(self._surface, request, driver, self._queue_depth()))

    def _send_kept(
        self,
        request: RpcRequest,
        call: RpcCall,
        timeout: threading.Timer,
        kept: list[Value],
        error: str | None,
    ) -> None:
        """Answer a write with its kept reply, or with the error its pass failed on."""
        if error is not None:
            self._surface.record_refusal(error)
            reply = error_reply(error)
        elif kept:
            reply = kept[0]
        else:
            reply = error_reply(f"the model surface failed on {request.verb}")
        call.complete(reply)
        timeout.cancel()


__all__ = [
    "DEFAULT_PVA_PORT",
    "DISPATCH",
    "MODEL_WRITE_VERBS",
    "RpcCall",
    "RpcFrontDoor",
    "instance_name",
    "pva_endpoint",
    "surface_reply",
]
