"""The runner that serves a composite on Channel Access and PVAccess.

:class:`ModelRunner` serves a composite: every channel the composite
declares is one of its variables, so the base class serves them all on both
transports from one configuration, and the model RPC is answered over the
composite's simulator view
(:meth:`~osprey.services.virtual_accelerator.serving.model_surface.ModelSurface.for_view`).

**This module imports the CA server extension**, through the serving
package (``lume_pva_apg``) it builds on. It is reached lazily
(``serving.runner``, never an eager re-export) precisely so that importing
the serving package stays cheap; see the package docstring.

Every model access happens on the run loop's thread. The composite is not
thread-safe, so a model RPC call taken on a p4p worker thread is answered by
a job the run loop runs, never on the thread that received it.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from collections.abc import Mapping
from functools import partial
from typing import TYPE_CHECKING, Any

from lume_pva_apg.runner import Runner
from p4p import Value
from p4p.server.thread import SharedPV

from osprey.services.virtual_accelerator.serving.model_rpc import (
    ERR_NOT_READY,
    ERR_TIMEOUT,
    REPLY_TYPE,
    RPC_PV,
    RPC_TIMEOUT_S,
    ModelRpcError,
    error_reply,
    ok_reply,
    parse_request,
)
from osprey.services.virtual_accelerator.serving.model_surface import ModelSurface
from osprey.services.virtual_accelerator.serving.runner_config import (
    HEALTH_KEYS,
    apply_safety,
    chromaticity_addresses,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from lume.model import LUMEModel
    from p4p.server import ServerOperation

    from osprey.services.virtual_accelerator.serving.model_rpc import RpcRequest

#: The pvAccess port a model RPC client reaches this server on when the
#: environment names none. p4p's own default, and therefore the port the
#: server binds in that case too.
DEFAULT_PVA_PORT = "5075"

#: The verbs that write. Only a refused *write* is what ``status`` reports as
#: the last refusal, so a refused read is answered and not recorded.
MODEL_WRITE_VERBS = frozenset({"set", "reset"})


def _instance_name() -> str:
    """This server's instance name: the host it answers as.

    Deployed, that is the container's hostname, which compose takes from the
    service key -- ``virtual-accelerator`` for the instance the model surface
    belongs to, and its own key for a second machine standing beside it. So a
    client holding a reply can tell which server produced it without either
    server being configured with a name for itself.
    """
    return socket.gethostname()


def _pva_endpoint() -> str:
    """Where a client reaches this server's model RPC: host and pvAccess port.

    The port is read from the same environment variable the pvAccess server
    binds from, so this reports the address in use rather than a second,
    separately configured copy of it that could disagree with it.
    """
    port = os.environ.get("EPICS_PVAS_SERVER_PORT", "").strip() or DEFAULT_PVA_PORT
    return f"{_instance_name()}:{port}"


class _RpcCall:
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


def _surface_reply(
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
        reply: Value = ok_reply(_dispatch(request, surface, driver))
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


def _dispatch(request: RpcRequest, surface: ModelSurface, driver: Any) -> Any:
    """Run ``request``'s verb on ``surface``, and return what it answered.

    Every verb the contract admits is dispatched here and nowhere else;
    ``parse_request`` has already refused anything that is not one of them,
    which is why the last verb needs no test of its own.
    """
    verb = request.verb
    if verb == "info":
        return surface.info()
    if verb == "get":
        return surface.get(request.names)
    if verb == "diff":
        # What the control system serves for an address, read from the
        # driver that serves it -- the same driver whose existence was
        # checked before the job was enqueued.
        return surface.diff(driver.getParam)
    if verb == "status":
        return surface.status()
    if verb == "set":
        return surface.set(request.values, request.token)
    return surface.reset(request.token)


class ModelRunner(Runner):
    """Serves a composite's channels on both transports, and its model RPC on PVA.

    The configuration is ``Runner.generate_config`` over the composite with
    the view's write safety applied
    (:func:`~osprey.services.virtual_accelerator.serving.runner_config.apply_safety`).
    A write pass reads back every served channel except those wired to the
    chromaticity output, which the next periodic pass publishes; a pass with
    no input values reads them all.
    A failed pass rolls the composite back to the state cached before it only
    when ``model.set`` had already succeeded: a refused ``set`` leaves the
    composite as it was. The model RPC's write verbs reply only after a
    publishing pass has run. Every pass that fails is recorded for
    ``status``. Every publishing pass's outcome, success or failure, is
    recorded in the surface's health record, which ``status`` reports.
    """

    def __init__(
        self,
        composite: LUMEModel,
        view: Mapping[str, Any],
        addresses_json: Mapping[str, Any],
        *,
        model_write_token: str | None,
        tick_interval_s: float | None = None,
        instance: str | None = None,
        failed_pass_tolerance: int | None = None,
    ) -> None:
        """Serve ``composite``, built from the simulator view ``view`` describes.

        Args:
            composite: the composite over the view.
            view: the view's ``variables.json`` document.
            addresses_json: the view's ``addresses.json`` document.
            model_write_token: the secret a model RPC write must present, or
                ``None`` to refuse every model write. Never logged.
            tick_interval_s: the period of the runner's own passes, or
                ``None`` for none.
            instance: the instance name the model RPC's ``status`` reports,
                or ``None`` for the host this server answers as.
            failed_pass_tolerance: how many consecutive failed publishing
                passes the health record still counts as ``degraded``, or
                ``None`` for the default of
                :data:`~osprey.services.virtual_accelerator.serving.runner_config.HEALTH_KEYS`.
        """
        self._addresses_json = addresses_json
        self._instance = instance
        self._model_write_token = model_write_token
        self._chromaticity = chromaticity_addresses(view)
        self._write_pass = False
        self._set_landed = False
        self._pass_started = False
        self._passes_run = 0
        self._first_pass_error: str | None = None
        config = apply_safety(Runner.generate_config(composite, prefix=""), view)
        if tick_interval_s is not None:
            config["tick_interval_s"] = tick_interval_s
        config["failed_pass_tolerance"] = (
            failed_pass_tolerance
            if failed_pass_tolerance is not None
            else HEALTH_KEYS["failed_pass_tolerance"]
        )
        super().__init__(model=composite, config=config)

    def first_pass(self) -> str | None:
        """Run the start-up publishing pass; the error it failed on, or ``None``.

        Call it before :meth:`run`, on the thread that will call :meth:`run`,
        so every model access stays on one thread. It runs queued items until
        one has run a publishing pass. The start-up item is always queued, and
        a jobs-only item ahead of it runs no pass, so it returns once that item
        has run.
        """
        while self._passes_run < 1:
            self._run_cycle(self.queue.get())
        return self._first_pass_error

    def _run_cycle(self, item: dict[str, Any]) -> None:
        """Note whether this pass carries input values, then run it and record its outcome."""
        self._write_pass = bool(item["values"])
        self._set_landed = False
        self._pass_started = False
        item["done"].append(self._record_pass)
        super()._run_cycle(item)

    def _set_cached_state(self, state: dict[str, Any]) -> None:
        """Note that this cycle runs a publishing pass, then cache ``state``."""
        self._pass_started = True
        super()._set_cached_state(state)

    def _record_pass(self, error: str | None) -> None:
        """Record a publishing pass's outcome; a cycle that ran no pass records nothing."""
        if not self._pass_started:
            return
        if self._passes_run == 0:
            self._first_pass_error = error
        self._passes_run += 1
        self._surface.record_pass(error)

    def _cycle_output_names(self) -> list[str]:
        """The roster read after ``model.set``; a write pass leaves the chromaticity out."""
        self._set_landed = True
        roster: list[str] = super()._cycle_output_names()
        if not self._write_pass:
            return roster
        return [name for name in roster if name not in self._chromaticity]

    def _reset_to_cached_state(self) -> None:
        """Restore the cached state only when the pass failed after ``model.set`` succeeded."""
        if self._set_landed:
            super()._reset_to_cached_state()

    def _create_model_info(self) -> None:
        """Serve the model RPC channel beside the base class's model info."""
        super()._create_model_info()
        self._surface = ModelSurface.for_view(
            self.model,
            self._addresses_json,
            instance=self._instance if self._instance is not None else _instance_name(),
            endpoint=_pva_endpoint(),
            model_write_token=self._model_write_token,
            failed_pass_tolerance=self.config["failed_pass_tolerance"],
        )
        channel = SharedPV(initial=REPLY_TYPE.wrap(""))
        channel.rpc(self._rpc)
        self.providers[f"{self.config['prefix']}{RPC_PV}"] = channel

    def _rpc(self, _channel: SharedPV, op: ServerOperation) -> None:
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
        driver = self.ca_driver
        if driver is None:
            op.done(error=ERR_NOT_READY)
            return

        call = _RpcCall(op)
        timeout = threading.Timer(RPC_TIMEOUT_S, call.complete, (error_reply(ERR_TIMEOUT),))
        timeout.daemon = True
        timeout.start()
        if request.verb not in MODEL_WRITE_VERBS:
            self._enqueue({}, jobs=[partial(self._reply_now, request, driver, call, timeout)])
            return
        kept: list[Value] = []
        self._enqueue({}, jobs=[partial(self._keep_reply, request, driver, kept)])
        self._enqueue({}, done=partial(self._send_kept, request, call, timeout, kept))

    def _reply_now(
        self, request: RpcRequest, driver: Any, call: _RpcCall, timeout: threading.Timer
    ) -> None:
        """Dispatch a read verb and answer the call, on the run loop's thread."""
        call.complete(_surface_reply(self._surface, request, driver, self.queue.qsize()))
        timeout.cancel()

    def _keep_reply(self, request: RpcRequest, driver: Any, kept: list[Value]) -> None:
        """Dispatch a write verb and keep its reply for the pass after it."""
        kept.append(_surface_reply(self._surface, request, driver, self.queue.qsize()))

    def _send_kept(
        self,
        request: RpcRequest,
        call: _RpcCall,
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


__all__ = ["ModelRunner"]
