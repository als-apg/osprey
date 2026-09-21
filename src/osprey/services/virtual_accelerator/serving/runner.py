"""The runner that serves the virtual accelerator's whole namespace.

One process, one model, one Channel Access server. :class:`CohostRunner`
takes the serving package's runner -- which knows how to serve a
:class:`~lume.model.LUMEModel`'s own variables -- and co-hosts the
facility's entire channel manifest alongside them, so a client sees one
namespace of ~2,900 channels rather than the handful a model happens to
describe. Everything specific to that arrangement is here; what a write to
a setpoint *means* is in
:mod:`~osprey.services.virtual_accelerator.serving.write_path`, which
imports no server library and is therefore testable without one.

**This module imports the CA server extension.** It is reached lazily
(``serving.runner``, never an eager re-export) precisely so that importing
the serving package stays cheap; see the package docstring.

Four decisions shape the subclass:

*The Channel Access namespace is contributed whole.* Every served CA PV
comes from the manifest-derived database, including the addresses that also
happen to be model variables. The base class would otherwise build a second
CA PV per model variable, from a spec derived from the variable rather than
from the manifest's record-type table -- a duplicate name that the database
merge rejects outright, and, were it not rejected, a BPM served without the
display precision its metre-scale readings need and without the boot value
the physics bridge pushed into the manifest's own spec. So the CA half of
the base class's per-variable PV creation is suppressed and the model's
variables are served on PVA alone.

*No served value derives from a model read.* The run loop's output pass --
read every variable back after a cycle and publish it -- is suppressed
entirely. Readings reach their PVs from the physics bridge instead, which
applies each BPM's seeded readout error to the value it pushes. Publishing
``model.get`` alongside that would overwrite the reading with the truth it
is supposed to differ from, and would do it a fraction of a second later, so
the divergence would come and go rather than hold.

*Every model access happens on the run loop's thread.* The model is
explicitly not thread-safe: two threads writing through it interleave their
lattice mutations and its rollback restores whichever snapshot it happens to
hold. The server thread therefore never touches it -- a co-hosted setpoint
write is enqueued and completed later from the loop -- and the physics hook
is reached through the model wrapper, so it too runs there.

*The runner claims no name of its own.* Control PVs are off: nothing is
served on Channel Access that the facility's channel manifest does not
describe. The names this process answers to beyond the manifest are
PVAccess-only and describe the model rather than the machine: the base
class's ``model_info``, and the model RPC channel the model surface is
questioned -- and, with the write token, written -- through.

A model variable the manifest serves -- see
:mod:`~osprey.services.virtual_accelerator.serving.model_surface` for where
that line is drawn -- is served on **both** transports: co-hosted on CA,
natively on PVA. A model-only variable is served on **neither**. It is
dropped from the configuration before the base constructor reads it, so no
PVA channel is built for it and the run loop does not read it back after a
cycle; the model RPC is the only way to reach it. Every other co-hosted
address -- most of the namespace, every magnet's paired ``:RB`` included --
has no model variable to build a PVA channel from, and is served on Channel
Access alone.

The model is always served through the write path's wrapper, a model with
no variables of its own included. So one model-only variable is always
there: the wrapper's stuck set, through which a stuck fault is set and
cleared at runtime -- the one fault a server with no lattice behind it can
take.

Keeping the two views of a served address in step is the write path's job,
not a model read's: every setpoint PV's put handler is replaced with one that
routes into the same write path the CA driver uses, and every value that path
publishes is committed on both views. So a write arriving on either transport
moves both, and a refused write moves neither.

The one place the two transports differ is how a client is told its write
finished, and there they differ for good reason: Channel Access completion
carries no status, so a refusal is signalled by an alarm and by the absence
of movement, while a PVA put is completed with the model's own error string.

What a client *reads* is synchronised by the same route rather than by a
second mechanism of its own. A BPM reading reaches its Channel Access PV
through the record shim the physics bridge pushes into, and that shim is
handed this module's publisher when its driver is attached, so one push
moves both views. Attaching also reconciles the two onto their boot values,
which are otherwise seeded from different places: Channel Access from the
manifest's PV specs, each PVA channel from the model variable it was built
from. So no channel this module serves moves on one transport and not on
the other -- not a setpoint, not its echo, and not a reading.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from collections.abc import Callable, Mapping
from functools import partial
from typing import TYPE_CHECKING, Any

import pcaspy
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
from osprey.services.virtual_accelerator.serving.model_surface import (
    ModelSurface,
    partition_variables,
)
from osprey.services.virtual_accelerator.serving.write_path import (
    RUNNER_CONFIG_POLICY,
    CohostWritePath,
    SetpointRoutedModel,
    physics_setpoint_addresses,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from lume.model import LUMEModel, Variable
    from p4p.server import ServerOperation

    from osprey.services.virtual_accelerator.serving.model_rpc import RpcRequest
    from osprey.services.virtual_accelerator.serving.pvdb import ServingRecords
    from osprey.services.virtual_accelerator.serving.write_path import BoundSetpoint

#: Protocols served. PVA carries the model's own variables and the
#: ``model_info`` structure; CA carries the co-hosted namespace. CA must be
#: enabled for the co-hosted database to be contributed at all.
DEFAULT_PROTOCOLS = ("ca", "pva")

#: Alarm raised on a setpoint whose write the model refused. Put-completion
#: can only ever report success, so this is the only signal a Channel Access
#: client gets that its write did not land.
REFUSAL_ALARM = (pcaspy.Alarm.WRITE_ALARM, pcaspy.Severity.INVALID_ALARM)

#: Reported to a PVA client that puts before the Channel Access driver
#: exists. The window is the tail of the base constructor: the PVA server is
#: listening from the moment it is created, and the driver every published
#: value is committed through is built after it. It is the model RPC's
#: refusal for the same window, not a second sentence that reads like it: a
#: client meets one window and should be told about it once.
NOT_READY = ERR_NOT_READY

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


def _refresh_nothing(changed: list[str]) -> None:
    """Propagate a landed model write no further.

    What a server with no physics behind it does: nothing derives a reading
    from the variables a model write moves, so there is nothing to recompute
    and nothing to push.
    """


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


def _unwrap_put(value: Any) -> Any:
    """The scalar a PVA client put, out of the structure it arrived in.

    A put carries a whole normative-type structure; its ``value`` field is
    the number written. Anything without one is passed through as-is and
    refused downstream on its own merits rather than here.
    """
    if isinstance(value, Value):
        try:
            return value["value"]
        except (KeyError, TypeError):
            return value
    return value


def _complete_put(op: ServerOperation, error: str | None) -> None:
    """Complete a PVA put, successfully or with the reason it failed.

    Unlike Channel Access, this carries a status: ``error`` is delivered to
    the client that issued the put and raised there. Called exactly once per
    put, on every outcome -- a put left uncompleted blocks its client until
    the client's own timeout expires.
    """
    op.done(error=error)


class CohostDriver(Runner.CaDriver):
    """Serves the co-hosted namespace, which the stock driver knows nothing of.

    The stock driver resolves a write's ``reason`` through the model's
    variables and refuses whatever it cannot find -- which here is every
    served PV, since the co-hosted database is the whole Channel Access
    namespace. So this intercepts *before* delegating and never delegates:
    the stock path would enqueue a bare ``model.set``, bypassing the physics
    hook that a setpoint write is supposed to run through.
    """

    def write(self, reason: str, value: Any) -> bool:
        """Delegate the whole decision to the runner's write path."""
        accepted: bool = self.runner.write_path.write(self, reason, value)
        return accepted


class CohostRunner(Runner):
    """Serves a model's variables and the facility's whole channel manifest.

    See the module docstring for the four decisions this makes, for which
    variables are served on both transports, on Channel Access alone and on
    neither, for how the two views of a doubly-served address are kept in
    step, and for the one place the two transports still differ: how a client
    is told its write finished.
    """

    ca_driver_cls = CohostDriver

    # Declared for the type checker alone: the base class sets it in its own
    # constructor, from a module that carries no annotations. No value, so
    # nothing exists at runtime that could shadow it.
    supports_ca: bool

    def __init__(
        self,
        model: LUMEModel,
        records: ServingRecords,
        *,
        on_setpoint: Callable[[str, float], None] | None = None,
        refresh: Callable[[list[str]], None] = _refresh_nothing,
        drive_limits: Mapping[str, tuple[float, float]] | None = None,
        bound_setpoints: Mapping[str, BoundSetpoint] | None = None,
        stuck_setpoints: frozenset[str] = frozenset(),
        model_write_token: str | None,
        backend_name: str,
        lattice_source: str,
        prefix: str = "",
        protocol: tuple[str, ...] = DEFAULT_PROTOCOLS,
    ) -> None:
        """Assemble the server around an already-built serving database.

        The database must be built, and every value that should be on the
        wire at boot pushed into it, *before* this runs: the CA server copies
        each spec when it creates the PV, so a value written into a spec
        afterwards is never served. That is why the physics bridge is bound
        to its records by the caller rather than here -- its first push of
        BPM readings is the boot state, and it has to land in the specs this
        constructor hands to the server.

        Args:
            model: the physics model. Its variables are served on PVA; its
                writable ones are what ``on_setpoint`` ultimately writes to.
                A model with no variables (no lattice in this process) is
                valid and serves the co-hosted namespace alone. Either way it
                is served wrapped in
                :class:`~osprey.services.virtual_accelerator.serving.write_path.SetpointRoutedModel`,
                which adds the stuck set as a model-only variable.
            records: the built serving database, from
                ``build_serving_pvdb``. Its records are pointed at the live
                driver here, once that driver exists.
            on_setpoint: the physics hook, called as
                ``on_setpoint(address, value)`` on the run loop's thread for
                each pyat-coupled setpoint write. ``None`` serves those
                setpoints as plain latches: they record what was written and
                propagate nothing, there being no physics to propagate into.
            refresh: called on the run loop's thread after a model RPC
                write lands, with the names it wrote -- the physics bridge's
                ``refresh``, which recomputes the readings those variables
                feed and pushes them. A model write reaches no setpoint, so
                nothing else in this process would notice it. The default
                propagates nothing, which is what a server with no physics
                behind it needs.
            drive_limits: ``{address: (low, high)}``; each written value is
                clamped into its band before anything else happens to it.
            bound_setpoints: what the served bindings document says each
                writable address does on readback, from
                :func:`~osprey.services.virtual_accelerator.serving.write_path.bound_setpoints`.
                Passed in rather than derived here because it takes the
                document and the model's variables, and this class resolves no
                tree: it is handed a built database and a built model. Empty
                (the default) is the lattice-free behaviour: every coupled
                readback echoes the value written, which is all a deployment
                with no document can say.
            stuck_setpoints: the apply-fault addresses stuck at boot, whose
                readbacks do not move. The write path and the wrapper both
                start from this set; a write to the wrapper's
                ``stuck_setpoints`` variable replaces it at runtime, and a
                reset of the wrapper restores it.
            model_write_token: the secret a model RPC write must present, or
                ``None`` to refuse every model write. Kept for the endpoint
                and never logged.
            backend_name: the model's backend, as the model RPC's ``status``
                verb reports it.
            lattice_source: where the model's lattice came from, as
                ``status`` reports it.
            prefix: prepended to every served name. Empty for a facility
                whose manifest addresses are already absolute.
            protocol: the protocols to serve.

        Raises:
            ValueError: Channel Access is not among the protocols. The
                co-hosted database is contributed on a hook that fires only
                for CA, so without it the facility's namespace is silently
                not served at all.
            RuntimeError: the serving database is empty, so no Channel Access
                server was created and there is nothing to serve.
        """
        if "ca" not in protocol:
            raise ValueError(
                "the co-hosted channel namespace is served over Channel Access: "
                f"'ca' must be among the protocols, got {list(protocol)}"
            )

        self._records = records
        # Kept for the model RPC endpoint. The token is what gates a model
        # write, so it is handed to the endpoint and never reaches a log.
        self._model_write_token = model_write_token
        self._backend_name = backend_name
        self._lattice_source = lattice_source
        self._refresh = refresh
        # The same set reached from the two ends of one tree: the manifest's
        # coupled partition, and the document's writable bindings. Their union
        # is what routes through the model, so a setpoint either end knows
        # about is served by the physics rather than latched -- and the
        # document's rule is what decides its readback (see the write path).
        physics_setpoints = physics_setpoint_addresses(records)
        bound = dict(bound_setpoints or {})
        routed = physics_setpoints | frozenset(bound)

        # Bound before the base constructor runs, because the base
        # constructor is what builds the driver, and the driver's first act
        # may be to consult this. `self._enqueue` is bound here and called
        # only later, once the base constructor has created the queue behind
        # it.
        self.write_path = CohostWritePath(
            records,
            enqueue=self._enqueue if on_setpoint is not None else None,
            physics_setpoints=physics_setpoints,
            bound_setpoints=bound,
            stuck_setpoints=stuck_setpoints,
            drive_limits=drive_limits,
            refusal_alarm=REFUSAL_ALARM,
            pva_post=self._post_pva,
        )

        # Wrapped whether or not there is a physics hook, a model with no
        # lattice behind it included: the wrapper carries the stuck set and
        # hands each new one to the write path, which is what decides the
        # next write to a setpoint. With no hook it routes nothing.
        model = SetpointRoutedModel(
            model,
            on_setpoint=on_setpoint,
            routed=routed,
            stuck_setpoints=stuck_setpoints,
            # What the write path actually routes, so the refusal a runtime
            # write meets names an address this server has no setpoint for
            # and nothing else.
            known_setpoints=self.write_path.setpoints,
            on_stuck_change=self.write_path.set_stuck_setpoints,
        )

        config = Runner.generate_config(model, prefix=prefix)
        config["protocol"] = list(protocol)
        config.update(RUNNER_CONFIG_POLICY)

        # A model-only variable is served on neither transport. The base
        # constructor builds a channel for every configured name, rejecting
        # names the model lacks but never names the configuration omits, so
        # leaving them out of the configuration is the whole of it. The
        # partition is of the model the base class is handed, wrapper
        # included, so a variable the wrapper declares lands on a side too.
        self.partition = partition_variables(model, records)
        for name in self.partition.model_only:
            del config["variables"][name]

        super().__init__(model=model, config=config)

        if self.ca_driver is None:
            raise RuntimeError(
                "no Channel Access server was created: the serving database is empty, "
                "so there is nothing for this runner to serve"
            )
        # Last, and only now: until the driver exists a record writes into
        # its spec (which is how the boot values above got there), and after
        # the server has created the PVs a spec write reaches nobody. The
        # publisher goes in here for the same reason it goes into the write
        # path above -- a value source pushes a reading once, and both views
        # of that address have to carry it.
        records.attach_driver(self.ca_driver, pva_post=self._post_pva)

    def _create_model_info(self) -> None:
        """Serve the model surface's RPC channel beside the base class's model info.

        This hook is the seam the RPC channel has to be built on. It runs
        late enough that everything the surface answers for exists -- the
        model, the partition, and every model variable's own PVA channel --
        and early enough that ``self.providers`` is still a dictionary the
        server has not been handed: a provider added after the base
        constructor creates the server is never served at all.

        The surface is built here rather than in the constructor for the
        same reason, and kept for the handler alone: nothing else in this
        runner reads the model.
        """
        super()._create_model_info()

        self._surface = ModelSurface(
            self.model,
            self.partition,
            self._records,
            backend_name=self._backend_name,
            lattice_source=self._lattice_source,
            instance=_instance_name(),
            endpoint=_pva_endpoint(),
            update_rate=self.update_rate,
            model_write_token=self._model_write_token,
            refresh=self._refresh,
        )

        # Open, because a closed channel refuses every operation, and holding
        # a resting value nobody reads: a reply is returned to the caller
        # that asked for it, never posted here for everyone.
        channel = SharedPV(initial=REPLY_TYPE.wrap(""))
        channel.rpc(self._rpc)
        self.providers[f"{self.config['prefix']}{RPC_PV}"] = channel

    def _rpc(self, _channel: SharedPV, op: ServerOperation) -> None:
        """Take one model RPC call, and hand the run loop the work.

        Runs on a p4p worker thread, which is no more allowed to touch the
        model than a put is: the verb is dispatched by a job the run loop
        runs, and this returns without waiting for it. Two refusals are
        answered here instead, because neither needs the model: a request
        that is not a request, and one that arrives before the driver the
        server commits values through exists.

        The call is answered by whichever gets there first -- the job, or
        the timer that stops waiting for it -- so a run loop that never
        reaches the job still answers its client, with the reason, instead
        of leaving it on its own timeout.
        """
        try:
            request = parse_request(op.value())
        except ModelRpcError as exc:
            # Nothing was dispatched: this is not the model refusing a call,
            # it is the call not being one.
            op.done(error=str(exc))
            return

        driver = self.ca_driver
        if driver is None:
            op.done(error=ERR_NOT_READY)
            return

        call = _RpcCall(op)
        timeout = threading.Timer(RPC_TIMEOUT_S, call.complete, (error_reply(ERR_TIMEOUT),))
        # Daemon, so a call still being waited on never holds up a shutdown.
        timeout.daemon = True
        timeout.start()
        self._enqueue({}, jobs=[partial(self._answer, request, driver, call, timeout)])

    def _answer(
        self,
        request: RpcRequest,
        driver: CohostDriver,
        call: _RpcCall,
        timeout: threading.Timer,
    ) -> None:
        """Dispatch one call's verb and answer it, on the run loop's thread.

        Dispatch is the one place a model RPC reaches the model, and it is
        reached from here alone. Every path replies: the
        run loop logs a job that raises and moves on to the next item, so a
        job that returned without answering would leave its client waiting
        out its own timeout with nothing to show for it.

        The reply goes out before the timer is cancelled, and not after,
        because an answer this method cannot produce is better delivered
        late by the timer than not at all.
        """
        surface = self._surface
        started = time.monotonic()
        try:
            surface.record_queue_depth(self.queue.qsize())
            reply = ok_reply(self._dispatch(request, surface, driver))
        except ModelRpcError as exc:
            # A refused write is recorded by the surface that refused it --
            # see ``ModelSurface._refusal`` -- so it is not recorded again
            # here, where a refused read would be recorded as a write.
            reply = error_reply(str(exc))
        except Exception as exc:  # noqa: BLE001 - the client is owed an answer, whatever failed
            text = f"the model surface failed on {request.verb}: {str(exc) or type(exc).__name__}"
            if request.verb in MODEL_WRITE_VERBS:
                surface.record_refusal(text)
            reply = error_reply(text)
        finally:
            surface.record_cycle((time.monotonic() - started) * 1000.0)
            call.complete(reply)
            timeout.cancel()

    def _dispatch(self, request: RpcRequest, surface: ModelSurface, driver: CohostDriver) -> Any:
        """Run ``request``'s verb on ``surface``, and return what it answered.

        Every verb the contract admits is dispatched here and nowhere else;
        ``parse_request`` has already refused anything that is not one of
        them, which is why the last verb needs no test of its own.
        """
        verb = request.verb
        if verb == "info":
            return surface.info()
        if verb == "get":
            return surface.get(request.names)
        if verb == "diff":
            # What the control system serves for an address, read from the
            # driver that serves it -- the same driver whose existence was
            # checked before this job was enqueued.
            return surface.diff(driver.getParam)
        if verb == "status":
            return surface.status()
        if verb == "set":
            return surface.set(request.values, request.token)
        return surface.reset(request.token)

    def _extend_pvdb(self) -> dict[str, dict[str, Any]]:
        """Contribute the whole co-hosted database.

        Called by the base constructor before the server is created, which is
        the only point at which the database can still be added to.
        """
        return dict(self._records.pvdb)

    def _add_pv(self, pv: str, var: Variable, ro: bool, prefix: str, handler: Any) -> None:
        """Serve one model variable on PVA only, and route its puts.

        The base implementation builds a PVA provider and a CA database entry
        from the same call, choosing each half by protocol. The CA half is
        suppressed here because the manifest already describes these
        addresses and contributes them itself -- see the module docstring.
        The suppression is done by scoping the flag the base implementation
        consults rather than by reimplementing the PVA half, so the PVA
        provider stays the base class's own code instead of a copy of it that
        would quietly drift.

        The writable half of the PVA provider is then re-pointed at the
        co-hosted write path. The stock put handler would enqueue a bare
        ``model.set`` for this one variable -- bypassing the physics hook, the
        drive-limit clamp this facility's bands come from, and the Channel
        Access view of the same address entirely. Installing the replacement
        through p4p's own ``put`` decorator is how the base class installs its
        reset handler, so nothing here reaches inside a fork of it.
        """
        supports_ca: bool = self.supports_ca
        self.supports_ca = False
        try:
            super()._add_pv(pv, var, ro=ro, prefix=prefix, handler=handler)
        finally:
            self.supports_ca = supports_ca

        channel: SharedPV | None = self.pvs.get(var.name)
        if ro or channel is None:
            # Read-only, or no PVA channel at all because PVA is not among
            # the served protocols. The stock handler refuses a put to a
            # read-only PV, which is what a Channel Access write to the same
            # address gets too.
            return
        channel.put(partial(self._put, var.name))

    def _put(self, address: str, _channel: SharedPV, op: ServerOperation) -> None:
        """Route a PVA put through the write path a CA write goes through.

        Runs on a p4p worker thread, which is no more allowed to touch the
        model than the Channel Access server thread is: like a write, a put
        that needs physics is enqueued here and completed later from the run
        loop. The published echo is the base class's own ``post``, reached
        through :meth:`_post_pva`, so this never posts a value of its own.
        """
        driver = self.ca_driver
        if driver is None:
            _complete_put(op, NOT_READY)
            return
        self.write_path.put(
            driver, address, _unwrap_put(op.value()), done=partial(_complete_put, op)
        )

    def _post_pva(self, address: str, value: Any) -> None:
        """Publish an accepted value on the PVA channel serving ``address``.

        Addresses the model does not describe -- most of the co-hosted
        namespace, including every magnet's paired ``:RB`` -- have no PVA
        channel and are silently skipped: there is no second view of them to
        keep in step. ``value`` is the client's own post-clamp value, packed
        into this variable's structure; nothing here reads the model.
        """
        channel: SharedPV | None = self.pvs.get(address)
        if channel is None:
            return
        channel.post(self._generate_value(address, value))

    def _post_outputs(self, out_values: dict[str, Any], ts: float) -> None:
        """Publish nothing.

        The run loop reads the served variables back at the end of a cycle
        (see :meth:`_cycle_output_names`) and offers them here to be
        published. Nothing is: a served reading comes
        from the physics bridge, which applies each BPM's seeded readout
        error on the way out, and publishing the model's own view alongside
        it would overwrite that reading with the truth it exists to differ
        from. Setpoints are published by the write path from the value the
        client wrote and the model accepted, which is likewise not a value
        read back out of the model.
        """

    def _cycle_output_names(self) -> list[str]:
        """The variables the run loop reads back after a cycle: the served ones.

        The base class reads back the model's whole roster. A model-only
        variable has no channel on either transport, so reading it back
        after every cycle would be a model read with nowhere to go. The
        served ones are read and then published nowhere either -- see
        :meth:`_post_outputs` -- but that is the output pass's decision, not
        the roster's.
        """
        return list(self.partition.served)

    def _reset_to_cached_state(self) -> None:
        """Roll nothing back.

        The run loop snapshots the model's settable variables before each
        cycle and writes them all back if the cycle fails. Here that would be
        actively harmful: the model already rolls a failed write back
        atomically and completely, and re-writing hundreds of retained
        currents through the physics hook would re-run each one's calibration
        and re-solve the lattice -- turning a refused write, which is
        supposed to be a complete no-op, into a machine-wide one.
        """


__all__ = [
    "DEFAULT_PROTOCOLS",
    "REFUSAL_ALARM",
    "CohostDriver",
    "CohostRunner",
]
