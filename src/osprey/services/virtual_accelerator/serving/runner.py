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

**Timed passes.** A view holding a ``fast`` or ``event`` channel gets one
scheduler thread beside the base class's ticker. It queues a timed pass at
the earlier of the next fast period and the composite's next deadline, never
more than one at a time, as a jobs-only queue item: the run loop runs the job
and no pass of its own. The job advances the composite, reads the names the
composite emitted and, when the fast period fell due, the ``fast`` names, and
posts them with a forced Channel
Access event (:mod:`~osprey.services.virtual_accelerator.serving.publication`).
Every other pass posts the two classes changed-only, except the names it
drained as emitted, so a step is published once, by whichever pass advanced
over it.

**Sliced posts.** A pass over a large view takes longer to post than the fast
period lasts, and the run loop runs one item at a time. So in a view with a
timed channel a pass posts :data:`POST_SLICE` values at a time, the timed and
forced names first, and between slices runs a timed pass if one fell due. The
names one drain of the composite's emission returned are one step, so they all
go in the first slice, however many there are. A
name that timed pass posted is dropped from the slices still to come, whose
value for it is the older one. The read that precedes the post is not sliced:
a timed pass waits for it.

**Queued writes.** The runner opens no batching window, so a write that
arrives alone is served at once. The plain writes already queued behind it
when its cycle starts, each to a channel the pass does not hold yet, are
served by that same pass: one ``model.set``, one readback, one post, and
every put completed by it. Merging stops at the first queued item that is
anything else, which is served next as it was queued, so nothing is served
out of its arrival order. A merged pass that fails is rolled back and its
writes are served one cycle each, so every put gets its own verdict.

Every model access happens on the run loop's thread. The composite is not
thread-safe, so a model RPC call taken on a p4p worker thread is answered by
a job the run loop runs, never on the thread that received it; that rule is
kept by the RPC handler,
:mod:`~osprey.services.virtual_accelerator.serving.model_rpc_handler`.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Collection
from functools import partial
from pathlib import Path
from queue import Empty
from typing import TYPE_CHECKING, Any

from lume_pva_apg.runner import Runner
from p4p.server.thread import SharedPV

from osprey.services.virtual_accelerator.serving.health import write_document
from osprey.services.virtual_accelerator.serving.model_rpc import REPLY_TYPE, RPC_PV
from osprey.services.virtual_accelerator.serving.model_rpc_handler import (
    RpcFrontDoor,
    instance_name,
    pva_endpoint,
)
from osprey.services.virtual_accelerator.serving.model_surface import ModelSurface
from osprey.services.virtual_accelerator.serving.publication import (
    changed_outputs,
    forced_names,
    next_wake,
    timed_pass_names,
    untimed_outputs,
)
from osprey.services.virtual_accelerator.serving.runner_config import (
    HEALTH_KEYS,
    apply_safety,
    event_addresses,
    fast_addresses,
    pass_readback,
    periodic_addresses,
    write_records,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from lume.model import LUMEModel

    from osprey_connectors.simulation.view import SimulatorView

LOG = logging.getLogger(__name__)

#: The name of the thread that queues timed passes.
SCHEDULER_THREAD = "osprey-va-timed-passes"

#: How many values one slice of a pass's post carries in a view with a timed
#: channel: few enough that a slice posts well inside the default fast period.
POST_SLICE = 2000


class ModelRunner(Runner):
    """Serves a composite's channels on both transports, and its model RPC on PVA.

    The configuration is ``Runner.generate_config`` over the composite with
    the view's write safety applied
    (:func:`~osprey.services.virtual_accelerator.serving.runner_config.apply_safety`).
    A write pass reads back every served channel except those the view says
    refresh ``periodic``, which the next periodic pass publishes. A write
    pass of physics settings alone reads back only the settings, their
    readbacks and what their models compute from their settings, the one
    part of the composite a setting changes
    (:func:`~osprey.services.virtual_accelerator.serving.runner_config.pass_readback`).
    Of what it reads, a write pass publishes the written channels and those
    whose value or undefined condition changed since they were last
    published; a pass with no input values reads and publishes them all.
    The writes queued when a write's cycle starts share its pass
    (:meth:`_run_cycle`).
    A failed pass rolls the composite back to the state cached before it only
    when ``model.set`` had already succeeded: a refused ``set`` leaves the
    composite as it was, and a model with ``restore`` takes the cached state
    back through it, so no engine sees the rollback as a client write. The
    model RPC's write verbs reply only after a publishing pass has run. Every
    publishing pass's outcome, success or failure, is recorded in the
    surface's health record, which ``status`` reports and which is rewritten
    to the health file after every pass when one is named. A timed pass
    records its outcome only when it differs from the last one recorded.
    """

    _cycle_requested_names: tuple[str, ...]

    #: When the ``fast`` names were last posted by a timed pass.
    _fast_posted_at: float = 0.0
    #: When a timed pass run between the slices of a post last posted them.
    _fast_sliced_at: float = 0.0
    #: Whether a sliced post is under way, so a timed pass run between its
    #: slices posts whole.
    _slicing: bool = False

    def __init__(
        self,
        composite: LUMEModel,
        view: SimulatorView,
        *,
        model_write_token: str | None,
        tick_interval_s: float | None = None,
        instance: str | None = None,
        health_file: Path | None = None,
        failed_pass_tolerance: int | None = None,
        fast_interval_s: float = 0.1,
    ) -> None:
        """Serve ``composite``, built from the simulator view ``view``.

        Args:
            composite: the composite over the view.
            view: the simulator view.
            model_write_token: the secret a model RPC write must present, or
                ``None`` to refuse every model write. Never logged.
            tick_interval_s: the period of the runner's own passes, or
                ``None`` for none.
            instance: the instance name the model RPC's ``status`` reports,
                or ``None`` for the host this server answers as.
            health_file: where the health record is rewritten, atomically,
                after every publishing pass, or ``None`` to keep it in
                process only.
            failed_pass_tolerance: how many consecutive failed publishing
                passes the health record still counts as ``degraded``, or
                ``None`` for the default of
                :data:`~osprey.services.virtual_accelerator.serving.runner_config.HEALTH_KEYS`.
            fast_interval_s: the fast period in seconds, the longest a timed
                pass waits for the next; used only when the view holds a
                ``fast`` or ``event`` channel.

        Raises:
            ValueError: ``fast_interval_s`` is not a number greater than zero.
        """
        if (
            isinstance(fast_interval_s, bool)
            or not isinstance(fast_interval_s, int | float)
            or not fast_interval_s > 0
        ):
            raise ValueError(
                f"the fast period must be a number greater than zero, got {fast_interval_s!r}"
            )
        self._fast_interval_s = float(fast_interval_s)
        self._view = view
        self._instance = instance
        self._health_file = health_file
        self._model_write_token = model_write_token
        self._periodic = periodic_addresses(view)
        texture_reach = getattr(composite, "texture_reach", None)
        self._setting_records = write_records(
            view, texture_reach() if callable(texture_reach) else {}
        )
        self._written: frozenset[str] = frozenset()
        self._write_pass = False
        self._served: dict[str, Any] = {}
        self._set_landed = False
        self._pass_started = False
        self._passes_run = 0
        self._first_pass_error: str | None = None
        self._fast = fast_addresses(view)
        self._timed_names = self._fast | event_addresses(view)
        self._needs_timed = bool(self._timed_names)
        self._forced: frozenset[str] = frozenset()
        self._last_ok: bool | None = None
        self._deadline: float | None = None
        self._timed_lock = threading.Lock()
        self._timed_pending = False
        self._scheduler: threading.Thread | None = None
        self._scheduler_stop: threading.Event | None = None
        self._rearm = threading.Event()
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
        """Serve ``item``, a write together with the plain writes queued behind it.

        A plain write carries values and nothing else: no reset, no job, no
        tick. When ``item`` is one, the plain writes already in the queue are
        taken into its pass, up to the first item that is not one or that
        writes a channel the pass already holds. That item is served next,
        unchanged, by the same rule, so arrival order holds and a channel
        written twice gets a pass per value. Nothing is waited for: an empty
        queue leaves ``item`` a pass of its own.

        A merged pass that succeeds completes every put it served. One that
        fails is rolled back as any failed pass is, records no outcome, and
        its writes are then served one cycle each in arrival order, so a put
        is refused only when its own write is.
        """
        following: dict[str, Any] | None = item
        while following is not None:
            item = following
            merged, following = self._queued_writes(item)
            if not merged:
                self._serve(item)
                continue
            writes = [item, *merged]
            outcome: list[str | None] = []
            self._serve(
                {
                    "values": {
                        name: value for write in writes for name, value in write["values"].items()
                    },
                    "done": [outcome.append],
                    "reset": False,
                    "jobs": [],
                    "tick": False,
                },
                record=False,
            )
            if outcome != [None]:
                for write in writes:
                    self._serve(write)
                continue
            for write in writes:
                for done in write["done"]:
                    try:
                        done(None)
                    except Exception as exc:
                        LOG.error("Error signalling put-completion: %s", exc)
            self._record_pass(None)

    def _queued_writes(
        self, item: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        """The queued plain writes ``item``'s pass takes, and the item taken off after them.

        The queue cannot be looked into, so the item that ends the merge is
        taken off it and returned for the caller to serve next; ``None`` when
        the queue ran empty first. An ``item`` that is not a plain write takes
        nothing.
        """
        merged: list[dict[str, Any]] = []
        if not _is_plain_write(item):
            return merged, None
        held = set(item["values"])
        while True:
            try:
                queued = self.queue.get_nowait()
            except Empty:
                return merged, None
            if not _is_plain_write(queued) or not held.isdisjoint(queued["values"]):
                return merged, queued
            held.update(queued["values"])
            merged.append(queued)

    def _serve(self, item: dict[str, Any], *, record: bool = True) -> None:
        """Note the input values this pass writes, then run it and record its outcome.

        ``record`` false leaves the outcome to the caller.
        """
        self._values = item["values"]
        self._set_landed = False
        self._pass_started = False
        if record:
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
        self._write_health(error)

    def _write_health(self, error: str | None) -> None:
        """Record ``error`` as the latest pass outcome and rewrite the health file."""
        self._last_ok = error is None
        document = self._surface.record_pass(error)
        if self._health_file is not None:
            write_document(self._health_file, document)

    def _cycle_snapshot_names(self, written: Collection[str], reset: bool) -> list[str]:
        """Snapshot only what a pass writes; the whole settable state for one that resets.

        The composite is put back by restoring the written setpoints, so a
        failed pass needs no more than their previous values, and a pass that
        writes nothing needs none.
        """
        if reset:
            names: list[str] = super()._cycle_snapshot_names(written, reset)
            return names
        variables = self.model.supported_variables
        return [name for name in written if name in variables and not variables[name].read_only]

    def _cycle_output_names(self) -> list[str]:
        """The roster read after ``model.set``, narrowed for a write pass.

        A write pass leaves the periodic addresses out, and reads only the
        records its setpoints can change when every one of them is known. The input
        values are read again here, after the base class merged the batch, so
        a write merged behind a jobs-only item still makes a write pass.
        """
        self._set_landed = True
        self._written = frozenset(getattr(self, "_values", ()))
        self._write_pass = bool(self._written)
        roster: list[str] = super()._cycle_output_names()
        if not self._write_pass:
            return roster
        changed = pass_readback(self._setting_records, self._written)
        return [
            name
            for name in roster
            if name not in self._periodic and (changed is None or name in changed)
        ]

    def _post_outputs(self, out_values: dict[str, Any], ts: float) -> None:
        """Post a pass's values; a write pass posts only what a client does not hold yet.

        A write pass posts the written channels and those whose value or
        undefined condition changed since they were last posted; any other
        pass posts every value but the ``fast`` and ``event`` ones, which it
        posts only when changed. Either way the names the composite emitted
        are drained here, read when the pass did not, posted, and forced as a
        Channel Access event, unless a read fault holds them; this pass
        published their step, so no timed pass posts it again. A name is
        recorded as posted only once the base class has published the pass,
        so a pass that fails before it publishes leaves the record as the
        wire is.
        """
        emitted = self._drain_emitted()
        missing = sorted(name for name in emitted if name not in out_values)
        if missing:
            out_values = {**out_values, **self.model.get(missing)}
            self._cycle_requested_names = (*self._cycle_requested_names, *missing)
        faulted = self._read_faulted()
        posting = out_values
        if self._write_pass or self._timed_names or emitted:
            udf_now = self._evaluate_severity(list(out_values))
            posting = untimed_outputs(
                out_values,
                self._served,
                self._timed_names,
                emitted - faulted,
                udf_now,
                self._udf,
                written=self._written if self._write_pass else None,
            )
        self._post_forced(posting, ts, forced_names((), emitted, faulted), emitted)
        self._note_deadline()

    def _post_forced(
        self,
        values: dict[str, Any],
        ts: float,
        forced: frozenset[str],
        emitted: frozenset[str] = frozenset(),
    ) -> None:
        """Post ``values``, forcing ``forced``; in slices when a timed pass could fall due.

        A view with no timed channel, and a post of one slice or less, posts
        whole. Otherwise the ``emitted`` names, one drain of the composite's
        emission and so one step, go first and all in the first slice however
        many there are, so no timed pass posts a later step between two of
        them; the other timed and forced names follow. Before each later slice
        a due timed pass runs; what it posted is left out of the rest, which
        holds the value read before it.
        """
        if not self._needs_timed or self._slicing or len(values) <= POST_SLICE:
            self._post_slice(values, ts, forced)
            return
        together = [name for name in values if name in emitted]
        first = together + [
            name
            for name in values
            if name not in emitted and (name in self._timed_names or name in forced)
        ]
        ahead = set(first)
        order = first + [name for name in values if name not in ahead]
        head = max(POST_SLICE, len(together))
        requested = self._cycle_requested_names
        superseded: set[str] = set()
        self._slicing = True
        try:
            for start in (0, *range(head, len(order), POST_SLICE)):
                if start:
                    superseded |= self._timed_between_slices()
                end = head if start == 0 else start + POST_SLICE
                part = {name: values[name] for name in order[start:end] if name not in superseded}
                self._cycle_requested_names = tuple(part)
                self._post_slice(part, ts, forced & part.keys())
        finally:
            self._slicing = False
            self._cycle_requested_names = requested

    def _post_slice(self, values: dict[str, Any], ts: float, forced: frozenset[str]) -> None:
        """Post ``values`` through the base class, forcing ``forced``; record them as served."""
        self._forced = forced
        try:
            Runner._post_outputs(self, values, ts)
        finally:
            self._forced = frozenset()
        self._served.update(values)

    def _timed_between_slices(self) -> frozenset[str]:
        """Run a timed pass if one is due; the names it posted, else none."""
        now = time.time()
        fast_due = now - self._fast_posted_at >= self._fast_interval_s
        if not fast_due and (self._deadline is None or self._deadline > now):
            return frozenset()
        requested = self._cycle_requested_names
        try:
            posted = self._run_timed(fast_due)
        finally:
            self._cycle_requested_names = requested
        if fast_due:
            self._fast_sliced_at = self._fast_posted_at
        return posted

    def _ca_forced_names(self) -> Collection[str]:
        """The names this post forces a Channel Access event for: set per post, else none."""
        return self._forced

    def _drain_emitted(self) -> frozenset[str]:
        """The names the composite emitted since it was last asked; none for a model without."""
        emitted = getattr(self.model, "emitted", None)
        return frozenset(emitted()) if callable(emitted) else frozenset()

    def _read_faulted(self) -> frozenset[str]:
        """The served names a ``frozen`` or ``disconnected`` read fault holds now.

        The composite keeps its active channel faults by address; a fault
        with a ``read`` changes the reading. A model without them has none.
        """
        faulted = getattr(self.model, "_faulted", None) or {}
        return frozenset(
            address for address, fault in faulted.items() if getattr(fault, "read", None)
        )

    def _note_deadline(self) -> None:
        """Take the composite's next deadline on the run loop; wake the scheduler if it moved."""
        if not self._needs_timed:
            return
        next_deadline = getattr(self.model, "next_deadline", None)
        deadline = next_deadline() if callable(next_deadline) else None
        if deadline != self._deadline:
            self._deadline = deadline
            self._rearm.set()

    def _timed_pass(self, fast_due: bool = True) -> None:
        """One timed pass, run on the run loop as a job; frees the scheduler to queue the next.

        A pass queued for the fast period while a sliced post was serving it
        finds the ``fast`` names posted between slices less than half a period
        ago, and leaves them to the next period.
        """
        if fast_due and time.time() - self._fast_sliced_at < self._fast_interval_s / 2:
            fast_due = False
        try:
            self._run_timed(fast_due)
        finally:
            with self._timed_lock:
                self._timed_pending = False
            self._rearm.set()

    def _run_timed(self, fast_due: bool) -> frozenset[str]:
        """Advance, read, post, record; the names posted.

        Reads the names the composite emitted and, when ``fast_due``, the
        ``fast`` names; a pass run for a deadline alone leaves the ``fast``
        names to the fast period, so their rate is the fast period's. It posts
        them
        through the base class, and forces a Channel Access event for each
        but those a read fault holds, which are posted only when changed. It
        caches no settable state, since it writes nothing. Its own errors are
        caught here, since the run loop only logs a job's; the outcome is
        recorded only when it differs from the last one recorded.
        """
        error: str | None = None
        posted: frozenset[str] = frozenset()
        try:
            self.model.advance()
            emitted = self._drain_emitted()
            fast = self._fast if fast_due else frozenset()
            names = sorted(timed_pass_names(fast, emitted))
            values = self.model.get(names) if names else {}
            faulted = self._read_faulted() & values.keys()
            if faulted:
                held = {name: values[name] for name in sorted(faulted)}
                udf_now = self._evaluate_severity(list(held))
                changed = changed_outputs(held, self._served, (), udf_now, self._udf)
                values = {
                    name: value
                    for name, value in values.items()
                    if name not in faulted or name in changed
                }
            self._cycle_requested_names = tuple(values)
            self._post_forced(values, time.time(), forced_names(fast, emitted, faulted), emitted)
            posted = frozenset(values)
            if fast_due:
                self._fast_posted_at = time.time()
            self._note_deadline()
        except Exception as exc:
            error = str(exc) or type(exc).__name__
            LOG.exception("Timed pass failed: (%s)", error)
        if (error is None) != self._last_ok:
            self._write_health(error)
        return posted

    def _queue_timed_pass(self, fast_due: bool) -> None:
        """Queue one timed pass, unless one is already waiting."""
        with self._timed_lock:
            if self._timed_pending:
                return
            self._timed_pending = True
        self._enqueue({}, jobs=[partial(self._timed_pass, fast_due)])

    def _start_ticker(self) -> None:
        """Start the base class's ticker, and the timed-pass scheduler when the view needs one.

        A view with neither a ``fast`` nor an ``event`` channel starts no
        scheduler. The stop event is held in the thread's closure, as the
        ticker's is, so a restart cannot leave an old thread waiting on an
        event nobody will set.
        """
        super()._start_ticker()
        if not self._needs_timed:
            return
        scheduler, stop = self._scheduler, self._scheduler_stop
        if scheduler is not None and scheduler.is_alive():
            if stop is not None and not stop.is_set():
                return
            scheduler.join(timeout=self._ticker_join_timeout())
        with self._timed_lock:
            self._timed_pending = False
        stop = threading.Event()
        rearm = self._rearm
        period = self._fast_interval_s

        def schedule() -> None:
            # Fast passes fall on a grid one period apart, so the rate holds
            # however long each pass takes; a grid fallen a period behind
            # starts again from now.
            grid = time.time()
            while True:
                rearm.clear()
                if stop.is_set():
                    return
                with self._timed_lock:
                    pending = self._timed_pending
                if pending:
                    rearm.wait(max(period, 1.0))
                    continue
                now = time.time()
                fast_due = grid + period
                due = next_wake(grid, period, self._deadline)
                if due > now:
                    rearm.wait(due - now)
                    continue
                is_fast = fast_due <= now
                if is_fast:
                    grid = fast_due if fast_due > now - period else now
                self._queue_timed_pass(is_fast)

        scheduler = threading.Thread(target=schedule, name=SCHEDULER_THREAD, daemon=True)
        self._scheduler_stop = stop
        self._scheduler = scheduler
        scheduler.start()

    def _stop_ticker(self) -> None:
        """Stop the timed-pass scheduler, if there is one, then the base class's ticker."""
        scheduler, stop = self._scheduler, self._scheduler_stop
        if scheduler is not None:
            if stop is not None:
                stop.set()
            self._rearm.set()
            scheduler.join(timeout=self._ticker_join_timeout())
            if scheduler.is_alive():
                LOG.warning("timed-pass scheduler did not stop within its join timeout")
            else:
                self._scheduler = None
                self._scheduler_stop = None
        super()._stop_ticker()

    def _reset_to_cached_state(self) -> None:
        """Restore the cached state only when the pass failed after ``model.set`` succeeded.

        A model with ``restore`` takes it back without a client write.
        """
        if not self._set_landed:
            return
        restore = getattr(self.model, "restore", None)
        if callable(restore):
            restore(self._cached_state)
        else:
            super()._reset_to_cached_state()

    def _create_model_info(self) -> None:
        """Serve the model RPC channel beside the base class's model info."""
        super()._create_model_info()
        self._surface = ModelSurface.for_view(
            self.model,
            self._view,
            instance=self._instance if self._instance is not None else instance_name(),
            endpoint=pva_endpoint(),
            model_write_token=self._model_write_token,
            failed_pass_tolerance=self.config["failed_pass_tolerance"],
        )
        channel = SharedPV(initial=REPLY_TYPE.wrap(""))
        channel.rpc(
            RpcFrontDoor(
                self._surface,
                enqueue=self._enqueue,
                driver=lambda: self.ca_driver,
                queue_depth=self.queue.qsize,
            )
        )
        self.providers[f"{self.config['prefix']}{RPC_PV}"] = channel


def _is_plain_write(item: dict[str, Any]) -> bool:
    """Whether a queue item carries values and nothing else: no reset, no job, no tick."""
    return bool(item["values"]) and not (item.get("reset") or item.get("jobs") or item.get("tick"))


__all__ = ["ModelRunner"]
