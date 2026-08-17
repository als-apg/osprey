"""The bridge's half of the document plane: a 0MQ proxy feeding the live-row buffer.

No plan runs inside this process. The RunEngine lives in the
``queueserver`` container (``qserver_startup.py``), which subscribes a
``bluesky.callbacks.zmq`` ``Publisher`` to it; this module runs the other end —
the ``Proxy`` that Publisher connects to, plus a ``RemoteDispatcher`` that
turns the republished stream back into rows in the existing ``live_rows``
buffer. That is how ``GET /runs/{id}/data`` keeps serving live data for a scan
the bridge is not executing.

**The proxy is the binding element.** The bridge binds both sockets and the
worker connects to the in-side, which is what makes key authentication
possible in the first place: the queueserver container is necessarily
dual-homed (it needs the accelerator network for Channel Access and Tiled), so
network placement alone protects nothing. CurveZMQ ``ServerCurve`` on the
in-side, with a pinned directory of accepted client public keys, is the only
thing standing between this socket and a rogue container injecting forged run
documents. Consequently, a configured-but-unauthenticatable document plane
does NOT fall back to an unencrypted socket — it does not come up at all, and
the bridge degrades to "no live rows" (Tiled still serves completed runs).

**Buffers are keyed by OSPREY's run id, not the RunEngine's uid.** The run uid
is chosen inside the worker; the bridge never sees it before the fact and
cannot map it back to the queue item an operator enqueued. The enqueue path
threads OSPREY's own run id through queueserver item metadata
(``queue_backend.RUN_ID_META_KEY``), queueserver merges that metadata into the
run's start document, and :class:`RunDocumentRouter` reads it back out there.
A start document without it (someone drove the worker directly) falls back to
the run uid, so such a run is still recorded — just under the only identity it
has. The same metadata path carries the plan stamp
(``queue_backend.PLAN_META_KEY``), read out here and handed to the recorder so
a run's rows stay self-describing for a reader holding only its id. Because a
run id can be reused — re-running an interrupted item keeps it — a start
document also invalidates any settled figure cached under that id
(``figure_cache.evict``), which is what lets the figure route trust its cache
without re-reading the source to check.

**Progress is computed here, not in the worker.** ``rows_seen`` comes from the
live buffer's true event count; ``expected_points`` comes from the plan
parameters captured at enqueue time (:func:`record_run_params`), with the
start document's own ``num_points`` as a fallback for plans that declare it.
When neither is available the progress record reports ``rows_seen`` with a
``None`` fraction — an honest "still going, total unknown" rather than a
fabricated denominator.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from . import figure_cache, live_rows
from .device_fields import is_read_only_device_field
from .live_rows import LiveRowRecorder
from .queue_backend import PLAN_META_KEY, RUN_ID_META_KEY

logger = logging.getLogger("osprey.services.bluesky_bridge.document_plane")

ZMQ_PUBLISH_ADDR_ENV = "BLUESKY_ZMQ_PUBLISH_ADDR"
"""Bind spec for the proxy's in-side — the socket the queueserver's
``Publisher`` connects to. The worker reads a variable of the same name holding
the *connect* form of the same socket (``tcp://bluesky-bridge:5567`` there,
``tcp://*:5567`` here). Absent means no document plane at all: the bridge
serves completed runs from Tiled and has no live rows."""

ZMQ_SUBSCRIBE_ADDR_ENV = "BLUESKY_ZMQ_SUBSCRIBE_ADDR"
"""Bind spec for the proxy's out-side, which this process's own
``RemoteDispatcher`` connects to. Defaults to a loopback address with a
kernel-assigned port — nothing outside this container subscribes."""

ZMQ_CURVE_SECRET_KEY_ENV = "BLUESKY_ZMQ_CURVE_SECRET_KEY"
"""Path to the proxy's own CURVE secret certificate — the server half of the
worker's ``BLUESKY_ZMQ_CURVE_SERVER_PUBLIC_KEY``. Required whenever the
document plane is configured at all."""

ZMQ_CURVE_CLIENT_PUBLIC_KEYS_ENV = "BLUESKY_ZMQ_CURVE_CLIENT_PUBLIC_KEYS"
"""Path to a directory of accepted client public certificates. Required
whenever the document plane is configured: pinning this folder is what
distinguishes "the worker may publish" from ``CURVE_ALLOW_ANY``, which
authenticates the transport but pins no identity."""

DEFAULT_SUBSCRIBE_ADDRESS = "tcp://127.0.0.1"
"""Loopback with a kernel-assigned port — the out-side default when
``BLUESKY_ZMQ_SUBSCRIBE_ADDR`` is unset."""

# How long `start()` waits for the dispatcher's polling task to exist before
# giving up on a clean handle to it. The task is created on the dispatcher's
# own loop in its own thread; without it there is nothing to cancel at stop.
_DISPATCHER_START_TIMEOUT = 10.0
_DISPATCHER_POLL_INTERVAL = 0.02

# Thread-join budget at shutdown. Both threads exit as soon as their socket is
# closed or their task cancelled; the bound only matters if 0MQ wedges, where
# leaking a daemon thread beats hanging the bridge's shutdown.
_STOP_JOIN_TIMEOUT = 5.0

# Runs tracked for document routing at once (descriptor->run maps and their
# recorders). A run is dropped as soon as its stop document lands, so this cap
# only bites on runs that never stop — a crashed worker, say — where the
# oldest is evicted rather than growing the map without limit.
_MAX_TRACKED_RUNS = 50

# Enqueued expected-point counts retained at once. Higher than the tracked-run
# cap because it is written at enqueue time and read for as long as the run
# stays visible in queue history.
_MAX_EXPECTED_POINTS = 200

# Parameter keys that can carry a per-sweep point count, most specific first.
_POINT_COUNT_KEYS = ("num_points", "num", "points")

_expected_lock = Lock()
_expected_points: OrderedDict[str, int] = OrderedDict()


# --------------------------------------------------------------------- progress


def expected_points_from_params(params: Mapping[str, Any] | None) -> int | None:
    """Best-effort event count for a plan invocation's parameters.

    Structural, not plan-specific: the catalog is open (facilities and the
    agent both add plans), so this reads the *shape* of a validated parameter
    mapping rather than switching on a plan name it would have to be taught
    about. Two shapes are recognized:

    - a list of per-axis mappings under ``axes``, each carrying its own point
      count — the rectangular grid shape (``prod`` of the counts);
    - a scalar point count at the top level, multiplied by the length of the
      one swept-device list if there is exactly one (a device list whose field
      name :func:`~osprey.services.bluesky_bridge.device_fields.is_read_only_device_field`
      claims is read at every point, never swept, so it does not count).

    Anything else returns ``None``, and ``None`` propagates all the way to the
    panel as "no fraction" — this number drives a progress bar and nothing
    else, so a wrong guess is worse than an absent one.

    Args:
        params: The plan's validated parameters, as the JSON-able mapping the
            queue item carries.

    Returns:
        The expected number of events, or ``None`` when the shape is not
        recognized or the counts are not positive integers.
    """
    if not isinstance(params, Mapping):
        return None

    axes = params.get("axes")
    if isinstance(axes, list) and axes and all(isinstance(axis, Mapping) for axis in axes):
        total = 1
        for axis in axes:
            count = _point_count(axis)
            if count is None:
                return None
            total *= count
        return total

    count = _point_count(params)
    if count is None:
        return None
    swept = _swept_device_lists(params)
    if len(swept) > 1:
        return None
    return count * len(swept[0]) if swept else count


def _point_count(params: Mapping[str, Any]) -> int | None:
    """The first positive integer point count named by :data:`_POINT_COUNT_KEYS`."""
    for key in _POINT_COUNT_KEYS:
        value = params.get(key)
        # bool is an int subclass; `snake_axes: true` is not a point count.
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _swept_device_lists(params: Mapping[str, Any]) -> list[list[Any]]:
    """Every list of device names in *params* that is swept rather than read.

    A plan that sweeps one device list serially (``orm`` sweeps each corrector
    over the same current range) produces ``count`` events per device. Two such
    lists are ambiguous — swept in sequence, or nested? — so the caller answers
    "don't know" rather than picking one. Which field names are read-only, and
    so excluded here, is decided solely by
    :func:`~osprey.services.bluesky_bridge.device_fields.is_read_only_device_field`.
    """
    return [
        value
        for key, value in params.items()
        if not is_read_only_device_field(key)
        and isinstance(value, list)
        and value
        and all(isinstance(item, str) for item in value)
    ]


def record_expected_points(run_id: str, expected: int | None) -> None:
    """Remember how many events *run_id* is expected to produce.

    ``None`` clears any previous entry rather than storing a placeholder, so
    "unknown" and "known to be N" stay distinguishable.
    """
    with _expected_lock:
        if expected is None:
            _expected_points.pop(run_id, None)
            return
        _expected_points[run_id] = expected
        _expected_points.move_to_end(run_id)
        while len(_expected_points) > _MAX_EXPECTED_POINTS:
            _expected_points.popitem(last=False)


def record_run_params(run_id: str, params: Mapping[str, Any] | None) -> int | None:
    """Derive and store *run_id*'s expected point count from its plan parameters.

    The enqueue path's one-call seam: the parameters an item was enqueued with
    are the only place the count can be known before the run starts, which is
    exactly when a queued item first needs a progress denominator.

    Returns:
        What was recorded — ``None`` when the parameter shape yielded nothing.
    """
    expected = expected_points_from_params(params)
    record_expected_points(run_id, expected)
    return expected


def get_expected_points(run_id: str) -> int | None:
    """The recorded expected point count for *run_id*, if any."""
    with _expected_lock:
        return _expected_points.get(run_id)


def progress(run_id: str) -> dict[str, Any] | None:
    """Progress for *run_id*, or ``None`` if nothing is known about it.

    Returns:
        ``{"rows_seen": int, "expected_points": int | None, "fraction": float |
        None, "complete": bool}``. ``fraction`` is clamped to 1.0 — a plan that
        emits more events than predicted reads as finished, never as 140%
        done — and is ``None`` whenever the expected count is unknown.
        ``complete`` is driven solely by the run's stop document, so a run that
        hit its expected count but never stopped is honestly still incomplete.
    """
    buffer = live_rows.get(run_id)
    expected = get_expected_points(run_id)
    if buffer is None and expected is None:
        return None

    rows_seen = int(buffer["total_seen"]) if buffer is not None else 0
    complete = bool(buffer is not None and not buffer["partial"])
    fraction: float | None = None
    if expected:
        fraction = min(1.0, rows_seen / expected)
    return {
        "rows_seen": rows_seen,
        "expected_points": expected,
        "fraction": fraction,
        "complete": complete,
    }


def _clear() -> None:
    """Drop every recorded expected-point count (test isolation only)."""
    with _expected_lock:
        _expected_points.clear()


# ----------------------------------------------------------------- doc routing


class RunDocumentRouter:
    """Fan one interleaved document stream out to per-run live-row recorders.

    ``LiveRowRecorder`` is single-run by construction (it latches one buffer
    key from the start document it sees). A remote stream carries every run the
    worker executes, so this router owns the multiplexing: it opens a recorder
    per start document, keyed by the OSPREY run id in that document's metadata,
    and routes later documents to it by following the document graph
    (``event.descriptor`` -> descriptor, ``descriptor.run_start`` -> start) —
    not by assuming documents arrive one run at a time.

    Never raises: this runs on the dispatcher's event loop, where an escaping
    exception would kill the polling task and silently end live rows for every
    subsequent run.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        # run uid -> recorder for that run
        self._recorders: OrderedDict[str, LiveRowRecorder] = OrderedDict()
        # descriptor uid -> owning run uid
        self._descriptors: dict[str, str] = {}

    def __call__(self, name: str, doc: dict[str, Any]) -> None:
        try:
            if name == "start":
                self._on_start(doc)
            elif name == "descriptor":
                self._on_descriptor(doc)
            elif name == "event":
                self._forward(self._run_of_event(doc), name, doc)
            elif name == "stop":
                self._on_stop(doc)
        except Exception:
            logger.warning("document router failed on %r document", name, exc_info=True)

    def _on_start(self, doc: dict[str, Any]) -> None:
        run_uid = doc.get("uid")
        if not run_uid:
            return
        run_id = doc.get(RUN_ID_META_KEY)
        key = run_id if isinstance(run_id, str) and run_id else run_uid
        if key == run_uid:
            logger.info(
                "run %s carries no %s — recording live rows under its run uid",
                run_uid,
                RUN_ID_META_KEY,
            )

        # A plan that declares its own point count (bluesky's standard scans
        # do) is a better denominator than nothing, but never overrides what
        # the enqueue path recorded: those parameters are what the operator
        # actually asked for.
        if get_expected_points(key) is None:
            declared = doc.get("num_points")
            if isinstance(declared, int) and not isinstance(declared, bool) and declared > 0:
                record_expected_points(key, declared)

        # Which plan produced the run travels with its rows: a reader holding
        # only a run id cannot otherwise tell what the columns mean. Read here,
        # beside the run id and for the same reason — the stamp is a document
        # detail, and keeping the extraction in this plane leaves `live_rows`
        # testable with synthetic documents.
        recorder = LiveRowRecorder(key=key, plan=doc.get(PLAN_META_KEY))
        with self._lock:
            self._recorders[run_uid] = recorder
            self._recorders.move_to_end(run_uid)
            while len(self._recorders) > _MAX_TRACKED_RUNS:
                evicted, _ = self._recorders.popitem(last=False)
                self._forget_descriptors(evicted)
        recorder("start", doc)
        # The rows under `key` are now this run's, so any settled figure cached
        # for that id was built from the previous occupant's rows — an operator
        # re-running an interrupted item reuses the OSPREY run id. Evicting here
        # is what keeps the figure route from validating its cache at read time,
        # which would cost the very Tiled search the cache exists to avoid. It
        # runs AFTER `recorder("start", ...)` returns, so `live_rows`' lock is
        # released before the cache's is taken (the two are never nested, in
        # either order) and so the buffer is already the new run's by the time
        # the generation bump admits figures computed from it.
        figure_cache.evict(key)

    def _on_descriptor(self, doc: dict[str, Any]) -> None:
        descriptor_uid = doc.get("uid")
        run_uid = doc.get("run_start")
        if not descriptor_uid or not run_uid:
            return
        with self._lock:
            if run_uid in self._recorders:
                self._descriptors[descriptor_uid] = run_uid

    def _on_stop(self, doc: dict[str, Any]) -> None:
        run_uid = doc.get("run_start")
        if not run_uid:
            return
        self._forward(run_uid, "stop", doc)
        # The run is over: its recorder and descriptor map have no further use.
        # The rows themselves stay in `live_rows`, which owns their retention.
        with self._lock:
            self._recorders.pop(run_uid, None)
            self._forget_descriptors(run_uid)

    def _run_of_event(self, doc: dict[str, Any]) -> str | None:
        descriptor_uid = doc.get("descriptor")
        if not descriptor_uid:
            return None
        with self._lock:
            return self._descriptors.get(descriptor_uid)

    def _forward(self, run_uid: str | None, name: str, doc: dict[str, Any]) -> None:
        if not run_uid:
            return
        with self._lock:
            recorder = self._recorders.get(run_uid)
        if recorder is not None:
            recorder(name, doc)

    def _forget_descriptors(self, run_uid: str) -> None:
        """Drop every descriptor mapping owned by *run_uid* (caller holds the lock)."""
        for descriptor_uid in [uid for uid, owner in self._descriptors.items() if owner == run_uid]:
            del self._descriptors[descriptor_uid]


# ------------------------------------------------------------------ the plane


class DocumentPlaneError(RuntimeError):
    """The document plane is configured but cannot be brought up safely."""


@dataclass(frozen=True)
class DocumentPlaneConfig:
    """Where the proxy binds and which keys authenticate its in-side.

    Attributes:
        publish_address: Bind spec for the in-side (documents arrive here).
        subscribe_address: Bind spec for the out-side (this process subscribes).
        server_secret_key: The proxy's own CURVE secret certificate.
        client_public_keys: Directory of accepted client public certificates.
            Required, not optional: pyzmq's alternative is ``CURVE_ALLOW_ANY``,
            which accepts any well-formed keypair and would let a rogue
            container on the shared network publish forged run documents. The
            pinned directory IS the authentication.
    """

    publish_address: str
    subscribe_address: str
    server_secret_key: Path
    client_public_keys: Path

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> DocumentPlaneConfig | None:
        """Read the document-plane configuration, or ``None`` if unconfigured.

        Args:
            env: Environment mapping to read; defaults to ``os.environ``.

        Returns:
            The configuration, or ``None`` when
            :data:`ZMQ_PUBLISH_ADDR_ENV` is unset (no document plane deployed).

        Raises:
            DocumentPlaneError: The plane is configured but either CURVE
                path — the proxy's secret certificate or the accepted-client
                directory — is unset or unreadable. Binding anyway would put a
                document socket a dual-homed container can reach behind either
                no authentication at all or ``CURVE_ALLOW_ANY``, so this is
                refused rather than degraded.
        """
        env = os.environ if env is None else env
        publish_address = (env.get(ZMQ_PUBLISH_ADDR_ENV) or "").strip()
        if not publish_address:
            return None

        secret_key = (env.get(ZMQ_CURVE_SECRET_KEY_ENV) or "").strip()
        if not secret_key:
            raise DocumentPlaneError(
                f"{ZMQ_PUBLISH_ADDR_ENV} is set but {ZMQ_CURVE_SECRET_KEY_ENV} is not — "
                "refusing to bind an unauthenticated document socket"
            )
        secret_path = Path(secret_key)
        if not secret_path.is_file():
            raise DocumentPlaneError(
                f"{ZMQ_CURVE_SECRET_KEY_ENV} points at {secret_key!r}, which is not a "
                "readable file — refusing to bind an unauthenticated document socket"
            )

        client_keys = (env.get(ZMQ_CURVE_CLIENT_PUBLIC_KEYS_ENV) or "").strip()
        if not client_keys:
            raise DocumentPlaneError(
                f"{ZMQ_PUBLISH_ADDR_ENV} is set but {ZMQ_CURVE_CLIENT_PUBLIC_KEYS_ENV} is "
                "not — refusing to accept any client key that happens to be well-formed"
            )
        client_path = Path(client_keys)
        if not client_path.is_dir():
            raise DocumentPlaneError(
                f"{ZMQ_CURVE_CLIENT_PUBLIC_KEYS_ENV} points at {client_keys!r}, which is "
                "not a directory — refusing to fall back to accepting any client key"
            )

        return cls(
            publish_address=publish_address,
            subscribe_address=(env.get(ZMQ_SUBSCRIBE_ADDR_ENV) or "").strip()
            or DEFAULT_SUBSCRIBE_ADDRESS,
            server_secret_key=secret_path,
            client_public_keys=client_path,
        )


class DocumentPlane:
    """A running 0MQ proxy plus the dispatcher that drains it into ``live_rows``.

    Both halves block (``Proxy.start`` runs a 0MQ forwarder,
    ``RemoteDispatcher.start`` runs its own event loop), so each gets a daemon
    thread and the bridge's async lifespan only ever calls :meth:`start` and
    :meth:`stop`.

    Args:
        config: Where to bind and which keys to accept.
        sink: Document callback fed by the dispatcher. Defaults to a fresh
            :class:`RunDocumentRouter`; tests pass their own to observe the
            raw stream.
    """

    def __init__(
        self,
        config: DocumentPlaneConfig,
        *,
        sink: Callable[[str, dict[str, Any]], Any] | None = None,
    ) -> None:
        self._config = config
        self._sink = sink if sink is not None else RunDocumentRouter()
        self._proxy: Any = None
        self._dispatcher: Any = None
        self._proxy_thread: threading.Thread | None = None
        self._dispatcher_thread: threading.Thread | None = None

    @property
    def config(self) -> DocumentPlaneConfig:
        """The configuration this plane was built from."""
        return self._config

    @property
    def publish_address(self) -> str | None:
        """The in-side address actually bound, once started."""
        return None if self._proxy is None else str(self._proxy.in_port)

    @property
    def subscribe_address(self) -> str | None:
        """The out-side address actually bound, once started."""
        return None if self._proxy is None else str(self._proxy.out_port)

    def start(self) -> DocumentPlane:
        """Bind the proxy and start draining it. Returns ``self``.

        Raises:
            DocumentPlaneError: Already started.
        """
        if self._proxy is not None:
            raise DocumentPlaneError("this document plane has already been started")

        from bluesky.callbacks.zmq import Proxy, RemoteDispatcher, ServerCurve

        curve = ServerCurve(
            secret_path=self._config.server_secret_key,
            client_public_keys=self._config.client_public_keys,
            allow=None,
        )
        self._proxy = Proxy(
            in_address=self._config.publish_address,
            out_address=self._config.subscribe_address,
            in_curve=curve,
        )
        self._proxy_thread = threading.Thread(
            target=self._run_proxy, name="osprey-doc-plane-proxy", daemon=True
        )
        self._proxy_thread.start()

        self._dispatcher = RemoteDispatcher(self._proxy.out_port)
        self._dispatcher.subscribe(self._sink)
        self._dispatcher_thread = threading.Thread(
            target=self._run_dispatcher, name="osprey-doc-plane-dispatcher", daemon=True
        )
        self._dispatcher_thread.start()
        self._await_dispatcher_task()

        logger.info(
            "document plane up: proxy in=%s out=%s, accepting client keys pinned in %s",
            self.publish_address,
            self.subscribe_address,
            self._config.client_public_keys,
        )
        return self

    def stop(self) -> None:
        """Tear both halves down. Safe to call on a plane that never started.

        Neither upstream object has a stop hook that works from another
        thread — ``Proxy`` has none at all and ``RemoteDispatcher.stop`` must
        run on its own loop — so shutdown reaches for their internals: closing
        the proxy's sockets makes its forwarder return (and its own ``finally``
        destroy the context), and cancelling the dispatcher's polling task on
        its own loop makes ``start()`` return and run its ``stop()``. Closing
        the proxy's *context* from this thread instead would deadlock against
        the CURVE authenticator thread that shares it.
        """
        dispatcher, self._dispatcher = self._dispatcher, None
        proxy, self._proxy = self._proxy, None
        dispatcher_thread, self._dispatcher_thread = self._dispatcher_thread, None
        proxy_thread, self._proxy_thread = self._proxy_thread, None

        if dispatcher is not None:
            task = dispatcher._task
            if task is not None:
                try:
                    dispatcher.loop.call_soon_threadsafe(task.cancel)
                except RuntimeError:
                    # Loop already closed — the dispatcher is on its way out.
                    logger.debug("document-plane dispatcher loop was already closed")
        if dispatcher_thread is not None:
            dispatcher_thread.join(timeout=_STOP_JOIN_TIMEOUT)

        if proxy is not None:
            for socket in (proxy._frontend, proxy._backend):
                try:
                    socket.close(linger=0)
                except Exception:
                    logger.debug("document-plane proxy socket close failed", exc_info=True)
        if proxy_thread is not None:
            proxy_thread.join(timeout=_STOP_JOIN_TIMEOUT)

        logger.info("document plane stopped")

    def _run_proxy(self) -> None:
        """Forward documents until :meth:`stop` closes the sockets under us."""
        try:
            self._proxy.start()
        except Exception:
            # Expected at shutdown: closing the sockets is how the forwarder is
            # made to return. A failure at any other time is a dead document
            # plane, which degrades telemetry and nothing else.
            logger.debug("document-plane proxy exited", exc_info=True)

    def _run_dispatcher(self) -> None:
        """Drain the proxy's out-side until :meth:`stop` cancels the poll task."""
        try:
            self._dispatcher.start()
        except (Exception, asyncio.CancelledError):
            # `CancelledError` is a BaseException, and cancelling the poll task
            # is exactly how `stop()` ends this thread — letting it escape
            # would print a traceback on every clean shutdown.
            logger.debug("document-plane dispatcher exited", exc_info=True)

    def _await_dispatcher_task(self) -> None:
        """Block until the dispatcher's poll task exists, so :meth:`stop` can cancel it."""
        dispatcher = self._dispatcher
        deadline = time.monotonic() + _DISPATCHER_START_TIMEOUT
        while time.monotonic() < deadline:
            if dispatcher is not None and dispatcher._task is not None:
                return
            time.sleep(_DISPATCHER_POLL_INTERVAL)
        logger.warning(
            "document-plane dispatcher did not start polling within %.0fs; "
            "live rows may be unavailable",
            _DISPATCHER_START_TIMEOUT,
        )


# ------------------------------------------------------- process-wide singleton

_plane: DocumentPlane | None = None


def start_from_env(env: Mapping[str, str] | None = None) -> DocumentPlane | None:
    """Start the process's document plane from the environment, if one is configured.

    The bridge lifespan's whole interaction with this module. Configuration
    errors propagate (see :meth:`DocumentPlaneConfig.from_env` — an
    unauthenticated socket is never an acceptable fallback), but a failure to
    actually bind or start is logged and swallowed: live rows are telemetry,
    and a taken port must not take the operator's whole bridge down with it.

    Returns:
        The running plane, or ``None`` when unconfigured or when startup failed.
    """
    global _plane

    if _plane is not None:
        return _plane

    config = DocumentPlaneConfig.from_env(env)
    if config is None:
        logger.info(
            "%s is unset — no document plane; live rows are unavailable and run data "
            "comes from Tiled",
            ZMQ_PUBLISH_ADDR_ENV,
        )
        return None

    plane = DocumentPlane(config)
    try:
        plane.start()
    except Exception:
        logger.error(
            "could not start the document plane on %s; the bridge will serve run data "
            "without live rows",
            config.publish_address,
            exc_info=True,
        )
        plane.stop()
        return None

    _plane = plane
    return _plane


def shutdown() -> None:
    """Stop the process's document plane, if one is running. Idempotent."""
    global _plane

    plane, _plane = _plane, None
    if plane is not None:
        plane.stop()
