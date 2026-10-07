"""The model RPC's verbs over a composite and its simulator view.

A composite's served side is the address set its simulator view's
``addresses.json`` lists: its ``channels`` and its ``status`` addresses.
Clients reach those through the control-system surface -- Channel Access,
and PVA for the same address -- exactly as they reach any other channel.
Every other name the surface answers for is a variable of one of the
composite's physics models, ``<model>/<name>``, reached through the
composite alone: the model RPC is its only surface.

:class:`ModelSurface` answers the model RPC's verbs on that split, and
:meth:`ModelSurface.for_view` builds it. The read verbs describe what the
surface answers for (``info``), read what the composite holds (``get``),
compare it with what the control system serves (``diff``) and report on the
server around it (``status``). Every verb runs on the run loop's thread --
the only thread that touches the composite -- and answers a plain JSON-able
dict. A refusal is a :class:`ModelRpcError` whose message a client shows
unchanged, and the composite's own refusal of a name -- a failed model's
included -- is the verb's refusal.

The write verbs change model variables, and only those. ``set`` writes the
values it is given; ``reset`` writes each drifted writable model variable
back to the value its model held when it was built, and touches no served
address. Both require the write token the server was configured with,
compared in constant time; a server configured without one refuses every
write. A refused write reaches no model: every check runs before the one
write that applies it, and its reason is kept for ``status``.

Nothing here imports the serving runtime -- the runner, the Channel Access
server or lume-pva -- so the verbs are decided and tested in process. The
verbs see the Channel Access side only through the one reader the runner
injects into ``diff``. The one server library reached at all is p4p, and
only indirectly: refusals are the RPC wire contract's
:class:`ModelRpcError`, and that module imports p4p for its wire types.
"""

from __future__ import annotations

import hmac
import math
import numbers
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any

from osprey.services.virtual_accelerator.serving.model_rpc import ModelRpcError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from lume.variables import Variable

    from osprey_connectors.simulation.composite import Composite

#: The side a name is on, as the model RPC's ``info`` verb reports it in
#: each entry's ``surface`` field.
SURFACE_SERVED = "served"
SURFACE_MODEL_ONLY = "model"

#: The refusal every write meets on a server configured without a token.
WRITES_DISABLED = "model writes are disabled"


class ModelSurface(ABC):
    """The model RPC's verbs.

    The runner builds one per server through :meth:`for_view` and calls each
    verb on the run loop's thread. It also feeds ``status`` what only it can
    see, through :meth:`record_cycle`, :meth:`record_queue_depth` and
    :meth:`record_refusal`.
    """

    def _start(
        self,
        *,
        instance: str,
        endpoint: str,
        clock: Callable[[], float],
        model_write_token: str | None,
    ) -> None:
        """Take the write token and start the state ``status`` reports."""
        self._write_token = model_write_token.encode() if model_write_token else None
        self._instance = instance
        self._endpoint = endpoint
        self._clock = clock
        self._started = clock()
        self._last_cycle_ms: float | None = None
        self._queue_depth = 0
        self._last_refused_write: str | None = None

    @classmethod
    def for_view(
        cls,
        composite: Composite,
        addresses_json: Mapping[str, Any],
        *,
        instance: str,
        endpoint: str,
        model_write_token: str | None,
        clock: Callable[[], float] = time.monotonic,
    ) -> ModelSurface:
        """Answer for a composite, keyed on its simulator view's address set.

        The served side is ``addresses_json``'s ``channels`` and ``status``;
        a physics model's own variables are reached as ``<model>/<name>``
        through ``composite.model_get`` and ``composite.model_set`` alone.

        Args:
            composite: the composite the runner serves. Read and written on
                the calling thread, which must be the run loop's.
            addresses_json: the view's ``addresses.json`` document.
            instance: this server's instance name.
            endpoint: where this server is reached.
            model_write_token: the token ``set`` and ``reset`` require.
                ``None`` or empty disables model writes.
            clock: seconds on a monotonic scale; ``uptime_s`` counts from
                the reading taken here.

        Returns:
            The surface. ``status`` reports ``instance``, ``endpoint``,
            ``last_cycle_ms``, ``queue_depth``, ``uptime_s`` and
            ``last_refused_write``.
        """
        return _ViewSurface(
            composite,
            addresses_json,
            instance=instance,
            endpoint=endpoint,
            model_write_token=model_write_token,
            clock=clock,
        )

    @abstractmethod
    def info(self) -> dict[str, Any]:
        """Describe every name the surface answers for; no value is read."""

    @abstractmethod
    def get(self, names: Iterable[str]) -> dict[str, Any]:
        """What the composite holds for ``names``."""

    @abstractmethod
    def diff(self, get_param: Callable[[str], Any]) -> dict[str, dict[str, Any]]:
        """Each served address's served value beside the value the composite holds."""

    def status(self) -> dict[str, Any]:
        """The server around the composite, as the runner last recorded it; no value is read.

        Returns:
            ``instance``, ``endpoint``, ``last_cycle_ms``, ``queue_depth``,
            ``uptime_s`` and ``last_refused_write``.
        """
        return {
            "instance": self._instance,
            "endpoint": self._endpoint,
            "last_cycle_ms": self._last_cycle_ms,
            "queue_depth": self._queue_depth,
            "uptime_s": self._clock() - self._started,
            "last_refused_write": self._last_refused_write,
        }

    def record_cycle(self, ms: float) -> None:
        """Record how long the run loop's latest cycle took, in milliseconds."""
        self._last_cycle_ms = float(ms)

    def record_queue_depth(self, n: int) -> None:
        """Record how many items the run loop's queue holds."""
        self._queue_depth = int(n)

    def record_refusal(self, text: str) -> None:
        """Record why the latest refused write was refused, as a client was told."""
        self._last_refused_write = str(text)

    @abstractmethod
    def set(self, values: Mapping[str, Any], token: str | None) -> list[str]:
        """Write model variables; the names written, in the order given."""

    @abstractmethod
    def reset(self, token: str | None) -> list[str]:
        """Write every drifted writable model variable back; the names reset."""

    def _authorize(self, token: str | None) -> None:
        """Refuse a write unless ``token`` is the configured one.

        The comparison is :func:`hmac.compare_digest` over UTF-8 bytes, so
        how long it takes says nothing about how much of the token matched,
        and a non-ASCII token is refused rather than raised on. Neither
        token appears in a refusal.
        """
        if self._write_token is None:
            raise self._refusal(WRITES_DISABLED)
        if not token:
            raise self._refusal("model write refused: no write token was presented")
        if not hmac.compare_digest(self._write_token, token.encode()):
            raise self._refusal("model write refused: the write token does not match")

    def _check(
        self, batch: Mapping[str, Any], checks: Iterable[tuple[str, Callable[[str], bool]]]
    ) -> None:
        """Refuse ``batch`` at the first check any name fails, naming every such name, sorted."""
        for reason, fails in checks:
            offenders = sorted(name for name in batch if fails(name))
            if offenders:
                raise self._refusal(f"{reason}: {', '.join(offenders)}")

    def _apply(self, write: Callable[[dict[str, Any]], Any], batch: dict[str, Any]) -> None:
        """One ``write`` of ``batch``, its validation failure turned into a refusal."""
        try:
            write(batch)
        except (ValueError, TypeError) as exc:
            raise self._refusal(str(exc) or type(exc).__name__) from exc

    def _refusal(self, text: str) -> ModelRpcError:
        """Record ``text`` for ``status`` and return the refusal carrying it."""
        self.record_refusal(text)
        return ModelRpcError(text)


class _ViewSurface(ModelSurface):
    """The model RPC's verbs over a composite, keyed on its view's address set.

    Built by :meth:`ModelSurface.for_view`. A served name is an address of
    ``addresses.json``; any other name is a model variable, ``<model>/<name>``,
    and the composite's refusal of one -- an unknown name, or a model that
    has failed -- is the verb's refusal, its text unchanged.
    """

    def __init__(
        self,
        composite: Composite,
        addresses_json: Mapping[str, Any],
        *,
        instance: str,
        endpoint: str,
        model_write_token: str | None,
        clock: Callable[[], float],
    ) -> None:
        self._composite = composite
        self._channels = [str(address) for address in addresses_json.get("channels", [])]
        self._served = [*self._channels, *(str(a) for a in addresses_json.get("status", []))]
        self._served_set = frozenset(self._served)
        self._start(
            instance=instance, endpoint=endpoint, clock=clock, model_write_token=model_write_token
        )

    def info(self) -> dict[str, Any]:
        """Describe every served address and every model variable; no value is read.

        Returns:
            ``variables``: one entry per served address -- the view's
            channels, then its status addresses -- with ``surface``
            :data:`SURFACE_SERVED`, then one per variable of each built
            model, named ``<model>/<name>``, with ``surface``
            :data:`SURFACE_MODEL_ONLY`. Each carries ``name``, ``unit``,
            ``value_range``, ``read_only`` and ``surface``.
        """
        declared = self._composite.supported_variables
        served = [_describe(declared[name], SURFACE_SERVED) for name in self._served]
        model = [
            _describe(variable, SURFACE_MODEL_ONLY, name=name)
            for name, (variable, _) in self._composite.model_variables().items()
        ]
        return {"variables": [*served, *model]}

    def get(self, names: Iterable[str]) -> dict[str, Any]:
        """Read served addresses from the composite and model variables from their model.

        Raises:
            ModelRpcError: the composite refuses a name; its text is the
                refusal, and nothing is returned.
        """
        requested = list(names)
        served = [name for name in requested if name in self._served_set]
        model = [name for name in requested if name not in self._served_set]
        try:
            values = dict(self._composite.get(served)) if served else {}
            if model:
                values.update(self._composite.model_get(model))
        except ValueError as exc:
            raise ModelRpcError(str(exc)) from exc
        return {name: values[name] for name in requested}

    def diff(self, get_param: Callable[[str], Any]) -> dict[str, dict[str, Any]]:
        """Each view channel's served value beside the value the composite holds.

        Args:
            get_param: reads the value the control system serves for an
                address -- the Channel Access driver's ``getParam``.

        Returns:
            ``{address: {"served": ..., "truth": ...}}`` for every channel of
            ``addresses.json``, the truth read without motion or readout in
            one batch.
        """
        if not self._channels:
            return {}
        truth = self._composite.held(self._channels)
        return {
            address: {"served": get_param(address), "truth": truth[address]}
            for address in self._channels
        }

    def set(self, values: Mapping[str, Any], token: str | None) -> list[str]:
        """Write model variables in one ``composite.model_set``.

        Checked in this order, each check refusing the whole call and naming
        every name it fails, sorted: the token; then any served address;
        then any number that is not finite. The composite's own refusal --
        a name it does not know, a model that has failed, a value the model
        refuses -- comes last, its text unchanged. An empty ``values``
        writes nothing.

        Returns:
            The names written, in the order given.

        Raises:
            ModelRpcError: the write is refused; the reason is what
                ``status`` reports as the last refusal.
        """
        self._authorize(token)
        batch = dict(values)
        checks: tuple[tuple[str, Callable[[str], bool]], ...] = (
            ("a served address, written through the control system", self._served_set.__contains__),
            ("not a finite value", lambda name: not _finite_or_not_a_number(batch[name])),
        )
        self._check(batch, checks)
        if not batch:
            return []
        self._apply(self._composite.model_set, batch)
        return list(batch)

    def reset(self, token: str | None) -> list[str]:
        """Write every drifted writable model variable back to its start value.

        A start value is what the model held when it was built at the active
        scenarios. The current values are read once and the drifted ones are
        written in one ``composite.model_set``. No served address -- a
        setpoint, a held value, a session write -- is written.

        Returns:
            The names reset, ``<model>/<name>``. An empty list -- nothing had
            drifted -- is not an error, and writes nothing.

        Raises:
            ModelRpcError: the token is refused, or the composite refuses the
                read or the write; its text is the refusal.
        """
        self._authorize(token)
        starts = {
            name: start
            for name, (variable, start) in self._composite.model_variables().items()
            if not variable.read_only and start is not None
        }
        if not starts:
            return []
        try:
            current = self._composite.model_get(list(starts))
        except ValueError as exc:
            raise self._refusal(str(exc)) from exc
        drifted = {name: start for name, start in starts.items() if current[name] != start}
        if drifted:
            self._apply(self._composite.model_set, drifted)
        return list(drifted)


def _finite_or_not_a_number(value: Any) -> bool:
    """False only for a real number that is not finite.

    A value that is not a number is left to the variable's own validation,
    which knows what it takes.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return True
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _describe(variable: Variable, surface: str, *, name: str | None = None) -> dict[str, Any]:
    """One ``info`` entry: the fields a client needs to address ``variable``.

    ``name`` is the name a client addresses it by, when that is not the
    variable's own.
    """
    value_range = getattr(variable, "value_range", None)
    return {
        "name": variable.name if name is None else name,
        "unit": getattr(variable, "unit", None),
        "value_range": None if value_range is None else [float(v) for v in value_range],
        "read_only": bool(variable.read_only),
        "surface": surface,
    }


__all__ = [
    "SURFACE_MODEL_ONLY",
    "SURFACE_SERVED",
    "WRITES_DISABLED",
    "ModelSurface",
]
