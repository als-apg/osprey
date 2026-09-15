"""Which of a model's variables the control system serves, and which it does not.

A facility's :class:`~lume.model.LUMEModel` declares its variables; the
facility's channel manifest declares the addresses the control system serves.
The two overlap but are not the same set, and this module draws the line
between them once, at boot:

* a variable is **served** iff its name is an address the serving database
  holds (a key of :attr:`ServingRecords.all
  <osprey.services.virtual_accelerator.serving.pvdb.ServingRecords.all>`).
  Clients reach it through the control-system surface -- Channel Access, and
  PVA for the same address -- exactly as they reach any other channel;
* every other variable is **model-only**. The model RPC is its only surface:
  no channel is served for it on either transport, so a client that does not
  speak the RPC cannot see it at all.

The rule is by name alone. A variable's ``read_only`` flag decides what a
client may do with it on its side of the line, never which side it is on: a
read-only BPM reading the manifest serves is served, and a writable
calibration factor the manifest does not list is model-only.

:class:`ModelSurface` answers the model RPC's verbs on that partition. The
read verbs describe the model (``info``), read what it holds (``get``),
compare it with what the control system serves (``diff``) and report on the
server around it (``status``). Every verb runs on the run loop's thread --
the only thread that touches the model -- and answers a plain JSON-able
dict. A refusal is a :class:`ModelRpcError` whose message a client shows
unchanged.

The write verbs change model-only variables, and only those. ``set`` writes
the values it is given; ``reset`` writes each model-only writable that has
drifted back to the seed it declared at boot. Both require the write token
the server was configured with, compared in constant time; a server
configured without one refuses every write. A refused write reaches no
model: every check runs before the one ``model.set`` that applies it, and
its reason is kept for ``status``. A write that lands ends with ``refresh``,
which tells whatever the server derives from the model -- the served BPM
readings -- which names changed. ``reset`` writes the seeds back through
``model.set`` and never calls the model's own ``reset``: that would also
return every served setpoint to its default, and a reset of the faults must
leave the machine where the control system put it.

Nothing here imports the serving runtime -- the runner, the Channel Access
server or lume-pva -- so the partition and the verbs are decided and tested
in process, the same way the write path is. The verbs see the Channel Access
side only through the one reader the runner injects into ``diff``. The one
server library reached at all is p4p, and only indirectly: refusals are the
RPC wire contract's :class:`ModelRpcError`, and that module imports p4p for
its wire types.
"""

from __future__ import annotations

import hmac
import math
import numbers
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from osprey.services.virtual_accelerator.serving.model_rpc import ModelRpcError
from osprey.services.virtual_accelerator.serving.write_path import (
    RUNNER_CONFIG_POLICY,
    STUCK_SETPOINTS_VARIABLE,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from lume.model import LUMEModel
    from lume.variables import Variable

    from osprey.services.virtual_accelerator.serving.pvdb import ServingRecords

#: The side of the partition a variable is on, as the model RPC's ``info``
#: verb reports it in each variable's ``surface`` field.
SURFACE_SERVED = "served"
SURFACE_MODEL_ONLY = "model-only"

#: The refusal every write meets on a server configured without a token.
WRITES_DISABLED = "model writes are disabled"


def _refresh_nothing(changed: Iterable[str]) -> None:
    """Refresh nothing: the server derives nothing from the model's variables."""


@dataclass(frozen=True)
class VariablePartition:
    """A model's declared variables, split by whether the manifest serves them.

    Every declared variable is on exactly one side, and each side keeps the
    model's declaration order and the model's own variable objects -- so a
    roster listed from either side is stable across boots and reports the
    model's units and ranges, not copies of them.

    Attributes:
        served: variables whose name is a served address, by name.
        model_only: every other declared variable, by name.
    """

    served: dict[str, Variable]
    model_only: dict[str, Variable]


def partition_variables(model: LUMEModel, records: ServingRecords) -> VariablePartition:
    """Split ``model``'s declared variables into served and model-only.

    Args:
        model: the model whose :attr:`~lume.model.LUMEModel.supported_variables`
            are partitioned. Read once; no value is read or written.
        records: the built serving database; a variable is served iff its
            name is a key of ``records.all``.

    Returns:
        The partition. Its dicts are new, so editing either never edits the
        model's declared namespace.
    """
    served_addresses = records.all
    served: dict[str, Variable] = {}
    model_only: dict[str, Variable] = {}
    for name, variable in model.supported_variables.items():
        side = served if name in served_addresses else model_only
        side[name] = variable
    return VariablePartition(served=served, model_only=model_only)


class ModelSurface:
    """The model RPC's verbs, answered from the model and its partition.

    The runner builds one per server, over the same model it serves -- the
    wrapped one, so the stuck set the wrapper owns is a variable here like
    any other -- and calls each verb on the run loop's thread. It also feeds
    ``status`` what only it can see, through :meth:`record_cycle`,
    :meth:`record_queue_depth` and :meth:`record_refusal`.
    """

    def __init__(
        self,
        model: LUMEModel,
        partition: VariablePartition,
        records: ServingRecords,
        *,
        backend_name: str,
        lattice_source: str,
        instance: str,
        endpoint: str,
        clock: Callable[[], float] = time.monotonic,
        update_rate: float = RUNNER_CONFIG_POLICY["update_rate"],
        model_write_token: str | None = None,
        refresh: Callable[[list[str]], None] = _refresh_nothing,
    ) -> None:
        """Answer for ``model``, as ``partition`` splits it.

        Args:
            model: the model the runner serves. Read on the calling thread,
                which must be the run loop's.
            partition: ``model``'s variables split into served and model-only,
                as :func:`partition_variables` decided them at boot.
            records: the built serving database the partition was drawn
                against.
            backend_name: the physics backend's name, as ``info`` and
                ``status`` report it.
            lattice_source: where the backend's lattice came from.
            instance: this server's instance name.
            endpoint: where this server is reached.
            clock: seconds on a monotonic scale; ``uptime_s`` counts from
                the reading taken here.
            update_rate: the run loop's batching rate in Hz, as the runner
                configured it; ``0.0`` runs every write as its own cycle.
            model_write_token: the token ``set`` and ``reset`` require.
                ``None`` or empty disables model writes: an unset token must
                never be matched by an empty one.
            refresh: called once after each write that lands, with the names
                it wrote, on the run loop's thread; the physics bridge's
                ``refresh``. The default does nothing.

        Each model-only writable's boot seed is read here, from its declared
        ``default_value``; one declaring none has no seed and ``reset``
        leaves it alone.
        """
        self._model = model
        self._partition = partition
        self._records = records
        self._write_token = model_write_token.encode() if model_write_token else None
        self._refresh = refresh
        self._seeds: dict[str, Any] = {
            name: variable.default_value
            for name, variable in partition.model_only.items()
            if not variable.read_only and variable.default_value is not None
        }
        self._backend_name = backend_name
        self._lattice_source = lattice_source
        self._instance = instance
        self._endpoint = endpoint
        self._clock = clock
        self._started = clock()
        self._update_rate = float(update_rate)
        self._last_cycle_ms: float | None = None
        self._queue_depth = 0
        self._last_refused_write: str | None = None

    def info(self) -> dict[str, Any]:
        """Describe every model variable; no value is read.

        Returns:
            ``backend`` and ``lattice_source``, and ``variables``: one entry
            per model variable -- the served side first, then the
            model-only side, each in declaration order -- carrying its
            ``name``, ``unit`` and ``value_range`` (``None`` for a kind that
            has neither), ``read_only`` and ``surface``
            (:data:`SURFACE_SERVED` or :data:`SURFACE_MODEL_ONLY`).
        """
        sides = (
            (SURFACE_SERVED, self._partition.served),
            (SURFACE_MODEL_ONLY, self._partition.model_only),
        )
        variables = [
            _describe(variable, surface) for surface, side in sides for variable in side.values()
        ]
        return {
            "backend": self._backend_name,
            "lattice_source": self._lattice_source,
            "variables": variables,
        }

    def get(self, names: Iterable[str]) -> dict[str, Any]:
        """What the model holds for ``names``, on either side of the partition.

        For a served address that is the model's un-faulted truth, which a
        fault on the control-system side may keep from what clients read.

        Raises:
            ModelRpcError: a name is not a model variable. The whole call is
                refused, naming every such name, before the model is read.
        """
        requested = list(names)
        declared = self._model.supported_variables
        unknown = sorted({name for name in requested if name not in declared})
        if unknown:
            raise ModelRpcError(f"not a model variable: {', '.join(unknown)}")
        return dict(self._model.get(requested))

    def diff(self, get_param: Callable[[str], Any]) -> dict[str, dict[str, Any]]:
        """Each served variable's served value beside the model's truth.

        Args:
            get_param: reads the value the control system serves for an
                address -- the Channel Access driver's ``getParam``.

        Returns:
            ``{name: {"served": ..., "truth": ...}}`` for every served model
            variable, the truth read from the model in one batch.
        """
        served = list(self._partition.served)
        if not served:
            return {}
        truth = self._model.get(served)
        return {name: {"served": get_param(name), "truth": truth[name]} for name in served}

    def status(self) -> dict[str, Any]:
        """The server around the model, as the runner last recorded it; no value is read."""
        return {
            "backend": self._backend_name,
            "lattice_source": self._lattice_source,
            "instance": self._instance,
            "endpoint": self._endpoint,
            "update_rate": self._update_rate,
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

    def set(self, values: Mapping[str, Any], token: str | None) -> list[str]:
        """Write model-only ``values`` in one ``model.set``, then refresh.

        Checked in this order, each check refusing the whole call and naming
        every name it fails, sorted: the token; then any served address
        (served addresses are written through the control system); then any
        read-only variable; then any name the model does not declare; then
        any number that is not finite. The model's own validation --
        ranges, allowed values, the text a variable takes -- runs last, in
        ``model.set``, and its message is the refusal. An empty ``values``
        writes nothing and refreshes nothing.

        Returns:
            The names written, in the order given.

        Raises:
            ModelRpcError: the write is refused. Nothing was written, and
                the reason is what ``status`` reports as the last refusal.
        """
        self._authorize(token)
        batch = dict(values)
        declared = self._model.supported_variables
        served = self._records.all
        checks: tuple[tuple[str, Callable[[str], bool]], ...] = (
            ("a served address, written through the control system", served.__contains__),
            ("read-only", lambda name: name in declared and bool(declared[name].read_only)),
            ("not a model variable", lambda name: name not in declared),
            ("not a finite value", lambda name: not _finite_or_not_a_number(batch[name])),
        )
        for reason, fails in checks:
            offenders = sorted(name for name in batch if fails(name))
            if offenders:
                raise self._refusal(f"{reason}: {', '.join(offenders)}")
        if not batch:
            return []
        self._apply(batch)
        written = list(batch)
        self._refresh(written)
        return written

    def reset(self, token: str | None) -> list[str]:
        """Write every drifted model-only writable back to its boot seed.

        The current values are read once. The drifted ones are written in
        one ``model.set``, then the stuck set -- if it drifted -- in its own,
        so the faults are restored before the serving side's stuck set moves.
        Served setpoints and read-only variables are never written, and the
        model's own ``reset`` is never called.

        Returns:
            The names reset, in declaration order. An empty list -- nothing
            had drifted -- is not an error, and writes and refreshes nothing.

        Raises:
            ModelRpcError: the token is refused, or the model refuses a seed.
                A refused token writes nothing; a refused stuck set leaves
                the faults already restored, and refreshed.
        """
        self._authorize(token)
        if not self._seeds:
            return []
        current = self._model.get(list(self._seeds))
        drifted = {name: seed for name, seed in self._seeds.items() if current[name] != seed}
        faults = {name: seed for name, seed in drifted.items() if name != STUCK_SETPOINTS_VARIABLE}
        stuck = {name: seed for name, seed in drifted.items() if name == STUCK_SETPOINTS_VARIABLE}
        restored: list[str] = []
        try:
            for batch in (faults, stuck):
                if batch:
                    self._apply(batch)
                    restored.extend(batch)
        finally:
            if restored:
                self._refresh(restored)
        return restored

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

    def _apply(self, batch: dict[str, Any]) -> None:
        """One ``model.set``, its validation failure turned into a refusal."""
        try:
            self._model.set(batch)
        except (ValueError, TypeError) as exc:
            raise self._refusal(str(exc) or type(exc).__name__) from exc

    def _refusal(self, text: str) -> ModelRpcError:
        """Record ``text`` for ``status`` and return the refusal carrying it."""
        self.record_refusal(text)
        return ModelRpcError(text)


def _finite_or_not_a_number(value: Any) -> bool:
    """False only for a real number that is not finite.

    A value that is not a number -- the stuck set's text -- is left to the
    variable's own validation, which knows what it takes.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return True
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _describe(variable: Variable, surface: str) -> dict[str, Any]:
    """One ``info`` entry: the fields a client needs to address ``variable``."""
    value_range = getattr(variable, "value_range", None)
    return {
        "name": variable.name,
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
    "VariablePartition",
    "partition_variables",
]
