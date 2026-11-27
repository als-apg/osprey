"""Bounded live-row buffer for run data still in flight or just completed.

The ``GET /runs/{id}/data`` route needs a source of truth for run data before
(or in place of) a Tiled server. Scans run in the queueserver worker, which
publishes its document stream over 0MQ; the bridge's document plane
(``document_plane.py``) subscribes to that stream and feeds the plain-dict
callback below, which keeps translated rows in a bounded buffer. This module is
deliberately import-clean of bluesky, so it can be unit-tested with synthetic
documents and no worker at all. Generalizes the concept in BELLA's
``services/experiment_config/live_rows.py``: this version has no
GEECS-specific legacy column mapping, and explicitly RETAINS completed runs
rather than evicting at a small run count, so a read arriving after the plan
finishes still succeeds without a Tiled server.

Document handling (bluesky's plain-dict document protocol):

- ``start``: begins a new buffer for ``doc["uid"]`` — or for the recorder's
  explicit ``key``, when the caller keys buffers by something other than the
  RunEngine's own uid (see :class:`LiveRowRecorder`) — marked partial, and
  stamped with the recorder's opaque ``plan`` value if it was given one.
- ``descriptor``: each ``data_keys`` entry that declares a non-empty string
  ``units`` records that unit under its column name, and each that declares a
  ``shape`` (:func:`declared_shapes`) records that shape. A later descriptor
  adds units and shapes, never clears them; a column whose data key declares
  none has none. Its ``object_keys`` (:func:`declared_objects`) records which
  device recorded each column; a later descriptor adds columns to a device,
  never removes them.
- ``event``: each event's ``doc["data"]`` becomes one row; columns are
  discovered incrementally in first-seen order across events. Each cell is
  stored as a JSON reading value or a refusal (:func:`json_cell`): an array
  reading is a list, a numpy scalar is a Python scalar, a value with no
  reading spelling is ``{"refused": "<reason>"}``. A key seen for the first time
  extends the column list and backfills every already-stored row with
  ``None`` in that column, so all stored rows stay aligned to the current
  column list; a later event missing an already-known column simply records
  ``None`` for it.
- ``stop``: flips ``partial`` to ``False``. This is the ONLY thing that ends
  the "still filling in" state — a buffer that never gets a stop doc (e.g.
  the process crashed mid-run) stays partial forever, which is the honest
  answer.

Bounding uses two independent knobs:

- :func:`max_bytes_per_run`: a hard cap on the bytes of values one run
  stores, so neither a runaway plan nor a wide array reading can grow the
  buffer without limit. A row counts :data:`VALUE_BYTES` for every value it
  holds as stored (each element of an array cell, nested arrays by their
  leaves) plus each string's length. A row is stored only while the run's
  stored bytes plus the row's stay within the cap; the first row that does not
  fit ends storing for the run, so the buffer is always a prefix of the run.
  ``total_seen`` keeps counting every event past this cap — the ``row_count``
  on the run-data route reports this *true* total even when the tail beyond
  the cap was never stored.
- :func:`max_runs`: number of run buffers retained at all, oldest-evicted
  (insertion-ordered ``OrderedDict``, ``move_to_end`` on every write) once
  exceeded. This is what lets a completed run's data survive to be read later.

Both are authored per facility (``bluesky.live_max_bytes_per_run`` and
``bluesky.live_max_runs`` in the build profile) and reach the bridge as env
vars, because how many runs are worth keeping in memory, and how much a run
records, are properties of the plans a facility runs and of the memory its
deployment host has. A recorder resolves both at its start document and holds
them for that run, so a cap never moves under a buffer that is already filling.

The recorder itself never raises: every document is handled inside a
try/except so a recorder bug can never abort a run (mirrors the runner
seam's own safety contract) — a bug here loses live-read fidelity for that
run, not the run itself.
"""

from __future__ import annotations

import logging
import operator
import os
import sys
from collections import OrderedDict
from datetime import UTC, date, datetime
from threading import Lock
from typing import Any

from osprey.config_guards import require_positive_int_str

logger = logging.getLogger(__name__)

#: Env var carrying how many run buffers this bridge retains at once.
#: Authored per facility in the build profile and rendered into the compose
#: file; the container cannot read `config.yml`, so the env var is the channel.
MAX_RUNS_ENV = "BLUESKY_LIVE_MAX_RUNS"

#: Used when the variable is unset — high enough that a completed run stays
#: readable well past its own completion.
DEFAULT_MAX_RUNS = 50

#: Env var carrying the hard per-run storage cap, in bytes.
MAX_BYTES_PER_RUN_ENV = "BLUESKY_LIVE_MAX_BYTES_PER_RUN"

#: Used when the variable is unset. A safety valve against a runaway plan or a
#: wide reading, not a normal-case limit (see `total_seen` above).
DEFAULT_MAX_BYTES_PER_RUN = 20_000_000

#: What one stored value counts against the byte cap: a Python float (24
#: bytes) plus its slot in the list that holds it (8 bytes).
VALUE_BYTES = 32

#: Why a cell of bytes that are not UTF-8 text is refused (:func:`json_cell`).
REFUSED_NOT_TEXT = "not text"

#: Why a cell with no reading spelling is refused (:func:`json_cell`).
REFUSED_NOT_A_READING = "not a number, text or array"

#: numpy kinds whose ``tolist()`` is already a reading: boolean, signed and
#: unsigned integer, floating point, and unicode text.
_READING_KINDS = frozenset("biufU")

#: numpy datetime64 units coarser than a day: such a value is a date only.
_DATE_UNITS = frozenset(("Y", "M", "W", "D"))


def max_runs() -> int:
    """How many run buffers this bridge retains at once, oldest evicted first.

    Read from `os.environ` on every call — never cached at import — so a test
    (and a re-exec'd process) sees the value its environment declares.

    Raises:
        ValueError: if the variable is set to something that is not an integer,
            or to an integer below 1. Retaining zero runs would evict every
            buffer the moment it was created, which is a misconfiguration
            rather than a policy.
    """
    return require_positive_int_str(os.environ.get(MAX_RUNS_ENV), DEFAULT_MAX_RUNS, MAX_RUNS_ENV)


def max_bytes_per_run() -> int:
    """How many bytes of values one run buffer stores before it stops growing.

    Rows past this cap are counted in ``total_seen`` and not stored, so the
    run-data route keeps reporting the true total over a truncated buffer.

    Raises:
        ValueError: if the variable is set to something that is not an integer,
            or to an integer below 1.
    """
    return require_positive_int_str(
        os.environ.get(MAX_BYTES_PER_RUN_ENV), DEFAULT_MAX_BYTES_PER_RUN, MAX_BYTES_PER_RUN_ENV
    )


def _cell_bytes(cell: Any) -> int:
    """What one stored cell counts against the byte cap (see :data:`VALUE_BYTES`)."""
    if isinstance(cell, list):
        return sum(_cell_bytes(value) for value in cell)
    if isinstance(cell, str):
        return VALUE_BYTES + len(cell)
    return VALUE_BYTES


_lock = Lock()
# Each buffer is keyed by run uid and holds the run's "columns" and "rows",
# whether it is "partial", how many events it has seen ("total_seen"), its
# "plan" stamp, each column's declared "units" and "shapes", and "objects",
# the columns each device recorded.
_buffers: OrderedDict[str, dict[str, Any]] = OrderedDict()


def get(run_uid: str) -> dict[str, Any] | None:
    """A snapshot of the live buffer for *run_uid* (or None if unknown).

    ``columns``, ``rows``, ``units``, ``shapes`` and ``objects`` are copies, so a caller cannot reach
    back into the buffer through them. A cell that is a list is not copied: it
    is shared with the buffer, and a caller must not mutate it. ``plan`` is
    whatever the recorder was handed and is returned by reference — this module
    never inspects or mutates it, and neither may a caller.
    """
    with _lock:
        buf = _buffers.get(run_uid)
        if buf is None:
            return None
        return {
            "columns": list(buf["columns"]),
            "rows": [list(row) for row in buf["rows"]],
            "partial": buf["partial"],
            "total_seen": buf["total_seen"],
            "plan": buf["plan"],
            "units": dict(buf["units"]),
            "shapes": {column: list(shape) for column, shape in buf["shapes"].items()},
            "objects": {device: list(keys) for device, keys in buf["objects"].items()},
        }


def declared_units(descriptor: Any) -> dict[str, str]:
    """The units one descriptor document declares, by data key.

    A data key declares a unit when its ``units`` is a non-empty string; any
    other entry, and any malformed descriptor, declares none.
    """
    data_keys = descriptor.get("data_keys") if isinstance(descriptor, dict) else None
    if not isinstance(data_keys, dict):
        return {}
    return {
        column: key["units"]
        for column, key in data_keys.items()
        if isinstance(key, dict) and isinstance(key.get("units"), str) and key["units"]
    }


def declared_shapes(descriptor: Any) -> dict[str, list[int | None]]:
    """The shapes one descriptor document declares, by data key.

    A data key declares a shape when its ``shape`` is a list whose every entry
    is an integral number >= 0 (any value ``operator.index`` accepts other than
    a bool), stored as an ``int``, or ``None`` for a ragged axis; ``[]`` is a
    scalar. Any other entry, and any malformed descriptor, declares none.
    """
    data_keys = descriptor.get("data_keys") if isinstance(descriptor, dict) else None
    if not isinstance(data_keys, dict):
        return {}
    return {
        column: shape
        for column, key in data_keys.items()
        if isinstance(key, dict) and (shape := _shape(key.get("shape"))) is not None
    }


def declared_objects(descriptor: Any) -> dict[str, list[str]]:
    """The columns each device recorded, as one descriptor document states them.

    A descriptor's ``object_keys`` maps each device the stream read to the data
    keys it recorded; an entry whose value is a list of strings declares them,
    any other entry, and any malformed descriptor, declares none.
    """
    object_keys = descriptor.get("object_keys") if isinstance(descriptor, dict) else None
    if not isinstance(object_keys, dict):
        return {}
    return {
        device: list(keys)
        for device, keys in object_keys.items()
        if isinstance(device, str)
        and isinstance(keys, list)
        and all(isinstance(key, str) for key in keys)
    }


def merge_objects(held: dict[str, list[str]], declared: dict[str, list[str]]) -> None:
    """Add *declared*'s columns to *held*, device by device, never removing one."""
    for device, keys in declared.items():
        columns = held.setdefault(device, [])
        columns.extend(key for key in keys if key not in columns)


def _shape(shape: Any) -> list[int | None] | None:
    """*shape* as plain ``int`` sizes and ``None`` ragged axes, or ``None`` if it is not one."""
    if not isinstance(shape, list):
        return None
    sizes: list[int | None] = []
    for size in shape:
        if size is None:
            sizes.append(None)
            continue
        if isinstance(size, bool):
            return None
        try:
            number = operator.index(size)
        except TypeError:
            return None
        if number < 0:
            return None
        sizes.append(number)
    return sizes


def json_cell(value: Any) -> Any:
    """One cell of a run's table as a JSON reading value or a refusal.

    The one rule both sources of a run apply, the live buffer as it stores a
    cell and the catalog snapshot as it builds its rows, so a live run and a
    stored run serve the same cell:

    - ``None``, a boolean, a number or text (numpy scalars of kind ``b i u f
      U`` included) is itself; NaN and infinity stay floats.
    - A missing value of any kind (numpy ``NaT``, pandas ``NaT``, pandas
      ``NA``) is ``None``, like a missing number.
    - A numpy array of kind ``b i u f U`` is ``tolist()``: nested lists of
      Python scalars, never walked element by element.
    - A list, a tuple, or a numpy array of kind ``S`` or ``O`` is a list, each
      element by this rule; one refused element refuses the whole cell.
    - Bytes (``bytes``, ``bytearray``, ``numpy.bytes_``) are their UTF-8
      text, or refused as ``not text``.
    - A date or a date and time (``datetime.date``, ``datetime.datetime``,
      ``pandas.Timestamp``, ``numpy.datetime64``) is ISO 8601 text in UTC:
      a zoned value converted to UTC, a value without a zone taken as UTC,
      ending in ``Z``, with a fraction of a second only when it is non-zero and
      without trailing zeros (``2026-11-24T10:00:00.5Z``); a date without a
      time is ``2026-11-24``.
    - Anything else (a complex number, a set, a dict, a ``Decimal``, a
      duration, numpy kinds ``c m V``, any other object) is refused as
      ``not a number, text or array``.

    A refusal is a fresh ``{"refused": "<reason>"}``, which no reading can be
    mistaken for, since every dict a run records is itself refused.
    """
    try:
        return _plain(value)
    except _Refused as refused:
        return {"refused": refused.reason}


class _Refused(Exception):
    """A value :func:`json_cell` has no reading spelling for."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _plain(value: Any) -> Any:
    """*value* as a JSON reading value, or raise `_Refused`."""
    if _is_pandas_missing(value):
        return None
    kind = getattr(getattr(value, "dtype", None), "kind", None)
    if isinstance(kind, str) and kind:
        return _plain_numpy(value, kind)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, datetime):
        return _utc_text(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray)):
        return _text(value)
    if isinstance(value, (list, tuple)):
        return [_plain(element) for element in value]
    raise _Refused(REFUSED_NOT_A_READING)


def _is_pandas_missing(value: Any) -> bool:
    """Whether *value* is pandas' ``NaT`` or ``NA``; never imports pandas."""
    pandas = sys.modules.get("pandas")
    return pandas is not None and (value is pandas.NaT or value is pandas.NA)


def _plain_numpy(value: Any, kind: str) -> Any:
    """A numpy scalar or array of dtype *kind* as a JSON reading value."""
    if kind in _READING_KINDS:
        return value.tolist()
    if kind in "Mm" and not getattr(value, "ndim", 0):
        import numpy as np

        if np.isnat(value):
            return None
    if kind not in "MSO":
        raise _Refused(REFUSED_NOT_A_READING)
    if getattr(value, "ndim", 0):
        return [_plain(element) for element in value]
    if kind == "M":
        return _datetime64_text(value)
    return _plain(value.item())


def _text(raw: bytes | bytearray) -> str:
    try:
        return bytes(raw).decode("utf-8")
    except UnicodeDecodeError:
        raise _Refused(REFUSED_NOT_TEXT) from None


def _utc_text(moment: datetime, extra_nanoseconds: int = 0) -> str:
    """*moment* as ISO 8601 text in UTC (see :func:`json_cell`).

    *extra_nanoseconds* are those below the microsecond a `datetime` cannot
    hold; a ``pandas.Timestamp`` carries its own as ``nanosecond``.
    """
    if moment.utcoffset() is not None:
        moment = moment.astimezone(UTC)
    nanoseconds = moment.microsecond * 1000 + getattr(moment, "nanosecond", 0) + extra_nanoseconds
    fraction = f".{nanoseconds:09d}".rstrip("0") if nanoseconds else ""
    return (
        f"{moment.date().isoformat()}T{moment.hour:02d}:{moment.minute:02d}"
        f":{moment.second:02d}{fraction}Z"
    )


def _datetime64_text(value: Any) -> str:
    """A `numpy.datetime64` scalar as ISO 8601 text in UTC (see :func:`json_cell`)."""
    import numpy as np

    unit, _ = np.datetime_data(value.dtype)
    if unit in _DATE_UNITS:
        day = value.astype("datetime64[D]").item()
        if not isinstance(day, date):
            raise _Refused(REFUSED_NOT_A_READING)
        return day.isoformat()
    microseconds = value.astype("datetime64[us]")
    moment = microseconds.item()
    if not isinstance(moment, datetime):
        raise _Refused(REFUSED_NOT_A_READING)
    below = int((value - microseconds).astype("timedelta64[ns]").astype("int64"))
    return _utc_text(moment, below)


def _clear() -> None:
    """Drop all buffers (test isolation only)."""
    with _lock:
        _buffers.clear()


class LiveRowRecorder:
    """Bluesky document callback recording rows into the bounded live buffer.

    One instance per run (mirrors BELLA's contract): it must see the run's
    start document, so it is opened at that document rather than partway
    through the stream.

    Args:
        key: Buffer key to record under, instead of the start document's own
            ``uid``. The document plane (``document_plane.py``) sets it to the
            OSPREY run id carried in the start document, because a run executed
            in the queueserver container has a RunEngine uid the bridge never
            chose and cannot map back to the item an operator enqueued. Left
            ``None``, the run uid IS the identity the read path looks up —
            which is the honest key for a run that carries no OSPREY id.
        plan: Which plan produced this run, carried alongside its rows so a
            reader that has only a run id can tell what the columns mean. The
            value is opaque here: this module stores it and hands it back in
            :func:`get`, and never interprets, validates, or copies it. The
            document plane extracts it from the start document (mirroring
            ``key``), which keeps this module testable with synthetic
            documents. ``None`` for a run whose start document carried no plan
            stamp — an honest "not known", never a placeholder.
    """

    def __init__(self, key: str | None = None, plan: Any | None = None) -> None:
        self._key = key
        self._plan = plan
        self._uid: str | None = None
        self._max_bytes: int = DEFAULT_MAX_BYTES_PER_RUN
        self._stored_bytes = 0
        self._full = False

    def __call__(self, name: str, doc: dict[str, Any]) -> None:
        try:
            if name == "start":
                self._on_start(doc)
            elif name == "descriptor":
                self._on_descriptor(doc)
            elif name == "event":
                self._on_event(doc)
            elif name == "stop":
                self._on_stop(doc)
        except Exception:
            # RunEngine thread — a recorder bug must never touch the plan.
            logger.warning("live-row recorder failed on %r doc", name, exc_info=True)

    def _on_start(self, doc: dict[str, Any]) -> None:
        uid = self._key or doc.get("uid")
        if not uid:
            return
        self._uid = uid
        # Both caps are resolved once, here, and not again for the rest of the
        # run: `_on_event` runs per event document on the RunEngine's thread,
        # and a cap that changes mid-run would move the boundary of a buffer
        # that is already half-filled under it.
        self._max_bytes = max_bytes_per_run()
        self._stored_bytes = 0
        self._full = False
        retained_runs = max_runs()
        with _lock:
            _buffers[uid] = {
                "columns": [],
                "rows": [],
                "partial": True,
                "total_seen": 0,
                "plan": self._plan,
                "units": {},
                "shapes": {},
                "objects": {},
            }
            _buffers.move_to_end(uid)
            while len(_buffers) > retained_runs:
                _buffers.popitem(last=False)

    def _on_descriptor(self, doc: dict[str, Any]) -> None:
        if not self._uid:
            return
        units = declared_units(doc)
        shapes = declared_shapes(doc)
        objects = declared_objects(doc)
        if not units and not shapes and not objects:
            return
        with _lock:
            buf = _buffers.get(self._uid)
            if buf is not None:
                buf["units"].update(units)
                buf["shapes"].update(shapes)
                merge_objects(buf["objects"], objects)

    def _on_event(self, doc: dict[str, Any]) -> None:
        if not self._uid:
            return
        data = doc.get("data") or {}
        with _lock:
            buf = _buffers.get(self._uid)
            if buf is None:
                return
            new_columns = [key for key in data if key not in buf["columns"]]
            if new_columns:
                buf["columns"].extend(new_columns)
                for row in buf["rows"]:
                    row.extend([None] * len(new_columns))
            buf["total_seen"] += 1
            if not self._full:
                row = [json_cell(data.get(col)) for col in buf["columns"]]
                row_bytes = sum(_cell_bytes(cell) for cell in row)
                if self._stored_bytes + row_bytes <= self._max_bytes:
                    buf["rows"].append(row)
                    self._stored_bytes += row_bytes
                else:
                    self._full = True
            _buffers.move_to_end(self._uid)

    def _on_stop(self, _doc: dict[str, Any]) -> None:
        if not self._uid:
            return
        with _lock:
            buf = _buffers.get(self._uid)
            if buf is not None:
                buf["partial"] = False
