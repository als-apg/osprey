"""The ORM cache: one model-derived orbit response matrix per model, built off the request path.

Each model has one entry. A build asks the SDK's ``model_response_matrix`` for every
corrector the model wires, in a background thread; a success swaps in a new immutable
snapshot under the cache lock and bumps the entry's version, a refusal or a timeout marks
the entry failed with the reason and drops the old matrix. Reads take the lock only to
copy a reference, so they never wait for a build.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

_log = logging.getLogger(__name__)

#: No build has run for the model yet.
IDLE = "idle"
#: A build is running; the previous snapshot, if any, is still served.
BUILDING = "building"
#: The last build succeeded; its snapshot is served.
READY = "ready"
#: The last build was refused or timed out; no snapshot is served.
FAILED = "failed"

#: Seconds a build may take before the SDK stops it.
DEFAULT_TIMEOUT_S = 120.0


@dataclass(frozen=True)
class OrmSnapshot:
    """One built response matrix of a model, never changed once made.

    ``rows`` are BPM position addresses, ``columns`` the solved corrector setpoint
    addresses, ``matrix[r, c]`` the response of row ``r`` to column ``c``; ``unsolved``
    maps each corrector setpoint address the model could not solve to the reason.
    """

    version: int
    rows: tuple[str, ...]
    columns: tuple[str, ...]
    matrix: np.ndarray
    unsolved: MappingProxyType[str, str]
    _row_at: dict[str, int] = field(repr=False, compare=False)
    _column_at: dict[str, int] = field(repr=False, compare=False)

    @classmethod
    def of(
        cls,
        version: int,
        rows: Sequence[str],
        columns: Sequence[str],
        matrix: Sequence[Sequence[float]],
        unsolved: dict[str, str],
    ) -> OrmSnapshot:
        """A snapshot over a copy of the SDK's result, its matrix made read-only."""
        array = np.array(matrix, dtype=float).reshape(len(rows), len(columns))
        array.flags.writeable = False
        return cls(
            version=version,
            rows=tuple(rows),
            columns=tuple(columns),
            matrix=array,
            unsolved=MappingProxyType(dict(unsolved)),
            _row_at={address: at for at, address in enumerate(rows)},
            _column_at={address: at for at, address in enumerate(columns)},
        )

    def slice(self, bpm_addrs: Sequence[str], corrector_addrs: Sequence[str]) -> np.ndarray:
        """The response of ``bpm_addrs`` to ``corrector_addrs``, in the order asked.

        Raises:
            KeyError: An address is not a row, or not a solved column, of this matrix.
        """
        rows = [_position(self._row_at, address, "BPM", self.unsolved) for address in bpm_addrs]
        columns = [
            _position(self._column_at, address, "corrector", self.unsolved)
            for address in corrector_addrs
        ]
        return self.matrix[np.ix_(rows, columns)]


def _position(at: dict[str, int], address: str, kind: str, unsolved: Any) -> int:
    try:
        return at[address]
    except KeyError:
        reason = unsolved.get(address)
        detail = f": unsolved, {reason}" if reason is not None else ""
        raise KeyError(f"{kind} {address} is not in the response matrix{detail}") from None


@dataclass(frozen=True)
class OrmState:
    """What a model's entry says now: what ``GET /api/state`` reports for its ORM."""

    status: str
    version: int
    built_at: float | None
    error: str | None
    unsolved: dict[str, str]


@dataclass
class _Entry:
    status: str = IDLE
    version: int = 0
    built_at: float | None = None
    error: str | None = None
    snapshot: OrmSnapshot | None = None
    done: threading.Event = field(default_factory=threading.Event)

    def __post_init__(self) -> None:
        self.done.set()


def _default_sdk() -> Any:
    from osprey.apps import sdk

    return sdk


class OrmCache:
    """The per-model ORM entries and the background builds that fill them.

    Args:
        sdk: The object holding ``closed_orbit_models`` and ``model_response_matrix``;
            the application SDK when omitted.
        timeout_s: Seconds each build may take before the SDK stops it.
        clock: The wall clock ``built_at`` is read from.
        on_ready: Called with the model's name after a successful build has swapped
            in its snapshot, outside the lock; what it raises is logged and dropped.
    """

    def __init__(
        self,
        sdk: Any = None,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        clock: Callable[[], float] = time.time,
        on_ready: Callable[[str], object] | None = None,
    ) -> None:
        self._sdk = _default_sdk() if sdk is None else sdk
        self._timeout_s = timeout_s
        self._clock = clock
        self._on_ready = on_ready
        self._lock = threading.Lock()
        self._entries: dict[str, _Entry] = {}

    def start(self) -> str | None:
        """Start the first closed-orbit model's build; its name, or None when there is none."""
        models: list[str] = list(self._sdk.closed_orbit_models())
        if not models:
            return None
        self.ensure(models[0])
        return models[0]

    def ensure(self, model: str) -> OrmState:
        """Start the model's build when it has never run; the entry's state either way."""
        with self._lock:
            entry = self._entries.setdefault(model, _Entry())
            if entry.status == IDLE:
                self._launch(model, entry)
            return _state_of(entry)

    def recompute(self, model: str) -> bool:
        """Start a new build of the model; False when one is already running."""
        with self._lock:
            entry = self._entries.setdefault(model, _Entry())
            if entry.status == BUILDING:
                return False
            self._launch(model, entry)
            return True

    def state(self, model: str) -> OrmState:
        """The model's entry as it stands; never waits for a build."""
        with self._lock:
            entry = self._entries.get(model)
            return _state_of(entry if entry is not None else _Entry())

    def snapshot(self, model: str) -> OrmSnapshot | None:
        """The model's current snapshot, or None when none is served."""
        with self._lock:
            entry = self._entries.get(model)
            return None if entry is None else entry.snapshot

    def join(self, model: str, timeout: float | None = None) -> bool:
        """Wait for the model's running build to end; False when ``timeout`` ran out."""
        with self._lock:
            entry = self._entries.get(model)
            if entry is None:
                return True
            done = entry.done
        return done.wait(timeout)

    def _launch(self, model: str, entry: _Entry) -> None:
        # Called under the lock.
        entry.status = BUILDING
        entry.done = threading.Event()
        threading.Thread(
            target=self._build,
            args=(model, entry.done),
            name=f"orm-build-{model}",
            daemon=True,
        ).start()

    def _build(self, model: str, done: threading.Event) -> None:
        try:
            try:
                response = self._sdk.model_response_matrix(model, None, timeout_s=self._timeout_s)
            except Exception as error:  # every failure is the entry's reason
                self._fail(model, error)
                return
            self._swap(model, response)
        finally:
            done.set()

    def _fail(self, model: str, error: Exception) -> None:
        reason = str(error) or type(error).__name__
        _log.warning("the response matrix of %s was not built: %s", model, reason)
        with self._lock:
            entry = self._entries[model]
            entry.status = FAILED
            entry.error = reason
            entry.built_at = None
            entry.snapshot = None

    def _swap(self, model: str, response: Any) -> None:
        try:
            built_at = self._clock()
            with self._lock:
                entry = self._entries[model]
                snapshot = OrmSnapshot.of(
                    entry.version + 1,
                    response.rows,
                    response.columns,
                    response.matrix,
                    response.unsolved,
                )
                entry.snapshot = snapshot
                entry.version = snapshot.version
                entry.status = READY
                entry.error = None
                entry.built_at = built_at
        except Exception as error:  # a malformed result fails the entry
            self._fail(model, error)
            return
        if self._on_ready is not None:
            try:
                self._on_ready(model)
            except Exception:  # the hook's failure is not the build's
                _log.exception("the ready hook for the response matrix of %s raised", model)


def _state_of(entry: _Entry) -> OrmState:
    snapshot = entry.snapshot
    return OrmState(
        status=entry.status,
        version=entry.version,
        built_at=entry.built_at,
        error=entry.error,
        unsolved={} if snapshot is None else dict(snapshot.unsolved),
    )
