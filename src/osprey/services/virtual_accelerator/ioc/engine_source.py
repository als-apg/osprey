"""Drives partition (c) ("static/noisy") IOC records from the in-image
SimulationEngine, and detects scenario switches on the bind-mounted
``data/simulation/`` directory.

Partition (c) is everything the record factory (``ioc.records``) builds as a
plain In-type record with no wired write behavior -- GOLDEN references,
STATUS flags, temperatures, pressures, and any other channel with no lattice
physics and no SP->RB echo pairing. This module is their only value source:
on each poll tick it reads the effective value for every partition-(c)
address through the SAME :class:`~osprey.simulation.engine.SimulationEngine`
used by the mock connector (never a second implementation -- see the module
docstring on ``engine.py``), and pushes it onto the record with ``.set()``.

Not every partition-(c) address is necessarily defined in the bind-mounted
``machine.json`` (only a scenario-relevant subset is -- see
``machine-json-lattice-augment``); addresses the engine doesn't serve fall
back to the same generic PV-taxonomy synthesis the mock connector itself uses
for unknown channels (``osprey.connectors.pv_taxonomy.classify_pv``), so mock
and VA present identical values for anything neither one has real data for.

Scenario-switch detection deliberately does its own (mtime, content-hash)
comparison of ``active_scenarios`` in addition to the engine's own internal
mtime-only tracking: the data directory is bind-mounted (not a single file)
specifically so an atomic-rename swap survives the mount, but the replacement
file's mtime can still collide with the old one at the host filesystem's
mtime granularity. The content hash is strictly more sensitive than mtime
alone and catches that edge case.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np

from osprey.connectors.pv_taxonomy import classify_pv
from osprey.services.virtual_accelerator.manifest import PARTITION_STATIC_NOISY, RECORD_TYPE_BINARY
from osprey.simulation.engine import SimulationEngine, engine_serves

ACTIVE_SCENARIOS_FILENAME = "active_scenarios"
DEFAULT_POLL_INTERVAL_S = 1.0
DEFAULT_NOISE_LEVEL = 0.01

_Signature = tuple[int, str]
_NO_SIGNATURE: _Signature = (-1, "")


class EngineSource:
    """Poll-driven bridge from a :class:`SimulationEngine` to a set of
    partition-(c) IOC records.

    Args:
        engine: The in-image simulation engine, already loaded from the
            bind-mounted ``machine.json``.
        channels: The namespace-union manifest's ``channels`` list (or any
            subset covering the addresses in ``static_noisy_records``) --
            used to look up each address's ``record_type``/``noise`` flag for
            the legacy-fallback synthesis path.
        static_noisy_records: The ``static_noisy`` dict from
            :func:`ioc.records.build_records` -- address -> softioc In-type
            record. This instance drives exactly these records; addresses in
            ``channels`` that aren't also keys here are ignored.
        data_dir: The bind-mounted ``data/simulation/`` directory (the mount
            unit). ``active_scenarios`` is read from directly under it, fresh
            on every poll tick -- never through a cached file handle, so a
            directory-level atomic-rename swap is always observed.
        noise_level: Relative noise fraction for the legacy-fallback
            synthesis path (mirrors ``MockConnector``'s own default).
        setpoint_echo_records: Optional ``address -> record`` map of sp-echo
            READBACK records whose current values are synced into the engine
            (``engine.write``) at the top of every poll tick, BEFORE the
            driven records are computed. This is what lets a machine-file
            expression channel (e.g. a camera response curve) depend on the
            latest accepted setpoint state when the engine is the only
            physics in the process -- the no-lattice entrypoint path wires
            it. With a PhysicsBridge, physics coupling flows through the
            bridge instead and this stays ``None`` (the default; behaviour
            unchanged). Addresses the engine does not serve are dropped at
            construction.
    """

    def __init__(
        self,
        engine: SimulationEngine,
        channels: list[dict],
        static_noisy_records: dict[str, Any],
        data_dir: Path,
        *,
        noise_level: float = DEFAULT_NOISE_LEVEL,
        setpoint_echo_records: dict[str, Any] | None = None,
    ) -> None:
        self._engine = engine
        self._records = dict(static_noisy_records)
        self._noise_level = noise_level
        self._rng = np.random.default_rng()
        self._legacy_base: dict[str, float] = {}
        self._setpoint_echo_records = {
            address: record
            for address, record in (setpoint_echo_records or {}).items()
            if engine_serves(engine, address)
        }

        self._channel_info: dict[str, tuple[str, bool]] = {
            c["address"]: (c["record_type"], bool(c["noise"]))
            for c in channels
            if c["partition"] == PARTITION_STATIC_NOISY and c["address"] in self._records
        }

        self._state_path = Path(data_dir) / ACTIVE_SCENARIOS_FILENAME
        self._last_signature: _Signature | None = None

    def poll_once(self) -> bool:
        """Run one poll iteration: detect a scenario switch (if any), then
        push a fresh value onto every driven record.

        Returns:
            True if this tick observed a changed ``active_scenarios``
            signature (a switch or an explicit re-assert); False otherwise.
            The first call always returns False (nothing to compare against
            yet), even though it establishes the baseline signature.
        """
        switched = self._detect_scenario_switch()
        # Sync accepted setpoint state into the engine first, so every
        # expression evaluated below sees this tick's setpoints.
        for address, record in self._setpoint_echo_records.items():
            try:
                self._engine.write(address, record.get())
            except Exception:  # noqa: BLE001 -- same isolation rationale as below
                import traceback

                print(
                    f"EngineSource: failed to sync setpoint echo {address!r} into the engine:",
                    file=sys.stderr,
                    flush=True,
                )
                traceback.print_exc()
        for address, record in self._records.items():
            try:
                record.set(self._read_value(address))
            except Exception:  # noqa: BLE001 -- see run_forever docstring
                import traceback

                print(
                    f"EngineSource: failed to set {address!r}, skipping this tick:",
                    file=sys.stderr,
                    flush=True,
                )
                traceback.print_exc()
        return switched

    async def run_forever(self, interval: float = DEFAULT_POLL_INTERVAL_S) -> None:
        """Poll indefinitely at ``interval`` seconds (for the IOC's asyncio
        dispatcher event loop; see ``softioc.asyncio_dispatcher``).

        ``poll_once()`` must never let one record's failure raise out of
        this loop: it is scheduled via
        ``asyncio.run_coroutine_threadsafe(...)`` in ``entrypoint.py`` and
        nothing retrieves the returned future's result, so an uncaught
        exception here would silently kill the poll loop forever after its
        first tick -- every static-noisy record would then be frozen at its
        record-default value with no visible error. See ``poll_once``'s
        per-record try/except.
        """
        import asyncio

        while True:
            self.poll_once()
            await asyncio.sleep(interval)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _detect_scenario_switch(self) -> bool:
        signature = self._compute_signature()
        first_tick = self._last_signature is None
        changed = not first_tick and signature != self._last_signature
        self._last_signature = signature

        if changed:
            # Force the engine to reparse active_scenarios even if ITS OWN
            # mtime-only signature didn't move (see module docstring): the
            # same escape hatch SimulationEngine.set_active_scenarios() uses
            # internally ("Force a re-read even if filesystem mtime
            # granularity hides the write" -- osprey/simulation/engine.py).
            self._engine._state_signature = ("", -1)  # noqa: SLF001

        return changed

    def _compute_signature(self) -> _Signature:
        try:
            mtime_ns = self._state_path.stat().st_mtime_ns
            content_hash = hashlib.sha256(self._state_path.read_bytes()).hexdigest()
        except FileNotFoundError:
            return _NO_SIGNATURE
        return (mtime_ns, content_hash)

    def _read_value(self, address: str) -> float | bool | str:
        if engine_serves(self._engine, address):
            return self._coerce(address, self._engine.read(address).value)
        return self._legacy_value(address)

    def _coerce(self, address: str, value: Any) -> float | bool | str:
        """Coerce an engine-served value to the record's actual type.

        ``SimulationEngine`` stores every channel's value as a plain float
        (see ``machine.json``), even for partition-(c) channels built as
        binary (``bi``) records (e.g. ``STATUS:VALID`` flags). Passing that
        raw float straight to a ``bi`` record's ``.set()`` raises
        ``TypeError`` inside softioc's ctypes conversion -- and because this
        runs inside ``poll_once()``'s single per-tick loop over every
        static-noisy record, one such mismatch kills the whole poll
        iteration (and, via ``run_forever``, every future tick) rather than
        just that one channel.
        """
        record_type, _noisy = self._channel_info.get(address, ("ai", False))
        if record_type == RECORD_TYPE_BINARY:
            return bool(value)
        return value

    def _legacy_value(self, address: str) -> float | bool:
        record_type, noisy = self._channel_info.get(address, ("ai", False))
        base = self._legacy_base.get(address)
        if base is None:
            base = classify_pv(address).base_value
            self._legacy_base[address] = base

        if record_type == RECORD_TYPE_BINARY:
            return bool(base)

        if not noisy:
            return base
        return base * (1.0 + float(self._rng.normal(0.0, self._noise_level)))
