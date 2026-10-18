"""The trajectory service's files: shot results and named references, per resolved model.

Everything lives under the directory ``OSPREY_TRAJECTORY_STATE_DIR`` names::

    shots/<model>/<shot_id>.json        one ShotResult each, newest kept
    references/<model>/<name>.json      one reference each
    references/<model>/current.json     {"name": <reference>} or {"name": null}

Every model, shot id and reference name matches :data:`NAME_PATTERN` and is
neither ``.`` nor ``..``, so no name can leave its directory; ``current`` is
not a reference name because its file is the pointer. A model keeps at most
``shots_per_model`` shot files; saving one more deletes the oldest.

A reference holds one ``(x, y)`` reading per BPM name. NaN is written as JSON
``null`` (every file is strict JSON) and read back as NaN. Loading a reference
matches it over the current BPM selection: :class:`ReferenceLoad` names the
matched BPMs, the selected BPMs the reference lacks (they read NaN) and the
reference BPMs outside the selection, counts the reference's NaN readings, and
carries the sha256 of the file's bytes so a step digest can bind the exact
content it was solved against. A reference matching no selected BPM is refused.

All disk I/O runs in a worker thread through :func:`asyncio.to_thread`, under
one :class:`asyncio.Lock` per store, and every file is written to a temporary
file in its own directory and moved into place with :func:`os.replace`, so a
reader sees either the previous complete file or the new one, never a part.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import time
from collections.abc import Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osprey.services.trajectory.state import ReferenceLoad, ShotResult
from osprey_connectors.safety_file import atomic_replace

__all__ = [
    "CURRENT_FILE",
    "FILE_MODE",
    "NAME_PATTERN",
    "SHOTS_PER_MODEL",
    "STATE_DIR_ENV_VAR",
    "InvalidName",
    "LoadedReference",
    "NoMatchingBpms",
    "ReferenceNotFound",
    "StoreUnconfigured",
    "TrajectoryStore",
]

#: The directory the deployment mounts for the service's shot files and references.
STATE_DIR_ENV_VAR = "OSPREY_TRAJECTORY_STATE_DIR"

#: Every model, shot id and reference name matches this.
NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

#: How many shot files each model keeps by default.
SHOTS_PER_MODEL = 100

#: The pointer to a model's current reference, beside the references.
CURRENT_FILE = "current.json"

#: The mode every shot file, reference and pointer is written with: the owner
#: writes, the group reads, nobody else. The deployment provisions the state
#: directory setgid to the host's group, so the group read is what lets the
#: host read a shot the service wrote under its own user.
FILE_MODE = 0o640

_CURRENT_NAME = CURRENT_FILE.removesuffix(".json")
_SUFFIX = ".json"


class StoreUnconfigured(RuntimeError):
    """``OSPREY_TRAJECTORY_STATE_DIR`` is unset or empty."""


class InvalidName(ValueError):
    """A model, shot id or reference name that does not match :data:`NAME_PATTERN`."""


class ReferenceNotFound(LookupError):
    """No reference of that name is stored for the model."""


class NoMatchingBpms(ValueError):
    """The reference names none of the selected BPMs."""


@dataclass(frozen=True)
class LoadedReference:
    """A reference matched over a BPM selection.

    Attributes:
        load: What matched, what is missing, what is unused, the NaN count and
            the content sha256.
        x: Horizontal reading by selected BPM, in selection order; NaN where the
            reference lacks the BPM or holds NaN.
        y: Vertical reading by selected BPM, likewise.
    """

    load: ReferenceLoad
    x: dict[str, float]
    y: dict[str, float]


def _check_name(name: object, what: str) -> str:
    if not isinstance(name, str) or not NAME_PATTERN.fullmatch(name) or name in {".", ".."}:
        raise InvalidName(
            f"{what} {name!r} is not a valid name: use 1 to 64 of A-Z a-z 0-9 . _ -, "
            "and not '.' or '..'"
        )
    return name


def _check_reference_name(name: object) -> str:
    _check_name(name, "reference name")
    if name == _CURRENT_NAME:
        raise InvalidName(f"reference name {_CURRENT_NAME!r} is reserved for the pointer file")
    return name  # type: ignore[return-value]


def _to_json_reading(value: float | None, where: str) -> float | None:
    if value is None:
        return None
    number = float(value)
    if math.isnan(number):
        return None
    if math.isinf(number):
        raise ValueError(f"reference reading {where} is infinite")
    return number


def _from_json_reading(value: Any) -> float:
    return math.nan if value is None else float(value)


def _write_atomic(path: Path, payload: bytes) -> None:
    """Replace ``path`` with ``payload`` at :data:`FILE_MODE`, whole or not at all.

    A failed flush raises: every write here reaches the operator, who can retry.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_replace(path, payload, mode=FILE_MODE, sync="required")


def _dumps(data: Any) -> bytes:
    return json.dumps(data, indent=2, sort_keys=True, allow_nan=False).encode()


class TrajectoryStore:
    """Shot files and references under one state directory.

    Args:
        root: The state directory; created on the first write.
        shots_per_model: How many shot files each model keeps, at least 1.
    """

    def __init__(self, root: Path | str, *, shots_per_model: int = SHOTS_PER_MODEL) -> None:
        if shots_per_model < 1:
            raise ValueError(f"shots_per_model must be at least 1, not {shots_per_model}")
        self.root = Path(root)
        self.shots_per_model = shots_per_model
        self._lock = asyncio.Lock()
        self._last_stamp_ns = 0

    @classmethod
    def from_env(cls, *, shots_per_model: int = SHOTS_PER_MODEL) -> TrajectoryStore:
        """The store rooted at ``OSPREY_TRAJECTORY_STATE_DIR``.

        Raises:
            StoreUnconfigured: The variable is unset or empty.
        """
        value = os.environ.get(STATE_DIR_ENV_VAR, "").strip()
        if not value:
            raise StoreUnconfigured(
                f"{STATE_DIR_ENV_VAR} is unset; the service keeps its shot files and "
                "references in that directory"
            )
        return cls(Path(value), shots_per_model=shots_per_model)

    # --- paths -----------------------------------------------------------------------

    def _shot_dir(self, model: str) -> Path:
        return self.root / "shots" / _check_name(model, "model")

    def _reference_dir(self, model: str) -> Path:
        return self.root / "references" / _check_name(model, "model")

    async def _run(self, fn: Any, *args: Any) -> Any:
        async with self._lock:
            return await asyncio.to_thread(fn, *args)

    # --- shots -----------------------------------------------------------------------

    async def save_shot(self, shot: ShotResult) -> Path:
        """Write ``shot`` as ``shots/<model>/<shot_id>.json`` and prune the model's oldest.

        Re-saving a shot id replaces its file and makes it the newest.

        Returns:
            The shot file's path.
        """
        directory = self._shot_dir(shot.model)
        path = directory / f"{_check_name(shot.shot_id, 'shot id')}{_SUFFIX}"
        payload = shot.model_dump_json(indent=2).encode()
        await self._run(self._save_shot_sync, path, payload)
        return path

    def _save_shot_sync(self, path: Path, payload: bytes) -> None:
        _write_atomic(path, payload)
        # Order by a stamp this store makes strictly increasing, so two saves in
        # one clock tick still prune in the order they were made.
        stamp = max(time.time_ns(), self._last_stamp_ns + 1)
        self._last_stamp_ns = stamp
        os.utime(path, ns=(stamp, stamp))
        for stale in self._shot_files_newest_first(path.parent)[self.shots_per_model :]:
            with suppress(FileNotFoundError):
                stale.unlink()

    @staticmethod
    def _shot_files_newest_first(directory: Path) -> list[Path]:
        if not directory.is_dir():
            return []
        files = [
            p for p in directory.iterdir() if p.suffix == _SUFFIX and not p.name.startswith(".")
        ]
        return sorted(files, key=lambda p: (p.stat().st_mtime_ns, p.name), reverse=True)

    async def load_shot(self, model: str, shot_id: str) -> ShotResult:
        """The stored shot ``shot_id`` of ``model``.

        Raises:
            LookupError: No such shot is stored (it may have been pruned).
        """
        path = self._shot_dir(model) / f"{_check_name(shot_id, 'shot id')}{_SUFFIX}"
        try:
            text = await self._run(path.read_text)
        except FileNotFoundError:
            raise LookupError(f"no shot {shot_id!r} is stored for model {model!r}") from None
        return ShotResult.model_validate_json(text)

    async def list_shots(self, model: str) -> list[ShotResult]:
        """Every stored shot of ``model``, newest first."""
        directory = self._shot_dir(model)

        def read_all() -> list[str]:
            return [p.read_text() for p in self._shot_files_newest_first(directory)]

        return [ShotResult.model_validate_json(text) for text in await self._run(read_all)]

    # --- references ------------------------------------------------------------------

    async def save_reference(
        self, model: str, name: str, readings: Mapping[str, tuple[float | None, float | None]]
    ) -> str:
        """Write the reference ``name`` of ``model``: one ``(x, y)`` reading per BPM.

        NaN or ``None`` readings are stored as ``null``; an infinite reading or
        an empty reference is refused.

        Returns:
            The sha256 of the written file's bytes.
        """
        directory = self._reference_dir(model)
        _check_reference_name(name)
        if not readings:
            raise ValueError(f"reference {name!r} has no BPM readings")
        body = {
            "model": model,
            "name": name,
            "readings": {
                str(bpm): {
                    "x": _to_json_reading(x, f"x at {bpm}"),
                    "y": _to_json_reading(y, f"y at {bpm}"),
                }
                for bpm, (x, y) in readings.items()
            },
        }
        payload = _dumps(body)
        await self._run(_write_atomic, directory / f"{name}{_SUFFIX}", payload)
        return hashlib.sha256(payload).hexdigest()

    async def load_reference(
        self, model: str, name: str, selection: Iterable[str]
    ) -> LoadedReference:
        """The reference ``name`` of ``model``, matched over the selected BPMs.

        Raises:
            ReferenceNotFound: No such reference is stored.
            NoMatchingBpms: The reference names none of the selected BPMs.
        """
        path = self._reference_dir(model) / f"{_check_reference_name(name)}{_SUFFIX}"
        try:
            payload: bytes = await self._run(path.read_bytes)
        except FileNotFoundError:
            raise ReferenceNotFound(
                f"no reference {name!r} is stored for model {model!r}"
            ) from None
        readings: dict[str, dict[str, Any]] = json.loads(payload)["readings"]
        selected = list(dict.fromkeys(selection))
        chosen = set(selected)
        matched = [bpm for bpm in selected if bpm in readings]
        if not matched:
            raise NoMatchingBpms(
                f"reference {name!r} of model {model!r} names none of the {len(selected)} "
                "selected BPMs"
            )
        x = {
            bpm: _from_json_reading(readings[bpm]["x"]) if bpm in readings else math.nan
            for bpm in selected
        }
        y = {
            bpm: _from_json_reading(readings[bpm]["y"]) if bpm in readings else math.nan
            for bpm in selected
        }
        load = ReferenceLoad(
            name=name,
            sha256=hashlib.sha256(payload).hexdigest(),
            matched=matched,
            missing_in_reference=[bpm for bpm in selected if bpm not in readings],
            unused_in_reference=[bpm for bpm in readings if bpm not in chosen],
            nan_count=sum(
                value is None for entry in readings.values() for value in (entry["x"], entry["y"])
            ),
        )
        return LoadedReference(load=load, x=x, y=y)

    async def list_references(self, model: str) -> list[str]:
        """The names of ``model``'s stored references, sorted."""
        directory = self._reference_dir(model)

        def names() -> list[str]:
            if not directory.is_dir():
                return []
            return sorted(
                p.stem
                for p in directory.iterdir()
                if p.suffix == _SUFFIX and not p.name.startswith(".") and p.name != CURRENT_FILE
            )

        listing: list[str] = await self._run(names)
        return listing

    async def current(self, model: str) -> str | None:
        """The name of ``model``'s current reference, ``None`` when none is set."""
        path = self._reference_dir(model) / CURRENT_FILE
        try:
            payload: bytes = await self._run(path.read_bytes)
        except FileNotFoundError:
            return None
        name: str | None = json.loads(payload).get("name")
        return name

    async def set_current(self, model: str, name: str | None) -> None:
        """Make ``name`` ``model``'s current reference; ``None`` clears it.

        Raises:
            ReferenceNotFound: No reference ``name`` is stored.
        """
        directory = self._reference_dir(model)
        if name is not None:
            _check_reference_name(name)
        reference = directory / f"{name}{_SUFFIX}"

        def write() -> None:
            if name is not None and not reference.is_file():
                raise ReferenceNotFound(f"no reference {name!r} is stored for model {model!r}")
            _write_atomic(directory / CURRENT_FILE, _dumps({"name": name}))

        await self._run(write)
