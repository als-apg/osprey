"""The trajectory service's files: shot results and named references, per target and model.

Everything lives under the directory ``OSPREY_TRAJECTORY_STATE_DIR`` names, in
one directory per control target the service is bound to::

    <target>/shots/<model>/<shot_id>.json        one ShotResult each, newest kept
    <target>/raw/<model>/<root_shot_id>.raw      one raw buffer per root shot (:mod:`.raw`)
    <target>/raw/<model>/evicted.json            the ids pruned most recently
    <target>/references/<model>/<name>.json      one reference each
    <target>/references/<model>/current.json     {"name": <reference>} or {"name": null}

A store bound to no target keeps the same tree at the state directory itself.
Two targets sharing a state directory never see each other's files: a shot or
reference stored for one target is not found by the other.

Every target, model, shot id and reference name matches :data:`NAME_PATTERN`
and is neither ``.`` nor ``..``, so no name can leave its directory;
``current`` is not a reference name because its file is the pointer.

A shot a chain made is a **root shot**; a shot derived from one names it as its
``parent_id`` and reads its root's raw buffer. Shots and raw buffers are kept
and pruned together, keyed by root shot: a target and model keep the
``shots_per_model`` root shots most recently saved or derived from — the raw
retention (``services.trajectory.raw_retention``) — each with its raw buffer
and every shot derived from it; saving past that deletes the oldest root's
shot files and buffer at once, so the latest shot is always kept. A pruned id
is remembered (the latest :data:`EVICTED_KEPT`) and its raw buffer answers
:class:`~.raw.RawEvicted`. A raw buffer is written before its shot file, so a
stored root shot that has a buffer never lacks it.

A reference holds one ``(x, y)`` reading per BPM name, the readout mode it
was taken in (a reference is never used across modes), the target it belongs
to and the target and model it was taken on (``source_target`` and
``source_model``; an imported reference belongs to the target it was imported
on and keeps where it came from). A stored reference whose file names another
target is refused (:class:`ForeignReference`). NaN is written as JSON
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

from osprey.services.trajectory.constants import DEFAULT_RAW_RETENTION
from osprey.services.trajectory.raw import (
    RAW_SUFFIX,
    RawBuffer,
    RawEvicted,
    decode_raw,
    encode_raw,
)
from osprey.services.trajectory.state import (
    Readout,
    ReferenceImport,
    ReferenceLoad,
    ReferenceReading,
    ShotResult,
)
from osprey_connectors.safety_file import atomic_replace

__all__ = [
    "CURRENT_FILE",
    "FILE_MODE",
    "NAME_PATTERN",
    "EVICTED_KEPT",
    "STATE_DIR_ENV_VAR",
    "ForeignReference",
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

#: How many pruned shot ids each model remembers as evicted.
EVICTED_KEPT = 1000

#: The pointer to a model's current reference, beside the references.
CURRENT_FILE = "current.json"

#: The mode every shot file, reference and pointer is written with: the owner
#: writes, the group reads, nobody else. The deployment provisions the state
#: directory setgid to the host's group, so the group read is what lets the
#: host read a shot the service wrote under its own user.
FILE_MODE = 0o640

_CURRENT_NAME = CURRENT_FILE.removesuffix(".json")
_SUFFIX = ".json"
_EVICTED_FILE = "evicted.json"


class StoreUnconfigured(RuntimeError):
    """``OSPREY_TRAJECTORY_STATE_DIR`` is unset or empty."""


class InvalidName(ValueError):
    """A model, shot id or reference name that does not match :data:`NAME_PATTERN`."""


class ReferenceNotFound(LookupError):
    """No reference of that name is stored for the model."""


class ForeignReference(ReferenceNotFound):
    """The stored reference file names another target than the store's."""


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


def _parent_of(path: Path) -> str | None:
    """The ``parent_id`` a shot file names, ``None`` for a root shot or an unreadable file."""
    try:
        parent = json.loads(path.read_bytes()).get("parent_id")
    except (OSError, ValueError, AttributeError):
        return None
    return parent if isinstance(parent, str) and parent else None


def _root_of(shot_id: str, parents: Mapping[str, str | None]) -> str:
    """The root shot of ``shot_id``, following ``parent_id`` through the stored shots."""
    seen = {shot_id}
    while (parent := parents.get(shot_id)) is not None and parent not in seen:
        seen.add(parent)
        shot_id = parent
    return shot_id


class TrajectoryStore:
    """Shot files and references of one target under one state directory.

    Args:
        root: The state directory; created on the first write.
        target: The control target the files belong to, ``None`` to keep them
            at ``root`` itself.
        shots_per_model: How many root shots each model keeps, with their raw
            buffers and derived shots, at least 1: the raw retention.

    Raises:
        InvalidName: ``target`` does not match :data:`NAME_PATTERN`.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        target: str | None = None,
        shots_per_model: int = DEFAULT_RAW_RETENTION,
    ) -> None:
        if shots_per_model < 1:
            raise ValueError(f"shots_per_model must be at least 1, not {shots_per_model}")
        self.root = Path(root)
        self.target = None if target is None else _check_name(target, "target")
        self.directory = self.root if self.target is None else self.root / self.target
        self.shots_per_model = shots_per_model
        self._lock = asyncio.Lock()
        self._last_stamp_ns = 0

    @classmethod
    def from_env(
        cls, *, target: str | None = None, shots_per_model: int = DEFAULT_RAW_RETENTION
    ) -> TrajectoryStore:
        """The store of ``target`` rooted at ``OSPREY_TRAJECTORY_STATE_DIR``.

        Raises:
            StoreUnconfigured: The variable is unset or empty.
        """
        value = os.environ.get(STATE_DIR_ENV_VAR, "").strip()
        if not value:
            raise StoreUnconfigured(
                f"{STATE_DIR_ENV_VAR} is unset; the service keeps its shot files and "
                "references in that directory"
            )
        return cls(Path(value), target=target, shots_per_model=shots_per_model)

    # --- paths -----------------------------------------------------------------------

    def _shot_dir(self, model: str) -> Path:
        return self.directory / "shots" / _check_name(model, "model")

    def _raw_dir(self, model: str) -> Path:
        return self.directory / "raw" / _check_name(model, "model")

    def _reference_dir(self, model: str) -> Path:
        return self.directory / "references" / _check_name(model, "model")

    async def _run(self, fn: Any, *args: Any) -> Any:
        async with self._lock:
            return await asyncio.to_thread(fn, *args)

    # --- shots -----------------------------------------------------------------------

    async def save_shot(self, shot: ShotResult, raw: RawBuffer | None = None) -> Path:
        """Write ``shot`` as ``shots/<model>/<shot_id>.json`` under the target, pruning the oldest.

        ``raw`` is a root shot's raw buffer, written first as
        ``raw/<model>/<shot_id>.raw``. Re-saving a shot id replaces its file
        and makes it the newest.

        Returns:
            The shot file's path.
        """
        directory = self._shot_dir(shot.model)
        name = _check_name(shot.shot_id, "shot id")
        path = directory / f"{name}{_SUFFIX}"
        payload = shot.model_dump_json(indent=2).encode()
        raw_path = self._raw_dir(shot.model) / f"{name}{RAW_SUFFIX}"
        raw_payload = None if raw is None else encode_raw(raw)
        await self._run(self._save_shot_sync, path, payload, raw_path, raw_payload)
        return path

    def _save_shot_sync(
        self, path: Path, payload: bytes, raw_path: Path, raw_payload: bytes | None
    ) -> None:
        if raw_payload is not None:
            _write_atomic(raw_path, raw_payload)
        _write_atomic(path, payload)
        # Order by a stamp this store makes strictly increasing, so two saves in
        # one clock tick still prune in the order they were made.
        stamp = max(time.time_ns(), self._last_stamp_ns + 1)
        self._last_stamp_ns = stamp
        os.utime(path, ns=(stamp, stamp))
        self._prune_sync(path.parent, raw_path.parent)

    def _prune_sync(self, shot_dir: Path, raw_dir: Path) -> None:
        """Keep the newest ``shots_per_model`` root shots, their buffers and derived shots."""
        files = self._shot_files_newest_first(shot_dir)
        parents: dict[str, str | None] = {}
        for file in files:
            parents[file.stem] = _parent_of(file)
        kept: list[str] = []
        evicted: list[str] = []
        for file in files:
            root = _root_of(file.stem, parents)
            if root not in kept and len(kept) < self.shots_per_model:
                kept.append(root)
            if root not in kept:
                with suppress(FileNotFoundError):
                    file.unlink()
                evicted.append(file.stem)
        if raw_dir.is_dir():
            for buffer in raw_dir.iterdir():
                if buffer.suffix != RAW_SUFFIX or buffer.name.startswith("."):
                    continue
                if buffer.stem not in kept:
                    with suppress(FileNotFoundError):
                        buffer.unlink()
                    evicted.append(buffer.stem)
        stored = {file.stem for file in files} - set(evicted)
        before = self._evicted_sync(raw_dir)
        if evicted or stored.intersection(before):
            remembered = [
                shot_id for shot_id in before if shot_id not in evicted and shot_id not in stored
            ]
            listing = [*remembered, *dict.fromkeys(evicted)][-EVICTED_KEPT:]
            _write_atomic(raw_dir / _EVICTED_FILE, _dumps(listing))

    @staticmethod
    def _evicted_sync(raw_dir: Path) -> list[str]:
        try:
            listing = json.loads((raw_dir / _EVICTED_FILE).read_bytes())
        except FileNotFoundError:
            return []
        return [str(shot_id) for shot_id in listing] if isinstance(listing, list) else []

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

    async def load_raw(self, model: str, shot_id: str) -> tuple[str, RawBuffer]:
        """The raw buffer ``shot_id`` of ``model`` was made from, and its root shot's id.

        A derived shot reads its root shot's buffer.

        Raises:
            RawEvicted: The shot, or its root, was pruned.
            LookupError: No such shot is stored, or its chain included no
                injection and so kept no buffer.
        """
        shot_dir = self._shot_dir(model)
        raw_dir = self._raw_dir(model)
        _check_name(shot_id, "shot id")

        def read() -> tuple[str, bytes]:
            evicted = set(self._evicted_sync(raw_dir))
            current = shot_id
            seen: set[str] = set()
            while True:
                path = shot_dir / f"{current}{_SUFFIX}"
                if not path.is_file():
                    if current in evicted:
                        raise RawEvicted(
                            f"the raw buffer of shot {shot_id!r} (model {model!r}) was pruned: "
                            "the store keeps only the raw retention's newest shots"
                        )
                    raise LookupError(f"no shot {shot_id!r} is stored for model {model!r}")
                parent = _parent_of(path)
                if parent is None or parent in seen:
                    break
                seen.add(current)
                current = parent
            try:
                return current, (raw_dir / f"{current}{RAW_SUFFIX}").read_bytes()
            except FileNotFoundError:
                raise LookupError(
                    f"shot {shot_id!r} of model {model!r} has no raw buffer: its chain "
                    "included no injection"
                ) from None

        root, payload = await self._run(read)
        return root, decode_raw(payload)

    async def list_shots(self, model: str) -> list[ShotResult]:
        """Every stored shot of ``model``, newest first."""
        directory = self._shot_dir(model)

        def read_all() -> list[str]:
            return [p.read_text() for p in self._shot_files_newest_first(directory)]

        return [ShotResult.model_validate_json(text) for text in await self._run(read_all)]

    # --- references ------------------------------------------------------------------

    async def save_reference(
        self,
        model: str,
        name: str,
        readings: Mapping[str, tuple[float | None, float | None]],
        *,
        readout: Readout = "tbt",
        source_target: str | None = None,
        source_model: str | None = None,
    ) -> str:
        """Write the reference ``name`` of ``model``: one ``(x, y)`` reading per BPM.

        The file belongs to this store's target; ``source_target`` and
        ``source_model`` record where its readings were taken, ``None`` when
        unknown. NaN or ``None`` readings are stored as ``null``; an infinite
        reading or an empty reference is refused.

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
            "readout": readout,
            "target": self.target,
            "source_target": source_target,
            "source_model": source_model,
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

    async def _read_reference(self, model: str, name: str) -> tuple[bytes, dict[str, Any]]:
        """The stored file of reference ``name``: its bytes and its body.

        Raises:
            ReferenceNotFound: No such reference is stored.
            ForeignReference: The file names another target than this store's.
        """
        path = self._reference_dir(model) / f"{_check_reference_name(name)}{_SUFFIX}"
        try:
            payload: bytes = await self._run(path.read_bytes)
        except FileNotFoundError:
            raise ReferenceNotFound(
                f"no reference {name!r} is stored for model {model!r}"
            ) from None
        body: dict[str, Any] = json.loads(payload)
        owner = body.get("target")
        if owner != self.target:
            raise ForeignReference(
                f"reference {name!r} of model {model!r} belongs to target {owner!r}, "
                f"not {self.target!r}; a reference is never used on another target"
            )
        return payload, body

    async def load_reference(
        self, model: str, name: str, selection: Iterable[str]
    ) -> LoadedReference:
        """The reference ``name`` of ``model``, matched over the selected BPMs.

        The load carries the reference's readout mode and its reading by
        selected BPM (``None`` where the reference lacks the BPM or holds NaN).

        Raises:
            ReferenceNotFound: No such reference is stored.
            ForeignReference: The file names another target than this store's.
            NoMatchingBpms: The reference names none of the selected BPMs.
        """
        payload, body = await self._read_reference(model, name)
        readings: dict[str, dict[str, Any]] = body["readings"]
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
            readout=body.get("readout"),
            readings={
                bpm: ReferenceReading(
                    x=readings[bpm]["x"] if bpm in readings else None,
                    y=readings[bpm]["y"] if bpm in readings else None,
                )
                for bpm in selected
            },
        )
        return LoadedReference(load=load, x=x, y=y)

    async def reference_file(self, model: str, name: str) -> ReferenceImport:
        """The stored reference ``name`` of ``model`` in full, as an import body.

        Importing the answer stores the same readings, readout and provenance
        under the same name.

        Raises:
            ReferenceNotFound: No such reference is stored.
            ForeignReference: The file names another target than this store's.
        """
        _payload, body = await self._read_reference(model, name)
        return ReferenceImport(
            name=name,
            readings={
                bpm: ReferenceReading(x=entry["x"], y=entry["y"])
                for bpm, entry in body["readings"].items()
            },
            readout=body["readout"],
            source_target=body.get("source_target"),
            source_model=body.get("source_model"),
        )

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
