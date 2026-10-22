"""Raw buffers: every included injection's turn series, kept beside its shot.

A chain that reads out keeps, for its shot, the samples it read from every
included injection — the trigger sample, then the turns tracked — as one
float32 array of shape ``[n_included, 3, n_bpm, raw_samples]``: per injection
the planes ``x``, ``y`` and ``sum`` in that order, per plane one row per BPM in
the shot's ``bpm_order``. Plots, extraction and pseudo-orbit use only these
samples, so the buffer is everything the shot was made from.

**The file** is the bytes ``GET /panel/api/shots/{shot_id}/raw`` serves
verbatim (``application/octet-stream``)::

    bytes 0..3        header length n, uint32 little-endian, a multiple of 4
    bytes 4..4+n      the header: UTF-8 JSON, padded with spaces to n bytes
    bytes 4+n..       the samples: float32 little-endian, C order

The header is ``{"shape": [n_included, 3, n_bpm, raw_samples],
"trigger_sample": <int>, "units": {"x": <str|null>, "y": <str|null>,
"sum": <str|null>}}``: the shape the samples are read back into, the sample
the injected shot arrives at (the one a plot marks; x stays the sample index)
and each plane's unit as the device table's channel facts state it (the unit a
plot converts by). The samples start on a 4-byte boundary, so a browser reads
them with ``new Float32Array(buffer, 4 + n)``.

The store keeps one file per **root shot** — the shot a chain made; a shot
derived from it reads its root's buffer — and prunes buffers and shots
together (:class:`~.store.TrajectoryStore`). A shot pruned that way answers
410 ``raw_evicted``.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "PLANES",
    "RAW_EVICTED",
    "RAW_MEDIA_TYPE",
    "RAW_SUFFIX",
    "RawBuffer",
    "RawEvicted",
    "RawHeader",
    "RawUnits",
    "decode_raw",
    "encode_raw",
    "units_of",
]

#: The planes of every injection, in buffer order.
PLANES = ("x", "y", "sum")

#: The refusal code of a shot whose raw buffer was pruned.
RAW_EVICTED = "raw_evicted"

#: The media type the raw route answers with.
RAW_MEDIA_TYPE = "application/octet-stream"

#: The suffix of a raw buffer file in the store.
RAW_SUFFIX = ".raw"

_LENGTH = struct.Struct("<I")
_ALIGN = 4
_DTYPE = np.dtype("<f4")


class RawEvicted(LookupError):
    """The shot's raw buffer was pruned with the shot: the retention moved past it."""


@dataclass(frozen=True)
class RawUnits:
    """The unit of each plane, ``None`` where the channel facts state none."""

    x: str | None = None
    y: str | None = None
    sum: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        """``{x, y, sum}``, as the header carries it."""
        return {"x": self.x, "y": self.y, "sum": self.sum}


def units_of(slots: Mapping[str, Any] | None) -> RawUnits:
    """Each plane's unit from a ``trajectory_slots`` entry's devices.

    A plane takes the first unit any device states for it, in device order;
    a plane no device states a unit for is ``None``.
    """
    devices = (slots or {}).get("devices") if isinstance(slots, Mapping) else None
    found: dict[str, str | None] = dict.fromkeys(PLANES)
    if isinstance(devices, Mapping):
        for device in devices.values():
            units = device.get("units") if isinstance(device, Mapping) else None
            if not isinstance(units, Mapping):
                continue
            for plane in PLANES:
                unit = units.get(plane)
                if found[plane] is None and isinstance(unit, str) and unit:
                    found[plane] = unit
    return RawUnits(**found)


@dataclass(frozen=True)
class RawHeader:
    """What a raw buffer's samples are.

    Attributes:
        shape: ``(n_included, 3, n_bpm, raw_samples)``.
        trigger_sample: The sample the injected shot arrives at.
        units: Each plane's unit.
    """

    shape: tuple[int, int, int, int]
    trigger_sample: int
    units: RawUnits

    def as_dict(self) -> dict[str, Any]:
        """The header as its JSON carries it."""
        return {
            "shape": list(self.shape),
            "trigger_sample": self.trigger_sample,
            "units": self.units.as_dict(),
        }


@dataclass(frozen=True, eq=False)
class RawBuffer:
    """One root shot's raw buffer: its header and its samples.

    Attributes:
        header: The shape, trigger sample and units.
        samples: float32, shaped as ``header.shape``.
    """

    header: RawHeader
    samples: np.ndarray

    @classmethod
    def of(
        cls,
        injections: Sequence[Sequence[np.ndarray]],
        *,
        trigger_sample: int,
        units: RawUnits,
    ) -> RawBuffer:
        """The buffer of the included injections, each its ``(x, y, sum)`` planes.

        Raises:
            ValueError: No injection, a plane that is not ``(n_bpm, raw_samples)``,
                or injections of different shapes.
        """
        if not injections:
            raise ValueError("a raw buffer holds at least one included injection")
        try:
            samples = np.stack([np.stack(list(planes)) for planes in injections]).astype(_DTYPE)
        except ValueError as exc:
            raise ValueError(f"the included injections differ in shape: {exc}") from exc
        if samples.ndim != 4 or samples.shape[1] != len(PLANES):
            raise ValueError(
                "each injection is three (n_bpm, raw_samples) planes; "
                f"got a buffer of shape {samples.shape}"
            )
        shape = tuple(int(n) for n in samples.shape)
        return cls(RawHeader(shape, int(trigger_sample), units), samples)  # type: ignore[arg-type]


def encode_raw(buffer: RawBuffer) -> bytes:
    """The buffer as its file: length, padded JSON header, float32 samples."""
    header = json.dumps(buffer.header.as_dict(), sort_keys=True, allow_nan=False).encode()
    header += b" " * (-len(header) % _ALIGN)
    samples = np.ascontiguousarray(buffer.samples, dtype=_DTYPE)
    if tuple(samples.shape) != buffer.header.shape:
        raise ValueError(
            f"the samples are shaped {samples.shape}, the header says {buffer.header.shape}"
        )
    return _LENGTH.pack(len(header)) + header + samples.tobytes()


def decode_raw(payload: bytes) -> RawBuffer:
    """The buffer a raw file holds.

    Raises:
        ValueError: The bytes are not a raw buffer file.
    """
    if len(payload) < _LENGTH.size:
        raise ValueError("a raw buffer file starts with its header length")
    (length,) = _LENGTH.unpack_from(payload)
    start = _LENGTH.size + length
    if length % _ALIGN or start > len(payload):
        raise ValueError(f"header length {length} does not fit a {len(payload)}-byte file")
    data = json.loads(payload[_LENGTH.size : start])
    shape = tuple(int(n) for n in data["shape"])
    units = data.get("units") or {}
    header = RawHeader(
        shape,  # type: ignore[arg-type]
        int(data["trigger_sample"]),
        RawUnits(**{plane: units.get(plane) for plane in PLANES}),
    )
    count = int(np.prod(shape))
    if len(payload) - start != count * _DTYPE.itemsize:
        raise ValueError(
            f"the header says {count} samples, the file holds "
            f"{(len(payload) - start) // _DTYPE.itemsize}"
        )
    samples = np.frombuffer(payload, dtype=_DTYPE, offset=start).reshape(shape)
    return RawBuffer(header, samples)
