"""The pairing check between a lattice deck and the export sampled over it.

A 2.0 export states four facts about the ring every calibration, nominal and
energy table in it was sampled over: how many elements the ring holds, the
digest of its family names, the energy it was solved at, and where its
parameter elements sit. The deck travels beside the export as its own file,
and nothing else ties the two together, so ``osprey mml import`` recomputes
the four from the deck it is given and refuses the pair when one disagrees.

:func:`lattice_fingerprint` recomputes them from the ring as the import reads
it -- every element kept, so the indices the Middle Layer carries hold -- and
:func:`check_fingerprint` compares that against what the export states.

The two sides spell the same facts differently, and the comparison is where
that is reconciled. MATLAB counts in doubles, writes a list of one as the bare
number it is, and reaches its model energy by a path of its own, so a whole
double is read as the count it is, a bare number as the one-entry list it is,
and the energy is compared to :data:`ENERGY_TOLERANCE_GEV`. Everything else is
compared exactly: the digest exists to refuse a ring whose names or order
moved, and a tolerance on it would refuse nothing.

A parameter element carries the ring's own properties rather than a piece of
beam line. A deck names its class; a ring read back from one tags it. Both
spellings are the same element.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = [
    "ENERGY_TOLERANCE_GEV",
    "FINGERPRINT_KEYS",
    "Mismatch",
    "Ring",
    "check_fingerprint",
    "lattice_fingerprint",
]

#: The facts a fingerprint holds, in the order they are compared: the cheapest
#: and coarsest first, so the field a message names is the plainest one that
#: disagrees.
FINGERPRINT_KEYS = ("elements", "famname_sha256", "energy_gev", "ringparam_indices")

#: How far apart two statements of one model energy may be, in GeV. One eV:
#: wide enough for the last places of two paths to the same number, narrow
#: enough to refuse a ring solved at another operating point.
ENERGY_TOLERANCE_GEV = 1e-9

#: The class a ring's parameter element carries, under either spelling.
_RINGPARAM = "RingParam"

_EV_PER_GEV = 1e9


@dataclass(frozen=True)
class Mismatch:
    """One fact the deck and the export do not agree on.

    Attributes:
        field: The name of the fact, one of :data:`FINGERPRINT_KEYS`.
        expected: What the export states, as it states it, or ``None`` when it
            states nothing for this fact.
        actual: What the deck holds, as :func:`lattice_fingerprint` recomputes it.
    """

    field: str
    expected: Any
    actual: Any


class Ring(Protocol):
    """A lattice deck's ring: its elements in saved order, and its model energy.

    What a fingerprint is recomputed from, and no more of a deck than that.
    """

    energy: float

    def __iter__(self) -> Iterator[Any]: ...


def lattice_fingerprint(ring: Ring) -> dict[str, Any]:
    """Recompute the four facts of a fingerprint from a lattice deck's ring.

    Args:
        ring: The ring as the import reads it, every saved element kept and in
            saved order, carrying the model energy in eV as ``energy``.

    Returns:
        ``elements``, ``famname_sha256``, ``energy_gev`` and
        ``ringparam_indices``, the indices one-based and in ring order.
    """
    names = [str(element.FamName) for element in ring]
    digest = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()
    return {
        "elements": len(names),
        "famname_sha256": digest,
        "energy_gev": float(ring.energy) / _EV_PER_GEV,
        "ringparam_indices": tuple(
            index for index, element in enumerate(ring, start=1) if _is_ringparam(element)
        ),
    }


def check_fingerprint(expected: dict, actual: dict) -> Mismatch | None:
    """Compare what an export states about its ring against what a deck holds.

    Args:
        expected: The ``lattice`` block of a virtual-accelerator export.
        actual: A fingerprint from :func:`lattice_fingerprint`.

    Returns:
        The first fact of :data:`FINGERPRINT_KEYS` the two do not agree on, or
        ``None`` when the deck is the ring the export was sampled over.
    """
    for field in FINGERPRINT_KEYS:
        stated = expected.get(field)
        found = actual[field]
        if field == "energy_gev":
            agrees = _agrees_within(stated, found, ENERGY_TOLERANCE_GEV)
        else:
            agrees = _read(field, stated) == found
        if not agrees:
            return Mismatch(field=field, expected=stated, actual=found)
    return None


def _is_ringparam(element: Any) -> bool:
    return _RINGPARAM in (getattr(element, "Class", None), getattr(element, "tag", None))


def _read(field: str, stated: Any) -> Any:
    """One stated fact in the spelling :func:`lattice_fingerprint` returns it in."""
    if field == "elements":
        return _as_index(stated)
    if field == "ringparam_indices":
        return _as_indices(stated)
    return stated


def _as_index(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _as_indices(value: Any) -> tuple[int, ...] | None:
    values = value if isinstance(value, list) else [value]
    indices = [_as_index(item) for item in values]
    if any(index is None for index in indices):
        return None
    return tuple(indices)  # type: ignore[arg-type]


def _agrees_within(stated: Any, found: float, tolerance: float) -> bool:
    if isinstance(stated, bool) or not isinstance(stated, (int, float)):
        return False
    return abs(float(stated) - found) <= tolerance
