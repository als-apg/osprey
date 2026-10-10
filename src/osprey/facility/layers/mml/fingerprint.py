"""The pairing check between a lattice deck and the export sampled over it.

A 2.0 export states four facts about the lattice every calibration, nominal and
energy table in it was sampled over: how many elements the lattice holds, the
digest of its family names, the energy it was solved at, and where its
parameter elements sit. The deck travels beside the export as its own file,
and nothing else ties the two together, so ``osprey facility import mml``
recomputes the four from the deck beside each export and holds the pair to
them.

:func:`lattice_fingerprint` recomputes them from the lattice as the import reads
it -- every element kept, so the indices the Middle Layer carries hold -- and
:func:`check_fingerprint` returns every fact the export states differently.

A disagreement on the element count, the digest or the parameter indices
refuses the pair: the deck is another lattice, and every calibration would be
bound to the wrong element. A disagreement on the energy alone is said and
refuses nothing (:attr:`Mismatch.refuses`): a facility restates its operating
energy without changing a single element, and the deck's own energy is the one
the model is solved at.

The two sides spell the same facts differently, and the comparison is where
that is reconciled. MATLAB counts in doubles, writes a list of one as the bare
number it is, and reaches its model energy by a path of its own, so a whole
double is read as the count it is, a bare number as the one-entry list it is,
and two energies within :data:`ENERGY_TOLERANCE_GEV` are the same energy.
Everything else is compared exactly: the digest exists to refuse a lattice
whose names or order moved, and a tolerance on it would refuse nothing.

A parameter element carries the lattice's own properties rather than a piece of
beam line. A deck names its class; a lattice read back from one tags it. Both
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
    "WARNED_KEYS",
    "Deck",
    "Mismatch",
    "check_fingerprint",
    "lattice_fingerprint",
]

#: The facts a fingerprint holds, in the order they are compared: the cheapest
#: and coarsest first, so the field a message names is the plainest one that
#: disagrees.
FINGERPRINT_KEYS = ("elements", "famname_sha256", "energy_gev", "ringparam_indices")

#: The facts whose disagreement is said and refuses nothing.
WARNED_KEYS = frozenset({"energy_gev"})

#: How far apart two statements of one model energy may be, in GeV. One eV:
#: wide enough for the last places of two paths to the same number, narrow
#: enough to tell a lattice solved at another operating point.
ENERGY_TOLERANCE_GEV = 1e-9

#: The class a lattice's parameter element carries, under either spelling.
_RINGPARAM = "RingParam"  # outside-format-quote

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

    @property
    def refuses(self) -> bool:
        """Whether the disagreement makes the deck another lattice than the export's."""
        return self.field not in WARNED_KEYS


class Deck(Protocol):
    """A deck's lattice: its elements in saved order, and its model energy.

    What a fingerprint is recomputed from, and no more of a deck than that.
    """

    energy: float

    def __iter__(self) -> Iterator[Any]: ...


def lattice_fingerprint(lattice: Deck) -> dict[str, Any]:
    """Recompute the four facts of a fingerprint from a deck's lattice.

    Args:
        lattice: The lattice as the import reads it, every saved element kept and in
            saved order, carrying the model energy in eV as ``energy``.

    Returns:
        ``elements``, ``famname_sha256``, ``energy_gev`` and
        ``ringparam_indices``, the indices one-based and in lattice order.
    """
    names = [str(element.FamName) for element in lattice]
    digest = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()
    return {
        "elements": len(names),
        "famname_sha256": digest,
        "energy_gev": float(lattice.energy) / _EV_PER_GEV,
        "ringparam_indices": tuple(
            index for index, element in enumerate(lattice, start=1) if _is_ringparam(element)
        ),
    }


def check_fingerprint(expected: dict, actual: dict) -> list[Mismatch]:
    """Compare what an export states about its lattice against what a deck holds.

    Args:
        expected: The ``lattice`` block of a virtual-accelerator export.
        actual: A fingerprint from :func:`lattice_fingerprint`.

    Returns:
        Every fact the two do not agree on, in :data:`FINGERPRINT_KEYS` order;
        empty when the deck is the lattice the export was sampled over.
    """
    found: list[Mismatch] = []
    for field in FINGERPRINT_KEYS:
        stated = expected.get(field)
        held = actual[field]
        if field == "energy_gev":
            agrees = _agrees_within(stated, held, ENERGY_TOLERANCE_GEV)
        else:
            agrees = _read(field, stated) == held
        if not agrees:
            found.append(Mismatch(field=field, expected=stated, actual=held))
    return found


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
