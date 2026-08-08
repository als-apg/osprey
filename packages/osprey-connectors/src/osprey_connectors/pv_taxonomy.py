"""Shared procedural-PV taxonomy for the mock connectors.

Both the mock control-system connector and the mock archiver connector need to
guess plausible defaults for PV names that the data-driven simulation engine
does *not* serve. They classify a PV by substring conventions in its name and
map it to a physical base value and engineering units.

Historically each connector hard-coded its own copy of this keyword ladder,
which had already drifted (e.g. the archiver lacked an ``energy`` branch and the
control connector did not recognise ``dcct`` as a beam-current monitor). This
module is the single source of truth: :func:`classify_pv` returns a
:class:`PVKind`, and each connector layers its own behaviour on top — the
archiver shapes a time series per :attr:`PVKind.name`, the control connector
seeds an initial value and reports units.
"""

from dataclasses import dataclass

__all__ = ["PVKind", "classify_pv"]


@dataclass(frozen=True)
class PVKind:
    """A procedural PV classification.

    Attributes:
        name: Canonical kind, used by the archiver to pick a synthesis shape.
        base_value: Plausible steady-state value for the kind.
        units: Engineering units string (empty for the generic default).
    """

    name: str
    base_value: float
    units: str


def classify_pv(pv_name: str) -> PVKind:
    """Classify a PV name into a :class:`PVKind` by naming convention.

    The checks are ordered: more specific kinds (beam current) are tested before
    more general ones (current). A name matching nothing falls through to the
    generic ``default`` kind.
    """
    lower = pv_name.lower()

    if ("beam" in lower and "current" in lower) or "dcct" in lower:
        return PVKind("beam_current", 500.0, "mA")
    if "current" in lower:
        return PVKind("current", 150.0, "A")
    if "voltage" in lower:
        return PVKind("voltage", 5000.0, "V")
    if "power" in lower:
        return PVKind("power", 50.0, "kW")
    if "pressure" in lower:
        return PVKind("pressure", 1e-9, "Torr")
    if "temp" in lower:
        return PVKind("temperature", 25.0, "°C")
    if "lifetime" in lower:
        return PVKind("lifetime", 10.0, "hours")
    if "position" in lower or "pos" in lower:
        return PVKind("position", 0.0, "mm")
    if "energy" in lower:
        return PVKind("energy", 1900.0, "MeV")
    return PVKind("default", 100.0, "")
