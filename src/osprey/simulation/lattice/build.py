"""Regenerate the demo facility's committed build artifacts.

Run::

    python -m osprey.simulation.lattice.build            # rewrite the artifacts
    python -m osprey.simulation.lattice.build --check    # refuse a stale one

The demo is a facility like any other: it ships a data tree, and a served
virtual accelerator reads its model out of that tree rather than out of code.
Where a real facility exports its deck and its calibrations from the Middle
Layer, the demo's are generated here from the ring
:func:`osprey.simulation.lattice.ring.build_ring` hand-ports, which is what
lets one bindings-driven path serve the demo and a facility alike.

Three files come out of one ring:

* ``data/lattice/als_u_ar.mat`` -- the canonical ``.mat`` the Lattice Dashboard
  reads. scipy stamps the wall clock into a ``.mat`` header, so it is not
  byte-stable and ``--check`` says nothing about it; the round-trip test guards
  its staleness instead.
* ``data/simulation/lattice.json`` -- the deck the served model loads, written
  by the emit lane's own lattice writer so the demo's deck and a facility's are
  rendered by one piece of code.
* ``data/simulation/va_bindings.json`` -- what each channel does to that deck.
  Its presence is what makes the tree a virtual-accelerator tree, and the
  addresses it binds ARE the manifest's ``pyat-coupled`` partition.

The two JSON files are build outputs: regenerable, never hand-edited, and
compared BYTE for byte by ``--check`` so a value that moved is caught even when
the document still parses. ``--check`` renders into a scratch directory and
writes nothing at all, so a stale tree stays stale until someone regenerates it
on purpose.

The physics the bindings state is the ring's own, one straight line per device:

* A **quadrupole** or **sextupole** scales the strength baked into its element
  by the fraction of nominal current it is driven at, so the calibration is
  ``PolynomB[n] = (baked / nominal) * I`` with no offset.
* A **dipole** is a trim coil: the bend's angle and gradient are held fixed and
  only the pure-dipole field error moves, ``PolynomB[0] = (angle / length) *
  (I / nominal - 1)``. At nominal current the error is exactly zero.
* A **corrector** kicks its own plane, ``KickAngle[plane] = I /
  AMPS_PER_RADIAN_KICK``, and carries no nominal-current scaling.
* A **monitor** reads a transverse axis of the closed orbit. Its channels are
  in metres and so is the orbit, so both of its curves are the identity.

Nominal currents come from the scenario seed beside the bindings --
``data/simulation/machine.json``, the same file the mock engine serves -- so
one number starts a channel off and calibrates it. A device the seed states no
value for is refused rather than calibrated against a guess.

This module reaches past the ring into the emit lane and the served tree's path
resolver, so it is the one part of this subpackage that is not importable from a
plain ``at`` + numpy environment. It is a build entry point rather than a
consumer API; :mod:`osprey.simulation.lattice` still exports only the ring and
the ``.mat`` helpers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import at

from osprey.services.mml.canonical import write_if_changed
from osprey.services.mml.emit.context import LATTICE_ARTIFACT, NO_EXPORTER, EmitContext
from osprey.services.mml.emit.va import emit_lattice
from osprey.services.virtual_accelerator.bindings import (
    Binding,
    BindingsDocument,
    Linear,
    Slice,
    dump_bindings,
)
from osprey.services.virtual_accelerator.manifest.paths import PACKAGE_PATHS
from osprey.simulation.facility_spec import ALS_U_AR, Family
from osprey.simulation.lattice.artifact import save_canonical_mat
from osprey.simulation.lattice.ring import build_ring

__all__ = [
    "AMPS_PER_RADIAN_KICK",
    "DEMO_SYSTEM",
    "DemoModelError",
    "build_demo_bindings",
    "digest_of",
    "main",
    "render_demo_model",
]

#: The system every bound address of the demo tree sits under.
DEMO_SYSTEM = "SR"

#: Amps of corrector current per radian of kick.
#:
#: Deterministic and sign-correct rather than a claim of physical realism, but
#: its magnitude carries weight on a low-emittance lattice: the ring's strong
#: sextupoles feed down into the orbit response, so a kick large enough to move
#: the closed orbit by millimetres stops responding antisymmetrically. At this
#: calibration a corrector's +-12 A range moves the orbit by tens of microns,
#: which keeps the response quasi-linear.
AMPS_PER_RADIAN_KICK = 1_000_000.0

#: How the demo tree spells a magnet or corrector current, and a monitor
#: reading. The demo's own address grammar, stated once here because the
#: bindings are what puts these addresses in the coupled partition -- nothing
#: upstream of this file can be asked what they are.
_CURRENT_ADDRESS = "{system}:MAG:{family}:{device}:CURRENT:{half}"
_POSITION_ADDRESS = "{system}:DIAG:{family}:{device}:POSITION:{axis}"

#: The ``KickAngle`` component each corrector family drives.
_CORRECTOR_PLANE: dict[str, int] = {"HCM": 0, "VCM": 1}

#: The transverse axes a monitor serves, and the address token for each.
_MONITOR_AXES: tuple[str, ...] = ("x", "y")

#: The polynomial coefficient each multipole order writes. A dipole is a trim
#: coil rather than a driven multipole, so it is calibrated before this table
#: is consulted and carries no row here.
_MULTIPOLE_INDEX: tuple[tuple[type, int], ...] = (
    (at.Quadrupole, 1),
    (at.Sextupole, 2),
)

#: What the generated documents say produced them: the module that renders
#: them, the ring they are rendered from, and the scenario seed every
#: calibration divides by. It names code and a path rather than input digests,
#: and carries no library version, which would rewrite both files on an
#: upgrade that changed neither.
_HEADER_LINES: tuple[str, str, str] = (
    "generator=osprey.simulation.lattice.build",
    "ring=osprey.simulation.lattice.ring.build_ring",
    "seed=data/simulation/machine.json",
)

#: Read size for a file digest; the lattice is JSON of a whole ring.
_CHUNK = 1 << 20


class DemoModelError(ValueError):
    """The demo tree cannot state a model for the ring it was asked about.

    Raised when the ring names a device the facility spec declares none or
    several times, when the scenario seed states no nominal current for one,
    when a device's baked strength leaves it with a calibration that carries
    no value, or when the deck was rendered without recording its digest.
    """


def digest_of(path: Path) -> str:
    """Return the lowercase hex sha256 of a file's bytes.

    Args:
        path: The file to hash.

    Returns:
        The digest, in the spelling the bindings document stamps.

    Raises:
        OSError: The file could not be read.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def build_demo_bindings(
    ring: at.Lattice, lattice_sha256: str, machine_channels: dict
) -> BindingsDocument:
    """Build the demo tree's bindings against one ring and one scenario seed.

    Args:
        ring: The deck the bindings describe, as the generator built it.
        lattice_sha256: Digest of the saved deck, which the document stamps so
            a served tree can refuse a lattice these bindings do not describe.
        machine_channels: The scenario seed's ``channels`` mapping, read for
            each device's nominal current.

    Returns:
        The document, its bindings ordered by setpoint address.

    Raises:
        DemoModelError: The ring is missing a declared device, the seed states
            no nominal for one, or a device's calibration would carry no value.
    """
    census = Counter(element.FamName for element in ring)
    elements = _elements_by_name(ring, census)
    bindings: list[Binding] = []
    for family in ALS_U_AR.families:
        for device in range(1, family.count + 1):
            name = ALS_U_AR.device_name("", family.name, device)
            element = elements.get(name)
            if element is None:
                raise DemoModelError(_unbindable(name, family.name, device, census[name]))
            bindings.extend(_device_bindings(family, device, name, element, machine_channels))
    return BindingsDocument(
        system=DEMO_SYSTEM,
        energy_gev=ALS_U_AR.energy_ev / 1e9,
        lattice_sha256=lattice_sha256,
        bindings=tuple(sorted(bindings, key=lambda binding: binding.setpoint_address)),
        provenance=" ".join(_HEADER_LINES),
    )


def render_demo_model(ring: at.Lattice) -> tuple[str, str]:
    """Render the demo tree's two generated documents, writing nothing.

    The deck goes through the emit lane's own lattice writer, into a scratch
    directory, so the text this returns is exactly what that lane would write
    and the digest the bindings stamp is taken over those same bytes.

    Args:
        ring: The deck to render, as the generator built it.

    Returns:
        The lattice text and the bindings text, in that order.

    Raises:
        DemoModelError: The bindings could not be stated for this ring.
        OSError: The scratch directory could not be written.
    """
    context = EmitContext(
        ao_sha256="",
        mapping_sha256="",
        exporter_version=NO_EXPORTER,
        provenance_string=" ".join(_HEADER_LINES),
        header_lines=_HEADER_LINES,
    )
    with tempfile.TemporaryDirectory(prefix="osprey-demo-model-") as scratch:
        lattice_text = emit_lattice(ring, Path(scratch) / LATTICE_ARTIFACT, context)
    digest = context.lattice_sha256
    if digest is None:  # pragma: no cover - emit_lattice records it or raises
        raise DemoModelError(
            "the deck was rendered without recording its digest, so the bindings "
            "would stamp no lattice for a served tree to check itself against"
        )
    machine = json.loads(PACKAGE_PATHS.machine_json.read_text(encoding="utf-8"))
    document = build_demo_bindings(ring, digest, machine["channels"])
    return lattice_text, dump_bindings(document)


def main(argv: Sequence[str] | None = None) -> int:
    """Regenerate the demo artifacts, or report that they have drifted.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.

    Returns:
        The process exit status: ``1`` when ``--check`` found a stale file,
        ``0`` otherwise.
    """
    parser = argparse.ArgumentParser(
        prog="python -m osprey.simulation.lattice.build",
        description="Regenerate the demo facility's lattice and virtual-accelerator bindings.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "compare the committed lattice.json and va_bindings.json with what this "
            "generator produces today, byte for byte, and write nothing"
        ),
    )
    args = parser.parse_args(argv)

    ring = build_ring()
    lattice_text, bindings_text = render_demo_model(ring)
    targets = (
        (PACKAGE_PATHS.lattice_json, lattice_text),
        (PACKAGE_PATHS.va_bindings, bindings_text),
    )

    if args.check:
        stale = [
            path
            for path, text in targets
            if not path.is_file() or path.read_bytes() != text.encode("utf-8")
        ]
        for path in stale:
            print(f"stale: {path}")
        if stale:
            print("Regenerate with: python -m osprey.simulation.lattice.build")
            return 1
        print(f"Up to date: {', '.join(path.name for path, _ in targets)}")
        return 0

    mat = save_canonical_mat(ring)
    for path, text in targets:
        write_if_changed(path, text)
    print(f"Wrote {len(ring)}-element {ALS_U_AR.name} ring to {mat}")
    for path, _ in targets:
        print(f"Wrote {path}")
    return 0


def _elements_by_name(
    ring: at.Lattice, census: Mapping[str, int]
) -> dict[str, at.elements.Element]:
    """Index a ring by element name, keeping only the names it carries once.

    A name two elements share says nothing about which one a channel writes,
    so it is left out here and the device that wanted it is refused by name.

    Args:
        ring: The deck to index.
        census: How many elements the ring carries under each name.

    Returns:
        The element behind each name the ring carries exactly once.
    """
    return {element.FamName: element for element in ring if census[element.FamName] == 1}


def _unbindable(name: str, family: str, device: int, count: int) -> str:
    """Say why a declared device binds nothing, counting what the ring holds.

    Args:
        name: The element name the device would bind.
        family: The family the facility spec declares the device under.
        device: The device's one-based number within that family.
        count: How many elements the ring carries under ``name``.

    Returns:
        The refusal message, which distinguishes an absent name from an
        ambiguous one because the two want opposite repairs.
    """
    if count == 0:
        return (
            f"the ring carries no element named {name!r}, which the facility spec "
            f"declares as device {device} of family {family!r}"
        )
    return (
        f"the ring carries {count} elements named {name!r}, so nothing says which one "
        f"device {device} of family {family!r} writes"
    )


def _device_bindings(
    family: Family, device: int, name: str, element: at.elements.Element, machine_channels: dict
) -> list[Binding]:
    """Build every binding one device carries."""
    if family.kind == "monitor":
        return [
            _monitor_binding(family, device, name, axis, machine_channels) for axis in _MONITOR_AXES
        ]
    return [_driven_binding(family, device, name, element, machine_channels)]


def _monitor_binding(
    family: Family, device: int, name: str, axis: str, machine_channels: dict
) -> Binding:
    """One transverse axis of one monitor, read back through the identity.

    The monitor's channels are in metres and the model's closed orbit is in
    metres, so both curves are the identity -- stated as data like any
    facility's, rather than left out as a conversion that happens to be free.

    The nominal is the reading the seed states, which is what a facility export
    carries for a monitor: the position the device sat at when the machine was
    read. It starts the channel off and the model drives it from there.
    """
    address = _POSITION_ADDRESS.format(
        system=DEMO_SYSTEM, family=family.name, device=f"{device:02d}", axis=axis.upper()
    )
    nominal = _require_seed(machine_channels, address, name)
    return Binding(
        kind="monitor",
        family=family.name,
        setpoint_address=address,
        readback_address=None,
        readback="inverse",
        element=name,
        attribute=axis,
        index=None,
        slices=(Slice(element=name, weight=1.0),),
        owner=family.name,
        calibration=Linear(gain=1.0, offset=0.0),
        monitor_inverse=Linear(gain=1.0, offset=0.0),
        nominal=nominal,
        energy_scaling="none",
        energy_table=None,
    )


def _driven_binding(
    family: Family, device: int, name: str, element: at.elements.Element, machine_channels: dict
) -> Binding:
    """One magnet or corrector: its setpoint, its echoed readback, its curve.

    A magnet driven by a supply current holds its integrated field, not its
    normalised strength, so the strength a given current is worth moves with
    the beam rigidity -- the same word every facility export states for these
    families. The demo ring has no energy knob, so the factor is one today and
    the served numbers do not move; what changes is that the document says
    what it means rather than leaving it to be assumed.
    """
    setpoint = _current_address(family, device, "SP")
    readback = _current_address(family, device, "RB")
    nominal = _require_seed(machine_channels, setpoint, name)
    _require_seed(machine_channels, readback, name)
    kind, attribute, index, calibration = _physics_of(family, name, element, nominal)
    return Binding(
        kind=kind,
        family=family.name,
        setpoint_address=setpoint,
        readback_address=readback,
        readback="identity",
        element=name,
        attribute=attribute,
        index=index,
        slices=(Slice(element=name, weight=1.0),),
        owner=family.name,
        calibration=calibration,
        monitor_inverse=None,
        nominal=nominal,
        energy_scaling="brho" if kind == "strength" else "none",
        energy_table=None,
    )


def _physics_of(
    family: Family, name: str, element: at.elements.Element, nominal: float
) -> tuple[str, str, int, Linear]:
    """Return what a driven device writes, and the line it writes it through.

    Raises:
        DemoModelError: The element is not a kind this generator can state a
            calibration for, or its calibration would carry no value.
    """
    if isinstance(element, at.Corrector):
        plane = _CORRECTOR_PLANE.get(family.name)
        if plane is None:
            raise DemoModelError(
                f"element {name!r} is a corrector, but family {family.name!r} names no plane "
                f"for it; the ring's corrector families are {sorted(_CORRECTOR_PLANE)}"
            )
        return "kick", "KickAngle", plane, Linear(gain=1.0 / AMPS_PER_RADIAN_KICK, offset=0.0)

    if isinstance(element, at.Dipole):
        # A trim coil about the nominal: the bend's angle and gradient stay put
        # and only the pure-dipole field error moves, through zero at nominal.
        slope = float(element.BendingAngle) / float(element.Length)
        gain = _nonzero(slope / _driving(nominal, name), name, "bending angle per unit length")
        return "strength", "PolynomB", 0, Linear(gain=gain, offset=-slope)

    for element_type, index in _MULTIPOLE_INDEX:
        if isinstance(element, element_type):
            baked = float(element.PolynomB[index])
            gain = _nonzero(baked / _driving(nominal, name), name, f"baked PolynomB[{index}]")
            return "strength", "PolynomB", index, Linear(gain=gain, offset=0.0)

    raise DemoModelError(
        f"element {name!r} is a {type(element).__name__}, which states no current-driven "
        "strength this generator knows how to calibrate"
    )


def _current_address(family: Family, device: int, half: str) -> str:
    return _CURRENT_ADDRESS.format(
        system=DEMO_SYSTEM, family=family.name, device=f"{device:02d}", half=half
    )


def _require_seed(machine_channels: dict, address: str, name: str) -> float:
    """Return the scenario seed's value for one address.

    Raises:
        DemoModelError: The seed states no nominal for that address, so the
            channel the binding would create is one the mock engine could not
            serve and the calibration would rest on a guess.
    """
    entry = machine_channels.get(address)
    if not isinstance(entry, dict) or "value" not in entry:
        raise DemoModelError(
            f"the scenario seed states no nominal for {address!r}, the channel element "
            f"{name!r} is driven by"
        )
    return float(entry["value"])


def _driving(nominal: float, name: str) -> float:
    """Return the nominal current a strength is scaled by.

    Raises:
        DemoModelError: The device is parked at zero current, where the
            fraction of nominal it is driven at is not defined.
    """
    if nominal == 0.0:
        raise DemoModelError(
            f"element {name!r} is seeded at zero current, so the fraction of nominal its "
            "strength scales with is not defined"
        )
    return nominal


def _nonzero(gain: float, name: str, what: str) -> float:
    """Return a calibration gain, refusing one that moves nothing.

    Raises:
        DemoModelError: The gain is zero, which is a channel that writes a
            value the lattice never sees.
    """
    if gain == 0.0:
        raise DemoModelError(
            f"element {name!r} has a zero {what}, so its channel would write a value the "
            "lattice never sees"
        )
    return gain


if __name__ == "__main__":
    raise SystemExit(main())
