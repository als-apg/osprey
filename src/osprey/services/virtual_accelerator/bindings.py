"""The VA bindings document, parsed into frozen dataclasses.

``va_bindings.json`` is the one file that says what a channel does to the
lattice: which element attribute a write lands in, through which calibration,
where and how its readback is served, and what the device's nominal value is.
The emit lane writes the file and the served virtual accelerator reads it, so
this module is the schema both halves agree on and the only place its rules
are stated.

The document carries the facts that make it self-checking -- the ``system`` it
describes, the ring's ``energy_gev`` and the ``lattice_sha256`` of the lattice
file the bindings were derived against -- and then one binding per address:

* ``strength`` writes a polynomial coefficient (``PolynomB``/``PolynomA``) at
  ``index``; ``kick`` writes ``KickAngle[0]`` or ``KickAngle[1]``; ``rf``
  writes every cavity's ``Frequency``; ``monitor`` reads a transverse axis;
  ``energy`` is the lattice-level knob and binds no element at all.
* ``slices`` names every element the value is written to, each with the weight
  it is multiplied by. A weight is any finite non-zero factor: an ordinary
  device weighs 1, a split device shares the value out as ``1/n`` a piece
  rather than replicating it, and a supply feeding magnets in series gives
  each magnet the fixed factor that puts it at its own strength. The first
  slice is the element the readback is read from, so ``element`` always
  repeats it.
* ``readback`` says how the readback value is produced and
  ``readback_address`` where it is served: ``identity`` is the written value on
  its own address, ``inverse`` is the value mapped back through
  ``monitor_inverse``, and ``same_as_setpoint`` is one address carrying both.
  A monitor serves its reading on its own address, always through the inverse.
* ``calibration`` converts hardware to physics and ``monitor_inverse`` physics
  to hardware. Both are either ``{kind: linear, gain, offset}`` or
  ``{kind: table, grid, values}``. The two directions are independent data --
  one is never derived from the other -- which is why a readback that needs an
  inverse is refused without one rather than falling back on an inversion.
* ``energy_scaling`` marks the bindings whose physics strength moves with the
  beam rigidity, and the ``energy`` binding's ``energy_table`` maps its
  hardware setpoint to beam energy.
* ``readout`` carries the facility's own calibration of one monitor's reading
  -- the numbers an error model perturbs around. It is the one optional key:
  a binding that states none omits it, and so does a document emitted for a
  facility that calibrates nothing.

Structure and rules only: every key but ``readout`` is required, and explicit
nulls stand for the slots a kind does not use, so a misspelt or missing key is
refused rather than silently defaulted. Nothing here evaluates a calibration, loads a lattice
or touches a channel -- that belongs to the layers above, which import these
dataclasses instead of respelling the document.

Nothing in this module names a family, an address token or a facility
constant: a binding carries all of those as data, which is what lets one
served path cover every facility. Pure stdlib; grids are plain tuples, so a
document is hashable and compares by value.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "ATTRIBUTES_BY_KIND",
    "BINDING_KINDS",
    "ELEMENT_KINDS",
    "ENERGY_SCALINGS",
    "PROVENANCE_KEY",
    "READBACK_RULES",
    "READOUT_KEYS",
    "Binding",
    "BindingsDocument",
    "BindingsError",
    "Calibration",
    "Linear",
    "Readout",
    "Slice",
    "Table",
    "dump_bindings",
    "load_bindings",
    "parse_bindings",
    "setpoints",
]

#: What a binding drives, in the order the document lists them.
BINDING_KINDS: tuple[str, ...] = ("strength", "kick", "monitor", "energy", "rf")

#: The kinds that bind a lattice element; ``energy`` is the lattice-level knob.
ELEMENT_KINDS: frozenset[str] = frozenset({"strength", "kick", "monitor", "rf"})

#: How a binding's readback value is produced.
READBACK_RULES: tuple[str, ...] = ("identity", "inverse", "same_as_setpoint")

#: Whether a binding's physics value moves with the beam rigidity.
ENERGY_SCALINGS: tuple[str, ...] = ("brho", "none")

#: The element attribute each kind may write, in pyAT's spelling; a monitor's
#: attribute is the transverse axis its reading comes from.
ATTRIBUTES_BY_KIND: dict[str, frozenset[str]] = {
    "strength": frozenset({"PolynomA", "PolynomB"}),
    "kick": frozenset({"KickAngle"}),
    "monitor": frozenset({"x", "y"}),
    "energy": frozenset(),
    "rf": frozenset({"Frequency"}),
}

#: The document key an emitter stamps its provenance under.
PROVENANCE_KEY = "_provenance"

#: The kinds whose attribute is indexed into.
_INDEXED_KINDS: frozenset[str] = frozenset({"strength", "kick"})

#: The kinds whose physics value may be rescaled with the beam rigidity.
_SCALED_KINDS: frozenset[str] = frozenset({"strength", "kick"})

#: The two transverse components of a kick.
_KICK_INDICES: tuple[int, ...] = (0, 1)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_TOP_REQUIRED = frozenset({"system", "energy_gev", "lattice_sha256", "bindings"})
_TOP_OPTIONAL = frozenset({PROVENANCE_KEY})
_BINDING_KEYS = frozenset(
    {
        "kind",
        "family",
        "setpoint_address",
        "readback_address",
        "readback",
        "element",
        "attribute",
        "index",
        "slices",
        "owner",
        "calibration",
        "monitor_inverse",
        "nominal",
        "energy_scaling",
        "energy_table",
    }
)
_BINDING_OPTIONAL = frozenset({"readout"})

#: What a monitor's readout may state, each one number for this device.
READOUT_KEYS: tuple[str, ...] = ("gain", "offset", "roll", "crunch")

_SLICE_KEYS = frozenset({"element", "weight"})
_LINEAR_KEYS = frozenset({"kind", "gain", "offset"})
_TABLE_KEYS = frozenset({"kind", "grid", "values"})
_NONE: frozenset[str] = frozenset()


class BindingsError(ValueError):
    """Raised when a bindings document has the wrong structure or rules.

    Args:
        key: Dotted path to the offending key, e.g.
            ``bindings[3].calibration.grid[2]``; the empty string is the
            document itself.
        message: What is wrong, phrased for the person reading the file.
        source: The file the document came from, when it came from one.
    """

    def __init__(self, key: str, message: str, source: str | None = None) -> None:
        super().__init__(message)
        self.key = key
        self.message = message
        self.source = source

    def __str__(self) -> str:
        """Return the refusal with its key, and its file when there is one."""
        located = f"{self.key}: {self.message}" if self.key else self.message
        return f"{self.source}: {located}" if self.source else located


@dataclass(frozen=True)
class Linear:
    """A straight-line conversion ``y = gain * x + offset``."""

    gain: float
    offset: float


@dataclass(frozen=True)
class Table:
    """A sampled conversion: piecewise linear through ``(grid, values)``.

    ``grid`` is strictly monotonic, so the curve is well defined everywhere;
    beyond either end the consumer extrapolates along the last segment.
    """

    grid: tuple[float, ...]
    values: tuple[float, ...]


#: Either shape a conversion may take.
Calibration = Linear | Table


@dataclass(frozen=True)
class Slice:
    """One element a binding writes, and the weight its value carries there.

    The written quantity is ``value * weight``; the readback divides the first
    slice's reading by that same weight, which is how a kick shared over ``n``
    pieces reads back as the whole kick.
    """

    element: str
    weight: float


@dataclass(frozen=True)
class Readout:
    """How one monitor's reading is calibrated, as the facility states it.

    These are the numbers the control system already applies before it
    publishes the reading, carried here so that something can perturb them:
    the gain and the offset are inside :attr:`Binding.calibration` and
    :attr:`Binding.monitor_inverse` too, which is exactly why the served path
    must not apply them a second time, and the roll and the crunch -- the
    rotation and the shear of the monitor pair -- are in neither curve and
    are applied by nothing today.

    **The seam.** A readout-error model belongs between the two halves of a
    monitor read: the solved physics coordinate, and the hardware reading
    :attr:`Binding.monitor_inverse` turns it into
    (``MonitorVariable._get``). It perturbs these numbers and re-derives the
    reading from the perturbed ones, so a monitor left unperturbed reproduces
    the reading exactly. Two facts such a model has to honour: the algebra is
    ``real = gain * (raw - offset)``, offset first; and where the served deck
    carries a monitor's rotation on the element itself, one rotation covers
    both planes and it is the horizontal family's, so applying each plane's
    own roll and crunch turns the pair twice and turns the vertical one by a
    number the control system never applies. Only a deck that carries no such
    rotation leaves each plane its own.

    Each number is this device's own and every one of them is optional: an
    absent number is not a zero but a fact the facility does not state, which
    a consumer reads as the value that changes nothing -- a gain of one, an
    offset, roll and crunch of zero.

    Attributes:
        gain: Dimensionless scale on the raw reading.
        offset: Subtracted from the raw reading, in the family's own hardware
            units -- what its readback answers in, not the physics units of
            the conversion beside it.
        roll: Rotation of the monitor pair, in radians.
        crunch: Shear of the monitor pair, dimensionless.
    """

    gain: float | None = None
    offset: float | None = None
    roll: float | None = None
    crunch: float | None = None

    @property
    def stated(self) -> tuple[str, ...]:
        """The keys this readout states, in :data:`READOUT_KEYS` order."""
        return tuple(name for name in READOUT_KEYS if getattr(self, name) is not None)


@dataclass(frozen=True)
class Binding:
    """One address and what it does to the lattice.

    Attributes:
        kind: One of :data:`BINDING_KINDS`.
        family: The family the address belongs to.
        setpoint_address: The address written, or read for a ``monitor``.
        readback_address: Where the readback is served; ``None`` means on
            :attr:`setpoint_address` itself.
        readback: One of :data:`READBACK_RULES`.
        element: The element read, ``None`` for ``energy``.
        attribute: The element attribute written, the axis read for a
            ``monitor``, ``None`` for ``energy``.
        index: The component of :attr:`attribute`, ``None`` where it has none.
        slices: Every element written, first the one read back.
        owner: The family the bound element is named after.
        calibration: Hardware to physics; ``None`` only for ``energy``.
        monitor_inverse: Physics back to hardware, as the facility measured it.
        nominal: The device's nominal hardware value.
        energy_scaling: One of :data:`ENERGY_SCALINGS`.
        energy_table: Hardware setpoint to beam energy, for ``energy`` alone.
        readout: How this monitor's reading is calibrated, for a ``monitor``
            alone; ``None`` where the facility states none of it.
    """

    kind: str
    family: str
    setpoint_address: str
    readback_address: str | None
    readback: str
    element: str | None
    attribute: str | None
    index: int | None
    slices: tuple[Slice, ...]
    owner: str | None
    calibration: Calibration | None
    monitor_inverse: Calibration | None
    nominal: float | None
    energy_scaling: str
    energy_table: Table | None
    readout: Readout | None = None

    @property
    def is_writable(self) -> bool:
        """Whether the binding is written; a ``monitor`` is read only."""
        return self.kind != "monitor"


@dataclass(frozen=True)
class BindingsDocument:
    """A structurally valid ``va_bindings.json``.

    Attributes:
        system: The system the bindings describe.
        energy_gev: The beam energy the lattice file is built at.
        lattice_sha256: Digest of the lattice file the bindings were derived
            against, checked when the ring is loaded.
        bindings: The bindings, in document order.
        provenance: The emitter's provenance stamp, when it carried one.
    """

    system: str
    energy_gev: float
    lattice_sha256: str
    bindings: tuple[Binding, ...]
    provenance: str | None = None


def parse_bindings(document: Any) -> BindingsDocument:
    """Parse an already-loaded bindings document.

    Args:
        document: The document as plain JSON types.

    Returns:
        The parsed document.

    Raises:
        BindingsError: A key is missing, unknown, or holds a value the schema
            refuses. The message names the key.
    """
    body = _dict(document, "")
    _keys(body, "", _TOP_REQUIRED, _TOP_OPTIONAL)
    bindings = tuple(
        _binding(entry, f"bindings[{position}]")
        for position, entry in enumerate(_list(body["bindings"], "bindings"))
    )
    parsed = BindingsDocument(
        system=body["system"],
        energy_gev=body["energy_gev"],
        lattice_sha256=body["lattice_sha256"],
        bindings=bindings,
        provenance=body.get(PROVENANCE_KEY),
    )
    _check_document(parsed)
    return parsed


def load_bindings(path: Path | str) -> BindingsDocument:
    """Read and parse a bindings file.

    Args:
        path: The file to read.

    Returns:
        The parsed document.

    Raises:
        FileNotFoundError: The file does not exist; the message names it.
        BindingsError: The text is not JSON, or the document is refused. The
            message names the file.
        OSError: The file could not be read.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"VA bindings file not found: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        document = json.loads(text, parse_constant=_refuse_constant)
    except ValueError as exc:
        raise BindingsError("", f"is not valid JSON: {exc}", source=str(path)) from exc
    try:
        return parse_bindings(document)
    except BindingsError as exc:
        raise BindingsError(exc.key, exc.message, source=str(path)) from exc


def dump_bindings(document: BindingsDocument) -> str:
    """Render a bindings document as the text the emit lane writes.

    The text is a pure function of the document -- sorted keys, no stamps of
    any kind -- so re-emitting unchanged inputs leaves the file untouched.
    The document is checked first, so a document this refuses is exactly a
    document :func:`load_bindings` would refuse.

    Args:
        document: The document to render.

    Returns:
        Canonical JSON text ending in one newline.

    Raises:
        BindingsError: The document breaks a schema rule.
    """
    _check_document(document)
    body: dict[str, Any] = {
        "system": document.system,
        "energy_gev": float(document.energy_gev),
        "lattice_sha256": document.lattice_sha256,
        "bindings": [_binding_json(binding) for binding in document.bindings],
    }
    if document.provenance is not None:
        body[PROVENANCE_KEY] = document.provenance
    return json.dumps(body, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def setpoints(document: BindingsDocument) -> tuple[str, ...]:
    """Return the written addresses of a document, in document order.

    Args:
        document: The document to list.

    Returns:
        One address per writable binding; a monitor's read-only address is
        not one of them.
    """
    return tuple(binding.setpoint_address for binding in document.bindings if binding.is_writable)


# -- structural helpers -------------------------------------------------------


def _dict(value: Any, key: str) -> dict:
    if not isinstance(value, dict):
        raise BindingsError(key, f"must be a mapping, got {_type_name(value)}")
    return value


def _list(value: Any, key: str) -> list:
    if not isinstance(value, list):
        raise BindingsError(key, f"must be a list, got {_type_name(value)}")
    return value


def _keys(body: dict, path: str, required: frozenset[str], optional: frozenset[str]) -> None:
    for name in body:
        if name not in required and name not in optional:
            raise BindingsError(_join(path, name), "unknown key")
    for name in sorted(required):
        if name not in body:
            raise BindingsError(_join(path, name), "required key is missing")


def _join(path: str, name: object) -> str:
    return f"{path}.{name}" if path else str(name)


def _type_name(value: Any) -> str:
    return "null" if value is None else type(value).__name__


def _shown(value: Any) -> str:
    """Name a refused value: the word itself when it is one, else its type."""
    return repr(value) if isinstance(value, str) else _type_name(value)


def _refuse_constant(token: str) -> Any:
    raise ValueError(f"non-finite token {token!r} is not valid here")


def _words(values: tuple[str, ...] | frozenset[str]) -> str:
    return ", ".join(sorted(values))


# -- parsing ------------------------------------------------------------------


def _binding(value: Any, path: str) -> Binding:
    body = _dict(value, path)
    _keys(body, path, _BINDING_KEYS, _BINDING_OPTIONAL)
    return Binding(
        kind=body["kind"],
        family=body["family"],
        setpoint_address=body["setpoint_address"],
        readback_address=body["readback_address"],
        readback=body["readback"],
        element=body["element"],
        attribute=body["attribute"],
        index=body["index"],
        slices=tuple(
            _slice(entry, f"{path}.slices[{position}]")
            for position, entry in enumerate(_list(body["slices"], f"{path}.slices"))
        ),
        owner=body["owner"],
        calibration=_curve(body["calibration"], f"{path}.calibration"),
        monitor_inverse=_curve(body["monitor_inverse"], f"{path}.monitor_inverse"),
        nominal=body["nominal"],
        energy_scaling=body["energy_scaling"],
        energy_table=_energy_table(body["energy_table"], f"{path}.energy_table"),
        readout=_readout(body["readout"], f"{path}.readout") if "readout" in body else None,
    )


def _readout(value: Any, path: str) -> Readout:
    """Parse a stated readout; the key is left out where none is stated."""
    body = _dict(value, path)
    _keys(body, path, _NONE, frozenset(READOUT_KEYS))
    stated = {name: body[name] for name in READOUT_KEYS if name in body}
    for name, value_ in stated.items():
        if value_ is None:
            raise BindingsError(
                f"{path}.{name}",
                f"the reading's {name}: must be a number, and a key the facility "
                "states nothing for is left out rather than written null",
            )
    return Readout(**stated)


def _slice(value: Any, path: str) -> Slice:
    body = _dict(value, path)
    _keys(body, path, _SLICE_KEYS, _NONE)
    return Slice(element=body["element"], weight=body["weight"])


def _curve(value: Any, path: str) -> Calibration | None:
    if value is None:
        return None
    body = _dict(value, path)
    kind = body.get("kind")
    if kind == "linear":
        _keys(body, path, _LINEAR_KEYS, _NONE)
        return Linear(gain=body["gain"], offset=body["offset"])
    if kind == "table":
        _keys(body, path, _TABLE_KEYS, _NONE)
        return Table(
            grid=tuple(_list(body["grid"], f"{path}.grid")),
            values=tuple(_list(body["values"], f"{path}.values")),
        )
    raise BindingsError(f"{path}.kind", f"must be 'linear' or 'table', got {_shown(kind)}")


def _energy_table(value: Any, path: str) -> Table | None:
    curve = _curve(value, path)
    if curve is None or isinstance(curve, Table):
        return curve
    raise BindingsError(f"{path}.kind", "must be 'table': an energy table is sampled, not linear")


# -- value rules --------------------------------------------------------------


def _text(value: Any, path: str, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise BindingsError(path, f"{what}: must be a non-empty string, got {_shown(value)}")
    return value


def _finite(value: Any, path: str, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise BindingsError(path, f"{what}: must be a number, got {_type_name(value)}")
    if not math.isfinite(value):
        raise BindingsError(path, f"{what}: must be a finite number, got {value}")
    return float(value)


def _one_of(value: Any, path: str, allowed: tuple[str, ...]) -> str:
    if value not in allowed:
        raise BindingsError(path, f"must be one of {_words(allowed)}, got {_shown(value)}")
    return str(value)


def _null(value: Any, path: str, why: str) -> None:
    if value is not None:
        raise BindingsError(path, f"must be null: {why}")


def _check_document(document: BindingsDocument) -> None:
    """Check a whole document, whether it was parsed or built in code."""
    if not isinstance(document, BindingsDocument):
        raise BindingsError("", f"must be a bindings document, got {_type_name(document)}")
    _text(document.system, "system", "the system the bindings describe")
    energy = _finite(document.energy_gev, "energy_gev", "the ring's beam energy")
    if energy <= 0.0:
        raise BindingsError("energy_gev", f"must be greater than zero, got {energy}")
    digest = document.lattice_sha256
    if not isinstance(digest, str) or not _SHA256.match(digest):
        raise BindingsError(
            "lattice_sha256",
            f"must be a 64-character lowercase hex sha256, got {_shown(digest)}",
        )
    if document.provenance is not None and not isinstance(document.provenance, str):
        raise BindingsError(
            PROVENANCE_KEY, f"must be a string, got {_type_name(document.provenance)}"
        )
    if not isinstance(document.bindings, tuple):
        raise BindingsError("bindings", f"must be a list, got {_type_name(document.bindings)}")

    addresses: dict[str, str] = {}
    fields: dict[tuple[str, str, int | None], str] = {}
    energy_family: str | None = None
    for position, binding in enumerate(document.bindings):
        path = f"bindings[{position}]"
        if not isinstance(binding, Binding):
            raise BindingsError(path, f"must be a binding, got {_type_name(binding)}")
        _check_binding(binding, path)
        if binding.kind == "energy":
            if energy_family is not None:
                raise BindingsError(
                    f"{path}.kind",
                    f"the energy knob is already bound by family {energy_family!r}",
                )
            energy_family = binding.family
        _claim_address(binding.setpoint_address, f"{path}.setpoint_address", binding, addresses)
        if binding.readback_address is not None:
            _claim_address(binding.readback_address, f"{path}.readback_address", binding, addresses)
        for slot, slice_ in enumerate(binding.slices):
            field = (slice_.element, str(binding.attribute), binding.index)
            owner = fields.get(field)
            if owner is not None:
                raise BindingsError(
                    f"{path}.slices[{slot}].element",
                    f"element {slice_.element!r} field {_field_name(binding)} "
                    f"is already written by family {owner!r}",
                )
            fields[field] = binding.family


def _claim_address(address: str, path: str, binding: Binding, addresses: dict[str, str]) -> None:
    owner = addresses.get(address)
    if owner is not None:
        raise BindingsError(path, f"address {address!r} is already bound by family {owner!r}")
    addresses[address] = binding.family


def _field_name(binding: Binding) -> str:
    """Name an element field the way the refusals write it."""
    if binding.index is None:
        return str(binding.attribute)
    return f"{binding.attribute}[{binding.index}]"


def _check_binding(binding: Binding, path: str) -> None:
    kind = _one_of(binding.kind, f"{path}.kind", BINDING_KINDS)
    _text(binding.family, f"{path}.family", "the family the address belongs to")
    _text(binding.setpoint_address, f"{path}.setpoint_address", "the address a binding is keyed by")
    _check_readback_rule(binding, path, kind)
    _check_element(binding, path, kind)
    _check_readback_address(binding, path, kind)
    _check_curves(binding, path, kind)
    _check_values(binding, path, kind)
    _check_readout(binding, path, kind)


def _check_readout(binding: Binding, path: str, kind: str) -> None:
    """Check the readout calibration, which only a monitor reading carries."""
    readout = binding.readout
    if readout is None:
        return
    if not isinstance(readout, Readout):
        raise BindingsError(f"{path}.readout", f"must be a readout, got {_type_name(readout)}")
    if kind != "monitor":
        raise BindingsError(
            f"{path}.readout",
            f"a {kind} binding drives a device and publishes no reading, so there is "
            "nothing for a readout calibration to correct",
        )
    for name in READOUT_KEYS:
        value = getattr(readout, name)
        if value is not None:
            _finite(value, f"{path}.readout.{name}", f"the reading's {name}")
    if not readout.stated:
        raise BindingsError(
            f"{path}.readout",
            "states none of "
            f"{_words(READOUT_KEYS)}: leave the key out where the facility states nothing",
        )


def _check_readback_rule(binding: Binding, path: str, kind: str) -> None:
    readback = _one_of(binding.readback, f"{path}.readback", READBACK_RULES)
    if kind == "monitor" and readback != "inverse":
        raise BindingsError(
            f"{path}.readback",
            "a monitor reading is served through monitor_inverse: must be 'inverse', "
            f"got {readback!r}",
        )
    if kind == "energy" and readback == "inverse":
        raise BindingsError(
            f"{path}.readback",
            "the energy knob converts through energy_table alone, so there is no physics "
            "value to map back: must be 'identity' or 'same_as_setpoint'",
        )


def _check_element(binding: Binding, path: str, kind: str) -> None:
    if kind not in ELEMENT_KINDS:
        _null(binding.element, f"{path}.element", "the energy knob binds no element")
        _null(binding.attribute, f"{path}.attribute", "the energy knob writes no attribute")
        _null(binding.index, f"{path}.index", "the energy knob writes no attribute")
        if binding.slices:
            raise BindingsError(f"{path}.slices", "must be empty: the energy knob binds no element")
        _null(binding.owner, f"{path}.owner", "the energy knob binds no element")
        return

    element = _text(binding.element, f"{path}.element", "the element the readback is read from")
    attribute = binding.attribute
    allowed = ATTRIBUTES_BY_KIND[kind]
    if attribute not in allowed:
        raise BindingsError(
            f"{path}.attribute",
            f"a {kind} binding writes {_words(allowed)}, got {_shown(attribute)}",
        )
    _check_index(binding, path, kind)
    _check_slices(binding, path, element)
    _text(binding.owner, f"{path}.owner", "the family the bound element is named after")


def _check_index(binding: Binding, path: str, kind: str) -> None:
    index = binding.index
    if kind not in _INDEXED_KINDS:
        _null(index, f"{path}.index", f"a {kind} binding's attribute has no components")
        return
    if isinstance(index, bool) or not isinstance(index, int):
        raise BindingsError(
            f"{path}.index",
            f"a {kind} binding names the component it writes: must be an integer, "
            f"got {_type_name(index)}",
        )
    if kind == "kick" and index not in _KICK_INDICES:
        raise BindingsError(f"{path}.index", f"a kick writes component 0 or 1, got {index}")
    if index < 0:
        raise BindingsError(f"{path}.index", f"must be 0 or more, got {index}")


def _check_slices(binding: Binding, path: str, element: str) -> None:
    if not binding.slices:
        raise BindingsError(
            f"{path}.slices", "must name at least one element: the one the binding writes"
        )
    seen: set[str] = set()
    for position, slice_ in enumerate(binding.slices):
        slot = f"{path}.slices[{position}]"
        if not isinstance(slice_, Slice):
            raise BindingsError(slot, f"must be a slice, got {_type_name(slice_)}")
        name = _text(slice_.element, f"{slot}.element", "the element this slice writes")
        if name in seen:
            raise BindingsError(f"{slot}.element", f"element {name!r} is already a slice here")
        seen.add(name)
        weight = _finite(slice_.weight, f"{slot}.weight", "the share this slice carries")
        if weight == 0.0:
            raise BindingsError(
                f"{slot}.weight", "must not be zero: a slice written with no weight moves nothing"
            )
    if binding.slices[0].element != element:
        raise BindingsError(
            f"{path}.slices[0].element",
            f"the first slice is the element read back: must be {element!r}, "
            f"got {binding.slices[0].element!r}",
        )


def _check_readback_address(binding: Binding, path: str, kind: str) -> None:
    address = binding.readback_address
    if kind == "monitor":
        _null(
            address, f"{path}.readback_address", "a monitor serves its reading on its own address"
        )
        return
    if binding.readback == "same_as_setpoint":
        _null(
            address,
            f"{path}.readback_address",
            "setpoint and readback share one address, which setpoint_address already names",
        )
        return
    _text(address, f"{path}.readback_address", "where the readback is served")
    if address == binding.setpoint_address:
        raise BindingsError(
            f"{path}.readback_address",
            "one address carrying both is readback 'same_as_setpoint' with no second address",
        )


def _check_curves(binding: Binding, path: str, kind: str) -> None:
    if kind == "energy":
        _null(
            binding.calibration,
            f"{path}.calibration",
            "the energy knob converts through energy_table alone",
        )
        _null(
            binding.monitor_inverse,
            f"{path}.monitor_inverse",
            "the energy knob reads back the value it was written",
        )
    else:
        if binding.calibration is None:
            raise BindingsError(
                f"{path}.calibration", "is required: it converts the written value to physics"
            )
        _check_curve(binding.calibration, f"{path}.calibration")
        _check_inverse(binding, path)


def _check_inverse(binding: Binding, path: str) -> None:
    inverse = binding.monitor_inverse
    if binding.readback == "identity":
        _null(
            inverse,
            f"{path}.monitor_inverse",
            "an identity readback serves the written value and applies no inverse",
        )
        return
    if binding.readback == "inverse" and inverse is None:
        raise BindingsError(
            f"{path}.monitor_inverse",
            "is required by readback 'inverse': the facility's own inverse is the only way "
            "back to hardware, and a calibration is never inverted to stand in for it",
        )
    if inverse is not None:
        _check_curve(inverse, f"{path}.monitor_inverse")


def _check_values(binding: Binding, path: str, kind: str) -> None:
    if binding.nominal is None:
        if kind != "monitor":
            raise BindingsError(
                f"{path}.nominal", f"is required: a {kind} binding has a nominal hardware value"
            )
    else:
        _finite(binding.nominal, f"{path}.nominal", "the device's nominal hardware value")

    scaling = _one_of(binding.energy_scaling, f"{path}.energy_scaling", ENERGY_SCALINGS)
    if scaling == "brho" and kind not in _SCALED_KINDS:
        raise BindingsError(
            f"{path}.energy_scaling",
            f"must be 'none': a {kind} binding carries no physics strength to rescale "
            "with the beam rigidity",
        )

    if kind == "energy":
        if binding.energy_table is None:
            raise BindingsError(
                f"{path}.energy_table",
                "is required: it is what maps the knob's hardware setpoint to beam energy",
            )
        _check_curve(binding.energy_table, f"{path}.energy_table")
    else:
        _null(
            binding.energy_table,
            f"{path}.energy_table",
            "only the energy knob maps a setpoint to beam energy",
        )


def _check_curve(curve: Any, path: str) -> None:
    if isinstance(curve, Linear):
        gain = _finite(curve.gain, f"{path}.gain", "the conversion's gain")
        if gain == 0.0:
            raise BindingsError(
                f"{path}.gain", "must not be zero: a conversion with no gain carries no value"
            )
        _finite(curve.offset, f"{path}.offset", "the conversion's offset")
        return
    if not isinstance(curve, Table):
        raise BindingsError(path, f"must be a linear or table conversion, got {_type_name(curve)}")

    grid = _numbers(curve.grid, f"{path}.grid", "the sampled points")
    values = _numbers(curve.values, f"{path}.values", "the sampled values")
    if len(values) != len(grid):
        raise BindingsError(
            f"{path}.values",
            f"must have as many entries as grid ({len(grid)}), got {len(values)}",
        )
    if len(grid) < 2:
        raise BindingsError(f"{path}.grid", f"must sample at least two points, got {len(grid)}")
    rising = grid[1] > grid[0]
    for position in range(1, len(grid)):
        step = grid[position] - grid[position - 1]
        if step == 0.0 or (step > 0.0) != rising:
            raise BindingsError(
                f"{path}.grid[{position}]",
                "must be strictly increasing or strictly decreasing throughout, "
                f"but {grid[position - 1]} is followed by {grid[position]}",
            )


def _numbers(values: Any, path: str, what: str) -> tuple[float, ...]:
    if not isinstance(values, tuple | list):
        raise BindingsError(path, f"{what}: must be a list, got {_type_name(values)}")
    return tuple(
        _finite(value, f"{path}[{position}]", what) for position, value in enumerate(values)
    )


# -- rendering ----------------------------------------------------------------


def _binding_json(binding: Binding) -> dict[str, Any]:
    body = _binding_body(binding)
    if binding.readout is not None:
        body["readout"] = {
            name: float(getattr(binding.readout, name)) for name in binding.readout.stated
        }
    return body


def _binding_body(binding: Binding) -> dict[str, Any]:
    return {
        "kind": binding.kind,
        "family": binding.family,
        "setpoint_address": binding.setpoint_address,
        "readback_address": binding.readback_address,
        "readback": binding.readback,
        "element": binding.element,
        "attribute": binding.attribute,
        "index": binding.index,
        "slices": [
            {"element": slice_.element, "weight": float(slice_.weight)} for slice_ in binding.slices
        ],
        "owner": binding.owner,
        "calibration": _curve_json(binding.calibration),
        "monitor_inverse": _curve_json(binding.monitor_inverse),
        "nominal": None if binding.nominal is None else float(binding.nominal),
        "energy_scaling": binding.energy_scaling,
        "energy_table": _curve_json(binding.energy_table),
    }


def _curve_json(curve: Calibration | None) -> dict[str, Any] | None:
    if curve is None:
        return None
    if isinstance(curve, Linear):
        return {"kind": "linear", "gain": float(curve.gain), "offset": float(curve.offset)}
    return {
        "kind": "table",
        "grid": [float(value) for value in curve.grid],
        "values": [float(value) for value in curve.values],
    }
