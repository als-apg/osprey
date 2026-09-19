"""What the virtual accelerator does with each exported family, decided by rule.

An export states, per family, the lattice type its Middle Layer binds to, the
element indices it binds, how hardware converts to physics, and -- for the
families a dipole ramp answers for -- the energy table. A deck travels beside
it. Between the two, most families decide themselves: a quadrupole drives a
gradient, a corrector drives a kick, a beam monitor reads an orbit. The rest
are questions, and a question is worth asking only when a person can answer it.

:func:`propose` reads one system's export block against its deck and returns
one :class:`~osprey.services.mml.mapping.schema.VAFamily` per family: a
``couple`` verdict naming what the model drives, or a ``latch`` verdict naming
why it drives nothing. Either may carry the one slot -- ``attype``,
``shared_field`` or ``escape_hatch`` -- that decides the family once a reviewer
answers it.

The rules run in this order, and the first one that settles a family wins:

* **Energy first.** A family the export marks an energy candidate never reaches
  the type table: a ramp that answers the same energy at every current is a
  constant, not a knob; a ramp that disagrees with the deck the export was
  sampled over is a pairing the reviewer has to look at; and only one family
  can be the knob, so a second one is a question.
* **A bend that corrects.** A dipole-typed family whose membership also names
  the correctors is a trim, not the energy knob. It binds a kick if its
  elements take one, and which plane is never inferred from the name.
* **The cavity.** A radio-frequency family couples when the deck holds a
  cavity, whatever indices the export states: cavities are bound by class,
  because the Middle Layer's index into them is not the model's.
* **The type table.** Every remaining family resolves its type token, folded
  to lower case. A token the table does not know is a question when the export
  binds elements and nothing at all when it does not. A dipole token the export
  left out of the candidate set drives no energy: there is one knob, and this
  family is either a second claim on it or a family with no ramp behind it.
* **Then the element rules**, for the families that bind an element field:
  the indices are read, the physics units are checked against the field the
  type implies, every bound element is checked for the attribute the model
  would write, a field two families both drive is a question, and a family the
  Middle Layer reaches through a function of its own is a question.

Two of those last rules are worth spelling out.

The units check passes when *either* the setpoint or the monitor states the
units the type implies, because a family often states its hardware on one side
and its physics on the other; the sibling that disagrees is carried on the
verdict's ``reason`` so the profile can list it. A slot opens only when neither
side states them.

A hook -- a special function, or a parameter group -- means the Middle Layer
reaches the family through code of its own rather than through the type and
index it states, so what the export says about the binding is not the whole
story. That is worth asking about whether or not the type token resolved, so
the hook rule runs for every family but the energy knob and the cavity, whose
bindings are class-wide and carry no element field to override. It latches:
the answer that couples such a family is the one that says to ignore the hook.

A family carrying an unanswered slot is always latched, because nothing may be
driven on an open question. Answering a slot is what turns it into a binding.

The rules read the deck itself for every fact about the ring -- its energy, its
cavities, the attributes of its elements -- and never the export's statement of
those facts, so an export whose lattice block states a refusal instead of the
four facts proposes exactly what a complete one does. Whether that deck is the
deck the export was sampled over is the pairing check's question, not this one.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from osprey.services.mml.mapping.schema import (
    ATTYPE_KIND,
    ESCAPE_HATCH_KIND,
    SHARED_FIELD_KIND,
    UNIT_CLASSES,
    VAFamily,
    VASlot,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from osprey.services.mml.family import FamilyView

__all__ = [
    "AT_HOOK_KEYS",
    "ATTYPE_TABLE",
    "CORRECTOR_MEMBERSHIP",
    "ENERGY_TOLERANCE_GEV",
    "FIELDS",
    "KICK_ATTRIBUTE",
    "attype_slot_opens",
    "carries",
    "index_rows",
    "is_cavity",
    "propose",
    "resolve_attype",
]

#: What each lattice type token drives, keyed by the token folded to lower
#: case: the element kind, and the element field a binding writes, which is
#: ``None`` for the kinds that bind no field of an element.
ATTYPE_TABLE: dict[str, tuple[str, str | None]] = {
    "quad": ("strength", "PolynomB[1]"),
    "k": ("strength", "PolynomB[1]"),
    "quadrupole": ("strength", "PolynomB[1]"),
    "sext": ("strength", "PolynomB[2]"),
    "k2": ("strength", "PolynomB[2]"),
    "sextupole": ("strength", "PolynomB[2]"),
    "octu": ("strength", "PolynomB[3]"),
    "k3": ("strength", "PolynomB[3]"),
    "skewquad": ("strength", "PolynomA[1]"),
    "ks": ("strength", "PolynomA[1]"),
    "ks1": ("strength", "PolynomA[1]"),
    "skewq": ("strength", "PolynomA[1]"),
    "hcm": ("kick", "KickAngle[0]"),
    "hcor": ("kick", "KickAngle[0]"),
    "vcm": ("kick", "KickAngle[1]"),
    "vcor": ("kick", "KickAngle[1]"),
    "bend": ("energy", None),
    "rf": ("rf", None),
    "rf cavity": ("rf", None),
    "x": ("monitor", "x"),
    "bpmx": ("monitor", "x"),
    "xturns": ("monitor", "x"),
    "y": ("monitor", "y"),
    "bpmy": ("monitor", "y"),
    "yturns": ("monitor", "y"),
}

#: How far the energy a family's ramp answers at its nominal current may sit
#: from the energy the deck is solved at, in GeV, before the pair is a question
#: for the reviewer rather than a knob for the model.
ENERGY_TOLERANCE_GEV = 1e-3

#: The membership word that marks a family as a corrector. A dipole-typed
#: family carrying it trims an orbit rather than setting the energy.
CORRECTOR_MEMBERSHIP = "COR"

#: The keys inside an ``AT`` block that mean the Middle Layer reaches the
#: family through code of its own rather than through its type and indices.
AT_HOOK_KEYS = ("SpecialFunctionGet", "SpecialFunctionSet", "ATParameterGroup")

#: The kinds that bind a field of a named element, and so answer to the
#: element rules; the energy knob and the cavity bind neither.
_ELEMENT_KINDS = ("strength", "kick", "monitor")

#: The class name a cavity carries, under every spelling a deck gives it.
_CAVITY = "RFCavity"

#: The attribute a corrector writes, and the one a dipole-typed trim is held to.
KICK_ATTRIBUTE = "KickAngle"

#: The fields an export writes a family's facts under, in the order a family
#: couples through them: a family that sets something is bound through what it
#: sets, and one that only reads is bound through what it reads.
FIELDS = ("Setpoint", "Monitor")

_EV_PER_GEV = 1e9

_NO_ELEMENT = "no lattice element"


class _Proposal:
    """One family part-way through the rules: what is known, and what is not.

    ``reason`` settles the family, so a family carrying one is latched;
    ``note`` says something worth reporting about a family that still couples.
    """

    def __init__(self, name: str, block: dict, view: FamilyView | None) -> None:
        self.name = name
        self.block = block
        self.view = view
        self.field: str | None = next((key for key in FIELDS if key in block), None)
        nominal = block.get("nominals")
        self.nominal: dict = (
            nominal.get(self.field) or {}
            if isinstance(nominal, dict) and self.field is not None
            else {}
        )
        self.candidate: bool = _is_energy_candidate(block)
        self.kind: str | None = None
        self.element_field: str | None = None
        self.rows: list[list[int]] = []
        self.reason: str | None = None
        self.note: str | None = None
        self.slot: VASlot | None = None

    @property
    def token(self) -> str:
        """The lattice type token the export states, as it states it."""
        value = self.nominal.get("at_type")
        return value.strip() if isinstance(value, str) else ""

    @property
    def couples(self) -> bool:
        """Whether the rules so far leave the model driving this family."""
        return self.reason is None and self.slot is None

    def latch(self, reason: str) -> None:
        """Settle the family on a reason, keeping the first one that settled it."""
        if self.reason is None:
            self.reason = reason

    def open(self, kind: str, question: str, reason: str) -> None:
        """Leave the family to a reviewer, keeping the first question asked."""
        if self.slot is None:
            self.slot = VASlot(kind=kind, question=question, answer=None)
            self.reason = reason

    def verdict(self) -> VAFamily:
        """The family as the mapping document carries it."""
        if self.couples:
            return VAFamily(
                verdict="couple",
                kind=self.kind,
                element_field=self.element_field,
                calibration=_calibration_kind(self.block, self.field),
                nominal_source=self.field,
                reason=self.note,
            )
        return VAFamily(verdict="latch", reason=self.reason, slot=self.slot)


def resolve_attype(token: Any) -> tuple[str, str | None] | None:
    """Return what a lattice type token drives.

    Args:
        token: The type token as an export states it; matching folds case and
            ignores surrounding blanks.

    Returns:
        The element kind and the element field a binding writes, or ``None``
        when the token is not one the table knows.
    """
    if not isinstance(token, str):
        return None
    return ATTYPE_TABLE.get(token.strip().lower())


def attype_slot_opens(token: Any, at_index: Any) -> bool:
    """Whether a type token leaves a family to a reviewer rather than a rule.

    A token the table does not know is a question when the export binds
    elements under it, because something is there to bind; with no elements
    bound there is nothing to ask about.

    Args:
        token: The type token as an export states it.
        at_index: The element indices stated beside it, in any of the shapes an
            export writes them: a row per device, a bare number, or nothing.

    Returns:
        Whether the family opens an ``attype`` slot on its token alone.
    """
    return resolve_attype(token) is None and bool(index_rows(at_index))


def propose(
    system_block: dict,
    ring: Sequence[Any],
    views: Mapping[str, FamilyView],
) -> dict[str, VAFamily]:
    """Decide what the virtual accelerator does with each family of one system.

    Args:
        system_block: One system's virtual-accelerator export block, carrying
            its ``families``.
        ring: The deck the export was sampled over, every element kept and in
            saved order, carrying the model energy in eV as ``energy``. Element
            indices are one-based into it, as the Middle Layer states them.
        views: The computed grain of the same system's families, keyed by the
            family token the export uses. A family with no view is read as
            stating no membership, no units and no hooks.

    Returns:
        One verdict per family of the block, in the block's own order.
    """
    families = system_block.get("families")
    if not isinstance(families, dict):
        return {}
    proposals = [
        _Proposal(name, block, views.get(name))
        for name, block in families.items()
        if isinstance(block, dict)
    ]

    deck_gev = _deck_energy_gev(ring)
    knob: str | None = None
    for proposal in proposals:
        if proposal.candidate:
            knob = _energy_rule(proposal, deck_gev, knob)
    for proposal in proposals:
        if proposal.candidate:
            continue
        if _is_bend_corrector(proposal):
            _corrector_rule(proposal, ring)
        else:
            _token_rule(proposal, ring, knob)

    _shared_field_rule(proposals, ring)
    _hook_rule(proposals)
    return {proposal.name: proposal.verdict() for proposal in proposals}


def _energy_rule(proposal: _Proposal, deck_gev: float | None, knob: str | None) -> str | None:
    """Decide a family the export marks an energy candidate.

    Returns:
        The family now holding the energy knob, which is the one held on entry
        unless this family takes it.
    """
    table = proposal.block.get("energy_table")
    if not isinstance(table, dict):
        proposal.latch("the export states no energy table")
        return knob

    values = [value for value in _numbers(table.get("values")) if math.isfinite(value)]
    if not values:
        proposal.latch("the energy table answers no energy")
        return knob
    if max(values) == min(values):
        proposal.latch("bend2gev is constant at this facility")
        return knob

    at_nominal = _number(table.get("energy_at_nominal"))
    if at_nominal is None or not math.isfinite(at_nominal):
        proposal.latch("the energy table answers no energy at the nominal current")
        return knob
    if deck_gev is not None and abs(at_nominal - deck_gev) >= ENERGY_TOLERANCE_GEV:
        proposal.latch(
            f"the ramp answers {_gev(at_nominal)} GeV at the nominal current "
            f"where the deck is solved at {_gev(deck_gev)} GeV"
        )
        return knob

    if knob is not None:
        proposal.open(
            ATTYPE_KIND,
            f"{knob} already answers the energy table; what does this family drive?",
            f"{knob} already answers the energy table",
        )
        return knob

    proposal.kind = "energy"
    return proposal.name


def _corrector_rule(proposal: _Proposal, ring: Sequence[Any]) -> None:
    """Decide a dipole-typed family whose membership names the correctors."""
    proposal.kind = "kick"
    if not _index_rule(proposal):
        return
    if not _attribute_rule(proposal, ring, KICK_ATTRIBUTE, None):
        return
    proposal.element_field = None
    proposal.open(
        ATTYPE_KIND,
        "this family bends and corrects; which plane of KickAngle does it drive?",
        "a bend that corrects states no plane",
    )


def _token_rule(proposal: _Proposal, ring: Sequence[Any], knob: str | None) -> None:
    """Decide a family by its lattice type token, then by its elements."""
    token = proposal.token
    resolved = resolve_attype(token)
    if resolved is None:
        if not index_rows(proposal.nominal.get("at_index")):
            proposal.latch(_NO_ELEMENT)
            return
        proposal.open(
            ATTYPE_KIND,
            f"ATType {token} is not one the table knows; what does this family drive?",
            f"ATType {token} is not one the table knows",
        )
        return

    proposal.kind, proposal.element_field = resolved
    if proposal.kind == "rf":
        if not any(is_cavity(element) for element in ring):
            proposal.latch("the deck holds no cavity")
        return
    if proposal.kind == "energy":
        if knob is None:
            proposal.latch("the export states no energy table")
        else:
            proposal.open(
                ATTYPE_KIND,
                f"{knob} already answers the energy table; what does this family drive?",
                f"{knob} already answers the energy table",
            )
        return

    if not _index_rule(proposal):
        return
    if not _units_rule(proposal):
        return
    attribute, index = _attribute_of(proposal.element_field)
    if attribute is not None:
        _attribute_rule(proposal, ring, attribute, index)


def _index_rule(proposal: _Proposal) -> bool:
    """Read the element indices a family binds. Rule (d)."""
    proposal.rows = index_rows(proposal.nominal.get("at_index"))
    if not proposal.rows:
        proposal.latch(_NO_ELEMENT)
        return False
    return True


def _units_rule(proposal: _Proposal) -> bool:
    """Check the physics units a family states against the units its kind implies."""
    stated = {field: _physics_units(proposal.view, field) for field in FIELDS}
    agreeing = [field for field, word in stated.items() if _is_unit_class(proposal.kind, word)]
    if agreeing:
        disagreeing = [field for field, word in stated.items() if word and field not in agreeing]
        if disagreeing:
            field = disagreeing[0]
            proposal.note = f"{field} states {stated[field]} where {agreeing[0]} states its units"
        return True

    spoken = ", ".join(f"{field} states {stated[field] or 'nothing'}" for field in FIELDS)
    proposal.open(
        ATTYPE_KIND,
        f"{spoken}, and neither of them {proposal.kind} units; what does this family drive?",
        f"neither field states {proposal.kind} units",
    )
    return False


def _attribute_rule(
    proposal: _Proposal,
    ring: Sequence[Any],
    attribute: str,
    index: int | None,
) -> bool:
    """Check that every bound element already carries the attribute a write needs."""
    for position in (position for row in proposal.rows for position in row):
        if not 1 <= position <= len(ring):
            proposal.latch(f"ATIndex {position} is past the end of a ring of {len(ring)}")
            return False
        element = ring[position - 1]
        if not carries(element, attribute, index):
            name = getattr(element, "FamName", "")
            pass_method = getattr(element, "PassMethod", "")
            proposal.latch(f"element {name} ({pass_method}) takes no {attribute}")
            return False
    return True


def _shared_field_rule(proposals: list[_Proposal], ring: Sequence[Any]) -> None:
    """Ask which family binds a field two of them both drive."""
    drivers: dict[tuple[int, str | None], list[_Proposal]] = {}
    for proposal in proposals:
        if not proposal.couples or proposal.kind not in _ELEMENT_KINDS:
            continue
        for position in (position for row in proposal.rows for position in row):
            drivers.setdefault((position, proposal.element_field), []).append(proposal)

    for (position, element_field), sharing in drivers.items():
        if len(sharing) < 2:
            continue
        element = getattr(ring[position - 1], "FamName", "")
        for proposal in sharing:
            others = ", ".join(other.name for other in sharing if other is not proposal)
            proposal.open(
                SHARED_FIELD_KIND,
                f"this family and {others} drive {element_field} of {element}; which one binds it?",
                f"{element_field} of {element} is driven by {others} as well",
            )


def _hook_rule(proposals: list[_Proposal]) -> None:
    """Ask about a family the Middle Layer reaches through code of its own."""
    for proposal in proposals:
        if proposal.kind in ("energy", "rf"):
            continue
        if proposal.slot is not None:
            continue
        hook = _hook(proposal)
        if hook is None:
            continue
        proposal.slot = VASlot(
            kind=ESCAPE_HATCH_KIND,
            question=f"the AT block reaches this family through {hook}; does the model drive it?",
            answer=None,
        )
        proposal.reason = f"the AT block reaches this family through {hook}"


def _hook(proposal: _Proposal) -> str | None:
    """The first hook named in the family's AT block or its coupled field's."""
    view = proposal.view
    if view is None:
        return None
    blocks = [view.body.get("AT")]
    if proposal.field is not None and proposal.field in view.fields:
        blocks.append(view.fields[proposal.field].body.get("AT"))
    for block in blocks:
        if not isinstance(block, dict):
            continue
        for key in AT_HOOK_KEYS:
            if block.get(key) not in (None, "", [], {}):
                return key
    return None


def _is_energy_candidate(block: dict) -> bool:
    value = _number(block.get("energy_candidate"))
    return value is not None and value != 0


def _is_bend_corrector(proposal: _Proposal) -> bool:
    """Whether a dipole-typed family's membership also names the correctors."""
    if resolve_attype(proposal.token) != ATTYPE_TABLE["bend"]:
        return False
    view = proposal.view
    membership = view.arrays.get("MemberOf") if view is not None else None
    if not isinstance(membership, (list, tuple)):
        return False
    return any(
        isinstance(word, str) and word.strip().upper() == CORRECTOR_MEMBERSHIP
        for word in membership
    )


def is_cavity(element: Any) -> bool:
    """Whether a deck element is a cavity, under every spelling of its class."""
    return _CAVITY in (
        type(element).__name__,
        getattr(element, "Class", None),
        getattr(element, "tag", None),
    )


def carries(element: Any, attribute: str | None, index: int | None) -> bool:
    """Whether an element already holds the attribute a write would reach for.

    An answer naming no attribute reaches for nothing, so every element holds
    what it asks for.
    """
    if attribute is None:
        return True
    value = getattr(element, attribute, None)
    if value is None:
        return False
    if index is None:
        return True
    return len(value) >= index + 1


def _attribute_of(element_field: str | None) -> tuple[str | None, int | None]:
    """The element attribute and index an element field names, if it names one."""
    if not element_field or not element_field.endswith("]"):
        return None, None
    attribute, _, tail = element_field.partition("[")
    return attribute, int(tail[:-1])


def _physics_units(view: FamilyView | None, field: str) -> str:
    """The physics units a field states, folded to lower case."""
    if view is None or field not in view.fields:
        return ""
    value = view.fields[field].body.get("PhysicsUnits")
    return value.strip().lower() if isinstance(value, str) else ""


def _is_unit_class(kind: str | None, word: str) -> bool:
    """Whether a units word is one of the kind's own.

    A kind with a listed vocabulary states one of its words; a strength states
    any other non-empty word, because it is measured in as many units as there
    are multipole orders.
    """
    if not word or kind is None:
        return False
    if kind in UNIT_CLASSES:
        return word in UNIT_CLASSES[kind]
    return not any(word in words for words in UNIT_CLASSES.values())


def _calibration_kind(block: dict, field: str | None) -> str | None:
    """How a family's coupled field converts hardware to physics."""
    if field is None:
        return None
    body = block.get(field)
    calibration = body.get("calibration") if isinstance(body, dict) else None
    kind = calibration.get("kind") if isinstance(calibration, dict) else None
    return kind if isinstance(kind, str) else None


def _deck_energy_gev(ring: Sequence[Any]) -> float | None:
    """The energy the deck is solved at, in GeV."""
    value = _number(getattr(ring, "energy", None))
    return None if value is None else value / _EV_PER_GEV


def index_rows(value: Any) -> list[list[int]]:
    """Read stated element indices as one row of slices per device.

    A bare number is the one row it is, and a slice an export writes as a
    not-a-number is a slice the device does not have.
    """
    rows = value if isinstance(value, (list, tuple)) else [value]
    read: list[list[int]] = []
    for row in rows:
        slices = row if isinstance(row, (list, tuple)) else [row]
        positions = [_index(item) for item in slices]
        kept = [position for position in positions if position is not None]
        if kept:
            read.append(kept)
    return read


def _index(value: Any) -> int | None:
    number = _number(value)
    if number is None or not math.isfinite(number) or number != int(number):
        return None
    return int(number)


def _numbers(value: Any) -> list[float]:
    values = value if isinstance(value, (list, tuple)) else [value]
    read: list[float] = []
    for item in values:
        if isinstance(item, (list, tuple)):
            read.extend(_numbers(item))
            continue
        number = _number(item)
        if number is not None:
            read.append(number)
    return read


def _number(value: Any) -> float | None:
    """One exported number, which an export may spell as a word."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _gev(value: float) -> str:
    """One energy, at the width a reviewer reads it at."""
    return f"{value:.6g}"
