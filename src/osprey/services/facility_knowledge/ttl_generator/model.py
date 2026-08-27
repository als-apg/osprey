"""Pure data model for the demo-machine knowledge graph.

Turns the expanded channel map of the control_assistant hierarchical channel
database into the three node populations a NARAD-convention Turtle file needs:

* :class:`Device` — one per ``(RING, SYSTEM, FAMILY, DEVICE)`` tuple.
* :class:`ChannelBinding` — one per channel address.
* :class:`SignalGroup` — one per ``(FAMILY, FIELD, SUBFIELD)`` combination; the
  ``narad_sem:SemanticSignal`` individuals the bindings point at.

This module is **pure**: standard library plus the (stdlib-only) seeder package,
for its NARAD prefix table — no ``rdflib``, no filesystem access, no
configuration lookups.  Serialisation and the read/write direction of each
signal group are decided elsewhere; :attr:`SignalGroup.direction` is
the slot a later pass fills in (see :meth:`GraphModel.with_directions`).

Everything is deterministic.  The same channel map always produces the same
model regardless of dict insertion order, because every collection is sorted by
an explicit key.  That matters downstream: the seed marker in the graph store
is a checksum over the generated Turtle text, so unstable ordering would make
every deploy look like a corpus change.

The address grammar is the six-token colon form used by the demo machine::

    RING:SYSTEM:FAMILY:DEVICE:FIELD:SUBFIELD
    SR:MAG:DIPOLE:01:CURRENT:SP
    SR:VAC:GAUGE:SR01:PRESSURE:RB
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from osprey.services.facility_knowledge.seeder import NARAD_PREFIXES

# ---------------------------------------------------------------------------
# Namespaces and fixed vocabulary
#
# The two NARAD namespaces are read from the seeder's prefix table — the one
# spelling the graph store registers with neosemantics and ``get_schema``
# reports to the agent — so the generated Turtle cannot drift from the IRIs a
# query is written against.  The table is plain strings, which is what keeps
# rdflib off this path.
# ---------------------------------------------------------------------------

#: Default facility token, used when a caller names none.  Every generated IRI
#: and identifier embeds the token, and :class:`GraphModel` carries the one it
#: was built with — the value is an argument, never a module global to patch.
FACILITY = "demo"

NARAD_PROPERTY_NS = NARAD_PREFIXES["narad_p"]
NARAD_SEM_NS = NARAD_PREFIXES["narad_sem"]
DEVICE_IRI_PREFIX = "https://narad.example.org/device/"
BINDING_IRI_PREFIX = "https://narad.example.org/binding/"

#: Every generated binding speaks EPICS Channel Access.
PROTOCOL = "ca"

#: Bindings are derived mechanically from the address grammar, not guessed.
CONFIDENCE = "high"

#: Rings in facility order; anything else sorts after these, alphabetically.
RING_ORDER: tuple[str, ...] = ("SR", "BR", "BTS")

_ADDRESS_FIELDS = ("ring", "system", "family", "device", "field", "subfield")
_ADDRESS_TOKEN_COUNT = len(_ADDRESS_FIELDS)
_DIGIT_RUN = re.compile(r"(\d+)")


# ---------------------------------------------------------------------------
# Address grammar
# ---------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class Address:
    """One parsed channel address.

    Args:
        ring: Ring token, e.g. ``SR``.
        system: System token, e.g. ``MAG``.
        family: Family token, e.g. ``DIPOLE``.
        device: Device instance token, e.g. ``01`` (or ``SR01`` for gauges).
        field: Field token, e.g. ``CURRENT``.
        subfield: Subfield token, e.g. ``SP``.
    """

    ring: str
    system: str
    family: str
    device: str
    field: str
    subfield: str

    @property
    def text(self) -> str:
        """The address rejoined into its canonical colon form."""
        return ":".join(
            (self.ring, self.system, self.family, self.device, self.field, self.subfield)
        )

    @property
    def device_key(self) -> tuple[str, str, str, str]:
        """Identity of the device this address belongs to."""
        return (self.ring, self.system, self.family, self.device)

    @property
    def signal_key(self) -> tuple[str, str, str]:
        """Identity of the semantic signal this address realises."""
        return (self.family, self.field, self.subfield)


def parse_address(addr: str) -> Address:
    """Parse a six-token colon address.

    Args:
        addr: Channel address, e.g. ``SR:MAG:DIPOLE:01:CURRENT:SP``.

    Returns:
        The parsed :class:`Address`.

    Raises:
        ValueError: If *addr* does not have exactly six colon-separated tokens,
            or if any token is empty.
    """
    tokens = addr.split(":")
    if len(tokens) != _ADDRESS_TOKEN_COUNT:
        raise ValueError(
            f"Channel address {addr!r} has {len(tokens)} colon-separated token(s), "
            f"expected {_ADDRESS_TOKEN_COUNT} "
            "(RING:SYSTEM:FAMILY:DEVICE:FIELD:SUBFIELD)"
        )
    empty = [name for name, tok in zip(_ADDRESS_FIELDS, tokens, strict=True) if not tok]
    if empty:
        raise ValueError(
            f"Channel address {addr!r} has an empty {', '.join(empty)} token; "
            "every token of RING:SYSTEM:FAMILY:DEVICE:FIELD:SUBFIELD must be non-empty"
        )
    return Address(*tokens)


# ---------------------------------------------------------------------------
# Hierarchy descriptions
#
# The hierarchical channel database is a tree of tokens, and most of its nodes
# carry a ``_description`` string.  The expanded channel map throws that prose
# away — it keeps only the addresses — so the generator reads it back off the
# raw tree.  Prose is keyed by the full path down to the node that carries it,
# never by the token alone: ``SR:MAG:QF:*:CURRENT:SP`` and
# ``BR:MAG:QF:*:CURRENT:SP`` share their last three tokens but describe two
# different machines.
# ---------------------------------------------------------------------------

#: Key that carries a node's prose in the hierarchical channel database.
DESCRIPTION_KEY = "_description"

#: Level types the hierarchical database understands.  A ``tree`` level is keyed
#: by its tokens; an ``instances`` level is one placeholder node whose tokens are
#: generated from an ``_expansion`` rule and carry no prose of their own.
TREE_LEVEL = "tree"
INSTANCE_LEVEL = "instances"

#: The level grammar this module reads, in address order.  It is the six-token
#: colon form with the device level generated rather than enumerated.
_EXPECTED_LEVELS: tuple[tuple[str, str], ...] = (
    ("ring", TREE_LEVEL),
    ("system", TREE_LEVEL),
    ("family", TREE_LEVEL),
    ("device", INSTANCE_LEVEL),
    ("field", TREE_LEVEL),
    ("subfield", TREE_LEVEL),
)


@dataclass(frozen=True)
class HierarchyDescriptions:
    """Prose from the hierarchical tree, one map per described level.

    Every map is keyed by the path of tokens down to the node the text sits on,
    with the generated device token left out — it names no node in the tree.
    Levels the tree leaves undescribed are simply absent from their map.

    Args:
        ring: Ring prose, keyed ``(RING,)``.
        system: System prose, keyed ``(RING, SYSTEM)``.
        family: Family prose, keyed ``(RING, SYSTEM, FAMILY)``.
        field: Field prose, keyed ``(RING, SYSTEM, FAMILY, FIELD)``.
        subfield: Subfield prose, keyed ``(RING, SYSTEM, FAMILY, FIELD, SUBFIELD)``.
    """

    ring: Mapping[tuple[str], str]
    system: Mapping[tuple[str, str], str]
    family: Mapping[tuple[str, str, str], str]
    field: Mapping[tuple[str, str, str, str], str]
    subfield: Mapping[tuple[str, str, str, str, str], str]


def resolve_hierarchy_descriptions(
    tree: Mapping[str, object],
    levels: Sequence[Mapping[str, object]],
) -> HierarchyDescriptions:
    """Read every ``_description`` out of a raw hierarchical tree.

    Pure and deterministic: no I/O, no ``rdflib``, the input is left untouched
    and each returned map iterates in sorted key order.

    Args:
        tree: The raw ``tree`` section of the hierarchical channel database —
            the unexpanded one, since the expanded channel map keeps only
            addresses.  Keys starting with ``_`` are metadata, not tokens.
        levels: The database's ``hierarchy.levels`` list, each entry a mapping
            with a ``name`` and a ``type``.

    Returns:
        The :class:`HierarchyDescriptions` for *tree*.  Text is trimmed of
        surrounding whitespace; nodes whose description is missing, blank or
        not a string contribute no key.

    Raises:
        ValueError: If *levels* is not the six-token grammar this module reads
            (RING, SYSTEM, FAMILY, DEVICE, FIELD, SUBFIELD, with DEVICE the one
            generated level).
    """
    _check_levels(levels)
    collected: dict[str, dict[tuple[str, ...], str]] = {
        name: {} for name, kind in _EXPECTED_LEVELS if kind == TREE_LEVEL
    }
    _collect_descriptions(tree, (), 0, collected)
    return HierarchyDescriptions(
        ring=dict(sorted(collected["ring"].items())),
        system=dict(sorted(collected["system"].items())),
        family=dict(sorted(collected["family"].items())),
        field=dict(sorted(collected["field"].items())),
        subfield=dict(sorted(collected["subfield"].items())),
    )


def _check_levels(levels: Sequence[Mapping[str, object]]) -> None:
    """Refuse a level list that is not the grammar the five maps are named for."""
    given = tuple((str(level.get("name", "")), str(level.get("type", ""))) for level in levels)
    if given != _EXPECTED_LEVELS:
        expected = ", ".join(f"{name} ({kind})" for name, kind in _EXPECTED_LEVELS)
        actual = ", ".join(f"{name} ({kind})" for name, kind in given) or "no levels"
        raise ValueError(f"Hierarchy level grammar is {actual}; this generator reads {expected}")


def _collect_descriptions(
    node: Mapping[str, object],
    path: tuple[str, ...],
    level_index: int,
    collected: dict[str, dict[tuple[str, ...], str]],
) -> None:
    """Walk one level of *node*, recording prose and descending into its children."""
    if level_index >= len(_EXPECTED_LEVELS):
        return
    name, kind = _EXPECTED_LEVELS[level_index]

    if kind == INSTANCE_LEVEL:
        # One placeholder node named for the level (``DEVICE``); its generated
        # tokens are not part of any description path, so the path is unchanged.
        for token, child in node.items():
            if token.upper() == name.upper() and isinstance(child, Mapping):
                _collect_descriptions(child, path, level_index + 1, collected)
        return

    for token, child in node.items():
        if token.startswith("_") or not isinstance(child, Mapping):
            continue
        key = (*path, token)
        text = child.get(DESCRIPTION_KEY)
        if isinstance(text, str) and text.strip():
            collected[name][key] = text.strip()
        _collect_descriptions(child, key, level_index + 1, collected)


def _lookup_description(source: Mapping[object, object] | None, key: object) -> str | None:
    """Return the prose *source* holds for *key*, or ``None``.

    The join is exact: a key the mapping does not hold contributes no text.
    Nothing is borrowed from a shorter path or a sibling token, because the
    prose is ring-qualified and a near-miss would describe a different machine.
    """
    if source is None:
        return None
    text = source.get(key)
    if isinstance(text, str) and text.strip():
        return text.strip()
    return None


# ---------------------------------------------------------------------------
# Deterministic ordering helpers
# ---------------------------------------------------------------------------


def _natural_key(token: str) -> tuple[tuple[int, int | str], ...]:
    """Sort key that orders digit runs numerically (``SR2`` before ``SR10``)."""
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part) for part in _DIGIT_RUN.split(token) if part
    )


def _ring_rank(ring: str) -> tuple[int, str]:
    """Facility order for a ring token: SR, BR, BTS, then the rest by name."""
    if ring in RING_ORDER:
        return (RING_ORDER.index(ring), "")
    return (len(RING_ORDER), ring)


def _device_sort_key(key: tuple[str, str, str, str]) -> tuple:
    """Order devices within a ring by (SYSTEM, FAMILY, natural DEVICE)."""
    _ring, system, family, device = key
    return (system, family, _natural_key(device))


def _facility_sort_key(key: tuple[str, str, str, str]) -> tuple:
    """Order devices across the whole facility (ring first, then in-ring order)."""
    return (_ring_rank(key[0]), _device_sort_key(key))


# ---------------------------------------------------------------------------
# Identifier derivations
# ---------------------------------------------------------------------------


def source_name(family: str, device: str) -> str:
    """Device source name: FAMILY concatenated with its DEVICE token.

    ``DIPOLE`` + ``01`` -> ``DIPOLE01``; ``GAUGE`` + ``SR01`` -> ``GAUGESR01``.
    """
    return f"{family}{device}"


def device_iri(ring: str, name: str, *, facility: str = FACILITY) -> str:
    """Canonical device IRI for *name* in *ring*, minted for *facility*."""
    return f"{DEVICE_IRI_PREFIX}{facility}_{ring}_{name}"


def device_id(ring: str, name: str, *, facility: str = FACILITY) -> str:
    """``narad_p:deviceId`` literal for *name* in *ring*, minted for *facility*."""
    return f"narad:device:{facility}:{ring}:{name}"


def source_section_id(ring: str, *, facility: str = FACILITY) -> str:
    """``narad_p:sourceSectionId`` literal for *ring* (ring token lowercased)."""
    return f"narad:section:{facility}:{ring.lower()}"


def binding_iri(
    ring: str, name: str, field: str, subfield: str, *, facility: str = FACILITY
) -> str:
    """Canonical ChannelBinding IRI, minted for *facility*."""
    return f"{BINDING_IRI_PREFIX}narad_endpoint_{facility}_{ring}_{name}_{field}_{subfield}"


def binding_id(ring: str, name: str, field: str, subfield: str, *, facility: str = FACILITY) -> str:
    """``narad_p:bindingId`` literal for a ChannelBinding, minted for *facility*."""
    return f"narad:binding:narad:endpoint:{facility}:{ring}:{name}:{field}_{subfield}"


def signal_name(family: str, field: str, subfield: str) -> str:
    """Local name of the SemanticSignal individual for a (FAMILY, FIELD, SUBFIELD) group.

    Lowercased, with ``-`` folded to ``_`` so the result is a legal Turtle local
    name: ``DIPOLE``/``CURRENT``/``SP`` -> ``dipole_current_sp`` and
    ``ION-PUMP``/``PRESSURE``/``RB`` -> ``ion_pump_pressure_rb``.
    """
    return f"{family}_{field}_{subfield}".lower().replace("-", "_")


def signal_iri(name: str) -> str:
    """IRI of a SemanticSignal individual, in the ``narad_sem`` namespace."""
    return f"{NARAD_SEM_NS}{name}"


# ---------------------------------------------------------------------------
# Node types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Device:
    """One accelerator device — a ``(RING, SYSTEM, FAMILY, DEVICE)`` tuple.

    Args:
        ring: Ring token; also the ``narad_p:sectionCode`` value.
        system: System token.
        family: Family token; also the ``narad_p:rawType`` value.
        device: Device instance token.
        source_name: ``narad_p:sourceName`` — FAMILY + DEVICE.
        section_code: ``narad_p:sectionCode`` — the ring token.
        raw_type: ``narad_p:rawType`` — the family token.
        ordinal_in_section: 1-based position within its ring.
        ordinal_in_facility: 1-based position across the whole facility.
        s_position_m: ``narad_p:sPositionM`` — the in-ring ordinal as a float.
            The demo machine has no lattice geometry, so this is a stand-in
            that at least preserves ordering along the ring.
        iri: Canonical device IRI.
        device_id: ``narad_p:deviceId`` literal.
        source_section_id: ``narad_p:sourceSectionId`` literal.
        binding_iris: IRIs of this device's bindings, in binding order.
        family_description: Prose for this device's ``(RING, SYSTEM, FAMILY)``
            path, or ``None`` when the caller supplied no hierarchy prose or
            the tree describes no such node.
        system_description: Prose for this device's ``(RING, SYSTEM)`` path.
        ring_description: Prose for this device's ring.
    """

    ring: str
    system: str
    family: str
    device: str
    source_name: str
    section_code: str
    raw_type: str
    ordinal_in_section: int
    ordinal_in_facility: int
    s_position_m: float
    iri: str
    device_id: str
    source_section_id: str
    binding_iris: tuple[str, ...]
    family_description: str | None = None
    system_description: str | None = None
    ring_description: str | None = None

    @property
    def key(self) -> tuple[str, str, str, str]:
        """Device identity tuple ``(RING, SYSTEM, FAMILY, DEVICE)``."""
        return (self.ring, self.system, self.family, self.device)

    @property
    def address(self) -> str:
        """Colon-joined device prefix, e.g. ``SR:MAG:DIPOLE:01``."""
        return ":".join(self.key)


@dataclass(frozen=True)
class ChannelBinding:
    """One channel address, bound to the device that owns it and the signal it realises.

    Args:
        address: The parsed address.
        full_pv: ``narad_p:fullPv`` — the address in colon form.
        protocol: ``narad_p:protocol`` — always :data:`PROTOCOL`.
        confidence: ``narad_p:confidence`` — always :data:`CONFIDENCE`.
        iri: Canonical binding IRI.
        binding_id: ``narad_p:bindingId`` literal.
        device_iri: IRI of the owning device.
        device_key: Identity tuple of the owning device.
        signal_key: ``(FAMILY, FIELD, SUBFIELD)`` of the signal group.
        signal_name: Local name of the SemanticSignal individual.
        signal_iri: IRI of the SemanticSignal individual.
        description: Prose for this channel, or ``None`` when the caller
            supplied none for this address.
        field_description: Prose for the address's ``(RING, SYSTEM, FAMILY,
            FIELD)`` path, or ``None``.
        subfield_description: Prose for the address's full ``(RING, SYSTEM,
            FAMILY, FIELD, SUBFIELD)`` path, or ``None``.
    """

    address: Address
    full_pv: str
    protocol: str
    confidence: str
    iri: str
    binding_id: str
    device_iri: str
    device_key: tuple[str, str, str, str]
    signal_key: tuple[str, str, str]
    signal_name: str
    signal_iri: str
    description: str | None = None
    field_description: str | None = None
    subfield_description: str | None = None


@dataclass(frozen=True)
class SignalGroup:
    """A SemanticSignal individual and the bindings that point at it.

    One group is minted per ``(FAMILY, FIELD, SUBFIELD)`` combination, so every
    member binding describes the same physical quantity on a different device.
    That is what lets a later pass decide read/write direction once per group
    and assert it holds for all members.

    A group carries no prose by design.  The hierarchical tree's field and
    subfield text is ring-qualified, and a group's key has no ring in it — the
    same ``(FAMILY, FIELD, SUBFIELD)`` in two rings would offer two texts.  The
    prose therefore hangs off :class:`ChannelBinding`, whose address resolves
    every text exactly.

    Args:
        family: Family token.
        field: Field token.
        subfield: Subfield token.
        name: Local name of the SemanticSignal individual.
        iri: IRI of the SemanticSignal individual.
        members: Addresses of the member bindings, in binding order.
        direction: ``"read"``, ``"write"`` (``direction.DIRECTION_READ`` /
            ``direction.DIRECTION_WRITE``), or ``None`` while undecided.
            This module never sets it.
    """

    family: str
    field: str
    subfield: str
    name: str
    iri: str
    members: tuple[str, ...]
    direction: str | None = None

    @property
    def key(self) -> tuple[str, str, str]:
        """Group identity tuple ``(FAMILY, FIELD, SUBFIELD)``."""
        return (self.family, self.field, self.subfield)


@dataclass(frozen=True)
class GraphModel:
    """The complete derived graph, in deterministic order.

    Args:
        facility: The facility token every identifier in this model was minted
            with.  Carrying it here is what stops a second reader — the emitter
            writing ``narad_p:facility`` — from having to look the value up
            somewhere else and disagreeing with the IRIs beside it.
        devices: Devices in facility order (ring, then SYSTEM/FAMILY/DEVICE).
        bindings: Bindings in device order, then by FIELD and SUBFIELD.
        signal_groups: Signal groups ordered by ``(FAMILY, FIELD, SUBFIELD)``.
    """

    facility: str
    devices: tuple[Device, ...]
    bindings: tuple[ChannelBinding, ...]
    signal_groups: tuple[SignalGroup, ...]

    def device_addresses(self) -> tuple[str, ...]:
        """Colon-joined device prefixes, in device order."""
        return tuple(device.address for device in self.devices)

    def binding_addresses(self) -> tuple[str, ...]:
        """Full channel addresses, in binding order."""
        return tuple(binding.full_pv for binding in self.bindings)

    def signal_groups_by_key(self) -> dict[tuple[str, str, str], SignalGroup]:
        """Signal groups keyed by ``(FAMILY, FIELD, SUBFIELD)``."""
        return {group.key: group for group in self.signal_groups}

    def with_directions(self, directions: Mapping[tuple[str, str, str], str]) -> GraphModel:
        """Return a copy whose signal groups carry the given directions.

        Args:
            directions: Direction per ``(FAMILY, FIELD, SUBFIELD)`` key.  Keys
                that name no group are ignored; groups the mapping omits keep
                the direction they already have.

        Returns:
            A new :class:`GraphModel`; the receiver is unchanged.
        """
        groups = tuple(
            replace(group, direction=directions.get(group.key, group.direction))
            for group in self.signal_groups
        )
        return replace(self, signal_groups=groups)


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def build_model(
    channel_map: Mapping[str, Mapping],
    *,
    facility: str = FACILITY,
    hierarchy_descriptions: HierarchyDescriptions | None = None,
    binding_descriptions: Mapping[str, str] | None = None,
) -> GraphModel:
    """Derive the graph model from an expanded channel map.

    Prose is optional and purely additive: with both description arguments left
    out, every node's text fields are ``None`` and the model is exactly the one
    the keys alone describe.  When prose is supplied it is attached by **exact
    join** — an address or path the caller has no text for keeps ``None``
    rather than borrowing a neighbour's.

    Args:
        channel_map: The expanded map from
            :class:`~osprey.services.channel_finder.databases.hierarchical.HierarchicalChannelDatabase`.
            Only the **keys** are read — they are the colon-grammar addresses —
            so the model does not depend on the value shape.
        facility: Facility token to mint every IRI and identifier with, and the
            token the returned model carries.  Defaults to :data:`FACILITY`, so
            the shipped demo corpus is unchanged.  Two models with different
            tokens can be built in one process: nothing is stored globally.
        hierarchy_descriptions: Prose from the hierarchical tree, as returned by
            :func:`resolve_hierarchy_descriptions`.  Its ring, system and family
            maps land on :class:`Device`; its field and subfield maps land on
            :class:`ChannelBinding`, which is the only node whose key carries
            the ring the text is qualified by.
        binding_descriptions: Per-channel prose keyed by full six-token address,
            for :attr:`ChannelBinding.description`.

    Returns:
        The derived :class:`GraphModel`.

    Raises:
        ValueError: If any key is not a valid six-token address.
    """
    addresses = [parse_address(addr) for addr in channel_map]

    # Devices first: ordinals depend on the sorted device population, and every
    # binding needs its device's source name to build its own IRI.
    device_keys = sorted({address.device_key for address in addresses}, key=_facility_sort_key)
    ordinal_in_section: dict[tuple[str, str, str, str], int] = {}
    seen_per_ring: dict[str, int] = {}
    for key in device_keys:
        ring = key[0]
        seen_per_ring[ring] = seen_per_ring.get(ring, 0) + 1
        ordinal_in_section[key] = seen_per_ring[ring]

    bindings_by_device: dict[tuple[str, str, str, str], list[ChannelBinding]] = {
        key: [] for key in device_keys
    }
    groups: dict[tuple[str, str, str], list[str]] = {}

    for address in sorted(addresses, key=_binding_sort_key):
        ring = address.ring
        name = source_name(address.family, address.device)
        group_name = signal_name(address.family, address.field, address.subfield)
        binding = ChannelBinding(
            address=address,
            full_pv=address.text,
            protocol=PROTOCOL,
            confidence=CONFIDENCE,
            iri=binding_iri(ring, name, address.field, address.subfield, facility=facility),
            binding_id=binding_id(ring, name, address.field, address.subfield, facility=facility),
            device_iri=device_iri(ring, name, facility=facility),
            device_key=address.device_key,
            signal_key=address.signal_key,
            signal_name=group_name,
            signal_iri=signal_iri(group_name),
            description=_lookup_description(binding_descriptions, address.text),
            field_description=_lookup_description(
                None if hierarchy_descriptions is None else hierarchy_descriptions.field,
                (ring, address.system, address.family, address.field),
            ),
            subfield_description=_lookup_description(
                None if hierarchy_descriptions is None else hierarchy_descriptions.subfield,
                (ring, address.system, address.family, address.field, address.subfield),
            ),
        )
        bindings_by_device[address.device_key].append(binding)
        groups.setdefault(address.signal_key, []).append(binding.full_pv)

    devices = tuple(
        _build_device(
            key,
            ordinal_in_section[key],
            index + 1,
            bindings_by_device[key],
            hierarchy_descriptions,
            facility=facility,
        )
        for index, key in enumerate(device_keys)
    )
    bindings = tuple(binding for key in device_keys for binding in bindings_by_device[key])
    signal_groups = tuple(
        SignalGroup(
            family=family,
            field=field,
            subfield=subfield,
            name=signal_name(family, field, subfield),
            iri=signal_iri(signal_name(family, field, subfield)),
            members=tuple(groups[(family, field, subfield)]),
        )
        for family, field, subfield in sorted(groups)
    )
    return GraphModel(
        facility=facility,
        devices=devices,
        bindings=bindings,
        signal_groups=signal_groups,
    )


def _binding_sort_key(address: Address) -> tuple:
    """Order bindings by their device, then by FIELD and SUBFIELD."""
    return (_facility_sort_key(address.device_key), address.field, address.subfield)


def _build_device(
    key: tuple[str, str, str, str],
    section_ordinal: int,
    facility_ordinal: int,
    bindings: list[ChannelBinding],
    descriptions: HierarchyDescriptions | None = None,
    *,
    facility: str = FACILITY,
) -> Device:
    """Assemble one :class:`Device` from its identity tuple and ordinals.

    The device's three texts resolve off its own ``(RING, SYSTEM, FAMILY)``
    path, which is unique per device population, so no tie-break is needed.
    """
    ring, system, family, device = key
    name = source_name(family, device)
    return Device(
        ring=ring,
        system=system,
        family=family,
        device=device,
        source_name=name,
        section_code=ring,
        raw_type=family,
        ordinal_in_section=section_ordinal,
        ordinal_in_facility=facility_ordinal,
        s_position_m=float(section_ordinal),
        iri=device_iri(ring, name, facility=facility),
        device_id=device_id(ring, name, facility=facility),
        source_section_id=source_section_id(ring, facility=facility),
        binding_iris=tuple(binding.iri for binding in bindings),
        family_description=_lookup_description(
            None if descriptions is None else descriptions.family, (ring, system, family)
        ),
        system_description=_lookup_description(
            None if descriptions is None else descriptions.system, (ring, system)
        ),
        ring_description=_lookup_description(
            None if descriptions is None else descriptions.ring, (ring,)
        ),
    )
