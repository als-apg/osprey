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
from collections.abc import Mapping
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

#: Facility token that appears in every generated IRI and identifier.
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


def device_iri(ring: str, name: str) -> str:
    """Canonical device IRI for *name* in *ring*."""
    return f"{DEVICE_IRI_PREFIX}{FACILITY}_{ring}_{name}"


def device_id(ring: str, name: str) -> str:
    """``narad_p:deviceId`` literal for *name* in *ring*."""
    return f"narad:device:{FACILITY}:{ring}:{name}"


def source_section_id(ring: str) -> str:
    """``narad_p:sourceSectionId`` literal for *ring* (ring token lowercased)."""
    return f"narad:section:{FACILITY}:{ring.lower()}"


def binding_iri(ring: str, name: str, field: str, subfield: str) -> str:
    """Canonical ChannelBinding IRI."""
    return f"{BINDING_IRI_PREFIX}narad_endpoint_{FACILITY}_{ring}_{name}_{field}_{subfield}"


def binding_id(ring: str, name: str, field: str, subfield: str) -> str:
    """``narad_p:bindingId`` literal for a ChannelBinding."""
    return f"narad:binding:narad:endpoint:{FACILITY}:{ring}:{name}:{field}_{subfield}"


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


@dataclass(frozen=True)
class SignalGroup:
    """A SemanticSignal individual and the bindings that point at it.

    One group is minted per ``(FAMILY, FIELD, SUBFIELD)`` combination, so every
    member binding describes the same physical quantity on a different device.
    That is what lets a later pass decide read/write direction once per group
    and assert it holds for all members.

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
        devices: Devices in facility order (ring, then SYSTEM/FAMILY/DEVICE).
        bindings: Bindings in device order, then by FIELD and SUBFIELD.
        signal_groups: Signal groups ordered by ``(FAMILY, FIELD, SUBFIELD)``.
    """

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


def build_model(channel_map: Mapping[str, Mapping]) -> GraphModel:
    """Derive the graph model from an expanded channel map.

    Args:
        channel_map: The expanded map from
            :class:`~osprey.services.channel_finder.databases.hierarchical.HierarchicalChannelDatabase`.
            Only the **keys** are read — they are the colon-grammar addresses —
            so the model does not depend on the value shape.

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
            iri=binding_iri(ring, name, address.field, address.subfield),
            binding_id=binding_id(ring, name, address.field, address.subfield),
            device_iri=device_iri(ring, name),
            device_key=address.device_key,
            signal_key=address.signal_key,
            signal_name=group_name,
            signal_iri=signal_iri(group_name),
        )
        bindings_by_device[address.device_key].append(binding)
        groups.setdefault(address.signal_key, []).append(binding.full_pv)

    devices = tuple(
        _build_device(key, ordinal_in_section[key], index + 1, bindings_by_device[key])
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
    return GraphModel(devices=devices, bindings=bindings, signal_groups=signal_groups)


def _binding_sort_key(address: Address) -> tuple:
    """Order bindings by their device, then by FIELD and SUBFIELD."""
    return (_facility_sort_key(address.device_key), address.field, address.subfield)


def _build_device(
    key: tuple[str, str, str, str],
    section_ordinal: int,
    facility_ordinal: int,
    bindings: list[ChannelBinding],
) -> Device:
    """Assemble one :class:`Device` from its identity tuple and ordinals."""
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
        iri=device_iri(ring, name),
        device_id=device_id(ring, name),
        source_section_id=source_section_id(ring),
        binding_iris=tuple(binding.iri for binding in bindings),
    )
