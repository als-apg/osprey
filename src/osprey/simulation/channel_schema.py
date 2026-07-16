"""Declarative per-family channel schema for the ``one-facility`` epic.

Declares, for each device family in :data:`osprey.simulation.facility_spec.
ALS_U_AR`, the tier-3 channel structure (FIELD/SUBFIELD levels of the
``RING:SYSTEM:FAMILY:DEVICE:FIELD:SUBFIELD`` address) and the human-facing
display metadata a downstream tier-DB generator (a later task) needs to emit
the three shipped tier-3 channel database formats --
``hierarchical.json``, ``in_context.json``, ``middle_layer.json`` (under
``src/osprey/templates/apps/control_assistant/data/channel_databases/tiers/
tier3/``) -- address-identically for every family.

Stdlib-only (``dataclasses``), zero third-party dependencies, matching the
style of :mod:`osprey.simulation.facility_spec`. This module is purely
declarative data: it performs no DB I/O and contains no generation logic.

Field/subfield conventions and per-subfield metadata (``DataType``,
``HWUnits``, description phrase) are sourced verbatim from the shipped
SR-facility ``middle_layer.json`` / ``in_context.json`` and from the
address-partitioning conventions in
:mod:`osprey.services.virtual_accelerator.manifest.classify`. Metadata is
NOT uniform across a family -- it is declared per FIELD:SUBFIELD:

* Analog channels (``CURRENT:{SP,RB,GOLDEN}`` for magnets/correctors;
  ``POSITION``/``GOLDEN``/``OFFSET`` ``{X,Y}`` for BPM): ``DataType =
  "double"``, ``HWUnits = "A"`` (magnet current) or ``"mm"`` (BPM position
  family), description = the SR DB's subfield prose (e.g. ``CURRENT:SP`` =
  "setpoint", ``POSITION:X`` = "horizontal").
* ``STATUS`` channels: ``DataType = "enum"``, ``HWUnits = ""``, description
  = the lowercased subfield word (``READY`` = "ready", etc).

SR magnets carry ``STATUS:{READY, ON, FAULT}``; SR correctors (HCM/VCM)
carry ``STATUS:{READY, ON}`` with **no** ``FAULT`` -- confirmed against
``in_context.json``, which has no ``SR:MAG:HCM:*:STATUS:FAULT`` /
``SR:MAG:VCM:*:STATUS:FAULT`` channels. ``SP``/``RB`` (magnets) and
``POSITION`` (BPM) feed the manifest's pyat-coupled partition; every other
field here is static-noisy (see ``classify.MAG_FAMILIES``).

The three genuinely-new ALS-U families the shipped databases have no
precedent for -- ``QFA``, ``SHF``, ``SHD`` -- are structural clones of their
closest existing analog (``QFA`` == ``QF``; ``SHF`` == ``SF``; ``SHD`` ==
``SD``: identical fields/subfields/metadata shape) with deterministic,
templated description/common-name prose -- no per-run LLM/network
dependency.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SubfieldSchema:
    """Declared tier-3 metadata for one FIELD:SUBFIELD leaf.

    Attributes:
        data_type: ``middle_layer.json`` ``DataType`` (``"double"`` for
            analog measurements/setpoints, ``"enum"`` for status booleans).
        hw_units: ``middle_layer.json`` ``HWUnits`` (e.g. ``"A"``, ``"mm"``,
            or ``""`` for unitless status booleans).
        description: Subfield-level ``_description`` phrase as it appears in
            the shipped SR DBs (e.g. ``"setpoint"``, ``"horizontal"``,
            ``"ready"``). Combined by the downstream generator with the
            family's :attr:`FamilyChannelSchema.description_template` to
            build the full per-channel description (as ``in_context.json``
            does, e.g. "... focusing quadrupole 01 current setpoint").
    """

    data_type: str
    hw_units: str
    description: str


@dataclass(frozen=True)
class FamilyChannelSchema:
    """Declared channel structure and display metadata for one device family.

    Attributes:
        system: Tier-3 ``SYSTEM`` tree token this family's channels live
            under -- ``"MAG"`` for magnets/correctors, ``"DIAG"`` for BPMs.
        fields: ``{field: {subfield: SubfieldSchema}}`` -- the complete
            FIELD/SUBFIELD levels of the
            ``RING:SYSTEM:FAMILY:DEVICE:FIELD:SUBFIELD`` address, each leaf
            carrying its own ``DataType``/``HWUnits``/description.
        display_name: Human-facing family name (e.g. ``"Focusing
            Quadrupole"``), used as the ``middle_layer.json`` family
            ``_description`` and as the noun phrase in generated channel
            descriptions.
        description_template: English description of one device,
            parametrized by ``{id}`` (zero-padded to two digits, matching
            ``in_context.json`` prose, e.g. "... focusing quadrupole 01").
        common_name_template: ``middle_layer.json`` ``CommonNames`` entry
            template, parametrized by ``{id}`` (e.g. ``"QF {id:02d}"``).
    """

    system: str
    fields: dict[str, dict[str, SubfieldSchema]]
    display_name: str
    description_template: str
    common_name_template: str

    def describe(self, device_id: int | str) -> str:
        """Render this family's device description for ``device_id``."""
        return self.description_template.format(id=int(device_id))

    def common_name(self, device_id: int | str) -> str:
        """Render this family's ``CommonNames`` entry for ``device_id``."""
        return self.common_name_template.format(id=int(device_id))


# ── System tokens ─────────────────────────────────────────────────────────
_MAG = "MAG"
_DIAG = "DIAG"

# ── Shared subfield metadata, sourced verbatim from middle_layer.json ──────
# Magnet/corrector current: a bipolar source with a commanded setpoint, a
# measured readback, and a stored golden reference value. SP/RB feed the
# manifest's pyat-coupled partition for SR magnets; GOLDEN is static-noisy.
_MAGNET_CURRENT: dict[str, SubfieldSchema] = {
    "SP": SubfieldSchema("double", "A", "setpoint"),
    "RB": SubfieldSchema("double", "A", "readback"),
    "GOLDEN": SubfieldSchema("double", "A", "golden setpoint"),
}

# STATUS subfields: enum/unitless, description = the lowercased subfield
# word. Magnets carry FAULT; correctors (HCM/VCM) do not (confirmed absent
# from in_context.json -- no SR:MAG:HCM|VCM:*:STATUS:FAULT channels).
_MAGNET_STATUS: dict[str, SubfieldSchema] = {
    "READY": SubfieldSchema("enum", "", "ready"),
    "ON": SubfieldSchema("enum", "", "on"),
    "FAULT": SubfieldSchema("enum", "", "fault"),
}
_CORRECTOR_STATUS: dict[str, SubfieldSchema] = {
    "READY": SubfieldSchema("enum", "", "ready"),
    "ON": SubfieldSchema("enum", "", "on"),
}

# BPM transverse-plane pair: POSITION (measured), GOLDEN (stored reference
# orbit), and OFFSET (measured minus golden) all share the same X/Y shape.
# POSITION feeds the manifest's pyat-coupled partition; GOLDEN/OFFSET are
# static-noisy.
_BPM_XY: dict[str, SubfieldSchema] = {
    "X": SubfieldSchema("double", "mm", "horizontal"),
    "Y": SubfieldSchema("double", "mm", "vertical"),
}
_BPM_STATUS: dict[str, SubfieldSchema] = {
    "VALID": SubfieldSchema("enum", "", "valid"),
    "CONNECTED": SubfieldSchema("enum", "", "connected"),
}


def _magnet_fields(*, has_fault: bool) -> dict[str, dict[str, SubfieldSchema]]:
    """Build the complete field/subfield structure for a magnet or corrector.

    Args:
        has_fault: Whether this family's ``STATUS`` carries ``FAULT`` (true
            for magnets, false for HCM/VCM correctors).
    """
    return {
        "CURRENT": dict(_MAGNET_CURRENT),
        "STATUS": dict(_MAGNET_STATUS if has_fault else _CORRECTOR_STATUS),
    }


_BPM_FIELDS: dict[str, dict[str, SubfieldSchema]] = {
    "POSITION": dict(_BPM_XY),
    "GOLDEN": dict(_BPM_XY),
    "OFFSET": dict(_BPM_XY),
    "STATUS": dict(_BPM_STATUS),
}


def _magnet_schema(
    display_name: str,
    description_template: str,
    common_name_template: str,
    *,
    has_fault: bool,
) -> FamilyChannelSchema:
    """Build a magnet/corrector schema entry (shared MAG-system boilerplate)."""
    return FamilyChannelSchema(
        system=_MAG,
        fields=_magnet_fields(has_fault=has_fault),
        display_name=display_name,
        description_template=description_template,
        common_name_template=common_name_template,
    )


CHANNEL_SCHEMA: dict[str, FamilyChannelSchema] = {
    "QF": _magnet_schema(
        "Focusing Quadrupole",
        "ALS-U Accumulator Ring focusing quadrupole {id:02d}",
        "QF {id:02d}",
        has_fault=True,
    ),
    "QD": _magnet_schema(
        "Defocusing Quadrupole",
        "ALS-U Accumulator Ring defocusing quadrupole {id:02d}",
        "QD {id:02d}",
        has_fault=True,
    ),
    # New family, no shipped-DB precedent: structural clone of QF (see
    # module docstring) -- same fields/subfields/metadata shape, only the
    # token and display/description prose differ.
    "QFA": _magnet_schema(
        "Achromat Focusing Quadrupole",
        "ALS-U Accumulator Ring achromat focusing quadrupole {id:02d}",
        "QFA {id:02d}",
        has_fault=True,
    ),
    "DIPOLE": _magnet_schema(
        "Dipole Bending Magnet",
        "ALS-U Accumulator Ring dipole bending magnet {id:02d}",
        "DIPOLE {id:02d}",
        has_fault=True,
    ),
    "SF": _magnet_schema(
        "Focusing Sextupole",
        "ALS-U Accumulator Ring focusing sextupole {id:02d}",
        "SF {id:02d}",
        has_fault=True,
    ),
    "SD": _magnet_schema(
        "Defocusing Sextupole",
        "ALS-U Accumulator Ring defocusing sextupole {id:02d}",
        "SD {id:02d}",
        has_fault=True,
    ),
    # New family, no shipped-DB precedent: structural clone of SF.
    "SHF": _magnet_schema(
        "Harmonic Focusing Sextupole",
        "ALS-U Accumulator Ring harmonic focusing sextupole {id:02d}",
        "SHF {id:02d}",
        has_fault=True,
    ),
    # New family, no shipped-DB precedent: structural clone of SD.
    "SHD": _magnet_schema(
        "Harmonic Defocusing Sextupole",
        "ALS-U Accumulator Ring harmonic defocusing sextupole {id:02d}",
        "SHD {id:02d}",
        has_fault=True,
    ),
    "HCM": _magnet_schema(
        "Horizontal Corrector Magnet",
        "ALS-U Accumulator Ring horizontal corrector magnet {id:02d}",
        "HCM {id:02d}",
        has_fault=False,
    ),
    "VCM": _magnet_schema(
        "Vertical Corrector Magnet",
        "ALS-U Accumulator Ring vertical corrector magnet {id:02d}",
        "VCM {id:02d}",
        has_fault=False,
    ),
    "BPM": FamilyChannelSchema(
        system=_DIAG,
        fields=dict(_BPM_FIELDS),
        display_name="Beam Position Monitor",
        description_template="ALS-U Accumulator Ring beam position monitor {id:02d}",
        common_name_template="BPM {id:02d}",
    ),
}


def schema_for(family: str) -> FamilyChannelSchema:
    """Return the :class:`FamilyChannelSchema` declared for ``family``.

    Raises:
        KeyError: If no schema is declared for that family.
    """
    return CHANNEL_SCHEMA[family]
