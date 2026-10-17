"""Facts of the facility build that are defined in exactly one place."""

from __future__ import annotations

import re
from pathlib import Path

from osprey.facility import DEFAULT_ROLE
from osprey.facility.combine import SET_VALUED
from osprey.facility.schema import core
from osprey.facility.sources import COMPUTED_SLOTS, KIND_CLASS, RECORD_FILES, SET_VALUED_SLOTS

FACILITY_DIR = Path(__file__).resolve().parents[2] / "src" / "osprey" / "facility"

#: Spellings of the default role a module would write instead of reading
#: ``DEFAULT_ROLE``. An empty default (a raw cell read as ``""``) names no role.
_ROLE_SPELLINGS = (
    re.compile(r"""get\(\s*['"]role['"]\s*,\s*['"][^'"]"""),
    re.compile(r"""\bor\s+['"]readback['"]"""),
    re.compile(r"""^\s*_?(DEFAULT_)?ROLE\s*(:[^=]+)?=""", re.M),
)


def test_combine_sorts_exactly_the_set_valued_rows_a_layer_may_state() -> None:
    for kind in RECORD_FILES.values():
        cls = KIND_CLASS[kind]
        assert {(cls, slot) for slot in SET_VALUED[kind]} == {
            row for row in SET_VALUED_SLOTS if row[0] == cls and row[1] not in COMPUTED_SLOTS[kind]
        }, kind
    assert SET_VALUED == {
        "place": frozenset({"names"}),
        "device": frozenset({"names"}),
        "channel": frozenset({"names", "tags", "former_addresses", "endpoint_of"}),
        "group": frozenset({"names", "members"}),
    }


def test_the_default_role_is_a_role_the_schema_knows() -> None:
    assert DEFAULT_ROLE in {member.value for member in core.RoleEnum}


def test_no_facility_module_spells_the_default_role_itself() -> None:
    hits = []
    for path in sorted(FACILITY_DIR.rglob("*.py")):
        if path == FACILITY_DIR / "__init__.py":
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in _ROLE_SPELLINGS:
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                hits.append(f"{path.relative_to(FACILITY_DIR)}:{line}")
    assert hits == [], "spells the default role instead of reading DEFAULT_ROLE: " + ", ".join(hits)


#: The modules whose value-type checks read ``LABELLED_TYPES``.
_SIMULATION_DIR = FACILITY_DIR.parents[2] / "packages/osprey-connectors/src/osprey_connectors"
_LABELLED_SPELLING = re.compile(r"""\(\s*['"]bool['"]\s*,\s*['"]enum['"]\s*\)""")


def test_the_labelled_value_types_are_defined_once() -> None:
    from osprey_connectors.simulation.values import GATED_TYPES, LABELLED_TYPES, VALUE_TYPES

    assert set(LABELLED_TYPES) | {"float", "int"} == set(GATED_TYPES) <= set(VALUE_TYPES)
    definition = _SIMULATION_DIR / "simulation/values.py"
    hits = []
    for root in (FACILITY_DIR, _SIMULATION_DIR):
        for path in sorted(root.rglob("*.py")):
            if path == definition:
                continue
            text = path.read_text(encoding="utf-8")
            for match in _LABELLED_SPELLING.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                hits.append(f"{path.name}:{line}")
    assert hits == [], "spells the labelled types instead of reading LABELLED_TYPES: " + ", ".join(
        hits
    )
