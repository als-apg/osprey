"""Write the loosenings table of the facility schema.

The facility schema starts from the NARAD seed schemas vendored under
``seeds/``. NARAD requires slots that fit one site's exports only, so each
``required: true`` slot of the four seeds gets exactly one row in
``src/osprey/facility/schema/loosenings.yaml`` saying what became of it:

* ``dropped`` — the facility file has no such slot;
* ``optional`` — the facility file keeps the slot, not required;
* ``renamed:<slot>`` — the facility file carries it as ``<slot>``;
* ``header`` — the facility file states it once, in its identity header.

The script reads only the vendored seeds. It stops when a seed requires a slot
the table below does not decide, or when the table decides a slot no seed
requires, so the committed table always covers the seeds exactly.

Usage (from the repository root)::

    uv run python scripts/facility_schema/loosenings.py           # write
    uv run python scripts/facility_schema/loosenings.py --check   # compare
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SEEDS_DIR = Path(__file__).resolve().parent / "seeds"
OUTPUT = REPO_ROOT / "src" / "osprey" / "facility" / "schema" / "loosenings.yaml"

#: The vendored seed schemas, in the order the table lists them.
SEED_FILES: tuple[str, ...] = (
    "canonical_ingest.yaml",
    "facility_bindings.yaml",
    "concept_vocabulary.yaml",
    "shared_semantics.yaml",
)

#: The shape every fate has.
FATE_PATTERN = re.compile(r"^(dropped|optional|header|renamed:[A-Za-z_][A-Za-z0-9_]*)$")

#: What became of each required seed slot: (seed file, slot) -> (fate, why).
FATES: dict[tuple[str, str], tuple[str, str]] = {
    # canonical_ingest
    ("canonical_ingest.yaml", "dataset_id"): (
        "dropped",
        "the dataset wrapper gives way to Facility",
    ),
    ("canonical_ingest.yaml", "title"): ("header", "identity name"),
    ("canonical_ingest.yaml", "facility"): ("header", "identity code"),
    ("canonical_ingest.yaml", "source_document"): (
        "renamed:provenance",
        "each record names its layer and file",
    ),
    ("canonical_ingest.yaml", "beamline_sections"): (
        "optional",
        "sections are places; a bare channel list has none",
    ),
    ("canonical_ingest.yaml", "devices"): (
        "optional",
        "a bare channel list has no devices",
    ),
    ("canonical_ingest.yaml", "source_id"): (
        "dropped",
        "provenance names the layer and file instead",
    ),
    ("canonical_ingest.yaml", "source_format"): (
        "renamed:layer",
        "the input layer that read the export",
    ),
    ("canonical_ingest.yaml", "section_id"): ("renamed:id", "a place id is its path"),
    ("canonical_ingest.yaml", "source_name"): (
        "renamed:names",
        "the source's own name is the first of the record's names",
    ),
    ("canonical_ingest.yaml", "device_id"): ("renamed:id", "a device id is its facility name"),
    ("canonical_ingest.yaml", "raw_type"): (
        "dropped",
        "a device's class comes from the vocabulary",
    ),
    ("canonical_ingest.yaml", "source_section_id"): (
        "optional",
        "a device's place is optional",
    ),
    ("canonical_ingest.yaml", "numeric_value"): (
        "dropped",
        "positions and lengths are plain numbers in metres",
    ),
    ("canonical_ingest.yaml", "unit"): (
        "optional",
        "a channel's unit is free text, never a closed list",
    ),
    ("canonical_ingest.yaml", "candidate_id"): (
        "renamed:id",
        "a channel id is its full address",
    ),
    ("canonical_ingest.yaml", "endpoint_kind"): (
        "dropped",
        "an address is always the full address, never a suffix",
    ),
    ("canonical_ingest.yaml", "token"): (
        "dropped",
        "an address is always the full address, never a suffix",
    ),
    ("canonical_ingest.yaml", "limit_id"): (
        "dropped",
        "limits records are keyed by channel address",
    ),
    ("canonical_ingest.yaml", "limit_kind"): (
        "dropped",
        "limits records carry min_value, max_value and max_step",
    ),
    ("canonical_ingest.yaml", "key"): (
        "renamed:attributes",
        "raw properties are keys of the record's attributes map",
    ),
    ("canonical_ingest.yaml", "property_key"): (
        "renamed:properties",
        "vocabulary property names",
    ),
    # facility_bindings
    ("facility_bindings.yaml", "bindings_dataset_id"): (
        "dropped",
        "the dataset wrapper gives way to Facility",
    ),
    ("facility_bindings.yaml", "facility"): ("header", "identity code"),
    ("facility_bindings.yaml", "canonical_source_id"): (
        "dropped",
        "one file holds devices and channels together",
    ),
    ("facility_bindings.yaml", "control_system"): (
        "dropped",
        "the control system is deployment configuration",
    ),
    ("facility_bindings.yaml", "bindings"): ("renamed:channels", "a binding is a channel"),
    ("facility_bindings.yaml", "binding_id"): ("renamed:id", "a channel id is its full address"),
    ("facility_bindings.yaml", "canonical_device_id"): (
        "renamed:on",
        "a channel belongs to one device, one place, or nothing",
    ),
    # concept_vocabulary
    ("concept_vocabulary.yaml", "vocabulary_id"): (
        "dropped",
        "the vocabulary is a schema module, not a dataset",
    ),
    ("concept_vocabulary.yaml", "class_name"): (
        "renamed:class",
        "a facility-added class names itself in class",
    ),
    ("concept_vocabulary.yaml", "role_name"): (
        "renamed:signal",
        "a channel names its vocabulary signal role in signal",
    ),
    ("concept_vocabulary.yaml", "group_name"): ("renamed:id", "a group id is its name"),
    ("concept_vocabulary.yaml", "section_id"): ("renamed:id", "a place id is its path"),
    # shared_semantics
    ("shared_semantics.yaml", "sem_dataset_id"): (
        "dropped",
        "the dataset wrapper gives way to Facility",
    ),
    ("shared_semantics.yaml", "facility"): ("header", "identity code"),
    ("shared_semantics.yaml", "canonical_source_id"): (
        "dropped",
        "one file holds devices and their classes together",
    ),
    ("shared_semantics.yaml", "semantic_devices"): (
        "renamed:devices",
        "a device record carries its own class",
    ),
    ("shared_semantics.yaml", "sem_device_id"): (
        "dropped",
        "a device record carries its own class",
    ),
    ("shared_semantics.yaml", "canonical_device_id"): (
        "renamed:id",
        "the semantic record merges into the device it names",
    ),
}


def required_slots(schema: dict) -> set[str]:
    """Every slot a LinkML schema requires, wherever it is declared.

    Args:
        schema: The parsed schema document.

    Returns:
        The names of the slots declared ``required: true`` at the top level, as
        a class attribute or in a class's ``slot_usage``.
    """
    found = {
        name
        for name, slot in (schema.get("slots") or {}).items()
        if isinstance(slot, dict) and slot.get("required") is True
    }
    for cls in (schema.get("classes") or {}).values():
        if not isinstance(cls, dict):
            continue
        for block in ("attributes", "slot_usage"):
            for name, slot in (cls.get(block) or {}).items():
                if isinstance(slot, dict) and slot.get("required") is True:
                    found.add(name)
    return found


def seed_required_slots(seeds_dir: Path = SEEDS_DIR) -> set[tuple[str, str]]:
    """Every ``(seed file, slot)`` pair the vendored seeds require.

    Args:
        seeds_dir: The directory holding the vendored seed schemas.

    Returns:
        One pair per required slot of each seed file.
    """
    pairs: set[tuple[str, str]] = set()
    for seed in SEED_FILES:
        schema = yaml.safe_load((seeds_dir / seed).read_text(encoding="utf-8"))
        pairs.update((seed, slot) for slot in required_slots(schema))
    return pairs


def render(seeds_dir: Path = SEEDS_DIR) -> str:
    """The loosenings table as the text of ``loosenings.yaml``.

    Args:
        seeds_dir: The directory holding the vendored seed schemas.

    Returns:
        The table, one row per required seed slot, in seed order then slot order.

    Raises:
        SystemExit: A seed requires a slot the table does not decide, the table
            decides a slot no seed requires, or a fate is malformed.
    """
    required = seed_required_slots(seeds_dir)
    undecided = sorted(required - FATES.keys())
    unknown = sorted(FATES.keys() - required)
    malformed = sorted(key for key, (fate, _) in FATES.items() if not FATE_PATTERN.match(fate))
    problems = [f"undecided required slot: {seed} {slot}" for seed, slot in undecided]
    problems += [f"decided slot no seed requires: {seed} {slot}" for seed, slot in unknown]
    problems += [f"malformed fate: {seed} {slot}" for seed, slot in malformed]
    if problems:
        raise SystemExit("\n".join(problems))

    order = {seed: index for index, seed in enumerate(SEED_FILES)}
    rows = [
        {"seed": seed, "slot": slot, "fate": FATES[seed, slot][0], "why": FATES[seed, slot][1]}
        for seed, slot in sorted(required, key=lambda pair: (order[pair[0]], pair[1]))
    ]
    header = (
        "# Generated by scripts/facility_schema/loosenings.py from the vendored NARAD\n"
        "# seed schemas; edit the script, never this file.\n"
        "# One row per required seed slot: dropped | optional | renamed:<slot> | header.\n"
    )
    body = yaml.safe_dump({"rows": rows}, sort_keys=False, width=100, allow_unicode=True)
    return header + body


def main(argv: list[str] | None = None) -> int:
    """Write the table, or with ``--check`` report whether the committed one is current.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.

    Returns:
        0 when the table was written or is current, 1 when ``--check`` finds it stale.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare instead of write")
    args = parser.parse_args(argv)
    text = render()
    if args.check:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else ""
        if current != text:
            print(f"{OUTPUT.relative_to(REPO_ROOT)} is stale; run {Path(__file__).name}")
            return 1
        return 0
    OUTPUT.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
