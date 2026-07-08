"""Builds the unified virtual-accelerator channel manifest.

Expands all three paradigm channel DBs at their build-resolved tier,
verifies they agree (the whole point of having three formats is that they
describe the same namespace), unions in the scenario-seed ``machine.json``
channels, reconciles the (currently broken) machine-state template against
the result, and classifies every address into a manifest partition plus an
EPICS record type.

Run as a script to (re)generate ``channel_manifest.json``::

    uv run python -m manifest.build
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from . import classify, loaders, paths


@dataclass(frozen=True)
class ManifestEntry:
    """One channel in the emitted manifest."""

    address: str
    ring: str
    system: str
    family: str
    device: str
    field: str
    subfield: str
    partition: str
    record_type: str
    noise: bool


def build_manifest() -> dict:
    """Build the full channel manifest as a JSON-serializable dict.

    Raises:
        loaders.ParadigmMismatchError: if the three paradigm DBs disagree on
            the address set they expand to at the build-resolved tier.
    """
    hier_channels = loaders.load_hierarchical_channels()
    hier_addresses = {c.address for c in hier_channels}

    in_context_addresses = loaders.load_in_context_addresses()
    middle_layer_addresses = loaders.load_middle_layer_addresses()

    if hier_addresses != in_context_addresses:
        only_hier = sorted(hier_addresses - in_context_addresses)[:10]
        only_in_context = sorted(in_context_addresses - hier_addresses)[:10]
        raise loaders.ParadigmMismatchError(
            "hierarchical vs in_context tier-3 address sets disagree: "
            f"only-in-hierarchical(sample)={only_hier} "
            f"only-in-in_context(sample)={only_in_context}"
        )
    if hier_addresses != middle_layer_addresses:
        only_hier = sorted(hier_addresses - middle_layer_addresses)[:10]
        only_ml = sorted(middle_layer_addresses - hier_addresses)[:10]
        raise loaders.ParadigmMismatchError(
            "hierarchical vs middle_layer tier-3 address sets disagree: "
            f"only-in-hierarchical(sample)={only_hier} "
            f"only-in-middle_layer(sample)={only_ml}"
        )

    machine_json_channels = loaders.load_machine_json_channels()
    # machine.json is expected to be a scenario-seed subset of the DB
    # namespace. A novel address here would be additive data, not an error --
    # but it's the one place a new channel could sneak in without ever
    # passing through the paradigm DBs, so it's surfaced in _metadata rather
    # than silently unioned in unremarked.
    novel_machine_json = sorted(set(machine_json_channels) - hier_addresses)

    all_addresses = hier_addresses | set(novel_machine_json)

    machine_state_candidates = loaders.load_machine_state_candidate_addresses()
    machine_state_valid = sorted(set(machine_state_candidates) & all_addresses)
    machine_state_invalid = sorted(set(machine_state_candidates) - all_addresses)

    entries: list[ManifestEntry] = []
    for ch in hier_channels:
        partition = classify.classify_partition(ch.path)
        record_type, noise = classify.derive_record_type(ch.path)
        entries.append(
            ManifestEntry(
                address=ch.address,
                ring=ch.path["ring"],
                system=ch.path["system"],
                family=ch.path["family"],
                device=ch.path["device"],
                field=ch.path["field"],
                subfield=ch.path["subfield"],
                partition=partition,
                record_type=record_type,
                noise=noise,
            )
        )

    # Novel machine.json-only addresses (currently none -- verified empty at
    # tier 3) carry no hierarchy path, so classify them from their
    # machine.json shape instead of the DB path.
    for address in novel_machine_json:
        chan = machine_json_channels[address]
        is_derived = "expr" in chan
        noise = (not is_derived) and bool(chan.get("noise", 0))
        entries.append(
            ManifestEntry(
                address=address,
                ring="",
                system="",
                family="",
                device="",
                field="",
                subfield="",
                partition=classify.PARTITION_STATIC_NOISY,
                record_type=classify.RECORD_TYPE_ANALOG,
                noise=noise,
            )
        )

    entries.sort(key=lambda e: e.address)

    by_ring: dict[str, int] = {}
    by_partition: dict[str, int] = {}
    setpoint_count = 0
    for e in entries:
        if e.ring:
            by_ring[e.ring] = by_ring.get(e.ring, 0) + 1
        by_partition[e.partition] = by_partition.get(e.partition, 0) + 1
        if e.subfield == "SP":
            setpoint_count += 1

    return {
        "_metadata": {
            "generator": "docker/virtual-accelerator/manifest",
            "source_tier": paths.DEFAULT_TIER,
            "total_channels": len(entries),
            "by_ring": by_ring,
            "by_partition": by_partition,
            "setpoint_count": setpoint_count,
            "machine_json_channel_count": len(machine_json_channels),
            "machine_json_novel_addresses": novel_machine_json,
            "machine_state_reconciliation": {
                "candidates_checked": len(machine_state_candidates),
                "valid": machine_state_valid,
                "invalid": machine_state_invalid,
            },
        },
        "channels": [asdict(e) for e in entries],
    }


def main() -> None:
    """CLI entry point: (re)generate channel_manifest.json on disk."""
    manifest = build_manifest()
    paths.MANIFEST_OUTPUT.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {manifest['_metadata']['total_channels']} channels to {paths.MANIFEST_OUTPUT}")


if __name__ == "__main__":
    main()
