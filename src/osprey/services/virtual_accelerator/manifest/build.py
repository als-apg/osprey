"""Builds the unified virtual-accelerator channel manifest.

Expands all three paradigm channel DBs at their build-resolved tier,
verifies they agree (the whole point of having three formats is that they
describe the same namespace), unions in the scenario-seed ``machine.json``
channels, reconciles the (currently broken) machine-state template against
the result, and classifies every address into a manifest partition plus an
EPICS record type.

Every source is anchored on a :class:`~.paths.ManifestPaths`, so the same
generator serves the bundled tree (the default, and what the container falls
back to) and the facility data tree ``osprey build`` is building from --
:func:`prepare_project_manifest` is the build-time entry point.

Run as a script to (re)generate ``channel_manifest.json``::

    uv run python -m osprey.services.virtual_accelerator.manifest.build
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from osprey.errors import BuildProfileError

from . import classify, loaders
from .paths import MANIFEST_OUTPUT, PACKAGE_PATHS, ManifestPaths

logger = logging.getLogger(__name__)

# Filenames the virtual-accelerator container looks for under its
# ``/data/simulation`` mount: ``VA_CHANNELS_FILE`` resolves relative names
# against the data dir, and drive limits are read from ``channel_limits.json``
# beside it (see services/virtual_accelerator/entrypoint.py).
MANIFEST_FILENAME = "channel_manifest.json"
LIMITS_FILENAME = "channel_limits.json"


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


def build_manifest(paths: ManifestPaths = PACKAGE_PATHS) -> dict:
    """Build the full channel manifest as a JSON-serializable dict.

    Args:
        paths: The data tree (and tier) to expand. Defaults to the bundled
            control-assistant tree, which is what the container's in-process
            regeneration and every runtime caller use.

    Raises:
        loaders.ParadigmMismatchError: if the three paradigm DBs disagree on
            the address set they expand to at the build-resolved tier.
    """
    # Three paradigms, deliberately — the ``graph`` paradigm is exempt from this
    # gate and stays exempt. The gate compares address sets expanded from tiered
    # database files; a graph project ships none, because its store is seeded
    # from the facility corpus TTL. Graph's identity contract is the same
    # promise enforced one step earlier, at the corpus: the PV-set equality
    # asserted by ``tests/services/facility_knowledge/test_demo_ttl_consistency
    # .py::test_demo_ttl_bindings_equal_the_channel_database`` — every channel
    # in the tier-3 database has a binding in the corpus and vice versa. So a
    # graph-mode project's namespace still agrees with the one this manifest is
    # built from; it is just pinned by the corpus test rather than here.
    hier_channels = loaders.load_hierarchical_channels(paths)
    hier_addresses = {c.address for c in hier_channels}

    in_context_addresses = loaders.load_in_context_addresses(paths)
    middle_layer_addresses = loaders.load_middle_layer_addresses(paths)

    if hier_addresses != in_context_addresses:
        only_hier = sorted(hier_addresses - in_context_addresses)[:10]
        only_in_context = sorted(in_context_addresses - hier_addresses)[:10]
        raise loaders.ParadigmMismatchError(
            f"hierarchical vs in_context tier-{paths.tier} address sets disagree: "
            f"only-in-hierarchical(sample)={only_hier} "
            f"only-in-in_context(sample)={only_in_context}"
        )
    if hier_addresses != middle_layer_addresses:
        only_hier = sorted(hier_addresses - middle_layer_addresses)[:10]
        only_ml = sorted(middle_layer_addresses - hier_addresses)[:10]
        raise loaders.ParadigmMismatchError(
            f"hierarchical vs middle_layer tier-{paths.tier} address sets disagree: "
            f"only-in-hierarchical(sample)={only_hier} "
            f"only-in-middle_layer(sample)={only_ml}"
        )

    machine_json_channels = loaders.load_machine_json_channels(paths=paths)
    # machine.json is expected to be a scenario-seed subset of the DB
    # namespace. A novel address here would be additive data, not an error --
    # but it's the one place a new channel could sneak in without ever
    # passing through the paradigm DBs, so it's surfaced in _metadata rather
    # than silently unioned in unremarked.
    novel_machine_json = sorted(set(machine_json_channels) - hier_addresses)

    all_addresses = hier_addresses | set(novel_machine_json)

    machine_state_candidates = loaders.load_machine_state_candidate_addresses(paths)
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
            "generator": "osprey.services.virtual_accelerator.manifest",
            "source_tier": paths.tier,
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


@dataclass(frozen=True)
class PreparedManifest:
    """A manifest built from a project's own data tree, ready to be written.

    Holding the built manifest (rather than re-deriving it at write time) is
    what makes the build step atomic: every way generation can fail -- absent
    sources, disagreeing paradigm DBs -- has already happened by the time a
    caller holds one of these, so the decision to wire ``VA_CHANNELS_FILE``
    into the project's ``.env`` can be made before anything is written.
    """

    manifest: dict
    limits_source: Path


def prepare_project_manifest(data_root: Path, tier: int) -> PreparedManifest | None:
    """Build the VA channel manifest from the data tree a build is using.

    The virtual accelerator otherwise serves the bundled control-assistant
    namespace regardless of what a project ships, because the container has
    no channel DBs mounted and the ``tiers/`` subtree is pruned from built
    projects. Build time is the one stage where a project's own paradigm DBs
    are still on disk.

    Args:
        data_root: The ``data/`` tree this build is sourcing from -- the
            profile's ``data:`` tree, or the bundle's own.
        tier: The build-resolved tier whose paradigm DBs to expand.

    Returns:
        The prepared manifest, or ``None`` when this tree cannot back one --
        in which case the caller wires nothing and the container keeps its
        package fallback. That is the common case: only the control-assistant
        bundle ships paradigm DBs. A facility tree that ships them but whose
        edits made them disagree also falls back, with a warning: profile
        data is user-owned and must not fail a build.

    Raises:
        BuildProfileError: if the tree ships the paradigm DBs but one of them
            is corrupt. Absent sources and disagreeing-but-valid sources fall
            back; unreadable ones must not. Falling back there would serve the
            bundled channel namespace to an operator who believes they are
            driving their own facility's -- for a control system that is worse
            than failing the build.
    """
    paths = ManifestPaths(data_root=data_root, tier=tier)
    missing = paths.missing_sources()
    if not paths.channel_limits.is_file():
        # Drive limits ship beside the manifest or the VA enforces none at
        # all -- generating one without the other is exactly the silent
        # unbounded-setpoint state this must never produce.
        missing.append(paths.channel_limits)
    if missing:
        logger.debug(
            "Virtual-accelerator manifest not generated from %s: missing %s",
            data_root,
            ", ".join(str(p.relative_to(data_root)) for p in missing),
        )
        return None

    try:
        manifest = build_manifest(paths)
    except loaders.ParadigmMismatchError as exc:
        logger.warning(
            "Virtual-accelerator manifest not generated from %s: the tier-%d "
            "paradigm channel databases disagree (%s). The virtual accelerator "
            "will serve the built-in channel set instead.",
            data_root,
            tier,
            exc,
        )
        return None
    except (json.JSONDecodeError, KeyError, OSError) as exc:
        culprit = _first_unreadable_source(paths)
        detail = (
            f"{culprit} is not readable as JSON"
            if culprit is not None
            else f"{type(exc).__name__}: {exc}"
        )
        raise BuildProfileError(
            f"virtual-accelerator channel manifest could not be built from the "
            f"data tree {data_root} at tier {tier}: {detail}. Repair the file, "
            f"or remove the tier-{tier} paradigm databases from the tree to "
            f"build against the framework's built-in channel set instead."
        ) from exc

    return PreparedManifest(manifest=manifest, limits_source=paths.channel_limits)


def _first_unreadable_source(paths: ManifestPaths) -> Path | None:
    """Name the source file behind a generation failure, for the error message.

    Only ever called on the failure path, where re-reading the tree costs
    nothing and turns an opaque ``KeyError: 'channel'`` from deep inside a
    database parser into the path the operator has to go fix. Walks
    :attr:`~.paths.ManifestPaths.required_sources` so it covers exactly what
    the generator read, however that set grows.
    """
    for path in paths.required_sources:
        try:
            json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return path
    return None


def write_project_manifest(prepared: PreparedManifest, project_data_dir: Path) -> Path:
    """Write a prepared manifest and its drive limits into ``data/simulation/``.

    Both files land in the directory the container already bind-mounts, so
    neither needs a compose change: ``VA_CHANNELS_FILE`` resolves relative
    names against the data dir, and the entrypoint reads ``channel_limits.json``
    from beside it. The limits file is copied rather than bind-mounted
    single-file, which fails at container init.

    The limits copy is taken from the built project when it has one (so any
    facility overlay applied to ``data/channel_limits.json`` is what the VA
    enforces), falling back to the source tree the manifest was prepared
    from.

    Returns:
        The path the manifest was written to.
    """
    simulation_dir = project_data_dir / "simulation"
    simulation_dir.mkdir(parents=True, exist_ok=True)

    limits_source = project_data_dir / LIMITS_FILENAME
    if not limits_source.is_file():
        limits_source = prepared.limits_source
    shutil.copy2(limits_source, simulation_dir / LIMITS_FILENAME)

    manifest_path = simulation_dir / MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(prepared.manifest, indent=2) + "\n")
    return manifest_path


def main() -> None:
    """CLI entry point: (re)generate channel_manifest.json on disk."""
    manifest = build_manifest()
    MANIFEST_OUTPUT.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {manifest['_metadata']['total_channels']} channels to {MANIFEST_OUTPUT}")


if __name__ == "__main__":
    main()
