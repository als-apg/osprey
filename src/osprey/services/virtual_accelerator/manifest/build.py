"""Builds the unified virtual-accelerator channel manifest.

Expands whichever paradigm channel DBs a tree stages at its build-resolved
tier, verifies the staged ones agree (the point of having several formats is
that they describe the same namespace), unions in the scenario-seed
``machine.json`` channels, reconciles the (currently broken) machine-state
template against the result, and classifies every address into a manifest
partition plus an EPICS record type.

Every source is anchored on a :class:`~.paths.ManifestPaths`, so the same
generator serves the bundled tree (the default, and the framework's own
tutorial machine) and the facility data tree ``osprey build`` is building from
-- :func:`prepare_project_manifest` is the build-time entry point. A project
deployment always serves a manifest built from its OWN tree: there is no path
by which a project's accelerator falls back to the bundled demo namespace.

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


def _pathless_entry(address: str, *, noise: bool) -> ManifestEntry:
    """One manifest entry for an address that carries no hierarchy path.

    Two sources produce these: an address seeded only by ``machine.json``, and
    every address of a tree that stages no ``hierarchical`` database. The
    identity keys are empty, which is what the file-backed manifest schema
    allows and what ``loaders.MANIFEST_CHANNEL_KEYS`` documents the cost of:
    setpoint and readback are paired on exactly those five keys, so a channel
    without them pairs with nothing. Classification cannot be better than the
    input, so the entry lands where an unclassifiable address belongs, in the
    partition the simulation engine drives.
    """
    return ManifestEntry(
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


class NoChannelSourcesError(RuntimeError):
    """A data tree names no channels at all at the tier being built.

    Distinct from a partial tree, which is ordinary: one staged paradigm
    database is a namespace, and the manifest is built from it. Zero is the
    case with no answer, and the build that hit it deploys an accelerator with
    nothing of the project's to serve.
    """


class CorruptChannelSourcesError(RuntimeError):
    """Every paradigm database the tree stages is present and unreadable.

    Deliberately NOT :class:`NoChannelSourcesError`: a database that is not
    there is a namespace this project did not ship, while one that is there and
    cannot be read is a namespace it meant to ship and got wrong. The two send
    an operator to different files, so they are different refusals -- the same
    split the channel roster draws between its ``missing-source`` and
    ``corrupt-source`` absences.
    """


@dataclass(frozen=True)
class CorruptParadigm:
    """One staged paradigm database that is present and could not be read.

    Attributes:
        paradigm: Which paradigm database this is.
        path: The file that could not be read.
        detail: One line saying why, from the parser that failed.
    """

    paradigm: str
    path: Path
    detail: str

    def describe(self, data_root: Path) -> str:
        """Name the file, relative to *data_root*, and why it could not be read."""
        return f"{self.paradigm} ({self.path.relative_to(data_root)}): {self.detail}"


@dataclass(frozen=True)
class ParadigmExpansion:
    """What the staged paradigm databases expanded to, and what did not.

    Attributes:
        addresses: The address set per paradigm that LOADED, in read order.
            Empty when the tree stages none, or when every staged one is
            corrupt.
        hierarchy_paths: The hierarchy path per address, from the
            ``hierarchical`` database when it loaded; empty otherwise.
        corrupt: The staged databases that are present and could not be read,
            in read order. Each contributed zero addresses.
    """

    addresses: dict[str, set[str]]
    hierarchy_paths: dict[str, dict[str, str]]
    corrupt: tuple[CorruptParadigm, ...]


def _hierarchical_expansion(paths: ManifestPaths) -> tuple[set[str], dict[str, dict[str, str]]]:
    """Expand the hierarchical database into its addresses and their paths.

    It is the one paradigm that declares a hierarchy path, and it is read once
    here so the manifest never re-opens a file this expansion already found
    unreadable.
    """
    by_address = {c.address: c.path for c in loaders.load_hierarchical_channels(paths)}
    return set(by_address), by_address


def _paradigm_addresses(paths: ManifestPaths) -> ParadigmExpansion:
    """Expand each paradigm database the tree stages into its address set.

    A staged database that cannot be loaded contributes NOTHING and is recorded
    as corrupt rather than propagating its parser's exception: one broken file
    out of several is a degraded namespace, and the manifest is still built
    from the databases that are left. Refusing is the job of the callers, and
    only when nothing usable is left.

    The broad ``except`` is the price of reusing the channel-finder parsers:
    a database class raises whatever its format's breakage produces, and a
    file that is valid JSON with the wrong shape surfaces as an
    ``AttributeError`` from deep inside one. So the ``try`` wraps the load call
    alone -- nothing downstream of it can be swallowed here.

    Returns:
        The expansion, whose ``addresses`` is empty when the tree stages no
        database at all or every staged one is corrupt.
    """
    loader_by_paradigm = {
        "hierarchical": lambda: _hierarchical_expansion(paths),
        "in_context": lambda: (loaders.load_in_context_addresses(paths), {}),
        "middle_layer": lambda: (loaders.load_middle_layer_addresses(paths), {}),
    }
    addresses: dict[str, set[str]] = {}
    hierarchy_paths: dict[str, dict[str, str]] = {}
    corrupt: list[CorruptParadigm] = []
    for name in paths.staged_paradigms:
        try:
            expanded, paths_by_address = loader_by_paradigm[name]()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "The tier-%d %s channel database at %s is present and could not be read "
                "(%s: %s), so it contributes no channels to the manifest.",
                paths.tier,
                name,
                paths.paradigm_databases[name],
                type(exc).__name__,
                exc,
            )
            corrupt.append(
                CorruptParadigm(
                    paradigm=name,
                    path=paths.paradigm_databases[name],
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        addresses[name] = expanded
        hierarchy_paths.update(paths_by_address)
    return ParadigmExpansion(
        addresses=addresses, hierarchy_paths=hierarchy_paths, corrupt=tuple(corrupt)
    )


def build_manifest(paths: ManifestPaths = PACKAGE_PATHS) -> dict:
    """Build the full channel manifest as a JSON-serializable dict.

    Built from the paradigm databases the tree STAGES at its tier, whichever
    subset that is. The project's namespace is what the project shipped: a tier
    naming one database is as complete an answer as a tier naming three, and
    the alternative to accepting it -- serving the framework's built-in demo
    namespace to a facility under its own name -- is the thing this must never
    do.

    What a subset costs is stated rather than hidden. The cross-paradigm
    agreement gate can only compare what is here, so a single database is taken
    at its word. And ``hierarchical`` is the one paradigm that declares a
    hierarchy path, so without it every channel carries empty identity keys and
    lands in the static-noisy partition; the setpoint/readback pairing those
    keys drive is not available. ``_metadata`` names the databases that fed the
    manifest, the ones that were absent, and the ones that were staged and
    could not be read, so a reader of the manifest alone can see which case
    they are in.

    That last group is the degradation stated twice: an unreadable database
    contributes no addresses AND is named with the reason it failed, so the
    census is never quietly short a database an operator believes fed it.

    Args:
        paths: The data tree (and tier) to expand. Defaults to the bundled
            control-assistant tree, which is what the container's in-process
            regeneration and every runtime caller use, and which stages all
            three.

    Raises:
        NoChannelSourcesError: if the tree stages no paradigm database at all.
        CorruptChannelSourcesError: if every database it stages is present and
            unreadable, so nothing usable is left to build from.
        loaders.ParadigmMismatchError: if the paradigm DBs it DOES stage and
            can read disagree on the address set they expand to.
    """
    # The ``graph`` paradigm is exempt from the agreement gate and stays exempt.
    # The gate compares address sets expanded from tiered database files; a
    # graph project ships none, because its store is seeded from the facility
    # corpus TTL. Graph's identity contract is the same promise enforced one
    # step earlier, at the corpus: the PV-set equality asserted by
    # ``tests/services/facility_knowledge/test_demo_ttl_consistency
    # .py::test_demo_ttl_bindings_equal_the_channel_database``. Every channel in
    # the tier-3 database has a binding in the corpus and vice versa, so a
    # graph-mode project's namespace still agrees with the one this manifest is
    # built from; it is just pinned by the corpus test rather than here.
    staged = paths.staged_paradigms
    if not staged:
        raise NoChannelSourcesError(
            f"no paradigm channel database is staged under {paths.tier_dir}: "
            f"there are no channels to build a manifest from"
        )

    expansion = _paradigm_addresses(paths)
    if not expansion.addresses:
        # Every staged database is present and unreadable. Named as corrupt
        # rather than counted as absent: the operator has files to repair, not
        # files to add.
        raise CorruptChannelSourcesError(
            f"every channel database staged under {paths.tier_dir} is present and could "
            "not be read: " + "; ".join(c.describe(paths.data_root) for c in expansion.corrupt)
        )

    reference_paradigm, addresses = next(iter(expansion.addresses.items()))
    for paradigm, other in list(expansion.addresses.items())[1:]:
        if other != addresses:
            only_reference = sorted(addresses - other)[:10]
            only_other = sorted(other - addresses)[:10]
            raise loaders.ParadigmMismatchError(
                f"{reference_paradigm} vs {paradigm} tier-{paths.tier} address sets "
                f"disagree: only-in-{reference_paradigm}(sample)={only_reference} "
                f"only-in-{paradigm}(sample)={only_other}"
            )

    # The hierarchy path is what a channel's identity keys are read from, and
    # only the hierarchical database declares one. Absent -- or staged and
    # unreadable -- every channel is classified the way a machine.json-only
    # address already is.
    path_by_address = expansion.hierarchy_paths

    machine_json_channels = loaders.load_machine_json_channels(paths=paths)
    # machine.json is expected to be a scenario-seed subset of the DB
    # namespace. A novel address here would be additive data, not an error --
    # but it's the one place a new channel could sneak in without ever
    # passing through the paradigm DBs, so it's surfaced in _metadata rather
    # than silently unioned in unremarked.
    novel_machine_json = sorted(set(machine_json_channels) - addresses)

    all_addresses = addresses | set(novel_machine_json)

    machine_state_candidates = loaders.load_machine_state_candidate_addresses(paths)
    machine_state_valid = sorted(set(machine_state_candidates) & all_addresses)
    machine_state_invalid = sorted(set(machine_state_candidates) - all_addresses)

    entries: list[ManifestEntry] = []
    for address in sorted(addresses):
        path = path_by_address.get(address)
        if path is None:
            entries.append(_pathless_entry(address, noise=False))
            continue
        partition = classify.classify_partition(path)
        record_type, noise = classify.derive_record_type(path)
        entries.append(
            ManifestEntry(
                address=address,
                ring=path["ring"],
                system=path["system"],
                family=path["family"],
                device=path["device"],
                field=path["field"],
                subfield=path["subfield"],
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
        entries.append(
            _pathless_entry(address, noise=(not is_derived) and bool(chan.get("noise", 0)))
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
            "source_paradigms": list(expansion.addresses),
            "absent_paradigms": list(paths.absent_paradigms),
            "corrupt_paradigms": [
                {
                    "paradigm": c.paradigm,
                    "path": str(c.path.relative_to(paths.data_root)),
                    "detail": c.detail,
                }
                for c in expansion.corrupt
            ],
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

    A project's accelerator simulates the project's own channels. Build time is
    the one stage where they are still on disk: the container mounts no channel
    databases, and the ``tiers/`` subtree is pruned from a built project. So
    whatever the tree stages is expanded here, or the caller has nothing to
    wire and the build refuses rather than letting the container fall back to
    the framework's built-in demo namespace.

    Args:
        data_root: The ``data/`` tree this build is sourcing from -- the
            profile's ``data:`` tree, or the bundle's own.
        tier: The build-resolved tier whose paradigm DBs to expand.

    Returns:
        The prepared manifest, or ``None`` when this tree cannot back one: it
        stages no paradigm database at this tier, the databases it stages name
        no channel, every one of them is present and unreadable, it is missing
        the scenario seed, the machine-state list or the drive limits, or the
        databases it does stage disagree. :func:`manifest_gap_reason` says
        which, in the words the refusal is written in. A caller deploying a
        virtual accelerator MUST refuse on ``None`` rather than continue.

        SOME of the staged databases being corrupt is not one of those cases:
        the manifest is built from the ones that are left, and ``_metadata``
        names the broken ones so the degrade is read rather than inferred from
        a channel count.

    Raises:
        BuildProfileError: if the scenario seed or the machine-state list --
            the sources a tree carries one of, rather than one per paradigm --
            cannot be read. There is nothing left to build from when either is
            broken, and reading past it would quietly serve a different channel
            set than the operator believes they are driving.
    """
    paths = ManifestPaths(data_root=data_root, tier=tier)
    if not paths.staged_paradigms:
        logger.debug(
            "Virtual-accelerator manifest not generated from %s: %s stages no "
            "paradigm channel database",
            data_root,
            paths.tier_dir,
        )
        return None

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
            "channel databases it stages disagree (%s). Repair them: a build "
            "deploying a virtual accelerator refuses rather than serving a "
            "channel set this project did not describe.",
            data_root,
            tier,
            exc,
        )
        return None
    except CorruptChannelSourcesError as exc:
        logger.warning(
            "Virtual-accelerator manifest not generated from %s: %s. Repair the "
            "file(s), or remove them from the tree: a build deploying a virtual "
            "accelerator refuses rather than serving a channel set this project "
            "did not describe.",
            data_root,
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
            f"or remove it from the tree so the manifest is built from the "
            f"databases that are left."
        ) from exc

    if not manifest["channels"]:
        # A staged database that expands to nothing is not a namespace, and the
        # file being present is what would otherwise let it past every check
        # above: `staged_paradigms` asks whether the file exists, not whether it
        # names a channel. Answering None here puts an empty tree on exactly the
        # path a tree with no databases at all takes.
        logger.debug(
            "Virtual-accelerator manifest not generated from %s: the databases it "
            "stages (%s) expand to no channels",
            data_root,
            ", ".join(paths.staged_paradigms),
        )
        return None

    return PreparedManifest(manifest=manifest, limits_source=paths.channel_limits)


def _staged_expansion_is_empty(paths: ManifestPaths, expansion: ParadigmExpansion) -> bool:
    """Whether everything *paths* stages expands to no channel at all.

    Only ever asked on the failure path, where re-reading the tree costs
    nothing. A tree whose sources cannot be read is NOT empty -- it has a
    different problem, and saying "expands to no channels" about it would send
    an operator to the wrong file. So a corrupt source answers no here and is
    named as corrupt by the caller instead.
    """
    if expansion.corrupt:
        return False
    try:
        addresses = set().union(*expansion.addresses.values())
        addresses |= set(loaders.load_machine_json_channels(paths=paths))
    except (json.JSONDecodeError, KeyError, OSError):
        return False
    return not addresses


def manifest_gap_reason(data_root: Path, tier: int) -> str:
    """Say why :func:`prepare_project_manifest` could back no manifest.

    The generator answers ``None`` for several different trees, and the build
    refusing on it has to name which one an operator is looking at. Re-checking
    the tree here costs nothing on a path that has already failed, and keeps
    the wording in the module that owns the sources rather than in the caller.

    Args:
        data_root: The ``data/`` tree this build sourced from.
        tier: The build-resolved tier whose paradigm databases were expanded.

    Returns:
        A phrase naming the absent files, the present-but-unreadable ones, or
        the disagreement between the ones that read cleanly. A corrupt database
        is never reported as an absent one: the two send an operator to
        different work.
    """
    paths = ManifestPaths(data_root=data_root, tier=tier)
    if not paths.staged_paradigms:
        return (
            f"no channel database is staged at tier {tier} "
            f"({', '.join(sorted(paths.absent_paradigms))} are all absent)"
        )
    missing = paths.missing_sources()
    if not paths.channel_limits.is_file():
        missing.append(paths.channel_limits)
    if missing:
        return "missing " + ", ".join(str(path.relative_to(data_root)) for path in missing)

    expansion = _paradigm_addresses(paths)
    if expansion.corrupt:
        unreadable = "; ".join(c.describe(data_root) for c in expansion.corrupt)
        if not expansion.addresses:
            return (
                f"every channel database staged at tier {tier} is present and could not "
                f"be read: {unreadable}"
            )
        also_unreadable = f"; also present and unreadable: {unreadable}"
    else:
        also_unreadable = ""

    if _staged_expansion_is_empty(paths, expansion):
        return (
            f"the channel databases staged at tier {tier} "
            f"({', '.join(expansion.addresses)}) name no channels{also_unreadable}"
        )
    return (
        f"the tier-{tier} channel databases it stages describe different channel sets"
        f"{also_unreadable}"
    )


def _first_unreadable_source(paths: ManifestPaths) -> Path | None:
    """Name the source file behind a generation failure, for the error message.

    Only ever called on the failure path, where re-reading the tree costs
    nothing and turns an opaque ``KeyError: 'channel'`` from deep inside a
    parser into the path the operator has to go fix. Walks
    :attr:`~.paths.ManifestPaths.required_sources` so it covers exactly what
    the generator read, however that set grows -- minus the paradigm
    databases, whose read failures no longer reach a caller: one of those
    degrades into a recorded corrupt source (see :func:`_paradigm_addresses`),
    so naming one here would blame the wrong file for the seed or
    machine-state list that actually failed.
    """
    paradigm_databases = set(paths.paradigm_databases.values())
    for path in paths.required_sources:
        if path in paradigm_databases:
            continue
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
