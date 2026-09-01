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

A graph-mode project stages no paradigm database at all -- its channels live in
the knowledge-graph corpus the deploy seeds the graph store from. For such a
tree the manifest's channel set is derived from the channel roster's graph
reader (:func:`osprey.channel_roster.registered_channels`, the one membership
authority), and ``_metadata`` names the corpus as the source.

A project may also AUTHOR its manifest outright, at the
``data/simulation/channel_manifest.json`` the framework reserves for it, and
then no derivation happens at all: the file is the answer, and the
``channel_limits.json`` beside it is what bounds the accelerator. So the source
precedence is staged paradigm databases, then the authored manifest, then the
graph -- most specific first, and each one only consulted when nothing above it
is staged.

Run as a script to (re)generate ``channel_manifest.json``::

    uv run python -m osprey.services.virtual_accelerator.manifest.build
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from osprey.errors import BuildProfileError

from . import classify, loaders
from .paths import (
    LIMITS_FILENAME,
    MANIFEST_FILENAME,
    MANIFEST_OUTPUT,
    PACKAGE_PATHS,
    ManifestPaths,
)

logger = logging.getLogger(__name__)


class ManifestSource(StrEnum):
    """Where a prepared manifest came from.

    Carried on the prepared manifest rather than re-derived from its
    ``_metadata``, because the two consumers that branch on it -- the writer
    and the build's fact -- must agree with the selection that actually
    happened, and an AUTHORED manifest's metadata is the facility's to write
    (it may say anything, or nothing at all).
    """

    #: Expanded from the paradigm channel databases the tree stages.
    PARADIGM = "paradigm"
    #: Read verbatim from the profile's own ``data/simulation/`` manifest.
    AUTHORED = "authored"
    #: Derived from the project's knowledge-graph corpus.
    GRAPH = "graph"


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

    return _finish_manifest(
        entries,
        paths,
        source_paradigms=list(expansion.addresses),
        absent_paradigms=list(paths.absent_paradigms),
        corrupt_paradigms=[
            {
                "paradigm": c.paradigm,
                "path": str(c.path.relative_to(paths.data_root)),
                "detail": c.detail,
            }
            for c in expansion.corrupt
        ],
    )


def _finish_manifest(
    entries: list[ManifestEntry],
    paths: ManifestPaths,
    *,
    source_paradigms: list[str],
    absent_paradigms: list[str],
    corrupt_paradigms: list[dict],
    source_corpus: str | None = None,
) -> dict:
    """Union the scenario seed in, reconcile machine state, and assemble the document.

    The half of manifest generation every source shares: whichever enumerator
    produced *entries* -- the paradigm databases or the knowledge-graph roster
    -- the scenario-seed union, the machine-state reconciliation, the census
    and the ``_metadata`` block are the same document, assembled once here so
    the two sources cannot drift in what they publish about themselves.

    Args:
        entries: One entry per enumerated channel, before the scenario seed.
        paths: The data tree the per-tree sources (``machine.json``, the
            machine-state list) are read from.
        source_paradigms: What fed the entries, in the paradigm vocabulary --
            the paradigm databases that loaded, or ``["graph"]``.
        absent_paradigms: The paradigm databases the tree did not stage. Empty
            for a graph-sourced manifest: graph mode stages no tier database
            by design, so there is nothing absent to report.
        corrupt_paradigms: The staged databases that could not be read, as
            rendered metadata rows.
        source_corpus: The knowledge-graph corpus the entries came from, as an
            operator would name it, or ``None`` for a database-sourced
            manifest -- the key appears only when a corpus actually fed it.
    """
    entries = list(entries)
    addresses = {entry.address for entry in entries}

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

    metadata: dict = {
        "generator": "osprey.services.virtual_accelerator.manifest",
        "source_tier": paths.tier,
        "total_channels": len(entries),
        "by_ring": by_ring,
        "by_partition": by_partition,
        "setpoint_count": setpoint_count,
        "machine_json_channel_count": len(machine_json_channels),
        "machine_json_novel_addresses": novel_machine_json,
        "source_paradigms": source_paradigms,
    }
    if source_corpus is not None:
        metadata["source_corpus"] = source_corpus
    metadata.update(
        {
            "absent_paradigms": absent_paradigms,
            "corrupt_paradigms": corrupt_paradigms,
            "machine_state_reconciliation": {
                "candidates_checked": len(machine_state_candidates),
                "valid": machine_state_valid,
                "invalid": machine_state_invalid,
            },
        }
    )
    return {"_metadata": metadata, "channels": [asdict(e) for e in entries]}


def _graph_roster(config: dict):
    """The channel roster, when this project's roster source is the knowledge graph.

    Answered through :mod:`osprey.channel_roster` -- the one membership
    authority -- rather than by a second corpus parser here. ``None`` when the
    project's roster source is not the graph at all (a database paradigm, or
    nothing configured), which tells the caller to keep the paradigm-database
    rules; a graph-mode project always gets a
    :class:`~osprey.channel_roster.records.RosterResult` back, absence
    included, so the refusal can name the corpus.

    Imported lazily: this module is imported inside the virtual-accelerator
    container, which never reads a roster source and should not pay for the
    roster package's import graph.
    """
    from osprey.channel_roster import (
        RosterAbsenceReason,
        RosterSourceKind,
        registered_channels,
        resolve_roster_source,
    )

    resolution = resolve_roster_source(config)
    graph_configured = (
        resolution.source is not None and resolution.source.kind is RosterSourceKind.GRAPH
    ) or (
        resolution.absence is not None
        and resolution.absence.reason is RosterAbsenceReason.GRAPH_NO_TTL
    )
    if not graph_configured:
        return None
    return registered_channels(config)


#: The paradigm-vocabulary name of the graph source in ``_metadata``. The same
#: token ``detect_pipeline_config`` answers for graph mode, so the manifest
#: names its source in the vocabulary every other build surface already uses.
GRAPH_SOURCE_PARADIGM = "graph"


def _graph_missing_sources(paths: ManifestPaths) -> list[Path]:
    """The per-tree files a graph-sourced manifest still needs, when absent.

    The corpus enumerates the channels, but the scenario seed, the
    machine-state list and the drive limits are one-per-tree sources every
    manifest ships beside -- the container refuses to boot without
    ``machine.json``, and a manifest without ``channel_limits.json`` beside it
    is the silent unbounded-setpoint state the paradigm path refuses too.
    """
    missing = [
        path
        for path in (paths.machine_json, paths.machine_state_channels, paths.channel_limits)
        if not path.is_file()
    ]
    return missing


def _prepare_graph_manifest(roster, paths: ManifestPaths) -> PreparedManifest | None:
    """Build the manifest from the knowledge-graph roster, or say no.

    Args:
        roster: The :class:`~osprey.channel_roster.records.RosterResult` the
            graph mode resolved to -- records, or the absence saying why there
            are none.
        paths: The data tree the per-tree sources are read from.

    Returns:
        The prepared manifest, or ``None`` when the corpus yields nothing (it
        is missing, unreadable, or declares no channels) or a per-tree source
        is absent. :func:`manifest_gap_reason` renders which, with the corpus
        named -- never the absent-paradigms wording, which would send an
        operator staging database files a graph project does not use.

    Raises:
        BuildProfileError: if the scenario seed or the machine-state list
            cannot be read -- the same rule as the paradigm path, for the same
            reason.
    """
    if roster.absence is not None:
        logger.warning(
            "Virtual-accelerator manifest not generated from the knowledge graph: %s",
            roster.absence.message(),
        )
        return None

    missing = _graph_missing_sources(paths)
    if missing:
        logger.debug(
            "Virtual-accelerator manifest not generated from %s: missing %s",
            paths.data_root,
            ", ".join(str(p.relative_to(paths.data_root)) for p in missing),
        )
        return None

    # The graph states each channel's address, direction and readback pairing
    # -- membership -- but no hierarchy path, and the identity keys are read
    # from nowhere else. So every entry is pathless: empty identity keys,
    # static-noisy partition, exactly what a tree without a hierarchical
    # database already gets. Nothing is invented to classify better than the
    # source can say; the build's fact states the cost. The manifest is a
    # namespace, so two bindings sharing one address are one channel: entries
    # are built from the unique address set, never once per record.
    entries = [
        _pathless_entry(address, noise=False)
        for address in sorted({record.address for record in roster.records})
    ]

    try:
        manifest = _finish_manifest(
            entries,
            paths,
            source_paradigms=[GRAPH_SOURCE_PARADIGM],
            absent_paradigms=[],
            corrupt_paradigms=[],
            source_corpus=roster.source.for_display(),
        )
    except (json.JSONDecodeError, KeyError, OSError) as exc:
        culprit = _first_unreadable_source(paths)
        detail = (
            f"{culprit} is not readable as JSON"
            if culprit is not None
            else f"{type(exc).__name__}: {exc}"
        )
        raise BuildProfileError(
            f"virtual-accelerator channel manifest could not be built from the "
            f"data tree {paths.data_root}: {detail}. Repair the file."
        ) from exc

    return PreparedManifest(
        manifest=manifest, limits_source=paths.channel_limits, source=ManifestSource.GRAPH
    )


@dataclass(frozen=True)
class PreparedManifest:
    """A manifest built from a project's own data tree, ready to be written.

    Holding the built manifest (rather than re-deriving it at write time) is
    what makes the build step atomic: every way generation can fail -- absent
    sources, disagreeing paradigm DBs, an unreadable authored file -- has
    already happened by the time a caller holds one of these, so the decision
    to wire ``VA_CHANNELS_FILE`` into the project's ``.env`` can be made before
    anything is written.

    Attributes:
        manifest: The manifest document, parsed.
        limits_source: The drive limits that ship beside it.
        source: Which of the three sources answered. The writer and the build's
            fact branch on this rather than on the manifest's ``_metadata``,
            which an authored manifest owns.
        manifest_path: The staged file the manifest was read from, for an
            authored one; ``None`` for a derived manifest, which exists only in
            memory until it is written.
    """

    manifest: dict
    limits_source: Path
    source: ManifestSource = ManifestSource.PARADIGM
    manifest_path: Path | None = None

    @property
    def channel_count(self) -> int:
        """How many channels the accelerator will serve.

        Read from ``_metadata`` when it states a count and from the channel
        list otherwise: a derived manifest always carries the census this
        package writes, while an authored one carries whatever its facility's
        generator wrote -- possibly no ``_metadata`` at all. The channels
        themselves are the one part every manifest must have.
        """
        metadata = self.manifest.get("_metadata") or {}
        total = metadata.get("total_channels")
        if isinstance(total, int):
            return total
        return len(self.manifest.get("channels", []))


def _authored_limits_source(paths: ManifestPaths) -> Path | None:
    """The drive limits an authored manifest is served under, or ``None``.

    Beside the manifest first -- a facility that writes its own simulation
    channel set writes the bounds for it in the same directory, and those are
    the ones the accelerator has to enforce. The tree-wide
    ``channel_limits.json`` is the fallback, so authoring a manifest without a
    limits file beside it still lands the project's limits in the render rather
    than none at all. ``None`` means the tree has neither, which is the silent
    unbounded-setpoint state every source path here refuses.
    """
    if paths.authored_limits.is_file():
        return paths.authored_limits
    if paths.channel_limits.is_file():
        return paths.channel_limits
    return None


def _read_authored_manifest(path: Path, data_root: Path) -> dict:
    """Read and check the manifest a profile staged, or refuse by name.

    Checked through :func:`loaders.load_manifest_file` -- the exact reader the
    container boots on -- so a file that would kill the accelerator at start is
    caught at build time instead, and by the same rules rather than by a second
    opinion written here. That reader takes a path and returns the channel
    list, so the document itself is read a second time; one extra parse of a
    file this build is about to copy anyway is cheaper than keeping a private
    copy of the schema in sync with the one the container enforces.

    A staged file that cannot be served is a REFUSAL, never a fall-through to
    derivation: deriving past it would quietly replace the facility's own
    channel set with one this package reconstructed, which is precisely what
    staging the file said not to do.

    Raises:
        BuildProfileError: if the file is not readable as a manifest, or reads
            as one that declares no channels.
    """
    relative = path.relative_to(data_root)
    remedy = (
        "Repair the file, or remove it so the manifest is derived from the rest of the data tree."
    )
    try:
        channels = loaders.load_manifest_file(path)
    except loaders.ManifestFileError as exc:
        raise BuildProfileError(
            f"the virtual-accelerator channel manifest this project stages at "
            f"{relative} could not be read: {exc}. {remedy}"
        ) from exc
    if not channels:
        raise BuildProfileError(
            f"the virtual-accelerator channel manifest this project stages at "
            f"{relative} declares no channels, so the accelerator has nothing of "
            f"the project's to serve. {remedy}"
        )
    document: dict = json.loads(path.read_text())
    return document


def _prepare_authored_manifest(paths: ManifestPaths) -> PreparedManifest | None:
    """Take the manifest the project staged, as staged.

    ``data/simulation/channel_manifest.json`` is a path the framework reserves
    for the profile, so a tree carrying one has already answered the question
    this module otherwise derives an answer to -- and it can answer it with
    things no derivation here can reconstruct: identity keys, setpoint/readback
    pairing, partitions a facility's own lattice knowledge assigned. Nothing is
    merged into it and nothing is recomputed from it: the file is the manifest.

    Returns:
        The prepared manifest, or ``None`` when the tree stages no authored
        manifest at all, or stages one with no drive limits anywhere to serve it
        under. :func:`manifest_gap_reason` renders which.

    Raises:
        BuildProfileError: if the staged manifest is present and cannot be read
            as one.
    """
    manifest_path = paths.authored_manifest
    if not manifest_path.is_file():
        return None

    manifest = _read_authored_manifest(manifest_path, paths.data_root)

    limits_source = _authored_limits_source(paths)
    if limits_source is None:
        logger.debug(
            "Virtual-accelerator manifest staged at %s not used: neither %s nor %s "
            "is present, and drive limits ship beside a manifest or the accelerator "
            "enforces none",
            manifest_path,
            paths.authored_limits,
            paths.channel_limits,
        )
        return None

    return PreparedManifest(
        manifest=manifest,
        limits_source=limits_source,
        source=ManifestSource.AUTHORED,
        manifest_path=manifest_path,
    )


def _authored_alternative(paths: ManifestPaths) -> str:
    """The clause naming the reserved path a tree could satisfy the build with.

    Appended to every refusal raised by a tree that stages no paradigm
    database, because for such a tree an authored manifest is the second way
    out and an operator reading only the first would never learn it exists.
    """
    return (
        f"; no manifest is staged at {paths.authored_manifest.relative_to(paths.data_root)} "
        f"either, which is the other way to give this accelerator a channel set"
    )


def prepare_project_manifest(
    data_root: Path, tier: int, *, config: dict | None = None
) -> PreparedManifest | None:
    """Build the VA channel manifest from the data tree a build is using.

    A project's accelerator simulates the project's own channels. Build time is
    the one stage where they are still on disk: the container mounts no channel
    databases, and the ``tiers/`` subtree is pruned from a built project. So
    whatever the tree stages is expanded here, or the caller has nothing to
    wire and the build refuses rather than letting the container fall back to
    the framework's built-in demo namespace.

    A tree that stages no paradigm database is not always a tree with no
    channels, so two more sources are consulted in order, most specific first:

    1. an AUTHORED ``data/simulation/channel_manifest.json`` -- a path the
       framework reserves for the profile -- taken as-is, with the
       ``channel_limits.json`` beside it as its limits (see
       :func:`_prepare_authored_manifest`);
    2. a graph-mode project's knowledge-graph corpus, when *config* is given
       and resolves the project's roster source to the graph (see
       :func:`_prepare_graph_manifest`).

    Staged paradigm databases always win over both, exactly as before. A
    ``config`` of ``None`` keeps every existing caller off the graph source; it
    does NOT hide an authored manifest, which is read off the tree and depends
    on no rendered configuration to be found.

    Args:
        data_root: The ``data/`` tree this build is sourcing from -- the
            profile's ``data:`` tree, or the bundle's own.
        tier: The build-resolved tier whose paradigm DBs to expand.
        config: The rendered project configuration, when the caller has one --
            what the roster resolves the graph corpus from. ``None`` skips the
            graph source entirely.

    Returns:
        The prepared manifest, or ``None`` when this tree cannot back one: it
        stages no channel source at all at this tier, the databases it stages
        name no channel, every one of them is present and unreadable, it is
        missing the scenario seed, the machine-state list or the drive limits,
        or the databases it does stage disagree. :func:`manifest_gap_reason`
        says which, in the words the refusal is written in. A caller deploying
        a virtual accelerator MUST refuse on ``None`` rather than continue.

        SOME of the staged databases being corrupt is not one of those cases:
        the manifest is built from the ones that are left, and ``_metadata``
        names the broken ones so the degrade is read rather than inferred from
        a channel count.

    Raises:
        BuildProfileError: if the scenario seed or the machine-state list --
            the sources a tree carries one of, rather than one per paradigm --
            cannot be read, or if an authored manifest is staged and cannot be
            read as one. There is nothing left to build from when any of them
            is broken, and reading past it would quietly serve a different
            channel set than the operator believes they are driving.
    """
    paths = ManifestPaths(data_root=data_root, tier=tier)
    if not paths.staged_paradigms:
        authored = _prepare_authored_manifest(paths)
        if authored is not None:
            return authored
        if config is not None:
            roster = _graph_roster(config)
            if roster is not None:
                return _prepare_graph_manifest(roster, paths)
        logger.debug(
            "Virtual-accelerator manifest not generated from %s: %s stages no "
            "paradigm channel database and no manifest is staged at %s",
            data_root,
            paths.tier_dir,
            paths.authored_manifest,
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

    return PreparedManifest(
        manifest=manifest, limits_source=paths.channel_limits, source=ManifestSource.PARADIGM
    )


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


def manifest_gap_reason(data_root: Path, tier: int, *, config: dict | None = None) -> str:
    """Say why :func:`prepare_project_manifest` could back no manifest.

    The generator answers ``None`` for several different trees, and the build
    refusing on it has to name which one an operator is looking at. Re-checking
    the tree here costs nothing on a path that has already failed, and keeps
    the wording in the module that owns the sources rather than in the caller.

    Args:
        data_root: The ``data/`` tree this build sourced from.
        tier: The build-resolved tier whose paradigm databases were expanded.
        config: The rendered project configuration, when the refusal is about a
            build that consulted the graph source -- pass the same value
            :func:`prepare_project_manifest` was given, so the reason describes
            the source that actually answered.

    Returns:
        A phrase naming the absent files, the present-but-unreadable ones, or
        the disagreement between the ones that read cleanly. A corrupt database
        is never reported as an absent one: the two send an operator to
        different work. A graph-mode tree gets the roster's own absence
        sentence -- the corpus named, with why it yielded nothing -- never the
        absent-paradigms wording, which would send its operator staging
        database files graph mode does not use. Every reason a tree staging no
        database gets also names the reserved manifest path, because staging
        one there is the other way to satisfy the build.
    """
    paths = ManifestPaths(data_root=data_root, tier=tier)
    if not paths.staged_paradigms:
        if paths.authored_manifest.is_file():
            # It is staged and readable -- otherwise the preparation raised
            # rather than answering None -- so the drive limits are what is
            # left to be missing.
            return "missing " + str(paths.authored_limits.relative_to(data_root))
        if config is not None:
            roster = _graph_roster(config)
            if roster is not None:
                if roster.absence is not None:
                    return roster.absence.message().rstrip(".") + _authored_alternative(paths)
                missing = _graph_missing_sources(paths)
                if missing:
                    return (
                        "missing "
                        + ", ".join(str(path.relative_to(data_root)) for path in missing)
                        + _authored_alternative(paths)
                    )
        return (
            f"no channel database is staged at tier {tier} "
            f"({', '.join(sorted(paths.absent_paradigms))} are all absent)"
            + _authored_alternative(paths)
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


def _copy_unless_same_file(source: Path, destination: Path) -> None:
    """Copy *source* onto *destination*, unless they are already one file.

    A build whose output zone IS the tree the manifest was read from -- the
    project's own ``data/`` -- would otherwise hand :func:`shutil.copy2` a file
    and itself, which raises. Nothing to do is the right answer there: the
    bytes are already where they are wanted.
    """
    if destination.exists() and destination.samefile(source):
        return
    shutil.copy2(source, destination)


def write_project_manifest(prepared: PreparedManifest, project_data_dir: Path) -> Path:
    """Write a prepared manifest and its drive limits into ``data/simulation/``.

    Both files land in the directory the container already bind-mounts, so
    neither needs a compose change: ``VA_CHANNELS_FILE`` resolves relative
    names against the data dir, and the entrypoint reads ``channel_limits.json``
    from beside it. The limits file is copied rather than bind-mounted
    single-file, which fails at container init.

    For a DERIVED manifest the limits copy is taken from the built project when
    it has one (so any facility overlay applied to ``data/channel_limits.json``
    is what the VA enforces), falling back to the source tree the manifest was
    prepared from.

    An AUTHORED manifest is exempt from that rule, in both halves. Its bytes
    are copied rather than re-serialized, so what the accelerator serves is the
    file the facility staged and reviewed; and its limits are the ones prepared
    with it -- the ``channel_limits.json`` staged beside it -- because the
    project's live limits file bounds the real machine, and a facility that
    authors a simulation channel set authors the bounds for it in the same
    place. Copying the live file over those would leave the accelerator
    enforcing limits its channel set was never written against.

    Returns:
        The path the manifest was written to.
    """
    simulation_dir = project_data_dir / "simulation"
    simulation_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = simulation_dir / MANIFEST_FILENAME

    if prepared.source is ManifestSource.AUTHORED and prepared.manifest_path is not None:
        _copy_unless_same_file(prepared.limits_source, simulation_dir / LIMITS_FILENAME)
        _copy_unless_same_file(prepared.manifest_path, manifest_path)
        return manifest_path

    limits_source = project_data_dir / LIMITS_FILENAME
    if not limits_source.is_file():
        limits_source = prepared.limits_source
    _copy_unless_same_file(limits_source, simulation_dir / LIMITS_FILENAME)

    manifest_path.write_text(json.dumps(prepared.manifest, indent=2) + "\n")
    return manifest_path


def main() -> None:
    """CLI entry point: (re)generate channel_manifest.json on disk."""
    manifest = build_manifest()
    MANIFEST_OUTPUT.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {manifest['_metadata']['total_channels']} channels to {MANIFEST_OUTPUT}")


if __name__ == "__main__":
    main()
