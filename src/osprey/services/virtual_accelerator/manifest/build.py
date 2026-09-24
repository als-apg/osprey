"""Builds the unified virtual-accelerator channel manifest.

Expands whichever paradigm channel DBs a tree stages at its build-resolved
tier, verifies the staged ones agree (the point of having several formats is
that they describe the same namespace), unions in the scenario-seed
``machine.json`` channels, reconciles the (currently broken) machine-state
template against the result, and classifies every address into a manifest
partition plus an EPICS record type.

The partition is read off the tree's own files, never off the address text.
``simulation/va_bindings.json`` is the authority for the pyat-coupled
partition: the addresses it binds are the ones a write steers the beam with,
and a tree carrying no bindings couples nothing. What is left is sp-echo where
the tree states a setpoint's readback -- the SP/RB halves of one device field
in a hierarchical database, the read-voted sibling of a write-voted field in a
middle-layer one -- and static-noisy everywhere else. One rule for every
facility: a partition keyed on ring, system or family names would be one
facility's naming convention masquerading as a physics fact.

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
authority), and ``_metadata`` names the corpus as the source. Staged paradigm
databases always win: the graph is consulted only when the tree stages none.

Run as a script to (re)generate ``channel_manifest.json``::

    uv run python -m osprey.services.virtual_accelerator.manifest.build
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Callable, Collection, Container, Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from osprey.errors import BuildProfileError

from ..bindings import BindingsDocument, load_bindings
from . import classify, loaders
from .classify import READBACK_SUBFIELD, SETPOINT_SUBFIELD
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
        hierarchy_levels: The level names that database DECLARES, in tree
            order; empty when it did not load or the tree stages none.
        corrupt: The staged databases that are present and could not be read,
            in read order. Each contributed zero addresses.
        middle_layer_channels: The middle-layer database's channel records, in
            read order; empty when it did not load or the tree stages none.
            They carry the signal group (system, family, field path) and the
            field metadata the read/write vote and the sp-echo pairing are
            decided from on a tree with no hierarchy path.
    """

    addresses: dict[str, set[str]]
    hierarchy_paths: dict[str, dict[str, str]]
    corrupt: tuple[CorruptParadigm, ...]
    hierarchy_levels: tuple[str, ...] = ()
    middle_layer_channels: tuple[dict, ...] = ()


@dataclass(frozen=True)
class _Expanded:
    """What one paradigm loader answered, in the shape the caller merges.

    Every paradigm answers this same shape so the loop over the staged ones
    stays one loop: the fields a given format cannot state are simply empty.
    """

    addresses: set[str]
    hierarchy_paths: dict[str, dict[str, str]]
    hierarchy_levels: tuple[str, ...] = ()
    middle_layer_channels: tuple[dict, ...] = ()


def _hierarchical_expansion(paths: ManifestPaths) -> _Expanded:
    """Expand the hierarchical database into its addresses, paths and levels.

    It is the one paradigm that declares a hierarchy path, and it is read once
    here so the manifest never re-opens a file this expansion already found
    unreadable.
    """
    channels, levels = loaders.load_hierarchical_database(paths)
    by_address = {c.address: c.path for c in channels}
    return _Expanded(addresses=set(by_address), hierarchy_paths=by_address, hierarchy_levels=levels)


def _middle_layer_expansion(paths: ManifestPaths) -> _Expanded:
    """Expand the middle-layer database into its addresses and its records.

    The records are kept, not just the addresses: on a tree that stages no
    hierarchical database they are the only statement of which addresses are
    the read and write halves of one device field, and of which addresses are
    measurements rather than setpoints.
    """
    channels = loaders.load_middle_layer_channels(paths)
    return _Expanded(
        addresses={channel["address"] for channel in channels},
        hierarchy_paths={},
        middle_layer_channels=tuple(channels),
    )


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
        "in_context": lambda: _Expanded(loaders.load_in_context_addresses(paths), {}),
        "middle_layer": lambda: _middle_layer_expansion(paths),
    }
    addresses: dict[str, set[str]] = {}
    hierarchy_paths: dict[str, dict[str, str]] = {}
    hierarchy_levels: tuple[str, ...] = ()
    middle_layer_channels: tuple[dict, ...] = ()
    corrupt: list[CorruptParadigm] = []
    for name in paths.staged_paradigms:
        try:
            expansion = loader_by_paradigm[name]()
        except Exception as exc:
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
        addresses[name] = expansion.addresses
        hierarchy_paths.update(expansion.hierarchy_paths)
        if expansion.hierarchy_levels:
            hierarchy_levels = expansion.hierarchy_levels
        if expansion.middle_layer_channels:
            middle_layer_channels = expansion.middle_layer_channels
    return ParadigmExpansion(
        addresses=addresses,
        hierarchy_paths=hierarchy_paths,
        corrupt=tuple(corrupt),
        hierarchy_levels=hierarchy_levels,
        middle_layer_channels=middle_layer_channels,
    )


# --- the partition rule --------------------------------------------------

#: The identity keys a setpoint and its readback are paired on: every
#: hierarchy level but the subfield, which is the one level they differ in.
#: Read off the single list of level names rather than respelled, so this and
#: ``serving/pvdb._channel_key`` -- which pairs the SERVED records on the same
#: five keys -- cannot drift apart.
_PAIR_LEVELS: tuple[str, ...] = classify.CLASSIFIER_LEVELS[:-1]

#: The direction vote that says an address is measured rather than written.
#: A measured address is the one a simulated machine adds noise to; a written
#: one reports back what it was given.
_READ_VOTE = "read"

#: What ``_metadata.partition_source`` says when nothing claimed the
#: pyat-coupled partition, because the tree carries no bindings document.
PARTITION_SOURCE_NONE = "none"


def _read_votes(channels: Collection[dict]) -> dict[str, str | None]:
    """The read/write direction voted for each middle-layer address.

    One vote per address, decided from the field it was read under by the same
    rule the MML mapping's own voter applies
    (:func:`osprey.services.mml.directions.field_vote`): the field's
    ``MemberOf`` tags first, then its name's suffix, then undecided. Reaching
    that function rather than restating the rule is the point -- a second copy
    of it here would let the manifest and the mapping disagree about which
    half of a device field an address is.

    Imported inside the call, and only where there is a vote to cast: the
    virtual-accelerator container imports this module and reads no MML export,
    so it never pays for that package's import graph.
    """
    if not channels:
        return {}

    from osprey.services.mml.directions import field_vote

    votes: dict[str, str | None] = {}
    for channel in channels:
        direction, _ = field_vote(channel["field"], channel.get("MemberOf"))
        votes[channel["address"]] = direction
    return votes


def _middle_layer_pairs(
    channels: Iterable[dict], votes: Mapping[str, str | None]
) -> dict[str, str]:
    """Setpoint to readback, for every pair a middle-layer database states.

    A middle-layer database groups its addresses by signal -- one list per
    ``(system, family, field path)``, one position per device -- so a
    write-voted group and a read-voted group of the same family that measure
    the same quantity are the read and write halves of one device field, and
    the pair is taken at each device position the two groups share. The
    position, not the place in the surviving list: a field that leaves a
    device blank, or names one address twice, keeps addresses whose places
    have shifted against its sibling field's.

    The same QUANTITY, not merely the same family, because a family may state
    a second write-voted field in other units beside its setpoint, or a
    boolean control beside it; the hardware units and the device count tell
    those apart, and a family name cannot. A group stating no hardware units
    states no quantity and pairs with nothing -- an absent unit is the absence
    of a statement, not a statement of equality. A group with no single such
    partner is left unpaired rather than guessed at, which serves it
    static-noisy: a write echoing onto a readback measuring something else
    would be a reading the facility never claimed.
    """
    groups: dict[tuple, dict[int, str]] = {}
    quantity: dict[tuple, tuple] = {}
    for channel in channels:
        slot = channel.get("slot")
        if slot is None:
            continue
        key = (
            channel["system"],
            channel["family"],
            channel["field"],
            tuple(channel["subfield"] or ()),
        )
        # One address per device position: an address listed at a position
        # another address already holds names no further device.
        groups.setdefault(key, {}).setdefault(slot, channel["address"])
        quantity.setdefault(key, (channel.get("HWUnits"), channel.get("Units")))

    by_family: dict[tuple, list[tuple]] = {}
    for key in groups:
        by_family.setdefault(key[:2], []).append(key)

    def vote(key: tuple) -> str | None:
        # Constant within a group: every address under one field was voted
        # from that field's own name and tags.
        return votes.get(next(iter(groups[key].values())))

    def stated_quantity(key: tuple) -> tuple | None:
        """What a group measures, or ``None`` where it states no units."""
        hw_units, units = quantity[key]
        return (hw_units, units) if hw_units else None

    pairs: dict[str, str] = {}
    for family_keys in by_family.values():
        reads = [key for key in family_keys if vote(key) == _READ_VOTE]
        for write_key in family_keys:
            if vote(write_key) != "write":
                continue
            measured = stated_quantity(write_key)
            if measured is None:
                continue
            candidates = [
                read_key
                for read_key in reads
                if stated_quantity(read_key) == measured
                and len(groups[read_key]) == len(groups[write_key])
            ]
            if len(candidates) != 1:
                continue
            readbacks = groups[candidates[0]]
            for slot, setpoint in groups[write_key].items():
                readback = readbacks.get(slot)
                # A family whose two halves are one address states a readback
                # that IS the setpoint; there is no second channel to echo
                # into, so it is not a pair.
                if readback is not None and readback != setpoint:
                    pairs[setpoint] = readback
    return pairs


def _hierarchical_pairs(path_by_address: Mapping[str, dict[str, str]]) -> dict[str, str]:
    """Setpoint to readback, for every pair a hierarchy path states.

    The two halves of one device field differ in exactly one level, the
    subfield, so a pair is the reserved ``SP``/``RB`` vocabulary found on two
    paths that agree on every other level -- the key the serving layer pairs
    its own records on.
    """
    halves: dict[tuple[str, ...], dict[str, str]] = {}
    for address, path in path_by_address.items():
        key = tuple(path[level] for level in _PAIR_LEVELS)
        halves.setdefault(key, {})[path["subfield"]] = address
    return {
        group[SETPOINT_SUBFIELD]: group[READBACK_SUBFIELD]
        for group in halves.values()
        if SETPOINT_SUBFIELD in group
        and READBACK_SUBFIELD in group
        and group[SETPOINT_SUBFIELD] != group[READBACK_SUBFIELD]
    }


def _unambiguous_pairs(stated: Mapping[str, str], addresses: Container[str]) -> dict[str, str]:
    """Keep the stated pairs that identify exactly one pair each.

    The manifest is a namespace, so two records naming one address are one
    channel, and a pair is taken only when nothing else claims either half. A
    pair is dropped -- both halves served static-noisy instead -- when the
    source states it ambiguously: a readback claimed by two setpoints, a
    setpoint that is itself another pair's readback, or a readback that is
    itself a setpoint. Shared by every source that states pairs, so none of
    them can be laxer than another about it.
    """
    claimed: dict[str, list[str]] = {}
    for setpoint, readback in stated.items():
        claimed.setdefault(readback, []).append(setpoint)
    return {
        setpoint: readback
        for setpoint, readback in stated.items()
        if readback in addresses
        and len(claimed[readback]) == 1
        and setpoint not in claimed
        and readback not in stated
    }


def _bound_entries(
    document: BindingsDocument, record: Callable[[str], tuple[str, bool]]
) -> dict[str, ManifestEntry]:
    """One entry per address the bindings document claims, keyed by address.

    These are the pyat-coupled partition, and the document is the only thing
    that puts an address in it. Each entry carries the pair-key shape: the
    identity keys are empty except ``device``, which holds the binding's own
    setpoint address, because the bindings -- not the address text and not a
    hierarchy path -- are what says these two addresses are one device's two
    halves. A written binding emits its setpoint as ``SP`` and its readback as
    ``RB``; one that serves both on a single address emits that address alone
    as ``SP``; a monitor emits its own address under the transverse axis it
    reads, ``X`` or ``Y``.

    Every entry is analog: a bound address carries a number by construction,
    whatever a channel database's record-type grammar would have said about
    its name. The noise flag still comes from the tree's own record-type rule
    (*record*), so a measured address jitters and a written one does not.

    Args:
        document: The tree's bindings.
        record: The record type and noise flag for one address, as the tree's
            own sources answer it.
    """
    entries: dict[str, ManifestEntry] = {}
    for binding in document.bindings:
        setpoint = binding.setpoint_address
        if binding.kind == "monitor":
            halves = {setpoint: (binding.attribute or "").upper()}
        elif binding.readback_address is None:
            halves = {setpoint: SETPOINT_SUBFIELD}
        else:
            halves = {
                setpoint: SETPOINT_SUBFIELD,
                binding.readback_address: READBACK_SUBFIELD,
            }
        for address, subfield in halves.items():
            entries[address] = _echo_entry(
                address,
                pair_key=setpoint,
                subfield=subfield,
                partition=classify.PARTITION_PYAT_COUPLED,
                noise=record(address)[1],
            )
    return entries


def _path_entry(
    address: str,
    path: dict[str, str],
    *,
    partition: str,
    record_type: str,
    noise: bool,
) -> ManifestEntry:
    """One manifest entry for an address whose hierarchy path is known."""
    return ManifestEntry(
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
    hierarchy path, so without it every channel carries empty identity keys;
    the setpoint/readback pairing those keys drive is then read off the
    middle-layer database's own signal groups instead, and a tree staging
    neither pairs nothing. ``_metadata`` names the databases that fed the
    manifest, the ones that were absent, and the ones that were staged and
    could not be read, so a reader of the manifest alone can see which case
    they are in.

    The pyat-coupled partition is the tree's ``simulation/va_bindings.json``
    and nothing else: a tree carrying no bindings serves no channel from a
    lattice model, whatever its addresses are called.

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
        bindings.BindingsError: if the tree carries a bindings document that
            breaks the schema. It names the file and the key, and the manifest
            is not built from a document whose meaning is in doubt.
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

    # A hierarchical database names its own levels, and the classifier reads
    # six particular names off each path. A tree levelled some other way
    # (system/family/sector/device/pv, say) states a hierarchy no rule here can
    # be evaluated against -- so nothing is classified, the reason is recorded,
    # and the build reports it. Guessing would put a facility's channels in
    # partitions derived from another facility's tokens; the KeyError this
    # replaces called a valid file unreadable.
    unclassified_reason: str | None = None
    missing_levels = [
        level for level in classify.CLASSIFIER_LEVELS if level not in expansion.hierarchy_levels
    ]
    if expansion.hierarchy_levels and missing_levels:
        unclassified_reason = (
            f"levels {'/'.join(expansion.hierarchy_levels)} lack {', '.join(missing_levels)}"
        )
        path_by_address = {}

    # The record type and the noise flag are the tree's own grammar, and they
    # are decided per address independently of the partition. A hierarchy path
    # states the record shape directly (a status flag is boolean and does not
    # jitter); a middle-layer database states no record shape at all, so every
    # address is analog and only the read/write vote is left to say whether it
    # is measured -- and therefore noisy -- or written.
    votes = _read_votes(expansion.middle_layer_channels)

    def record_for(address: str) -> tuple[str, bool]:
        path = path_by_address.get(address)
        if path is not None:
            return classify.derive_record_type(path)
        return classify.RECORD_TYPE_ANALOG, votes.get(address) == _READ_VOTE

    # The bindings document is the only thing that claims the pyat-coupled
    # partition, and it claims its addresses whether or not a channel database
    # named them: the manifest's coupled entries ARE the bindings file, which
    # is the equality the deployed model is built against.
    document = load_bindings(paths.va_bindings) if paths.va_bindings.is_file() else None
    bound = _bound_entries(document, record_for) if document is not None else {}

    # What is left pairs the way the tree states pairs: a hierarchy path says
    # it with the reserved SP/RB subfields, a middle-layer database with a
    # read-voted sibling group. A pair with a bound half is not an echo pair --
    # the bindings already gave that address a model to answer from.
    stated = (
        _hierarchical_pairs(path_by_address)
        if path_by_address
        else _middle_layer_pairs(expansion.middle_layer_channels, votes)
    )
    stated = {
        setpoint: readback
        for setpoint, readback in stated.items()
        if setpoint not in bound and readback not in bound
    }
    pairs = _unambiguous_pairs(stated, addresses)
    echo_readbacks = set(pairs.values())

    entries: list[ManifestEntry] = list(bound.values())
    for address in sorted(addresses):
        if address in bound or address in echo_readbacks:
            # Claimed by the bindings, or emitted below beside its setpoint.
            continue
        readback = pairs.get(address)
        partition = (
            classify.PARTITION_SP_ECHO if readback is not None else classify.PARTITION_STATIC_NOISY
        )
        record_type, noise = record_for(address)
        path = path_by_address.get(address)
        if path is not None:
            entries.append(
                _path_entry(
                    address, path, partition=partition, record_type=record_type, noise=noise
                )
            )
            if readback is not None:
                readback_type, readback_noise = record_for(readback)
                entries.append(
                    _path_entry(
                        readback,
                        path_by_address[readback],
                        partition=partition,
                        record_type=readback_type,
                        noise=readback_noise,
                    )
                )
        elif readback is not None:
            entries.append(
                _echo_entry(address, pair_key=address, subfield=SETPOINT_SUBFIELD, noise=noise)
            )
            entries.append(
                _echo_entry(
                    readback,
                    pair_key=address,
                    subfield=READBACK_SUBFIELD,
                    noise=record_for(readback)[1],
                )
            )
        else:
            entries.append(_pathless_entry(address, noise=noise))

    return _finish_manifest(
        entries,
        paths,
        source_paradigms=list(expansion.addresses),
        absent_paradigms=list(paths.absent_paradigms),
        unclassified_reason=unclassified_reason,
        partition_source=(
            str(paths.va_bindings.relative_to(paths.data_root))
            if document is not None
            else PARTITION_SOURCE_NONE
        ),
        bindings_novel_addresses=sorted(set(bound) - addresses),
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
    partition_source: str = PARTITION_SOURCE_NONE,
    bindings_novel_addresses: list[str] | None = None,
    source_corpus: str | None = None,
    unclassified_reason: str | None = None,
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
        partition_source: What claimed the pyat-coupled partition -- the
            tree-relative path of the bindings document that did, or
            :data:`PARTITION_SOURCE_NONE` for a tree carrying none, whose
            channels are all sp-echo or static-noisy. Always recorded, so a
            reader of the manifest alone can tell an accelerator with no model
            behind it from one whose model file went missing.
        bindings_novel_addresses: Addresses the bindings claim that no channel
            database named. Normally empty -- one emit run writes both files --
            and recorded when it is not, because those channels reach the
            served namespace from the bindings alone.
        source_corpus: The knowledge-graph source the entries came from (the
            channel search index built from the corpus), as an operator would
            name it, or ``None`` for a database-sourced manifest -- the key
            appears only when the graph actually fed it.
        unclassified_reason: Why no entry could be classified, or ``None``
            when classification ran. The key appears only when it happened,
            and the build's census report prints it.
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
        if e.subfield == SETPOINT_SUBFIELD:
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
        "partition_source": partition_source,
    }
    if bindings_novel_addresses:
        metadata["bindings_novel_addresses"] = bindings_novel_addresses
    if source_corpus is not None:
        metadata["source_corpus"] = source_corpus
    if unclassified_reason is not None:
        metadata["unclassified_reason"] = unclassified_reason
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
    # Graph mode with no readable corpus is still graph mode: both absences
    # come back so the refusal can name the corpus keys, or the broken line.
    graph_configured = (
        resolution.source is not None and resolution.source.kind is RosterSourceKind.GRAPH
    ) or (
        resolution.absence is not None
        and resolution.absence.reason
        in (RosterAbsenceReason.GRAPH_NO_TTL, RosterAbsenceReason.GRAPH_MALFORMED)
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


def _echo_entry(
    address: str,
    *,
    pair_key: str,
    subfield: str,
    partition: str = classify.PARTITION_SP_ECHO,
    noise: bool = False,
) -> ManifestEntry:
    """One half of a pair whose pair key is the setpoint's own address.

    The container pairs a setpoint with its readback on the five identity keys
    ``(ring, system, family, device, field)``, differing only in subfield
    (``serving/pvdb._channel_key``). Where the source states the pair itself
    rather than a hierarchy path -- the knowledge-graph roster, a
    middle-layer database, and the bindings document, which pairs addresses no
    path could be trusted to agree with -- the pair is keyed on the one thing
    that identifies it, the setpoint's own address carried in ``device``, and
    the other four keys stay empty, exactly as on a pathless entry. That is
    enough for the IOC to pair the two halves, which is all either partition
    needs of the keys.
    """
    return ManifestEntry(
        address=address,
        ring="",
        system="",
        family="",
        device=pair_key,
        field="",
        subfield=subfield,
        partition=partition,
        record_type=classify.RECORD_TYPE_ANALOG,
        noise=noise,
    )


def _graph_entries(records) -> list[ManifestEntry]:
    """Manifest entries for a knowledge-graph roster.

    The graph states each channel's address, direction and -- where the
    corpus groups a setpoint with the readback reporting it -- its readback.
    Every stated pair becomes a setpoint-echo pair (:func:`_echo_entry`): a
    write to the setpoint echoes onto the readback, physics-free, which is
    what a plan driving that setpoint against the accelerator needs to observe
    its own write. Everything else is pathless and static-noisy: the graph
    carries no hierarchy path, the identity keys are read from nowhere else,
    and nothing is invented to classify better than the source can say; the
    build's fact states the cost.

    The manifest is a namespace, so two records sharing one address are one
    channel, and a readback is emitted once, beside its setpoint. A pair is
    dropped -- both halves served static-noisy instead -- when the corpus
    states it ambiguously: a readback claimed by two setpoints, a setpoint that
    is itself another pair's readback, or a readback that is itself a
    setpoint.

    Args:
        records: The roster's records, in source order.
    """
    stated: dict[str, str] = {}
    for record in records:
        if record.readback is not None:
            stated.setdefault(record.address, record.readback)
    addresses = {record.address for record in records}
    pairs = _unambiguous_pairs(stated, addresses)
    readbacks = set(pairs.values())

    entries: list[ManifestEntry] = []
    for address in sorted(addresses):
        if address in readbacks:
            continue
        readback = pairs.get(address)
        if readback is None:
            entries.append(_pathless_entry(address, noise=False))
            continue
        entries.append(_echo_entry(address, pair_key=address, subfield=SETPOINT_SUBFIELD))
        entries.append(_echo_entry(readback, pair_key=address, subfield=READBACK_SUBFIELD))
    return entries


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

    entries = _graph_entries(roster.records)

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

    return PreparedManifest(manifest=manifest, limits_source=paths.channel_limits)


@dataclass(frozen=True)
class PreparedManifest:
    """A manifest built from a project's own data tree, ready to be written.

    Holding the built manifest (rather than re-deriving it at write time) is
    what makes the build step atomic: every way generation can fail -- absent
    sources, disagreeing paradigm DBs -- has already happened by the time a
    caller holds one of these, so the decision to wire ``VA_CHANNELS_FILE``
    into the project's ``.env`` can be made before anything is written.

    ``model_sources`` are the files describing the simulated accelerator
    itself -- the lattice and the bindings that tie channels to it -- which
    ship beside the manifest so the container finds the whole model under the
    one directory it mounts. It is empty for a tree that serves no virtual
    accelerator.
    """

    manifest: dict
    limits_source: Path
    model_sources: tuple[Path, ...] = ()


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
    channels: a graph-mode project's channels live in its knowledge-graph
    corpus. When *config* is given and resolves the project's roster source to
    the graph, such a tree gets its manifest from the roster's graph reader
    instead (see :func:`_prepare_graph_manifest`). Staged paradigm databases
    always win over the graph, exactly as before; a ``config`` of ``None``
    keeps every existing caller on the paradigm-database rules alone.

    Args:
        data_root: The ``data/`` tree this build is sourcing from -- the
            profile's ``data:`` tree, or the bundle's own.
        tier: The build-resolved tier whose paradigm DBs to expand.
        config: The rendered project configuration, when the caller has one --
            what the roster resolves the graph corpus from. ``None`` skips the
            graph source entirely.

    Returns:
        The prepared manifest, or ``None`` when this tree cannot back one: it
        stages no paradigm database at this tier, the databases it stages name
        no channel, every one of them is present and unreadable, it is missing
        the scenario seed, the machine-state list, the drive limits or the
        lattice its bindings point into, or the databases it does stage
        disagree. :func:`manifest_gap_reason` says
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
        if config is not None:
            roster = _graph_roster(config)
            if roster is not None:
                return _prepare_graph_manifest(roster, paths)
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

    model_files = (paths.lattice_json, paths.va_bindings)
    return PreparedManifest(
        manifest=manifest,
        limits_source=paths.channel_limits,
        # Read off the required sources rather than re-deciding which model
        # files a tree carries, so what ships beside the manifest is exactly
        # what the build just refused to go without.
        model_sources=tuple(path for path in paths.required_sources if path in model_files),
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
        database files graph mode does not use.
    """
    paths = ManifestPaths(data_root=data_root, tier=tier)
    if not paths.staged_paradigms:
        if config is not None:
            roster = _graph_roster(config)
            if roster is not None:
                if roster.absence is not None:
                    return roster.absence.message().rstrip(".")
                missing = _graph_missing_sources(paths)
                if missing:
                    return "missing " + ", ".join(
                        str(path.relative_to(data_root)) for path in missing
                    )
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
    """Write a prepared manifest, its drive limits and its model into ``data/simulation/``.

    Every file lands in the directory the container already bind-mounts, so
    none of them needs a compose change: ``VA_CHANNELS_FILE`` resolves relative
    names against the data dir, and the entrypoint reads ``channel_limits.json``
    from beside it. The limits file is copied rather than bind-mounted
    single-file, which fails at container init.

    The limits copy is taken from the built project when it has one (so any
    facility overlay applied to ``data/channel_limits.json`` is what the VA
    enforces), falling back to the source tree the manifest was prepared
    from.

    The lattice and the bindings are copied byte for byte: the digest recorded
    for the lattice when it was emitted is the digest of these bytes, so
    anything that re-serialises them describes a different ring than the one
    the provenance names.

    Returns:
        The path the manifest was written to.
    """
    simulation_dir = project_data_dir / "simulation"
    simulation_dir.mkdir(parents=True, exist_ok=True)

    limits_source = project_data_dir / LIMITS_FILENAME
    if not limits_source.is_file():
        limits_source = prepared.limits_source
    shutil.copy2(limits_source, simulation_dir / LIMITS_FILENAME)

    for source in prepared.model_sources:
        destination = simulation_dir / source.name
        # A tree built in place already holds its model where the container
        # reads it, and copying a file onto itself is the one copy that fails.
        if destination.exists() and destination.samefile(source):
            continue
        shutil.copy2(source, destination)

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
