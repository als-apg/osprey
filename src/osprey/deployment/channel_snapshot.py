"""Build-time channel snapshot decision for web channel suggestions.

Web panels offer a typeahead over the control-system addresses a project knows
about. The addresses come from the project's Channel Finder database — or, on
the graph paradigm, from the Turtle corpus the build stages for the graph
store — and a browser can reach neither, so the build emits a static snapshot
of them next to the rendered service files instead.

This module answers the single question the build needs: *should a snapshot be
written, and what goes in it?* :func:`compute_channel_snapshot` returns one
:class:`SnapshotDecision`; every consumer reads that object rather than
re-deriving the predicate, so the compose fragment, the mount, and the file on
disk can never disagree about whether a snapshot exists.

The decision fails soft in almost every direction. A project that configures no
channel source at all, one whose source is empty, too large to be useful as a
typeahead, or unreadable — a database file that cannot be opened, a graph
corpus that is missing, unreadable or empty, or an rdflib that will not import
and so cannot parse one — gets no snapshot at all. The build itself is never
blocked, because a missing autocomplete list is a degraded panel, not a broken
deployment. The one exception is a ``pipeline_mode`` naming a paradigm that
does not exist: that is a configuration mistake rather than a degraded panel,
so it stops the build.

Path preconditions: a relative ``database.path`` is resolved against the process
working directory, which the build sets to the project root before generating
compose files. A relative ``services.graphdb.ttl_path`` is render-relative
instead, and resolves against the ``config_dir`` recorded by
:func:`~osprey.deployment.compose_generator.prepare_compose_files` — falling
back to the directory of the ``config.yml`` this process runs against, which is
``OSPREY_CONFIG`` when that is set, else ``build/config.yml`` under the working
directory when that file exists, else the working directory itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from osprey.utils.logger import get_logger

logger = get_logger("deployment.channel_snapshot")

#: Upper bound on snapshot size when the project does not set one. Well past any
#: real facility catalog, and small enough that the browser-side typeahead stays
#: responsive on the largest one anybody has pointed at OSPREY so far.
DEFAULT_MAX_CHANNELS = 50000

#: Named in the skip message so a project that trips the guard knows what to raise.
MAX_CHANNELS_CONFIG_KEY = "web.channel_suggestions.max_channels"

#: Named in the skip message when the feature is switched off explicitly.
ENABLED_CONFIG_KEY = "web.channel_suggestions.enabled"

#: The ``narad_p:fullPv`` predicate every ``ChannelBinding`` carries — the
#: control-system address of that binding, and the only thing the graph-mode
#: snapshot reads out of a Turtle corpus.
#:
#: ``services/facility_knowledge/seeder/ttl_seeder.py:48-58`` spells the same
#: IRI as ``_NARAD_P + "fullPv"``, and ``seeder/graph_seeder.py``'s
#: ``NARAD_PREFIXES`` is the prefix table of record for the ``narad_p:``
#: namespace. This is a plain string here deliberately rather than an import of
#: either module: importing anything from ``osprey.services.facility_knowledge``
#: executes that package's ``__init__``, which pulls ``okf/bundle.py`` and with
#: it ``osprey.services.qmd`` into the deployment import graph.
_FULL_PV_IRI = "https://narad.example.org/property/fullPv"


@dataclass(frozen=True)
class SnapshotDecision:
    """Whether to emit a channel snapshot, and what it contains.

    Attributes:
        emit: True when a snapshot file should be written.
        channels: Sorted, deduplicated control-system addresses. Empty unless
            ``emit`` is True — a decision not to emit carries no payload.
        count: How many distinct addresses the source yielded. Reported even
            when ``emit`` is False, so a caller can say why nothing was written.
        source_path: Resolved path of the database or corpus the addresses came
            from, or None when no source was configured.
    """

    emit: bool
    channels: list[str] = field(default_factory=list)
    count: int = 0
    source_path: Path | None = None


#: What one source branch hands back: the addresses it found and the path they
#: came from, or the no-emit decision that stands in their place. Each branch
#: owns its own fail-soft exits that way, leaving the caller only the guards
#: that apply to every source alike.
_SourcedChannels = tuple[list[str], Path] | SnapshotDecision


def _suggestions_section(config: dict) -> dict:
    """Read the ``web.channel_suggestions`` block, tolerating any shape.

    An absent block is not an opt-out: the feature is on by default, and the
    keys are only written explicitly by newer generated configs.

    Args:
        config: Full project configuration dictionary.

    Returns:
        The block as a dict, or an empty dict when it is absent or not a mapping.
    """
    web = config.get("web") or {}
    if not isinstance(web, dict):
        return {}
    section = web.get("channel_suggestions") or {}
    return section if isinstance(section, dict) else {}


def _max_channels(section: dict) -> int:
    """Resolve the size guard, falling back to the default on an unusable value."""
    raw = section.get("max_channels", DEFAULT_MAX_CHANNELS)
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning(
            f"Ignoring unusable {MAX_CHANNELS_CONFIG_KEY} value {raw!r}; "
            f"using the default of {DEFAULT_MAX_CHANNELS}."
        )
        return DEFAULT_MAX_CHANNELS


def _render_dir(config: dict) -> Path | None:
    """Directory a render-relative configured path is authored against.

    :func:`osprey.deployment.compose_generator.prepare_compose_files`
    (``compose_generator.py:2832``) records ``config_dir`` — the directory the
    loaded ``config.yml`` sits in — before any render helper runs, and that is
    what ``services.graphdb.ttl_path`` resolves against: the one render-relative
    key (see ``osprey/utils/config_paths.py:30-38``).

    Returning None lets :func:`~osprey.utils.config_paths.resolve_render_relative_path`
    fall back to the ``config.yml`` this process runs against — ``OSPREY_CONFIG``
    when set, else ``build/config.yml`` under the working directory when that
    file exists, else the working directory itself
    (:func:`osprey_connectors.workspace.resolve_config_path`).

    Deliberately NOT the ``project_root`` rung of
    :func:`~osprey.deployment.compose_generator._render_anchor_dir`:
    ``project_root`` is the repo root, which is the wrong anchor for a
    render-relative key.

    Args:
        config: Full project configuration dictionary.

    Returns:
        The recorded config directory, or None when the config carries none.
    """
    raw = config.get("config_dir")
    if isinstance(raw, str) and raw.strip():
        return Path(raw)
    return None


def _load_channel_records(pipeline_type: str, db_config: dict, db_path: Path) -> list[dict]:
    """Load a channel database and return its channel records.

    Constructing any of the database classes loads the file, so a bad path or
    malformed content surfaces here as an exception for the caller to absorb.
    The database classes are imported lazily to keep the deployment import graph
    free of the Channel Finder service.

    Args:
        pipeline_type: Pipeline name from ``detect_pipeline_config``.
        db_config: The pipeline's ``database`` block (``path``, and ``type`` for
            the in-context pipeline).
        db_path: Resolved path to the database file.

    Returns:
        Channel records, each carrying at least a ``channel`` key and usually an
        ``address``.

    Raises:
        PipelineModeError: If ``pipeline_type`` is not a known paradigm.
    """
    database: Any
    if pipeline_type == "in_context":
        from osprey.services.channel_finder.databases import (
            FlatChannelDatabase,
            TemplateChannelDatabase,
        )

        db_type = db_config.get("type", "template")
        if db_type == "flat":
            database = FlatChannelDatabase(str(db_path))
        else:
            database = TemplateChannelDatabase(str(db_path))
    elif pipeline_type == "hierarchical":
        from osprey.services.channel_finder.databases import HierarchicalChannelDatabase

        database = HierarchicalChannelDatabase(str(db_path))
    elif pipeline_type == "middle_layer":
        from osprey.services.channel_finder.databases import MiddleLayerDatabase

        database = MiddleLayerDatabase(str(db_path))
    else:
        from osprey.services.channel_finder.core.exceptions import PipelineModeError

        raise PipelineModeError(f"Unknown channel finder pipeline '{pipeline_type}'")

    records: list[dict] = database.get_all_channels()
    return records


def _graph_channels(config: dict) -> _SourcedChannels:
    """Read the graph paradigm's addresses out of the Turtle corpus it stages.

    The graph store is a disposable mirror of a Turtle corpus that sits on the
    build host under ``services.graphdb.ttl_path``, so the addresses are read
    out of that file. Nothing here dials the store: the corpus is the source of
    truth the deploy seeds it from.

    Args:
        config: Full project configuration dictionary.

    Returns:
        The sorted addresses and the resolved corpus path, or a no-emit
        decision: quietly for a project that configures no ``ttl_path``,
        because its graph store is external, and with a warning for a malformed
        ``services.graphdb`` block, a corpus that cannot be read, or an rdflib
        that will not import.
    """
    from osprey.deployment.graphdb_service import resolve_graphdb_service_config

    try:
        settings = resolve_graphdb_service_config(config)
    except ValueError as e:
        logger.warning(
            f"The services.graphdb block is malformed ({e}); not emitting a channel snapshot."
        )
        return SnapshotDecision(emit=False)

    if settings is None or settings.ttl_path is None:
        logger.debug(
            "No services.graphdb.ttl_path is configured; the graph corpus is what a "
            "channel snapshot would be derived from, so not emitting one."
        )
        return SnapshotDecision(emit=False)

    from osprey.utils.config_paths import resolve_render_relative_path

    ttl_path = resolve_render_relative_path(settings.ttl_path, _render_dir(config))

    try:
        from rdflib import Graph, Literal, URIRef
    except ImportError:
        logger.warning(
            f"rdflib is not importable, so the graph corpus at {ttl_path} cannot be read; "
            "not emitting a channel snapshot. rdflib is a core dependency, so this "
            "environment is incomplete — reinstall it with: "
            "pip install --upgrade osprey-framework."
        )
        return SnapshotDecision(emit=False, source_path=ttl_path)

    try:
        graph = Graph()
        # The format is forced rather than guessed from the extension, as both
        # other readers of this key force it — the deploy's n10s import and the
        # TTL seeder — and rdflib would otherwise hand a corpus named ``.rdf``
        # to its XML parser.
        graph.parse(str(ttl_path), format="turtle")
        channels = sorted(
            {
                str(o)
                for o in graph.objects(None, URIRef(_FULL_PV_IRI))
                if isinstance(o, Literal) and str(o)
            }
        )
    except Exception as e:
        logger.warning(
            f"Could not read the graph corpus at {ttl_path} ({e}); not emitting a channel snapshot."
        )
        return SnapshotDecision(emit=False, source_path=ttl_path)

    return channels, ttl_path


def _database_channels(pipeline_type: str | None, db_config: dict | None) -> _SourcedChannels:
    """Read a Channel Finder paradigm's addresses out of its database file.

    A database whose records name both an address and a channel keeps its
    ``address``; one that names only ``channel`` (the hierarchical and
    middle-layer pipelines synthesize an address from the channel name)
    contributes that instead.

    Args:
        pipeline_type: Pipeline name from ``detect_pipeline_config``, or falsy
            when the project configures no channel finder database.
        db_config: That pipeline's ``database`` block, or falsy likewise.

    Returns:
        The sorted addresses and the resolved database path, or a no-emit
        decision: quietly when no database is configured, and with a warning
        when the configured one cannot be read.
    """
    if not pipeline_type or not db_config:
        logger.debug("No channel finder database is configured; not emitting a channel snapshot.")
        return SnapshotDecision(emit=False)

    db_path = Path(db_config["path"])
    if not db_path.is_absolute():
        db_path = Path.cwd() / db_path

    try:
        records = _load_channel_records(pipeline_type, db_config, db_path)
        addresses = {record.get("address", record["channel"]) for record in records}
        channels = sorted(address for address in addresses if address)
    except Exception as e:
        logger.warning(
            f"Could not read the channel database at {db_path} ({e}); "
            "not emitting a channel snapshot."
        )
        return SnapshotDecision(emit=False, source_path=db_path)

    return channels, db_path


def compute_channel_snapshot(config: dict) -> SnapshotDecision:
    """Decide whether the build should emit a channel snapshot, and with what.

    A snapshot is emitted when the project configures a channel source — a
    Channel Finder database, or the Turtle corpus of the graph paradigm —
    ``web.channel_suggestions.enabled`` is not switched off, that source is
    readable, and it holds at least one and at most
    ``web.channel_suggestions.max_channels`` distinct addresses.

    The addresses — not the human-facing channel names — are what a panel writes
    into a control-system request, so those are what the snapshot carries. A
    database whose records name both keeps its ``address``; one that names only
    ``channel`` (the hierarchical and middle-layer pipelines synthesize an
    address from the channel name) contributes that instead.

    On the graph paradigm the addresses come from the corpus named by
    ``services.graphdb.ttl_path``, which the build stages for the graph store.
    That file is parsed as Turtle — the format is forced, exactly as the
    deploy's import of the same corpus forces it — and every ``narad_p:fullPv``
    literal in it is one address. The feature switch, the emptiness check and
    the size guard then apply as they do to a database file. A project that
    configures no ``ttl_path``, because its graph store is external, gets no
    snapshot quietly; a corpus that cannot be read, a malformed
    ``services.graphdb`` block, or an rdflib that will not import degrades to no
    snapshot with a warning.

    Args:
        config: Full project configuration dictionary, as the build already
            holds it.

    Returns:
        The decision. An unreadable or malformed source is logged as a warning
        and yields ``emit=False`` rather than raising.

    Raises:
        PipelineModeError: If ``channel_finder.pipeline_mode`` names a paradigm
            that does not exist.
    """
    section = _suggestions_section(config)

    if not section.get("enabled", True):
        logger.debug(f"{ENABLED_CONFIG_KEY} is off; not emitting a channel snapshot.")
        return SnapshotDecision(emit=False)

    from osprey.services.channel_finder.utils.detection import detect_pipeline_config

    pipeline_type, db_config = detect_pipeline_config(config)

    sourced = (
        _graph_channels(config)
        if pipeline_type == "graph"
        else _database_channels(pipeline_type, db_config)
    )
    if isinstance(sourced, SnapshotDecision):
        return sourced

    channels, source_path = sourced
    count = len(channels)

    if count == 0:
        # An empty snapshot is not a smaller suggestion list, it is a typeahead
        # that never suggests anything — so there is nothing worth writing.
        logger.debug(
            f"The channel source at {source_path} holds no channels; "
            "not emitting a channel snapshot."
        )
        return SnapshotDecision(emit=False, source_path=source_path)

    max_channels = _max_channels(section)
    if count > max_channels:
        logger.warning(
            f"The channel source at {source_path} holds {count} channels, above the "
            f"{MAX_CHANNELS_CONFIG_KEY} limit of {max_channels}; not emitting a channel "
            "snapshot. Raise that limit to include it."
        )
        return SnapshotDecision(emit=False, count=count, source_path=source_path)

    logger.debug(f"Channel snapshot: {count} channels from {source_path}.")
    return SnapshotDecision(emit=True, channels=channels, count=count, source_path=source_path)
