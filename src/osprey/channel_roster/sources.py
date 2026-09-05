"""Which source a build's channel roster is enumerated from, and where it sits.

One question, answered once: *given this project's configuration, what does the
roster read?* The answer is settled by :func:`detect_pipeline_config` and
nothing else --- the graph paradigm reads the search index the build derives
from the Turtle corpus it stages for the graph store, every other paradigm
reads its channel-finder database file, and a project that configures neither
gets an honest
:class:`~osprey.channel_roster.records.RosterAbsence` rather than a silent
empty roster.

The mode decides, never the artifacts. A hierarchical project that also runs a
graph store still resolves its database: probing ``services.graphdb.ttl_path``
independently of the mode is how a facility ends up with two disagreeing
enumerations of its own channels, which is the whole reason this package
exists. For the same reason the graph store is never dialed here: the files on
disk are what the deploy builds from the project's own configuration, so a
graph project naming neither a corpus nor an index is an absence that names the
config keys that would have declared one, not a network probe.

This module also owns the path rule the roster needs, moved here so the
snapshot, the plan-device derivation and the build's fact lines resolve one
configured string to one file: :func:`resolve_database_path` anchors
``database.path`` on the process working directory, which the build sets to the
project root before generating service files. The graph paradigm's index is
render-relative instead, and
:func:`~osprey.deployment.graphdb_service.resolve_graph_index_path` is the one
resolver for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osprey.channel_roster.records import (
    RosterAbsence,
    RosterAbsenceReason,
    RosterSource,
    RosterSourceKind,
)
from osprey.deployment.graphdb_service import GRAPHDB_INDEX_PATH_CONFIG_KEY
from osprey.utils.logger import get_logger

logger = get_logger("channel_roster.sources")

#: The paradigm name :func:`detect_pipeline_config` answers on the mode alone,
#: because a graph store is a service rather than a database file.
GRAPH_PARADIGM = "graph"

#: The keys a graph-mode project declares its roster with, named in the absence
#: it gets when it declares none of them. ``ttl_path`` is the corpus this build
#: stages; ``index_path`` is where the build writes the search index derived
#: from it, which is what the roster actually reads; ``uri`` is how a project
#: points at a store it does not run --- and a store is not enumerable from the
#: build host, so naming it is the remedy for an operator who expected one, not
#: a source.
GRAPH_CORPUS_CONFIG_KEYS: tuple[str, ...] = (
    "services.graphdb.ttl_path",
    GRAPHDB_INDEX_PATH_CONFIG_KEY,
    "services.graphdb.uri",
)


@dataclass(frozen=True, slots=True)
class RosterSourceResolution:
    """What this build's roster reads, or the reason it reads nothing.

    Attributes:
        source: The resolved source, or ``None`` when there is none.
        absence: Why there is no source, or ``None`` when there is one.
        paradigm: The ``detect_pipeline_config`` paradigm the source belongs to
            (``"graph"``, ``"hierarchical"``, ``"in_context"``,
            ``"middle_layer"``). Carried because a database reader needs the
            paradigm to know which database class opens the file.
        db_config: The paradigm's ``database`` block, for the reader that opens
            it --- the in-context paradigm's ``type`` key picks between the flat
            and template databases. ``None`` for the graph paradigm.

    Raises:
        ValueError: If the resolution says both or neither (a source *and* an
            absence, or neither), or names a source without the paradigm that
            reads it. A caller branching on ``source is None`` would otherwise
            silently take the wrong arm.
    """

    source: RosterSource | None = None
    absence: RosterAbsence | None = None
    paradigm: str | None = None
    db_config: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if (self.source is None) == (self.absence is None):
            raise ValueError(
                "A roster source resolution names exactly one of a source or an absence."
            )
        if self.source is not None and not self.paradigm:
            raise ValueError("A resolved roster source must name the paradigm that reads it.")


def _render_dir(config: dict) -> Path | None:
    """Directory a render-relative configured path is authored against.

    :func:`osprey.deployment.compose_generator.prepare_compose_files` records
    ``config_dir`` --- the directory the loaded ``config.yml`` sits in --- before
    any render helper runs, and that is what ``services.graphdb.ttl_path``
    resolves against: the one render-relative key (see
    :mod:`osprey.utils.config_paths`).

    Returning None lets :func:`~osprey.utils.config_paths.resolve_render_relative_path`
    fall back to the ``config.yml`` this process runs against --- ``OSPREY_CONFIG``
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


def resolve_database_path(db_config: dict) -> Path:
    """Resolve a paradigm's ``database.path`` to a file on disk.

    Anchored on the process working directory rather than the render, because
    the build sets its working directory to the project root before generating
    service files and a database file is a project artifact, not a rendered one.

    Args:
        db_config: The paradigm's ``database`` block, carrying ``path``.

    Returns:
        Absolute path to the database file.
    """
    db_path = Path(db_config["path"])
    if not db_path.is_absolute():
        db_path = Path.cwd() / db_path
    return db_path


def _graph_source(config: dict) -> RosterSourceResolution:
    """Resolve the graph paradigm's search index, or say why there is none.

    The roster reads the search index the build derives from the corpus, not the
    corpus itself: one file, resolved by
    :func:`~osprey.deployment.graphdb_service.resolve_graph_index_path`, so the
    roster and every other index reader look in the place the build wrote.

    The store is never dialed: a graph-mode project that declares no roster at
    all gets an absence naming :data:`GRAPH_CORPUS_CONFIG_KEYS`, which is what
    an operator has to edit, rather than a connection attempt whose failure
    would be reported as an empty facility. "Declares none" means no
    ``ttl_path`` *and* no explicit ``index_path``: ``index_path`` defaults
    rather than being optional, so a ``uri``-only project -- one pointing at a
    store it does not run and staging no corpus -- would otherwise be handed a
    defaulted path to a file nothing in this deployment ever builds. An explicit
    ``index_path`` is a declaration in its own right, and names the source even
    with no corpus staged here.

    A block the service resolver cannot read is its own absence
    (:attr:`~osprey.channel_roster.records.RosterAbsenceReason.GRAPH_MALFORMED`),
    carrying the resolver's complaint: "you declared no corpus" and "the line
    you declared it with cannot be read" send an operator to different edits.
    Fail-soft either way -- the build stays browse-only and the web body says
    why -- because no source was named, so none is there to be corrupt.

    Args:
        config: Full project configuration dictionary.

    Returns:
        The index source, or a
        :attr:`~osprey.channel_roster.records.RosterAbsenceReason.GRAPH_NO_TTL`
        / :attr:`~osprey.channel_roster.records.RosterAbsenceReason.GRAPH_MALFORMED`
        absence.
    """
    # Private, deliberately: asking "did the operator write this key?" has to
    # read the block exactly as the resolver that defaults it does, and a second
    # copy of the services.graphdb lookup here would disagree with it on the
    # shapes YAML allows (a bare ``graphdb:``, a non-mapping value).
    from osprey.deployment.graphdb_service import (
        _graphdb_block,
        resolve_graph_index_path,
        resolve_graphdb_service_config,
    )

    try:
        settings = resolve_graphdb_service_config(config)
    except ValueError as e:
        logger.warning(
            f"The services.graphdb block is malformed ({e}), so it names no readable "
            "knowledge-graph corpus to enumerate channels from."
        )
        return RosterSourceResolution(
            absence=RosterAbsence(
                reason=RosterAbsenceReason.GRAPH_MALFORMED,
                config_keys=GRAPH_CORPUS_CONFIG_KEYS,
                detail=str(e),
            )
        )

    declares_index = "index_path" in (_graphdb_block(config) or {})
    if settings is None or (settings.ttl_path is None and not declares_index):
        return RosterSourceResolution(
            absence=RosterAbsence(
                reason=RosterAbsenceReason.GRAPH_NO_TTL,
                config_keys=GRAPH_CORPUS_CONFIG_KEYS,
            )
        )

    return RosterSourceResolution(
        source=RosterSource(
            kind=RosterSourceKind.GRAPH,
            path=resolve_graph_index_path(config, _render_dir(config)),
            spelled=str(settings.index_path),
        ),
        paradigm=GRAPH_PARADIGM,
    )


def resolve_roster_source(config: dict) -> RosterSourceResolution:
    """Decide what this project's channel roster is enumerated from.

    The paradigm is read from :func:`~osprey.services.channel_finder.utils.detection.detect_pipeline_config`
    verbatim --- explicit ``channel_finder.pipeline_mode`` first, then the
    database paths that are actually configured --- and the source follows from
    it alone:

    - ``graph`` reads the staged Turtle corpus (:func:`_graph_source`).
    - Any other paradigm reads its own ``database.path``, even when the project
      also runs a graph store: the mode names the roster, and a second
      enumeration read off a block the mode did not select is the divergence
      this package exists to end.
    - A project that configures no paradigm at all gets the
      :attr:`~osprey.channel_roster.records.RosterAbsenceReason.NO_SOURCE`
      absence, and the build stays browse-only.

    Args:
        config: Full project configuration dictionary, as the build holds it.

    Returns:
        The resolution: exactly one of a source or an absence.

    Raises:
        PipelineModeError: If ``channel_finder.pipeline_mode`` names a paradigm
            that does not exist. Deliberately not absorbed into an absence: a
            typo'd mode is a configuration mistake with a fix, and reporting it
            as "this facility has no channels" would hide it behind a plausible
            state.
    """
    from osprey.services.channel_finder.utils.detection import detect_pipeline_config

    paradigm, db_config = detect_pipeline_config(config)

    if paradigm == GRAPH_PARADIGM:
        return _graph_source(config)

    if not paradigm or not db_config or not db_config.get("path"):
        return RosterSourceResolution(absence=RosterAbsence(reason=RosterAbsenceReason.NO_SOURCE))

    return RosterSourceResolution(
        source=RosterSource(
            kind=RosterSourceKind.DATABASE,
            path=resolve_database_path(db_config),
            spelled=str(db_config["path"]),
        ),
        paradigm=paradigm,
        db_config=db_config,
    )
