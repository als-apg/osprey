"""Knowledge CLI commands for OSPREY Facility Knowledge (OKF).

Two kinds of verb live here, and the pair that seeds is the pair worth telling
apart.  ``regen-index``, ``validate`` and ``seed-from-ttl`` operate on an OKF
bundle: a directory of markdown documents on disk.  ``seed-graph`` operates on
the deployed Neo4j graph store, loading the same NARAD TTL into it through
neosemantics.  One TTL, two destinations.

``build-ttl`` stands upstream of both: it derives that TTL from the project's
channel database, so a facility with no hand-written corpus still has one to
seed.  ``compile-ontology`` stands upstream of ``build-ttl`` in turn: it is the
authoring verb, turning a LinkML schema into the compiled FAMILY-to-class table
that ``build-ttl`` emits the corpus against.

Note: this module is intentionally importable without ``rdflib`` or the
``neo4j`` driver in the import graph.  Both are core dependencies, so this is
about keeping the CLI's import graph small, not about optional installs: any
rdflib or neo4j usage must be guarded by a lazy import inside the command body.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import click

from .output import fail, note, report, warn

#: Default token for ``build-ttl --facility``.  Spelled out rather than imported
#: so that ``osprey knowledge`` does not pull the TTL generator into its import
#: graph for every verb; ``tests/cli/test_knowledge_build_ttl.py`` pins it to
#: ``ttl_generator.model.FACILITY`` so the two cannot drift apart.
DEFAULT_FACILITY = "demo"


@click.group()
def knowledge() -> None:
    """Manage OKF facility knowledge bundles."""


def _resolve_bundle(bundle: Path | None) -> Path:
    """Return *bundle*, or fall back to ``facility_knowledge.bundle_path`` from config.

    Commands accept an optional BUNDLE argument; when omitted, the bundle root is
    read from the ``facility_knowledge.bundle_path`` config key and resolved by the
    shared rule (``~`` expanded, then relative values taken against the config.yml
    directory) so the CLI opens the same bundle as the MCP server and the OKF panel.
    This is the single source of that fallback rule and its error.

    Raises:
        click.UsageError: When *bundle* is None and the config key is unset.
    """
    if bundle is not None:
        return bundle

    from osprey.services.facility_knowledge.bundle_path import resolve_bundle_path
    from osprey.utils.config import get_config_value

    raw = get_config_value("facility_knowledge.bundle_path", None)
    if raw is None:
        raise click.UsageError(
            "No bundle path given and facility_knowledge.bundle_path is not set in config."
        )
    return resolve_bundle_path(raw)


@knowledge.command("regen-index")
@click.argument(
    "bundle",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=False,
    default=None,
)
def regen_index(bundle: Path | None) -> None:
    """Regenerate index.md files throughout an OKF bundle.

    BUNDLE is the path to the root directory of an OKF bundle.
    When omitted, facility_knowledge.bundle_path from the OSPREY
    config is used.

    Processes directories deepest-first so child descriptions propagate
    to parent indexes.  The bundle-root index.md receives an
    okf_version frontmatter block (OKF §11); all others have none
    (OKF §6).  Running the command a second time produces bit-identical
    output (idempotent).
    """
    bundle = _resolve_bundle(bundle)

    from osprey.services.facility_knowledge.okf.index import regenerate_indexes

    written = regenerate_indexes(bundle)
    for path in written:
        report(str(path))
    report(f"Wrote {len(written)} index file(s).")


@knowledge.command("validate")
@click.argument(
    "bundle",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=False,
    default=None,
)
def validate(bundle: Path | None) -> None:
    """Validate all OKF documents in a bundle.

    BUNDLE is the path to the root directory of an OKF bundle.
    When omitted, facility_knowledge.bundle_path from the OSPREY
    config is used.

    Every *.md file is checked:

    \b
      - index.md files are validated against OKF §6/§11.
      - All other .md files are parsed and their frontmatter is validated
        at the 'authoring' level (requires type, title and description).

    All files are checked even if earlier failures are found.  A
    per-file report is printed and the command exits non-zero if any
    file fails.
    """
    bundle = _resolve_bundle(bundle)

    from osprey.services.facility_knowledge.okf.document import OKFDocument, OKFDocumentError
    from osprey.services.facility_knowledge.okf.index import OKFIndexError, validate_index

    failures: list[tuple[Path, str]] = []

    for md_path in sorted(bundle.rglob("*.md")):
        if md_path.name == "index.md":
            try:
                validate_index(md_path, bundle_root=bundle)
            except (OKFIndexError, OKFDocumentError) as exc:
                failures.append((md_path, str(exc)))
        else:
            try:
                text = md_path.read_text(encoding="utf-8")
                doc = OKFDocument.parse(text)
                doc.validate("authoring")
            except (OKFDocumentError, ValueError) as exc:
                failures.append((md_path, str(exc)))

    if not failures:
        report(f"All files in {bundle} are valid.")
        return

    fail(
        f"{len(failures)} file(s) failed validation",
        "\n".join(f"{path}: {msg}" for path, msg in failures),
    )
    raise SystemExit(1)


@knowledge.command("seed-from-ttl")
@click.argument("ttl", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument(
    "bundle",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite existing stub files even when their content differs.",
)
def seed_from_ttl(ttl: Path, bundle: Path, force: bool) -> None:
    """Seed OKF stub documents from a NARAD/als-ontology TTL file.

    TTL is the path to a Turtle RDF file produced by NARAD or als-ontology.

    BUNDLE is the path to the root directory of an OKF bundle.  One stub
    .md file is written per device node in the TTL, placed at
    <bundle>/<local-iri-name>.md.

    Idempotency rules (applied per stub):

    \b
    - File absent              → write and report "written".
    - File present, same body  → skip and report "unchanged".
    - File present, diff body, no --force → skip and report "differs, use --force".
    - File present, diff body, --force   → overwrite and report "overwritten".

    The TTL is read with rdflib, a core dependency.  If rdflib is missing the
    installation is incomplete, and a clean error is printed — no traceback.
    """
    # rdflib is imported lazily inside the seeder, so an absent one surfaces
    # either here at import time or inside seed_from_ttl itself.  Both mean the
    # same broken environment, so both get the same repair hint.
    try:
        from osprey.services.facility_knowledge.seeder.ttl_seeder import seed_from_ttl as _seed

        stubs = _seed(ttl)
    except ImportError as exc:
        raise click.ClickException(
            f"rdflib is not importable: {exc}\n"
            "It is a core dependency, so this environment is incomplete. "
            "Reinstall it with: pip install --upgrade osprey-framework"
        ) from exc

    from osprey.services.facility_knowledge.okf.bundle import OKFBundle
    from osprey.services.facility_knowledge.seeder import local_name

    okf_bundle = OKFBundle(bundle)

    written = skipped_same = skipped_differs = overwritten = 0

    for stub in stubs:
        # Derive a filesystem-safe OKF §2 concept ID from the device IRI's
        # local name.  Reuse the seeder's single derivation rule (handles both
        # '/' and '#' separators) so placement and identity never diverge.
        concept_id = local_name(stub.resource)
        concept_path = okf_bundle.resolve_concept_path(concept_id)

        if concept_path.exists():
            existing = concept_path.read_text(encoding="utf-8")
            if existing == stub.body:
                note(f"unchanged   {concept_path.name}")
                skipped_same += 1
                continue
            if not force:
                note(f"differs     {concept_path.name}")
                skipped_differs += 1
                continue
            concept_path.write_text(stub.body, encoding="utf-8")
            note(f"overwritten {concept_path.name}")
            overwritten += 1
        else:
            concept_path.parent.mkdir(parents=True, exist_ok=True)
            concept_path.write_text(stub.body, encoding="utf-8")
            note(f"written     {concept_path.name}")
            written += 1

    parts = []
    if written:
        parts.append(f"{written} written")
    if overwritten:
        parts.append(f"{overwritten} overwritten")
    if skipped_same:
        parts.append(f"{skipped_same} unchanged")
    if skipped_differs:
        parts.append(f"{skipped_differs} left alone")
    report(", ".join(parts) + "." if parts else "Nothing to do.")

    # The one line that asks the operator for a decision, so it is a warning and
    # not another entry in the per-file record above.
    if skipped_differs:
        warn(
            f"{skipped_differs} file(s) already exist with different content",
            "They were left as they are. Re-run with --force to overwrite them.",
        )


# ---------------------------------------------------------------------------
# seed-graph
# ---------------------------------------------------------------------------


def _resolve_ttl(ttl: Path | None) -> Path:
    """Return *ttl*, or fall back to ``services.graphdb.ttl_path`` from config.

    A path typed on the command line is used as typed, which makes it relative
    to the shell's directory like every other filename an operator types.  A
    path read from config is resolved by
    :func:`~osprey.services.facility_knowledge.bundle_path.resolve_bundle_path`
    instead: ``~`` expanded, absolute left alone, relative taken against the
    directory holding ``config.yml``.  That helper is named for the key it was
    written for, but the rule it spells is the config-relative rule for the
    whole project, and the shipped config template promises exactly it for
    ``ttl_path``.  Sharing the one implementation is what keeps this verb and
    the deploy-time staging step opening the same file, from whatever directory
    either was started in.

    Args:
        ttl: The path given as the command's argument, or ``None``.

    Returns:
        The TTL file to import.

    Raises:
        click.UsageError: When *ttl* is None and the config key is unset.
        click.ClickException: When the configured path names no file.
    """
    if ttl is not None:
        return ttl

    from osprey.utils.config import get_config_value
    from osprey.utils.config_paths import resolve_render_relative_path

    raw = get_config_value("services.graphdb.ttl_path", None)
    if raw is None:
        raise click.UsageError(
            "No TTL path given and services.graphdb.ttl_path is not set in config."
        )

    # Render-relative, like the deploy's own seeding (container_lifecycle
    # `_graphdb_ttl_text`): the corpus is an artifact of the render.
    path = resolve_render_relative_path(raw)
    if not path.is_file():
        raise click.ClickException(
            f"services.graphdb.ttl_path names a file that is not there: {path}\n"
            f"It was read as '{raw}' and resolved against the config file's directory."
        )
    return path


def _graphdb_block() -> Mapping[str, Any] | None:
    """Return the ``services.graphdb`` config block, or ``None``.

    ``None`` covers a project with no block at all and a block YAML parsed as
    something other than a mapping.  Both resolve to the shipped connection
    defaults rather than a refusal: an operator pointing this verb at a store
    on the default port has a working command, and a store that is not there
    fails later with the address it tried.
    """
    from osprey.utils.config import get_config_value

    block = get_config_value("services.graphdb", None)
    return block if isinstance(block, Mapping) else None


def _graphdb_port_base() -> int:
    """Return the port base this project resolved.

    Handed to :func:`resolve_graphdb_connection` so a store with no explicit
    ``port_host`` is dialed on the ``graphdb_bolt`` slot of *this* deployment's
    block -- the number the compose service publishes it on. Without it the dial
    would go to the layout's default base and, on a host running two
    deployments, reach the other one's graph store.

    Returns:
        ``deployment.port_base``, or the layout default when the project sets
        none.
    """
    from osprey.port_layout import resolve_port_base
    from osprey.utils.config import get_config_value

    return resolve_port_base({"deployment": get_config_value("deployment", {})})


def _connection_failure(exc: Exception, uri: str) -> click.ClickException:
    """Turn a driver exception into a CLI error with a remedy on it.

    One failure is worth naming on its own, because its remedy is a variable
    rather than the store: a rejected credential.  Everything else keeps the
    driver's own message next to the address that was dialed, which says more
    than a guess at which of the store, the network or the corpus was at
    fault.

    Args:
        exc: What came out of the session.
        uri: The address that was dialed.

    Returns:
        The exception to raise, so the CLI prints one line instead of a
        traceback.
    """
    from osprey.deployment.graphdb_service import GRAPHDB_PASSWORD_ENV

    if _is_auth_failure(exc):
        return click.ClickException(
            f"The graph store at {uri} rejected the credentials.\n"
            f"Set {GRAPHDB_PASSWORD_ENV} to the store's password. 'osprey up' mints one "
            "and writes it to the project's .env file."
        )
    return click.ClickException(
        f"Cannot use the graph store at {uri}: {exc}\n"
        "Check that it is running ('osprey up'), and that services.graphdb points at the "
        "store you meant."
    )


def _is_auth_failure(exc: BaseException) -> bool:
    """Whether *exc* is the driver's "credentials rejected" error.

    The import is lazy and its failure is an answer rather than an error: this
    is only ever asked about an exception that came out of the driver, so a
    driver that cannot be imported means the exception came from somewhere
    else and is not an auth failure.
    """
    try:
        from neo4j.exceptions import AuthError
    except ImportError:  # pragma: no cover - the driver is a core dependency
        return False
    return isinstance(exc, AuthError)


def _refuse_or_continue(graph_seeder: ModuleType, session: Any, *, ttl: Path, digest: str) -> bool:
    """Read the store's state and decide whether seeding may go ahead.

    The three states a run without ``--force`` can end in before anything is
    written: unchanged (this file is already in there), differs (some other
    file is), and unmanaged-partial (data with no marker on it). The fourth,
    seeded, is the one this function lets through.

    Args:
        graph_seeder: The seeder module, passed in so the caller's single lazy
            import serves this helper too.
        session: An open driver session.
        ttl: The file about to be imported, for the messages.
        digest: That file's checksum, to compare the marker against.

    Returns:
        True when the caller should stop with nothing done (unchanged).

    Raises:
        SystemExit: On either refusal, after printing what was found.
    """
    marker = graph_seeder.read_marker(session)
    if marker == digest:
        report(f"Unchanged. The graph store already holds {ttl}.")
        return True

    if marker is not None:
        fail(
            "The graph store holds a different corpus.",
            f"Its seed marker reads {marker[:12]}, and {ttl} checksums to {digest[:12]}.",
            "Re-run with --force to wipe the store and import this file.",
        )
        raise SystemExit(1)

    resources = graph_seeder.resource_count(session)
    if resources:
        fail(
            "The graph store holds data that OSPREY did not seed.",
            f"It has {resources} Resource nodes and no seed marker, which is what a "
            "crashed import or a store seeded outside OSPREY looks like.",
            "Re-run with --force to wipe the store and import this file.",
        )
        raise SystemExit(1)

    return False


def _bake_prompt_snapshot(session: Any) -> None:
    """Bake the just-verified store's schema into the rendered agent prompts.

    Follows every outcome that leaves the store holding a corpus this verb
    vouches for — seeded, overwritten, unchanged — so a render rebuilt since
    the last seed gets its placeholder filled back in.  Never fails the verb:
    an agent whose prompt kept the placeholder reads the schema through its
    tools at run time, which is the same behavior an unseeded render has.

    Args:
        session: The verb's open driver session.
    """
    from osprey.services.facility_knowledge.seeder import prompt_snapshot
    from osprey.utils.workspace import resolve_config_path

    config_file = Path(resolve_config_path())
    if not config_file.is_file():
        note(
            "No rendered project here, so the agent-prompt schema snapshot was left "
            "alone. Run this verb from the deployment repo to bake it."
        )
        return

    try:
        patched = prompt_snapshot.bake_snapshot(session, config_file.parent)
    except Exception as exc:  # noqa: BLE001 — the seed already succeeded; report and move on
        warn(
            f"The schema snapshot could not be baked into the agent prompt ({exc}); "
            "the agent will read the schema through its tools instead."
        )
        return
    if patched:
        note(f"Schema snapshot baked into {prompt_snapshot.describe_patched(patched)}.")


@knowledge.command("seed-graph")
@click.argument(
    "ttl",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=False,
    default=None,
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Wipe the store first, then bootstrap and import from scratch.",
)
def seed_graph(ttl: Path | None, force: bool) -> None:
    """Load a NARAD/als-ontology TTL into the deployed graph store.

    This verb seeds the deployed graph store; seed-from-ttl builds an OKF
    document bundle. Both read the same TTL file and write to different
    places.

    TTL is the Turtle file to import. When omitted, services.graphdb.ttl_path
    from the OSPREY config is used, resolved against the config file's own
    directory (the same rule facility_knowledge.bundle_path follows).

    The store is dialed at services.graphdb.uri, or at the bolt port the
    deployment publishes, with the password in GRAPHDB_PASSWORD.

    What a run reports, and when:

    \b
      seeded             The store held no data, so it was bootstrapped and
                         imported.
      unchanged          The store already holds this exact file. Nothing to do.
      differs            The store holds a different corpus. Nothing was
                         touched; --force replaces it.
      unmanaged-partial  The store holds data with no seed marker, which is
                         what a crashed import or a store seeded outside OSPREY
                         looks like. Nothing was touched; --force replaces it.
      overwritten        --force wiped the store and imported the file.

    The seed marker carrying the file's checksum is written only after the
    import reports success, so an import that dies partway is reported as
    unmanaged-partial on the next run rather than mistaken for a good seed.

    Every refusal exits non-zero and never touches the store's data.
    """
    from osprey.deployment.graphdb_service import resolve_graphdb_connection
    from osprey.services.facility_knowledge.seeder import graph_seeder

    ttl = _resolve_ttl(ttl)
    try:
        text = ttl.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise click.ClickException(f"Cannot read {ttl}: {exc}") from exc

    # The seed marker's identity: the checksum of the TTL text as imported.
    # Computed by the seeder rather than here, so this verb and the deploy-time
    # staging step read one another's markers rather than re-seeding on sight.
    digest = graph_seeder.ttl_sha256(text)

    # Read off the same text, so the marker's provenance and its digest can only
    # ever describe the one corpus. A file built before the header existed
    # declares nothing, and the snapshot then states nothing.
    direction_source = graph_seeder.parse_direction_source(text)

    try:
        connection = resolve_graphdb_connection(_graphdb_block(), base=_graphdb_port_base())
    except ValueError as exc:
        raise click.ClickException(f"Cannot work out which graph store to dial: {exc}") from exc

    try:
        with graph_seeder.open_session(
            connection.uri, connection.username, connection.password
        ) as session:
            if force:
                graph_seeder.wipe(session)
                note("Wiped the store. Its graph config and seed marker went with the data.")
            elif _refuse_or_continue(graph_seeder, session, ttl=ttl, digest=digest):
                # Unchanged — but a rebuild since the last seed resets the
                # rendered agent prompt to its placeholder, so the snapshot is
                # re-baked even when the corpus is not.
                _bake_prompt_snapshot(session)
                return

            bootstrapped = graph_seeder.bootstrap(session)
            if not bootstrapped.ok:
                fail(
                    "The graph store is set up differently from what OSPREY expects.",
                    bootstrapped.message,
                    "Re-run with --force to wipe the store and set it up from scratch.",
                )
                raise SystemExit(1)
            note(bootstrapped.message)

            imported = graph_seeder.import_ttl(session, text)
            if not imported.ok:
                fail(
                    f"The import of {ttl} did not finish.",
                    f"neo4j reported {imported.termination_status} after "
                    f"{imported.triples_loaded} of {imported.triples_parsed} triples: "
                    f"{imported.extra_info}",
                    "Fix the file, then re-run with --force to clear the partial import.",
                )
                raise SystemExit(1)

            # Only now: a marker written before this point would label a
            # half-imported graph as a good seed, and every later run would
            # report it as unchanged.
            graph_seeder.write_marker(session, digest, direction_source)

            state = "Overwritten" if force else "Seeded"
            report(f"{state}. The graph store at {connection.uri} now holds {ttl}.")
            note(f"{imported.triples_loaded} triples loaded.")
            _bake_prompt_snapshot(session)
    except click.ClickException:
        raise
    except graph_seeder.MissingN10sPluginError as exc:
        # The store answered — this is not a connection failure, and folding it
        # into one would send the operator to check an address that works. The
        # seeder's message names the plugin gap and the server version.
        fail(
            "The graph store cannot import RDF.",
            str(exc),
            "Nothing was seeded; the store's data was not touched.",
        )
        raise SystemExit(1) from exc
    except ImportError as exc:
        # The driver is a core dependency, so this means a broken install; the
        # seeder's own message says how to repair it.
        raise click.ClickException(str(exc)) from exc
    except Exception as exc:
        # Everything the driver can raise becomes one line with a remedy on it.
        # An operator who cannot reach their store is not helped by a stack of
        # frames from inside the driver.
        raise _connection_failure(exc, connection.uri) from exc


# ---------------------------------------------------------------------------
# build-ttl
# ---------------------------------------------------------------------------


CHANNEL_DB_CONFIG_KEY = "channel_finder.pipelines.hierarchical.database.path"
"""Config key naming the hierarchical channel database ``build-ttl`` reads."""

PIPELINE_MODE_CONFIG_KEY = "channel_finder.pipeline_mode"
"""Config key naming the channel-finder paradigm the project runs."""

DESCRIPTIONS_FILENAME = "in_context.json"
"""What the in-context database is called where the two sit side by side."""


def _paradigm_databases(directory: Path) -> str:
    """Say which channel databases *directory* actually holds.

    A project is rendered for one channel-finder paradigm and ships only that
    paradigm's database — the others are pruned at build time.  So "the file
    is not there" is nearly always "this project is an in-context project",
    and naming what *is* there turns a dead end into a diagnosis.

    Args:
        directory: Where the missing database was looked for.

    Returns:
        One sentence, ready to drop into an error message.
    """
    try:
        names = sorted(item.name for item in directory.iterdir() if item.suffix == ".json")
    except OSError:
        names = []
    if not names:
        return f"There are no channel databases in {directory}."
    return f"{directory} holds {', '.join(names)}."


def _resolve_channel_db(channel_db: Path | None) -> Path:
    """Return *channel_db*, or the one the project's config names.

    A path typed on the command line is used as typed.  A path read from
    config follows the same rule every other configured path in this module
    follows: ``~`` expanded, relative values taken against the directory
    holding ``config.yml``.  In a rendered project that lands on
    ``data/channel_databases/hierarchical.json``.

    Args:
        channel_db: The ``--channel-db`` value, or ``None``.

    Returns:
        The database file to read.

    Raises:
        click.UsageError: When nothing was given and the config key is unset.
        click.ClickException: When the configured path names no file.
    """
    if channel_db is not None:
        return channel_db

    from osprey.services.facility_knowledge.bundle_path import resolve_bundle_path
    from osprey.utils.config import get_config_value

    if get_config_value(PIPELINE_MODE_CONFIG_KEY, None) == "graph":
        # A graph project has no hierarchical database path configured at all,
        # so without this arm the run dies on the unset key below — which reads
        # as a broken project rather than as the posture it is.
        raise click.UsageError(
            "This project runs the graph channel-finder paradigm, which ships no "
            "channel database: the seeded graph store is the database.\n"
            "build-ttl derives a corpus from database files, so name them both: "
            "--channel-db PATH for a hierarchical database and --descriptions PATH "
            "for the in-context database of the same machine."
        )

    raw = get_config_value(CHANNEL_DB_CONFIG_KEY, None)
    if not isinstance(raw, str) or not raw:
        raise click.UsageError(
            f"No channel database given and {CHANNEL_DB_CONFIG_KEY} is not set in config.\n"
            "Name the database with --channel-db PATH, or run this inside a project whose "
            "config sets that key."
        )

    path = resolve_bundle_path(raw)
    if not path.is_file():
        raise click.ClickException(
            f"No channel database at {path}.\n"
            f"That is {CHANNEL_DB_CONFIG_KEY} ('{raw}') resolved against the config file's "
            f"directory. {_paradigm_databases(path.parent)}\n"
            "build-ttl reads the hierarchical-paradigm database, the one whose addresses "
            "are RING:SYSTEM:FAMILY:DEVICE:FIELD:SUBFIELD; a project built for another "
            "paradigm does not ship it. Name a hierarchical database with --channel-db PATH."
        )
    return path


def _load_channel_map(db_path: Path) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Expand *db_path* into its flat address map, keeping the file's own blocks.

    Expansion throws the tree away — the channel map holds addresses and
    nothing else — but the tree is where the per-level prose lives, so the raw
    payload is handed back beside the map rather than re-read later.  The
    database class's own value dict is untouched: the prose is read from the
    file's ``tree`` block, not from the expanded entries.

    Args:
        db_path: A hierarchical channel database.

    Returns:
        Tuple of (the expanded map keyed by colon-grammar address, the raw
        database payload — its ``tree`` and ``hierarchy`` blocks).

    Raises:
        click.ClickException: When the file cannot be read or expanded.
    """
    from osprey.services.channel_finder.databases.hierarchical import HierarchicalChannelDatabase

    try:
        raw = json.loads(db_path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError("the file is not a JSON object")
        return HierarchicalChannelDatabase(str(db_path)).channel_map, raw
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise click.ClickException(
            f"Cannot read the channel database at {db_path}: {exc}\n"
            "build-ttl expects a hierarchical database: a 'hierarchy' block naming the "
            "levels, and a 'tree' block holding them."
        ) from exc


def _resolve_descriptions(descriptions: Path | None, db_path: Path) -> Path:
    """Return *descriptions*, or the in-context database beside *db_path*.

    The fallback is a convenience for the OSPREY source tree, where the two
    databases of one machine sit together under ``tiers/tier3/``.  A rendered
    project materializes only the paradigm it runs and prunes the rest, so
    there is nothing to default to there and both files have to be named.

    Args:
        descriptions: The ``--descriptions`` value, or ``None``.
        db_path: The hierarchical database already resolved.

    Returns:
        The in-context database to read the per-channel prose from.

    Raises:
        click.UsageError: When nothing was given and *db_path* has no
            in-context neighbour.
    """
    if descriptions is not None:
        return descriptions

    sibling = db_path.parent / DESCRIPTIONS_FILENAME
    if sibling.is_file():
        return sibling

    raise click.UsageError(
        f"No descriptions database given and there is no {DESCRIPTIONS_FILENAME} "
        f"beside {db_path}.\n"
        "That neighbour is how the OSPREY source tree keeps the two databases of one "
        "machine; a rendered project ships only the paradigm it runs, so name both "
        "files there: --channel-db PATH --descriptions PATH."
    )


def _load_binding_descriptions(descriptions_path: Path) -> Mapping[str, str]:
    """Read the per-channel prose out of an in-context database.

    Args:
        descriptions_path: An in-context database: a ``channels`` list whose
            entries carry an ``address`` and a ``description``.

    Returns:
        The prose keyed by full six-token address.  An entry with no address or
        no text contributes no key, so the join stays exact.

    Raises:
        click.ClickException: When the file cannot be read or is not one.
    """
    try:
        payload = json.loads(descriptions_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise click.ClickException(
            f"Cannot read the descriptions database at {descriptions_path}: {exc}"
        ) from exc

    channels = payload.get("channels") if isinstance(payload, Mapping) else None
    if not isinstance(channels, list):
        raise click.ClickException(
            f"The descriptions database at {descriptions_path} has no 'channels' list.\n"
            "build-ttl expects an in-context database: {'channels': [{'address': ..., "
            "'description': ...}]}."
        )

    prose: dict[str, str] = {}
    for entry in channels:
        if not isinstance(entry, Mapping):
            continue
        address = entry.get("address")
        text = entry.get("description")
        if isinstance(address, str) and isinstance(text, str) and text.strip():
            prose[address] = text.strip()
    return prose


def _resolve_hierarchy_descriptions(raw: Mapping[str, Any], db_path: Path) -> Any:
    """Read the per-level prose out of a raw hierarchical database payload.

    Args:
        raw: The database payload, as returned by :func:`_load_channel_map`.
        db_path: Where it came from, for the error message.

    Returns:
        The ``HierarchyDescriptions`` for the file's tree.

    Raises:
        click.ClickException: When the file's levels are not the six-token
            grammar this verb reads.
    """
    from osprey.services.facility_knowledge.ttl_generator.model import (
        resolve_hierarchy_descriptions,
    )

    hierarchy = raw.get("hierarchy") if isinstance(raw.get("hierarchy"), Mapping) else {}
    levels = hierarchy.get("levels") or []
    tree = raw.get("tree") if isinstance(raw.get("tree"), Mapping) else {}
    try:
        return resolve_hierarchy_descriptions(tree, levels)
    except (ValueError, AttributeError, TypeError) as exc:
        raise click.ClickException(
            f"Cannot read the level prose in {db_path}: {exc}\n"
            "build-ttl reads the six-token grammar: RING, SYSTEM, FAMILY, DEVICE, "
            "FIELD, SUBFIELD."
        ) from exc


def _load_ontology_table(ontology: Path | None) -> Any:
    """Return the FAMILY-to-class table to emit against.

    The table is the compiled JSON form of a LinkML schema, so a ``.yaml`` or
    ``.yml`` path is the authoring source handed over one step too early: say
    so, rather than only that the file is not valid JSON.

    Args:
        ontology: The ``--ontology`` value, or ``None`` for the shipped table.

    Returns:
        The validated table.

    Raises:
        click.ClickException: When a given table is malformed or unreadable.
    """
    from osprey.services.facility_knowledge.ttl_generator.ontology_map import (
        OntologyMapError,
        load_demo_ontology,
        load_ontology,
    )

    if ontology is None:
        return load_demo_ontology()
    try:
        return load_ontology(ontology)
    except OntologyMapError as exc:
        message = f"Cannot use the ontology table {ontology}: {exc}"
        if ontology.suffix.lower() in {".yaml", ".yml"}:
            message += (
                "\nA LinkML schema must be compiled first: "
                "`osprey knowledge compile-ontology <schema.yaml> <table.json>`, "
                "then pass the JSON here."
            )
        raise click.ClickException(message) from exc
    except OSError as exc:
        raise click.ClickException(f"Cannot read the ontology table {ontology}: {exc}") from exc


def _assign_directions(graph_model: Any, limits: Path | None) -> tuple[Any, Any]:
    """Stamp every signal with a direction, and say where it came from.

    Args:
        graph_model: The model built from the channel database.
        limits: The ``--limits`` value, or ``None`` to resolve from config.

    Returns:
        Tuple of (annotated model, direction report).

    Raises:
        click.ClickException: When the limits file is missing, unreadable, or
            disagrees with itself about a signal.
    """
    from osprey.services.facility_knowledge.ttl_generator.direction import (
        LIMITS_DATABASE_CONFIG_KEY,
        DirectionConflictError,
        LimitsFileMissing,
        resolve_and_assign,
    )

    try:
        return resolve_and_assign(graph_model, limits)
    except LimitsFileMissing as exc:
        raise click.ClickException(
            f"The --limits file is not there: {exc.path}\n"
            "Give the path to a channel_limits.json, or drop --limits to read "
            f"{LIMITS_DATABASE_CONFIG_KEY} from the config instead — and when that names "
            "no file either, the PV grammar decides."
        ) from exc
    except DirectionConflictError as exc:
        raise click.ClickException(
            f"The channel limits disagree about one signal: {exc}\n"
            "Every address in a signal group must be writable or none of them, because "
            "the corpus records the direction once per signal rather than once per address."
        ) from exc
    except (OSError, ValueError) as exc:
        raise click.ClickException(f"Cannot read the channel limits: {exc}") from exc


@knowledge.command("build-ttl")
@click.argument("output", type=click.Path(dir_okay=False, writable=True, path_type=Path))
@click.option(
    "--channel-db",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Hierarchical channel database to read. Defaults to the one the project's config names.",
)
@click.option(
    "--descriptions",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="In-context database whose per-channel prose the corpus carries. "
    "Defaults to the in_context.json beside --channel-db.",
)
@click.option(
    # Deliberately not exists=True: when the file is absent this verb says what
    # the fallback would have been, which click's own "does not exist" cannot.
    "--limits",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="channel_limits.json deciding which signals are written. Defaults to the config's.",
)
@click.option(
    "--ontology",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="FAMILY-to-class table, in its compiled JSON form. Defaults to the demo-machine "
    "table shipped with OSPREY, which is compiled from a LinkML schema.",
)
@click.option(
    "--facility",
    default=DEFAULT_FACILITY,
    show_default=True,
    help="Facility token embedded in every IRI and identifier the corpus mints, "
    "and written onto each device as narad_p:facility. Name your own facility "
    "when the corpus is not the demo machine's.",
)
def build_ttl(
    output: Path,
    channel_db: Path | None,
    descriptions: Path | None,
    limits: Path | None,
    ontology: Path | None,
    facility: str,
) -> None:
    """Derive a NARAD-convention TTL corpus from the channel database.

    OUTPUT is the Turtle file to write. Missing parent directories are
    created, and an existing file is overwritten.

    The corpus describes the machine the channel databases already describe:
    one device node per device, one channel binding per address, one class per
    device family, a read or write direction on every signal, and the prose
    from both databases carried onto the nodes it describes — which is what
    lets the graph be searched by meaning rather than by address alone. It is
    what 'osprey knowledge seed-graph' loads into the deployed graph store, and
    what the OSPREY agent searches once it is in there.

    Where each input comes from when you do not name it:

    \b
      --channel-db  channel_finder.pipelines.hierarchical.database.path from
                    the OSPREY config, resolved against the config file's own
                    directory. In a rendered project that is
                    data/channel_databases/hierarchical.json.
      --descriptions
                    The in_context.json sitting beside the file --channel-db
                    named. That neighbour is how the OSPREY source tree keeps
                    the two databases of one machine, under tiers/tier3/. A
                    rendered project ships only the paradigm it runs, so name
                    both files yourself there — and a graph-mode project ships
                    neither, because its store is the database.
      --limits      control_system.limits_checking.database_path from the
                    config. This file is what tells a readback from a
                    setpoint, so the corpus and the write-safety layer agree
                    on which channels are written. When it names no file, the
                    PV grammar decides instead: a :SP subfield writes and
                    everything else reads. Every run reports which of the two
                    it used.
      --ontology    The demo-machine table shipped with OSPREY. Give your own
                    when your device families are not the demo machine's --
                    authored as a LinkML schema and turned into the table this
                    flag reads by 'osprey knowledge compile-ontology'.
      --facility    The token 'demo', which is what the shipped demo corpus
                    carries. Every IRI and identifier in the file embeds it,
                    so name your own facility when the corpus is not the demo
                    machine's.

    Then load the file into the store:

    \b
      osprey knowledge seed-graph <output>

    To have every deployment seed this corpus, point services.graphdb.ttl_path at
    the file you wrote. That key is deliberately not where OUTPUT defaults to:
    it usually names a hand-curated corpus, and a bare run of this verb would
    overwrite it.
    """
    from osprey.services.facility_knowledge.ttl_generator import emitter
    from osprey.services.facility_knowledge.ttl_generator.model import build_model
    from osprey.services.facility_knowledge.ttl_generator.ontology_map import UnknownFamilyError

    db_path = _resolve_channel_db(channel_db)
    descriptions_path = _resolve_descriptions(descriptions, db_path)
    channel_map, raw_database = _load_channel_map(db_path)
    binding_descriptions = _load_binding_descriptions(descriptions_path)
    hierarchy_descriptions = _resolve_hierarchy_descriptions(raw_database, db_path)

    try:
        graph_model = build_model(
            channel_map,
            facility=facility,
            hierarchy_descriptions=hierarchy_descriptions,
            binding_descriptions=binding_descriptions,
        )
    except ValueError as exc:
        raise click.ClickException(
            f"The channel database at {db_path} holds an address build-ttl cannot read: {exc}\n"
            "Every address must be six colon-separated tokens: "
            "RING:SYSTEM:FAMILY:DEVICE:FIELD:SUBFIELD."
        ) from exc

    ontology_table = _load_ontology_table(ontology)

    # Directions first: the emitter refuses a model whose signals have none,
    # rather than writing a corpus that reads every setpoint as a readback.
    graph_model, direction_report = _assign_directions(graph_model, limits)

    try:
        written = emitter.write_turtle(
            graph_model,
            ontology_table,
            output,
            direction_source=direction_report.source,
        )
    except emitter.UndirectedSignalError as exc:
        raise click.ClickException(
            f"The corpus was not written because a signal has no direction: {exc}"
        ) from exc
    except UnknownFamilyError as exc:
        raise click.ClickException(
            f"{exc}\nOr point --ontology at a table that covers this machine's families."
        ) from exc
    except OSError as exc:
        raise click.ClickException(f"Cannot write {output}: {exc}") from exc

    report(f"Wrote {written}.")
    note(
        f"{len(graph_model.devices)} devices, {len(graph_model.bindings)} channel bindings, "
        f"{len(graph_model.signal_groups)} signals."
    )
    note(direction_report.message)
    note(f"Load it with: osprey knowledge seed-graph {written}")


def _replace_file(output: Path, text: str) -> None:
    """Replace *output* with *text*, or leave it exactly as it was.

    ``Path.write_text`` truncates its target before the first byte lands, so a
    write that fails part-way -- a full disk, a killed process -- destroys a
    committed artifact and leaves nothing behind to compile from.  The rendered
    text goes to a sibling temporary file instead, which is renamed over
    *output* once it is complete: a reader only ever sees the whole old table
    or the whole new one.  The sibling has to be a sibling for the rename to stay
    inside one filesystem, and it is removed again if anything fails before the
    rename.

    Args:
        output: File to replace.  Its directory must already exist.
        text: What to write, verbatim: UTF-8, with no newline translation, so
            the bytes on disk are the ones ``--check`` recompiles and compares.

    Raises:
        OSError: The temporary file could not be written, or the rename failed.
            *output* is untouched in either case.
    """
    handle = tempfile.NamedTemporaryFile(  # closed by the `with` below
        "w",
        encoding="utf-8",
        newline="",
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    )
    try:
        with handle:
            handle.write(text)
        os.replace(handle.name, output)
    except OSError:
        Path(handle.name).unlink(missing_ok=True)
        raise


@knowledge.command("compile-ontology")
@click.argument("source", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output", type=click.Path(dir_okay=False, path_type=Path))
@click.option(
    "--check",
    "check",
    is_flag=True,
    default=False,
    help="Compare OUTPUT against a fresh compile of SOURCE and fail on drift. Writes nothing.",
)
def compile_ontology(source: Path, output: Path, check: bool) -> None:
    """Compile an authored LinkML schema into the ontology table build-ttl reads.

    SOURCE is the LinkML schema to compile. OUTPUT is the JSON table to write;
    an existing file is replaced, and only once the new table is complete.

    The schema is where a machine's device vocabulary is authored: one class
    per kind of device, 'is_a' naming its parent, 'aliases' listing the
    synonyms a search should also answer to, and a 'DeviceFamily' enum mapping
    every FAMILY token of the channel database onto one of those classes. This
    verb turns that into the compiled table the corpus is emitted against, and
    refuses a schema the table cannot represent rather than writing one that
    only fails when something later tries to load it.

    What is written is deterministic: classes and families in sorted order,
    synonyms sorted and de-duplicated, a trailing newline, and a header line
    saying the file is generated. Compiling an unchanged schema twice therefore
    leaves 'git diff' silent, which is what makes a committed table reviewable.

    Then emit the corpus against it:

    \b
      osprey knowledge build-ttl <corpus.ttl> --ontology <output>

    With --check nothing is written at all. OUTPUT is compared against a fresh
    compile of SOURCE, and a run whose two sides disagree fails, naming the
    classes, synonyms and families that differ. That is the form for CI and for
    a pre-commit hook: it is what proves a committed table still matches the
    schema it says it came from, since neither file announces the drift on its
    own.

    The 'knowledge' extra (linkml-runtime) is required. A clean error is
    printed when it is absent -- no traceback.
    """
    try:
        from osprey.services.facility_knowledge.ontology_compiler import (
            OntologyCompileError,
            check_artifact,
            compile_schema,
            render_json,
        )
        from osprey.services.facility_knowledge.ttl_generator.ontology_map import OntologyMapError
    except ImportError as exc:  # linkml_runtime absent
        raise click.ClickException(
            f"The 'knowledge' extra is required for compile-ontology: {exc}\n"
            "Install it with: pip install 'osprey-framework[knowledge]'"
        ) from exc

    try:
        if check:
            # Reading OUTPUT is the only I/O this branch does, so every failure
            # of it is a failure to read -- never a failure to write.
            write_one = f"Write one first: osprey knowledge compile-ontology {source} {output}"
            try:
                drift = check_artifact(source, output)
            except FileNotFoundError as exc:
                raise click.ClickException(
                    f"There is no compiled table at {output} to check against {source}.\n"
                    f"{write_one}"
                ) from exc
            except UnicodeDecodeError as exc:
                raise click.ClickException(
                    f"{output} is not UTF-8 text, so it cannot be the compiled table.\n{write_one}"
                ) from exc
            except OSError as exc:
                raise click.ClickException(f"Cannot read {output}: {exc}") from exc
            if drift:
                # The report is complete as it stands: its last line already
                # names the command that regenerates OUTPUT.
                raise click.ClickException("\n".join(drift))
            report(f"{output} is up to date with {source}.")
            return

        compiled = compile_schema(source)
        rendered = render_json(compiled.payload, source)
        try:
            _replace_file(output, rendered)
        except OSError as exc:
            raise click.ClickException(f"Cannot write {output}: {exc}") from exc
    except ImportError as exc:  # linkml_runtime absent (raised inside compile_schema)
        raise click.ClickException(
            f"The 'knowledge' extra is required for compile-ontology: {exc}\n"
            "Install it with: pip install 'osprey-framework[knowledge]'"
        ) from exc
    except OntologyCompileError as exc:
        raise click.ClickException(f"Cannot compile the schema: {exc}") from exc
    except OntologyMapError as exc:
        raise click.ClickException(
            f"The schema compiled, but the ontology it describes does not stand up: {exc}"
        ) from exc

    report(f"Wrote {output}.")
    note(f"{len(compiled.table.classes)} classes, {len(compiled.table.family_to_class)} families.")
    note(
        f"Emit a corpus against it with: osprey knowledge build-ttl <corpus.ttl> "
        f"--ontology {output}"
    )
