"""Knowledge CLI commands for OSPREY Facility Knowledge (OKF).

``regen-index``, ``validate`` and ``seed-from-ttl`` operate on an OKF bundle: a
directory of markdown documents on disk.  ``seed-from-ttl`` writes one device
stub per device of the build's graph view, ``data/graph/facility.ttl``.  The
graph store and the channel search index are filled from that same view by
``osprey build && osprey up``, with no verb of their own.

Note: this module is intentionally importable without ``rdflib`` or the
``neo4j`` driver in the import graph.  Both are core dependencies, so this is
about keeping the CLI's import graph small, not about optional installs: any
rdflib or neo4j usage must be guarded by a lazy import inside the command body.
"""

from __future__ import annotations

from pathlib import Path

import click

from osprey_connectors.config import get_config_value

from .output import fail, note, report, warn
from .repo_resolver import repo_option


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
      - each index.md must match what regen-index writes for its directory,
        and every directory holding pages must have one.

    All files are checked even if earlier failures are found.  A
    per-file report is printed and the command exits non-zero if any
    file fails.
    """
    bundle = _resolve_bundle(bundle)

    from osprey.services.facility_knowledge.okf.document import OKFDocument, OKFDocumentError
    from osprey.services.facility_knowledge.okf.index import (
        OKFIndexError,
        check_indexes,
        validate_index,
    )

    failures: list[tuple[Path, str]] = []
    failed_indexes: set[Path] = set()

    for md_path in sorted(bundle.rglob("*.md")):
        if md_path.name == "index.md":
            try:
                validate_index(md_path, bundle_root=bundle)
            except (OKFIndexError, OKFDocumentError) as exc:
                failures.append((md_path, str(exc)))
                failed_indexes.add(md_path)
        else:
            try:
                text = md_path.read_text(encoding="utf-8")
                doc = OKFDocument.parse(text)
                doc.validate("authoring")
            except (OKFDocumentError, ValueError) as exc:
                failures.append((md_path, str(exc)))

    drifted = [drift for drift in check_indexes(bundle) if drift.path not in failed_indexes]
    failures.extend((drift.path, f"index.md {drift.message}") for drift in drifted)

    if not failures:
        report(f"All files in {bundle} are valid.")
        return

    cause = "\n".join(f"{path}: {msg}" for path, msg in failures)
    if drifted:
        fail(
            f"{len(failures)} file(s) failed validation",
            cause,
            f"Rebuild the indexes: osprey knowledge regen-index {bundle}",
        )
    else:
        fail(f"{len(failures)} file(s) failed validation", cause)
    raise SystemExit(1)


def _build_graph_view(repo: Path | None) -> Path:
    """Return the graph view the build wrote for the deployment repo in use.

    The repo is found by the shared walk from *repo*, or from the working
    directory when *repo* is None, and the view sits at
    ``build/data/graph/facility.ttl`` under its root.

    Raises:
        click.ClickException: When no deployment repo encloses the working
            directory, or its build has written no graph view.
    """
    from osprey.cli.repo_resolver import RepoNotFoundError, find_repo_root
    from osprey.facility.views.graph import GRAPH_FILE
    from osprey_connectors.workspace import BUILD_DIR_NAME

    try:
        repo_root = find_repo_root(repo)
    except RepoNotFoundError as exc:
        raise click.ClickException(
            "No deployment repo found. Pass the repo with --repo or the Turtle file with --ttl."
        ) from exc
    view = repo_root / BUILD_DIR_NAME / "data" / "graph" / GRAPH_FILE
    if not view.is_file():
        raise click.ClickException(f"No graph view at {view}. Run osprey build first.")
    return view


@knowledge.command("seed-from-ttl")
@click.argument(
    "bundle",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--ttl",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Turtle file to read. Default: build/data/graph/facility.ttl in the deployment repo.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite existing stub files even when their content differs.",
)
@repo_option
def seed_from_ttl(bundle: Path, ttl: Path | None, force: bool, repo: Path | None) -> None:
    """Seed OKF stub documents from the build's graph view.

    BUNDLE is the path to the root directory of an OKF bundle.  One stub
    .md file is written per device node in the TTL, placed at
    <bundle>/<local-iri-name>.md.

    --ttl names the Turtle file to read.  Without it the verb reads the graph
    view 'osprey build' wrote, build/data/graph/facility.ttl in the deployment
    repo it is run in or the one --repo names; any other NARAD Turtle file is
    accepted too.  Each stub's device_id is the facility file's device id, so the build links the
    page to its device.

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
    if ttl is None:
        ttl = _build_graph_view(repo)

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
