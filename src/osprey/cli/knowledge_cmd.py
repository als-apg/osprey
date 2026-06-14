"""Knowledge CLI commands for OSPREY Facility Knowledge (OKF).

Provides operations on OKF bundle directories: index regeneration,
validation, and seed-from-ontology (the latter added by later tasks).

Note: this module is intentionally importable WITHOUT the ``knowledge``
extra (rdflib) installed.  Any rdflib usage must be guarded by a lazy
import inside the command body.
"""

from __future__ import annotations

from pathlib import Path

import click


@click.group()
def knowledge() -> None:
    """Manage OKF facility knowledge bundles."""


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
    When omitted, ``facility_knowledge.bundle_path`` from the OSPREY
    config is used.

    Processes directories deepest-first so child descriptions propagate
    to parent indexes.  The bundle-root ``index.md`` receives an
    ``okf_version`` frontmatter block (OKF §11); all others have none
    (OKF §6).  Running the command a second time produces bit-identical
    output (idempotent).
    """
    if bundle is None:
        from osprey.utils.config import get_config_value

        raw = get_config_value("facility_knowledge.bundle_path", None)
        if raw is None:
            raise click.UsageError(
                "No bundle path given and facility_knowledge.bundle_path is not set in config."
            )
        bundle = Path(raw).expanduser().resolve()

    from osprey.services.facility_knowledge.okf.index import regenerate_indexes

    written = regenerate_indexes(bundle)
    for path in written:
        click.echo(str(path))
    click.echo(f"Wrote {len(written)} index file(s).")


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
    When omitted, ``facility_knowledge.bundle_path`` from the OSPREY
    config is used.

    Every ``*.md`` file is checked:

    * ``index.md`` files are validated against OKF §6/§11 via
      ``validate_index``.
    * All other ``.md`` files are parsed and their frontmatter is
      validated at the ``"authoring"`` level (requires ``type``,
      ``title``, and ``description``).

    All files are checked even if earlier failures are found.  A
    per-file report is printed and the command exits non-zero if any
    file fails.
    """
    if bundle is None:
        from osprey.utils.config import get_config_value

        raw = get_config_value("facility_knowledge.bundle_path", None)
        if raw is None:
            raise click.UsageError(
                "No bundle path given and facility_knowledge.bundle_path is not set in config."
            )
        bundle = Path(raw).expanduser().resolve()

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
        click.echo(f"All files in {bundle} are valid.")
        return

    click.echo(f"{len(failures)} file(s) failed validation:", err=True)
    for path, msg in failures:
        click.echo(f"  {path}: {msg}", err=True)
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
    ``.md`` file is written per device node in the TTL, placed at
    ``<bundle>/<local-iri-name>.md``.

    Idempotency rules (applied per stub):

    \b
    - File absent              → write and report "written".
    - File present, same body  → skip and report "unchanged".
    - File present, diff body, no --force → skip and report "differs, use --force".
    - File present, diff body, --force   → overwrite and report "overwritten".

    The ``knowledge`` extra (rdflib) is required.  A clean error is printed
    when it is absent — no traceback.
    """
    try:
        from osprey.services.facility_knowledge.seeder.ttl_seeder import seed_from_ttl as _seed
    except ImportError as exc:  # rdflib absent
        raise click.ClickException(
            f"The 'knowledge' extra is required for seed-from-ttl: {exc}\n"
            "Install it with: pip install 'osprey-framework[knowledge]'"
        ) from exc

    try:
        stubs = _seed(ttl)
    except ImportError as exc:  # rdflib absent (raised inside seed_from_ttl)
        raise click.ClickException(
            f"The 'knowledge' extra is required for seed-from-ttl: {exc}\n"
            "Install it with: pip install 'osprey-framework[knowledge]'"
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
                click.echo(f"  unchanged  {concept_path.name}")
                skipped_same += 1
                continue
            if not force:
                click.echo(
                    f"  differs    {concept_path.name}  (use --force to overwrite)",
                    err=True,
                )
                skipped_differs += 1
                continue
            concept_path.write_text(stub.body, encoding="utf-8")
            click.echo(f"  overwritten {concept_path.name}")
            overwritten += 1
        else:
            concept_path.parent.mkdir(parents=True, exist_ok=True)
            concept_path.write_text(stub.body, encoding="utf-8")
            click.echo(f"  written    {concept_path.name}")
            written += 1

    parts = []
    if written:
        parts.append(f"{written} written")
    if overwritten:
        parts.append(f"{overwritten} overwritten")
    if skipped_same:
        parts.append(f"{skipped_same} unchanged")
    if skipped_differs:
        parts.append(f"{skipped_differs} skipped (differs; use --force)")
    click.echo(", ".join(parts) + "." if parts else "Nothing to do.")
