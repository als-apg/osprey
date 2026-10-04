"""Knowledge CLI commands for OSPREY Facility Knowledge (OKF).

``regen-index``, ``validate`` and ``seed-from-ttl`` operate on an OKF bundle: a
directory of markdown documents on disk.  ``seed-from-ttl`` writes one device
stub per device of the build's graph view, ``data/graph/facility.ttl``.  The
graph store and the channel search index are filled from that same view by
``osprey build && osprey up``, with no verb of their own.

``compile-ontology`` is the authoring verb: it turns a LinkML schema into the
compiled FAMILY-to-class table.

Note: this module is intentionally importable without ``rdflib`` or the
``neo4j`` driver in the import graph.  Both are core dependencies, so this is
about keeping the CLI's import graph small, not about optional installs: any
rdflib or neo4j usage must be guarded by a lazy import inside the command body.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import click

from osprey_connectors.config import get_config_value

from .output import fail, note, report, warn


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
    """Seed OKF stub documents from the build's graph view.

    TTL is the graph view 'osprey build' writes, data/graph/facility.ttl under
    the render, or any other NARAD Turtle file.  Each stub's device_id is the
    facility file's device id, so the build links the page to its device.

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
    """Compile an authored LinkML schema into the ontology table.

    SOURCE is the LinkML schema to compile. OUTPUT is the JSON table to write;
    an existing file is replaced, and only once the new table is complete.

    The schema is where a machine's device vocabulary is authored: one class
    per kind of device, 'is_a' naming its parent, 'aliases' listing the
    synonyms a search should also answer to, and a 'DeviceFamily' enum mapping
    every FAMILY token of the channel database onto one of those classes. This
    verb turns that into the compiled table, and refuses a schema the table cannot represent rather than writing one that
    only fails when something later tries to load it.

    What is written is deterministic: classes and families in sorted order,
    synonyms sorted and de-duplicated, a trailing newline, and a header line
    saying the file is generated. Compiling an unchanged schema twice therefore
    leaves 'git diff' silent, which is what makes a committed table reviewable.

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
