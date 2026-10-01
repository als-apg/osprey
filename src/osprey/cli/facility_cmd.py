"""Facility description commands.

``osprey facility validate`` runs every check ``osprey build`` makes of the
repo's ``data/facility/`` tree and renders every facility view, and writes
nothing. The checks are the build's own stages, run to the first that fails;
every error of that stage is printed, sorted, one line each on stderr. A clean
tree is then taken through the render of the repo's main profile, made in a
temporary directory with the build's ``--skip-deps`` semantics and discarded, so
every view :func:`osprey.facility.render.render_facility_outputs` writes is
checked against a real render without a file of it reaching the repo. Persona
and image renders are checked by ``osprey build`` alone.

``osprey facility import mml EXPORT...`` writes MML exports as the mml layer's
sources under ``data/facility/imported/mml/`` and seeds each authored file that
does not exist yet. An authored record source merges against the layer's
records, so the verb stops before it reads an export while one is present and
prints the ``rm`` line of each: every file of ``records/`` and ``decks/``,
``models.yaml``, and each of ``seeds.yaml``, ``limits.yaml``, ``identity.yaml``
and ``measurement/`` that does not open with the layer's own header line.
``fixes.yaml``, ``classes.yaml`` and ``knowledge/`` are never in the way.
``--print-exporter`` prints the MATLAB exporter the layer ships and needs
neither a repo nor an export.

Note: the facility package and the build's render are imported inside the
command body, so ``osprey --help`` does not load them.
"""

from __future__ import annotations

from pathlib import Path

import click

from .output import fail, report
from .phase_reporter import current_reporter
from .repo_resolver import PROFILE_FILENAME, find_repo_root, repo_option


@click.group()
def facility() -> None:
    """Import into and check the facility description under data/facility/."""


@facility.command("validate")
@repo_option
@click.pass_context
def validate(ctx: click.Context, repo: Path | None) -> None:
    """Check data/facility/ and render every facility view in memory; writes nothing.

    Exits 0 when the tree builds, every kept response export passes its check
    and every view renders; otherwise prints each error line to stderr and
    exits 1. Each model with a kept response export prints one
    ``response check <model>:`` line to stderr, pass or fail.
    """
    import tempfile

    from osprey.errors import BuildProfileError
    from osprey.facility.build import LATER_STAGES
    from osprey.facility.render import facility_digest
    from osprey.facility.response_check import check_responses
    from osprey.facility.response_check import report as report_responses
    from osprey.facility.validate import report, run_stages

    from .build_cmd import _render_project, _render_zones, _SharedRenderInputs
    from .build_profile_resolve import resolve_build_document
    from .profile_conventions import PROJECT_MIRROR_DIR, facility_mirror_violation
    from .templates.manager import TemplateManager
    from .variant_selection import resolve_variant_selection

    repo_root = find_repo_root(repo)
    name = repo_root.name
    profile_path = repo_root / PROFILE_FILENAME

    mirror_stop = facility_mirror_violation(repo_root / PROJECT_MIRROR_DIR)
    if mirror_stop is not None:
        raise mirror_stop

    variant = resolve_variant_selection(repo_root)
    overlays: tuple[Path, ...] = (variant.path,) if variant.path is not None else ()
    resolved = resolve_build_document(profile_path, None, overlays)
    build_profile = resolved.profile
    data_root = build_profile.resolved_data_root(repo_root)
    if data_root is None:
        raise RuntimeError("a resolved profile names no data root")
    facility_dir = data_root / "facility"

    result = run_stages(facility_dir, project_name=name, later=LATER_STAGES)
    if not result.ok:
        report(result.errors)
        ctx.exit(1)
    document = result.validated.document
    if document is None:
        raise RuntimeError("the stages ran clean without producing the document")

    try:
        responses = check_responses(facility_dir, document)
    except (OSError, ValueError) as error:
        fail("The response check cannot run.", str(error))
        ctx.exit(1)
    if not report_responses(responses):
        ctx.exit(1)

    shared = _SharedRenderInputs(
        repo_root=repo_root,
        build_dir=_render_zones(repo_root).build_dir,
        runtime_root=None,
        project_deps=list(build_profile.dependencies or []),
        skip_deps=True,
        manager=TemplateManager(),
        va_manifests={},
        va_reported=set(),
        graph_indexes={},
        graph_facts_reported=set(),
        model_facts_reported=set(),
        facility=document,
        facility_sha256=facility_digest(facility_dir),
        profile_overlays=overlays,
    )
    # The render narrates what it builds on stdout; that narration is the
    # build's, so it is captured and dropped here and only trouble, which goes
    # to stderr, is printed.
    try:
        with (
            current_reporter().out().capture(),
            tempfile.TemporaryDirectory(prefix="osprey-facility-") as scratch,
        ):
            _render_project(
                shared,
                resolved,
                profile_path=profile_path,
                project_name=name,
                output_dir=Path(scratch),
                deployment=False,
                progress=lambda *_args: None,
                repair=False,
            )
    except (BuildProfileError, ValueError) as error:
        fail("The profile does not render.", str(error))
        ctx.exit(1)


@facility.group("import")
def import_group() -> None:
    """Write an export as sources under data/facility/imported/."""


#: The authored files that merge against a layer's records whoever wrote them,
#: relative to ``data/facility/``: single files, then directories taken whole.
_RECORD_FILES = ("models.yaml",)
_RECORD_DIRS = ("decks",)


def _facility_dir(repo_root: Path) -> Path:
    """The ``data/facility`` directory of the repo's main profile."""
    from .build_profile_resolve import resolve_build_document
    from .variant_selection import resolve_variant_selection

    variant = resolve_variant_selection(repo_root)
    overlays: tuple[Path, ...] = (variant.path,) if variant.path is not None else ()
    resolved = resolve_build_document(repo_root / PROFILE_FILENAME, None, overlays)
    data_root = resolved.profile.resolved_data_root(repo_root)
    if data_root is None:
        raise RuntimeError("a resolved profile names no data root")
    return data_root / "facility"


def _files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.rglob("*") if path.is_file())


def _mml_seeded(path: Path) -> bool:
    """Whether a file opens with the header line the mml layer seeds files under."""
    from osprey.facility.layers.mml.seed import HEADER

    # Only the newline ends the first line: str.splitlines would also end it at
    # a form feed or a Unicode line separator and pass a line that runs on.
    first = path.read_text(encoding="utf-8", errors="replace").split("\n", 1)[0]
    return first == HEADER


def _authored_record_sources(facility_dir: Path) -> list[Path]:
    """The authored files that would merge against the mml layer's records, sorted.

    Every file of ``records/`` and ``decks/`` and ``models.yaml`` count
    whatever they hold; ``seeds.yaml``, ``limits.yaml``, ``identity.yaml`` and
    the files of ``measurement/`` count unless the mml layer seeded them.
    """
    from osprey.facility.layers.mml.seed import (
        IDENTITY_FILE,
        LIMITS_FILE,
        MEASUREMENT_DIR,
        SEEDS_FILE,
    )

    found = sorted((facility_dir / "records").glob("*.yaml"))
    found += [facility_dir / name for name in _RECORD_FILES if (facility_dir / name).is_file()]
    for name in _RECORD_DIRS:
        found += _files(facility_dir / name)
    seedable = [
        facility_dir / name
        for name in (SEEDS_FILE, LIMITS_FILE, IDENTITY_FILE)
        if (facility_dir / name).is_file()
    ]
    seedable += _files(facility_dir / MEASUREMENT_DIR)
    found += [path for path in seedable if not _mml_seeded(path)]
    return sorted(found)


def _shown(path: Path, repo_root: Path) -> str:
    """A path as the verb prints it: relative to the repo when it is inside it."""
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def _print_exporter(ctx: click.Context, _param: click.Parameter, value: bool) -> None:
    if not value or ctx.resilient_parsing:
        return
    from importlib.resources import files

    exporter = files("osprey.facility.layers.mml").joinpath("mml_export.m")
    click.echo(exporter.read_text(encoding="utf-8"), nl=False)
    ctx.exit(0)


@import_group.command("mml")
@click.argument(
    "exports",
    metavar="EXPORT...",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=Path),
)
@click.option(
    "--print-exporter",
    is_flag=True,
    is_eager=True,
    expose_value=False,
    callback=_print_exporter,
    help="Print the MATLAB exporter and exit; needs no repo and no EXPORT.",
)
@repo_option
@click.pass_context
def import_mml(ctx: click.Context, exports: tuple[Path, ...], repo: Path | None) -> None:
    """Write MML exports as sources under data/facility/imported/mml/.

    EXPORT is an export's <stem>.ao.json file; its sibling files are read from
    beside it. Exits 1 and writes no record while the mapping is a draft, has
    an undecided slot, has the wrong structure or fails its check, while
    an authored record source is present, or while the profile does not
    resolve.
    """
    import shlex

    from osprey.errors import BuildProfileError
    from osprey.facility.layers.mml.importer import MappingProblems
    from osprey.facility.layers.mml.importer import import_mml as run_import
    from osprey.facility.layers.mml.mapping import MAPPING_FILE, MappingError

    repo_root = find_repo_root(repo)
    try:
        facility_dir = _facility_dir(repo_root)
    except (BuildProfileError, ValueError, RuntimeError) as error:
        fail("The profile does not resolve.", str(error))
        ctx.exit(1)

    present = _authored_record_sources(facility_dir)
    if present:
        noun = "file" if len(present) == 1 else "files"
        click.echo(f"import mml: authored-present: {len(present)} {noun}", err=True)
        for path in present:
            click.echo(f"rm {shlex.quote(_shown(path, repo_root))}", err=True)
        ctx.exit(1)

    try:
        for path in run_import(list(exports), facility_dir):
            report(f"wrote {_shown(path, repo_root)}")
    except MappingProblems as stop:
        raise MappingProblems(Path(_shown(stop.path, repo_root)), stop.problems) from None
    except MappingError as error:
        mapping = _shown(facility_dir / MAPPING_FILE, repo_root)
        fail(f"{mapping} is not a valid mapping document.", str(error))
        ctx.exit(1)
