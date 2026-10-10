"""Facility description commands.

``osprey facility validate`` runs every check ``osprey build`` makes of the
repo's ``data/facility/`` tree and renders every facility view, and writes
nothing. The checks are the build's own stages, run to the first that fails;
every error of that stage is printed, sorted, one line each on stderr. A clean
tree is then taken through the render of the repo's main profile, made in a
temporary directory with the build's ``--skip-deps`` semantics and discarded, so
every view :func:`osprey.facility.render.render_facility_outputs` writes is
checked against a real render without a file of it reaching the repo. On a clean
tree the response check runs before the render and prints one line per kept
response export. When the check left rows out of the comparison, a second line
for that model follows on stderr, giving the number of rows left out and the
count per reason (unwired, no width, unsolved, table calibration); it does not
change the verdict or the exit code. Persona and image renders are checked by ``osprey build``
alone.

``osprey facility show [--json] [ID]`` builds in memory as ``validate`` does
and prints what the build holds: the identity, the records per kind and the
wiring records per model, each model's engine, served flag and solve setting,
and each view with its path and whether the main render, the ``config.yml``
``osprey build`` writes into ``build/``, carries it, with the reason when it
does not. With an ID it prints that record, its provenance and the fixes
applied to it. ``--json`` prints one document on stdout and every other line on
stderr; an error leaves stdout empty.

``osprey facility import mml EXPORT...`` writes MML exports as the mml layer's
sources under ``data/facility/imported/mml/`` and seeds each authored file that
does not exist yet. An authored record source merges against the layer's
records, so the verb stops before it reads an export while one is present and
prints the ``rm`` line of each: every file of ``records/`` and ``decks/``,
``models.yaml``, and each of ``seeds.yaml``, ``limits.yaml``, ``identity.yaml``
and ``measurement/`` that does not open with the layer's own header line.
``fixes.yaml``, ``classes.yaml``, ``scenarios/`` and ``knowledge/`` are never in
the way. After the import it prints an ``rm`` line, on stderr, for each scenario
file without the layer's header line that names something the imported facility
does not have, and deletes none.
``--print-exporter`` prints the MATLAB exporter the layer ships and needs
neither a repo nor an export.

Note: the facility package and the build's render are imported inside the
command body, so ``osprey --help`` does not load them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from .output import fail, report
from .phase_reporter import current_reporter
from .repo_resolver import PROFILE_FILENAME, find_repo_root, repo_option

if TYPE_CHECKING:
    from .build_profile_load import LoadedProfile


@click.group()
def facility() -> None:
    """Import into, check and show the facility description under data/facility/."""


@facility.command("validate")
@repo_option
@click.pass_context
def validate(ctx: click.Context, repo: Path | None) -> None:
    """Check data/facility/ and render every facility view in memory; writes nothing.

    Exits 0 when the tree builds, every kept response export passes its check
    and every view renders; otherwise prints each error line to stderr and
    exits 1. Each model with a kept response export prints one
    ``response check <model>:`` line to stderr, pass or fail. When the check
    left rows out of the comparison, a second line for that model follows on
    stderr, giving the number of rows left out and the count per reason
    (unwired, no width, unsolved, table calibration); it does not change the
    verdict or the exit code.
    """
    _build_in_memory(ctx, repo)


@dataclass(frozen=True)
class _InMemoryBuild:
    """What a build made in memory leaves behind once its render is discarded.

    Attributes:
        repo_root: The deployment repo.
        document: The facility file.
        facility_dir: The repo's ``data/facility`` directory.
        rendered_config: The main render's ``config.yml``, as a nested mapping;
            empty unless the caller asked for it.
        primary_config: Where ``osprey build`` writes that ``config.yml``.
    """

    repo_root: Path
    document: dict[str, Any]
    facility_dir: Path
    rendered_config: dict[str, Any]
    primary_config: Path


def _build_in_memory(
    ctx: click.Context, repo: Path | None, *, read_config: bool = False
) -> _InMemoryBuild:
    """Run every check of ``osprey build`` and render the main profile in a scratch directory.

    Exits 1 through ``ctx`` on the first failing check, after printing its
    lines on stderr. ``read_config`` keeps the render's ``config.yml`` before
    the scratch directory goes.
    """
    import tempfile

    from osprey.errors import BuildProfileError
    from osprey.facility.build import LATER_STAGES
    from osprey.facility.render import facility_digest
    from osprey.facility.response_check import check_responses
    from osprey.facility.response_check import report as report_responses
    from osprey.facility.validate import report, run_stages

    from .build_cmd import _render_project, _render_zones, _rendered_config, _SharedRenderInputs
    from .profile_conventions import (
        PROJECT_MIRROR_DIR,
        facility_mirror_violation,
        handwritten_limits_violation,
    )
    from .templates.manager import TemplateManager

    repo_root = find_repo_root(repo)
    profile_path = repo_root / PROFILE_FILENAME

    mirror_stop = facility_mirror_violation(repo_root / PROJECT_MIRROR_DIR)
    if mirror_stop is not None:
        raise mirror_stop

    try:
        resolved, overlays = _main_profile(repo_root)
        name = _project_name(resolved, repo_root)
        facility_dir = _facility_dir(resolved, repo_root)
    except (BuildProfileError, ValueError, RuntimeError) as error:
        fail("The profile does not resolve.", str(error))
        ctx.exit(1)
    build_profile = resolved.profile
    limits_stop = handwritten_limits_violation(facility_dir.parent, repo_root)
    if limits_stop is not None:
        raise limits_stop

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

    build_dir = _render_zones(repo_root, name).build_dir
    shared = _SharedRenderInputs(
        repo_root=repo_root,
        build_dir=build_dir,
        runtime_root=None,
        project_deps=list(build_profile.dependencies or []),
        skip_deps=True,
        manager=TemplateManager(),
        graph_indexes={},
        graph_facts_reported=set(),
        model_facts_reported=set(),
        facility=document,
        facility_sha256=facility_digest(facility_dir),
        profile_overlays=overlays,
        tolerance_warnings=False,
    )
    # The render narrates what it builds on stdout; that narration is the
    # build's, so it is captured and dropped here and only trouble, which goes
    # to stderr, is printed.
    try:
        with (
            current_reporter().out().capture(),
            tempfile.TemporaryDirectory(prefix="osprey-facility-") as scratch,
        ):
            render_dir = _render_project(
                shared,
                resolved,
                profile_path=profile_path,
                project_name=name,
                output_dir=Path(scratch),
                deployment=False,
                progress=lambda *_args: None,
                repair=False,
            )
            rendered = _rendered_config(render_dir) if read_config else {}
    except (BuildProfileError, ValueError) as error:
        fail("The profile does not render.", str(error))
        ctx.exit(1)
    return _InMemoryBuild(
        repo_root=repo_root,
        document=document,
        facility_dir=facility_dir,
        rendered_config=rendered,
        primary_config=build_dir / "config.yml",
    )


#: Each record kind of the facility file: its key in the file and its id field.
_RECORD_KINDS: tuple[tuple[str, str, str], ...] = (
    ("place", "places", "id"),
    ("device", "devices", "id"),
    ("channel", "channels", "id"),
    ("group", "groups", "id"),
    ("model", "models", "name"),
)


def _counts(document: dict[str, Any]) -> dict[str, Any]:
    """The records per kind, and the wiring records per model."""
    counts: dict[str, Any] = {
        key: len(document.get(key) or []) for _kind, key, _id in _RECORD_KINDS[:4]
    }
    counts["wiring"] = {
        str(model["name"]): len(model.get("wiring") or [])
        for model in sorted(document.get("models") or [], key=lambda model: str(model["name"]))
    }
    return counts


def _views(built: _InMemoryBuild, served: list[str]) -> list[dict[str, Any]]:
    """Every view with its path and whether the main render carries it.

    ``reason`` is present only on a view the render does not carry.
    """
    from osprey.facility import views

    inputs = views.ViewInputs(
        doc=built.document,
        rendered_config=built.rendered_config,
        facility_dir=built.facility_dir,
        served=served,
    )
    rows = []
    for view in views.VIEWS:
        written, reason = view.written_when(inputs)
        row: dict[str, Any] = {
            "name": view.name,
            "path": (Path("data") / view.path).as_posix(),
            "written": written,
        }
        if not written:
            row["reason"] = reason
        rows.append(row)
    return rows


def _overview(built: _InMemoryBuild) -> dict[str, Any]:
    """The ``facility show`` document of the whole facility."""
    from osprey.facility.served import resolve_served
    from osprey.facility.views.facts import facts_document

    served = resolve_served(built.rendered_config, built.document)
    project_name = built.rendered_config.get("project_name")
    facts = facts_document(built.document, served, str(project_name) if project_name else None)
    return {
        "identity": facts["identity"],
        "counts": _counts(built.document),
        "models": facts["models"],
        "views": _views(built, served),
    }


def _record(document: dict[str, Any], record_id: str) -> dict[str, Any] | str:
    """The ``facility show ID`` document, or the line that says why there is none."""
    found = [
        (kind, record)
        for kind, key, id_field in _RECORD_KINDS
        for record in document.get(key) or []
        if str(record.get(id_field)) == record_id
    ]
    if not found:
        return f"facility show: no record {record_id}"
    if len(found) > 1:
        kinds = [f"a {kind}" for kind, _record in found]
        named = ", ".join(kinds[:-1]) + f" and {kinds[-1]}"
        return f"facility show: {record_id} names {named}"
    ((kind, record),) = found
    provenance = dict(record.get("provenance") or {})
    return {
        "record": {key: value for key, value in record.items() if key != "provenance"},
        "kind": kind,
        "provenance": provenance,
        "fixes_applied": list(provenance.get("fixes") or []),
    }


def _yaml_lines(value: Any) -> list[str]:
    import yaml

    text = yaml.safe_dump(value, sort_keys=False, allow_unicode=True, default_flow_style=False)
    return text.rstrip("\n").splitlines()


def _print_overview(document: dict[str, Any], primary_config: str) -> None:
    from .output import section

    identity = document["identity"]
    section(
        "identity",
        [(str(key), value) for key, value in identity.items() if value is not None],
    )
    counts = document["counts"]
    rows: list[tuple[str, object]] = [
        (key, value) for key, value in counts.items() if key != "wiring"
    ]
    rows += [(f"wiring {model}", value) for model, value in counts["wiring"].items()]
    section("counts", rows)
    section(
        "models",
        [
            (
                model["name"],
                f"{model['engine']}, {'served' if model['served'] else 'not served'}, "
                f"solve {model['solve'] if model['solve'] is not None else 'unset'}",
            )
            for model in document["models"]
        ],
    )
    section(
        f"views of {primary_config}",
        [
            (
                view["name"],
                f"{view['path']}, written"
                if view["written"]
                else f"{view['path']}, not written: {view['reason']}",
            )
            for view in document["views"]
        ],
    )


def _print_record(document: dict[str, Any]) -> None:
    record = document["record"]
    id_field = "name" if document["kind"] == "model" else "id"
    fields = {key: value for key, value in record.items() if key != id_field}
    provenance = {key: value for key, value in document["provenance"].items() if key != "fixes"}
    fixes = [f"  {fix['op']}: {fix['why']}" for fix in document["fixes_applied"]]
    lines = [
        f"{document['kind']} {record[id_field]}",
        *(f"  {line}" for line in _yaml_lines(fields)),
        "provenance",
        *(f"  {line}" for line in _yaml_lines(provenance)),
        "fixes applied",
        *(fixes or ["  none"]),
    ]
    for line in lines:
        report(line)


@facility.command("show")
@click.argument("record_id", metavar="[ID]", required=False)
@click.option("--json", "as_json", is_flag=True, help="Print one JSON document on stdout.")
@repo_option
@click.pass_context
def show(ctx: click.Context, record_id: str | None, as_json: bool, repo: Path | None) -> None:
    """Print the facility the repo builds, or the record ID names.

    Builds in memory as ``osprey facility validate`` does and exits 1 like it
    on an error. Without ID it prints the identity, the records per kind, the
    wiring per model, each model's engine, served flag and solve setting, and
    each view with its path and whether the main render carries it. With ID it
    prints that record with its provenance and the fixes applied to it; an ID
    that names no record, or more than one, exits 1. Under ``--json`` stdout
    holds one document and every other line goes to stderr.
    """
    import json
    from contextlib import nullcontext

    from . import output

    with output.machine_mode() if as_json else nullcontext():
        built = _build_in_memory(ctx, repo, read_config=True)
        if record_id is None:
            document = _overview(built)
        else:
            found = _record(built.document, record_id)
            if isinstance(found, str):
                click.echo(found, err=True)
                ctx.exit(1)
            document = found
    if as_json:
        click.echo(json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False))
    elif record_id is None:
        _print_overview(document, _shown(built.primary_config, built.repo_root))
    else:
        _print_record(document)


@facility.group("import")
def import_group() -> None:
    """Write an export as sources under data/facility/imported/."""


#: The authored files that merge against a layer's records whoever wrote them,
#: relative to ``data/facility/``: single files, then directories taken whole.
_RECORD_FILES = ("models.yaml",)
_RECORD_DIRS = ("decks",)

#: The directory of scenario sources, relative to ``data/facility/``.
_SCENARIOS_DIR = "scenarios"


def _main_profile(repo_root: Path) -> tuple[LoadedProfile, tuple[Path, ...]]:
    """The repo's resolved main profile and the overlays it was resolved with."""
    from .build_profile_resolve import resolve_build_document
    from .variant_selection import resolve_variant_selection

    variant = resolve_variant_selection(repo_root)
    overlays: tuple[Path, ...] = (variant.path,) if variant.path is not None else ()
    return resolve_build_document(repo_root / PROFILE_FILENAME, None, overlays), overlays


def _project_name(resolved: LoadedProfile, repo_root: Path) -> str:
    """The project's name as the build is given it: the profile's ``project_name:``.

    The checkout's folder name is read nowhere after ``osprey init``, so a
    profile that states none is refused here exactly as ``osprey build``
    refuses it, with the line to add.

    Raises:
        BuildProfileError: When the profile states no ``project_name``.
    """
    from .build_cmd import _profile_project_name

    return _profile_project_name(resolved.profile.project_name, repo_root)


def _facility_dir(resolved: LoadedProfile, repo_root: Path) -> Path:
    """The ``data/facility`` directory under a resolved profile's data root."""
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
    resolve. After the import it lists each scenario file that names
    something the facility no longer has, as rm lines, and deletes nothing.

    The MATLAB exporter mml_export.m ships with OSPREY and writes the export
    this command reads; no facility writes its own.

    Get the script with osprey facility import mml --print-exporter >
    mml_export.m, then copy it onto the MATLAB path of the Middle Layer host.

    Run your Middle Layer setpath for one sub-machine, load its simulator
    model, then run mml_export in MATLAB. The lattice is saved before the
    export samples anything. To write to another folder, run
    mml_export('/path/to/exports').

    \b
    Each run writes six files, named from AD.Machine and AD.SubMachine,
    lowercased:
      <machine>.<submachine>.lattice.mat    the lattice (THERING)
      <machine>.<submachine>.ao.json        the Accelerator Objects (getao)
      <machine>.<submachine>.ad.json        the Accelerator Data (getad)
      <machine>.<submachine>.va.json        calibrations, energy facts, nominals
      <machine>.<submachine>.response.json  the orbit response matrix
      <machine>.<submachine>.model.json     tune, chromaticity, dispersion

    Repeat the run for every sub-machine; each run writes its own six files.

    Import the .ao.json files, for example osprey facility import mml
    mymachine.storagering.ao.json mymachine.ltb.ao.json. Each one's siblings
    are read from beside it, and its sub-machine becomes the system name.

    The .model.json file is not imported.

    \b
    Requirements:
      - MATLAB R2016b or newer, started with its Java runtime.
      - The Middle Layer on the path and set up for the sub-machine.
      - The sub-machine's simulator model loaded, so THERING holds its
        lattice; the export refuses without it.
    """
    import shlex

    from osprey.errors import BuildProfileError
    from osprey.facility.layers.mml.importer import MappingProblems
    from osprey.facility.layers.mml.importer import import_mml as run_import
    from osprey.facility.layers.mml.mapping import MAPPING_FILE, MappingError
    from osprey.facility.validate import stale_scenarios

    repo_root = find_repo_root(repo)
    try:
        resolved = _main_profile(repo_root)[0]
        project_name = _project_name(resolved, repo_root)
        facility_dir = _facility_dir(resolved, repo_root)
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

    stale = sorted(
        path
        for path in (
            facility_dir / _SCENARIOS_DIR / f"{name}.yaml"
            for name in stale_scenarios(facility_dir, project_name=project_name)
        )
        if not _mml_seeded(path)
    )
    if stale:
        click.echo("these scenario files name channels that no longer exist:", err=True)
        for path in stale:
            lines = [f"  rm {shlex.quote(_shown(path, repo_root))}"]
            attached = path.with_suffix("")
            if attached.is_dir():
                lines.append(f"  rm -r {shlex.quote(_shown(attached, repo_root) + '/')}")
            click.echo("\n".join(lines), err=True)
