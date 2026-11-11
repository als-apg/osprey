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
change the verdict or the exit code. A device whose stated s lies outside its
periodic model's deck prints one warning on stderr; it does not change the exit
code. A scenario that moves a readback beyond its setpoint's tolerance, and a
lane whose float setpoints declare no tolerance, print the build's
settle-tolerance warning on stderr; it does not change the exit code. Render and image renders are checked by ``osprey build`` alone.

A profile that states ``machine:`` mounts a machine root: every verb reads and
writes machine facts there, through
:meth:`~osprey.cli.build_profile_model.BuildProfile.resolved_facility_roots`, and
keeps the deployment's ``knowledge/``, ``ariel/`` and ``benchmarks/`` under
``data/facility/``. ``validate`` then stops on any other entry of
``data/facility/`` (``machine-restated``).

``osprey facility show [--json] [ID]`` builds in memory as ``validate`` does
and prints what the build holds: the identity, the records per kind and the
wiring records per model, each model's engine, served flag and solve setting,
and each view with its path and whether the main render, the ``config.yml``
``osprey build`` writes into ``build/``, carries it, with the reason when it
does not. With an ID it prints that record, its provenance and the fixes
applied to it. ``--json`` prints one document on stdout and every other line on
stderr; an error leaves stdout empty.

``osprey facility import`` runs one step before every format reads its source,
:func:`_replace_or_refuse`, and that step is the one place a format's verb
inspects the facility tree (while mounted, the machine root with the
deployment's content directories; each file is removed under its own root):

==========================================  ==========================================
what the tree holds                         what the step does
==========================================  ==========================================
a hand-written record the import writes     refuses, naming it; nothing changes
a hand-written file of a kind the format    refuses, naming it; nothing changes
seeds once
bundled files                               replaces them and says so in one line
anything else of the deployment's own       keeps it
==========================================  ==========================================

A hand-written record is one the import writes when it carries the same id: a
channel, a device or a group by its ``id``, and a model, a deck
(``decks/<MODEL>.*``) and a measurement file (``measurement/<MODEL>.yaml``) by
the model's name. Every other hand-written record stays, so a machine written by
hand sits beside an import layer and the layer can be imported again. To compare
ids the step has the format derive its records first, which it does only when
the tree holds hand-written records of those kinds; a stop of the format then
comes before the bundled files are replaced. The seed-once kinds (seed, limit,
identity) are refused by file.

The knowledge indexes are then settled, and a blank index is left when no page
remains.

``osprey facility import mml EXPORT...`` writes MML exports as the mml layer's
sources under ``data/facility/imported/mml/`` and seeds each authored file that
does not exist yet. It writes channel, device, group, model, deck, seed, limit,
identity and measurement records.
``--print-exporter`` prints the MATLAB exporter the layer ships and needs
neither a repo nor an export.

``osprey facility import list FILE`` writes a CSV channel list as the list
layer's sources, ``data/facility/imported/list/channels.yaml`` and, when a row
states a device's ``s``, ``devices.yaml`` beside it. It writes channel and device
records. Every row the layer refuses prints one ``source-invalid`` line on
stderr, and the verb then writes no layer file and exits 1.

Note: the facility package and the build's render are imported inside the
command body, so ``osprey --help`` does not load them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from .output import fail, report
from .phase_reporter import current_reporter
from .repo_resolver import PROFILE_FILENAME, find_repo_root, repo_option

if TYPE_CHECKING:
    from osprey.facility.bundled import Classification
    from osprey.facility.records import FacilityRecords
    from osprey.facility.roots import FacilityRoots

    from .build_profile_load import LoadedProfile


@click.group()
def facility() -> None:
    """Import into, check and show the facility description under data/facility/.

    A profile that states machine: reads machine facts from that root and keeps
    only knowledge/, ariel/ and benchmarks/ under data/facility/.
    """


@facility.command("validate")
@repo_option
@click.pass_context
def validate(ctx: click.Context, repo: Path | None) -> None:
    """Check data/facility/ and render every facility view in memory; writes nothing.

    A profile that states machine: validates that machine root, with the
    deployment's knowledge/, ariel/ and benchmarks/ from data/facility/; any
    other entry of data/facility/ then stops with machine-restated.

    Exits 0 when the tree builds, every kept response export passes its check
    and every view renders; otherwise prints each error line to stderr and
    exits 1. Each model with a kept response export prints one
    ``response check <model>:`` line to stderr, pass or fail. When the check
    left rows out of the comparison, a second line for that model follows on
    stderr, giving the number of rows left out and the count per reason
    (unwired, no width, unsolved, table calibration); it does not change the
    verdict or the exit code. A device whose stated s lies outside its
    periodic model's deck prints one warning per device on stderr; it does
    not change the exit code. A scenario that moves a readback beyond its
    setpoint's tolerance, and a lane whose float setpoints declare no
    tolerance, print the build's settle-tolerance warning on stderr; it does
    not change the exit code.
    """
    _build_in_memory(ctx, repo)


@dataclass(frozen=True)
class _InMemoryBuild:
    """What a build made in memory leaves behind once its render is discarded.

    Attributes:
        repo_root: The deployment repo.
        document: The facility file.
        facility_dir: The machine root: ``machine:`` when the profile states
            it, otherwise the repo's ``data/facility`` directory.
        content_dir: The deployment's ``data/facility`` when it is apart from
            ``facility_dir``, otherwise ``None``.
        rendered_config: The main render's ``config.yml``, as a nested mapping;
            empty unless the caller asked for it.
        primary_config: Where ``osprey build`` writes that ``config.yml``.
    """

    repo_root: Path
    document: dict[str, Any]
    facility_dir: Path
    content_dir: Path | None
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

    from osprey.facility.build import LATER_STAGES
    from osprey.facility.layers.mml.response_check import check_responses
    from osprey.facility.layers.mml.response_check import report as report_responses
    from osprey.facility.render import facility_digest
    from osprey.facility.validate import report, report_warnings, run_stages
    from osprey_connectors.errors import BuildProfileError

    from .build_cmd import (
        _content_dir,
        _render_project,
        _render_zones,
        _rendered_config,
        _SharedRenderInputs,
        _stated_machine_route,
    )
    from .build_environment import _plan_project_venv
    from .profile_conventions import (
        PROJECT_MIRROR_DIR,
        facility_mirror_violation,
        handwritten_limits_violation,
        machine_restated_violation,
    )
    from .templates.manager import TemplateManager

    repo_root = find_repo_root(repo)
    profile_path = repo_root / PROFILE_FILENAME

    # The mirror guard checks the deployment's project/ before anything is
    # resolved; a stated machine root only changes where its remedy points.
    mirror_stop = facility_mirror_violation(
        repo_root / PROJECT_MIRROR_DIR, authoring_route=_stated_machine_route(profile_path)
    )
    if mirror_stop is not None:
        raise mirror_stop

    try:
        resolved, overlays = _main_profile(repo_root)
        name = _project_name(resolved, repo_root)
        roots = _facility_roots(resolved, repo_root)
    except (BuildProfileError, ValueError, RuntimeError) as error:
        fail("The profile does not resolve.", str(error))
        ctx.exit(1)
    build_profile = resolved.profile
    # The limits guard checks the deployment's data root; a mounted machine
    # only changes where its remedy says the limits go.
    data_root = build_profile.resolved_data_root(repo_root)
    assert data_root is not None  # resolved with the roots above
    limits_stop = handwritten_limits_violation(
        data_root, repo_root, machine_root=roots.machine if roots.mounted else None
    )
    if limits_stop is not None:
        raise limits_stop
    restated_stop = machine_restated_violation(roots)
    if restated_stop is not None:
        raise restated_stop

    facility_dir = roots.machine
    content_dir = _content_dir(roots)
    result = run_stages(
        facility_dir, project_name=name, later=LATER_STAGES, content_dir=content_dir
    )
    if not result.ok:
        report(result.errors)
        ctx.exit(1)
    report_warnings(result.validated.warnings)
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
        project_deps=list(_plan_project_venv(build_profile, install=False).extra_deps),
        skip_deps=True,
        manager=TemplateManager(),
        graph_indexes={},
        graph_facts_reported=set(),
        model_facts_reported=set(),
        facility=document,
        facility_sha256=facility_digest(facility_dir, content_dir=content_dir),
        profile_overlays=overlays,
        facility_roots=roots,
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
        content_dir=content_dir,
        rendered_config=rendered,
        primary_config=build_dir / "config.yml",
    )


def _counts(records: FacilityRecords) -> dict[str, Any]:
    """The records per kind, and the wiring records per model."""
    from osprey.facility.records import RECORD_KINDS

    counts: dict[str, Any] = {
        key: len(records.of_kind(kind)) for kind, key, _id in RECORD_KINDS[:4]
    }
    counts["wiring"] = {
        str(model["name"]): len(model.get("wiring") or [])
        for model in sorted(records.of_kind("model"), key=lambda model: str(model["name"]))
    }
    return counts


def _views(built: _InMemoryBuild, served: list[str]) -> list[dict[str, Any]]:
    """Every view with its path and whether the main render carries it.

    ``reason`` is present only on a view the render does not carry.
    """
    from osprey.facility import views
    from osprey.facility.records import FacilityRecords

    inputs = views.ViewInputs(
        records=FacilityRecords(built.document),
        rendered_config=built.rendered_config,
        facility_dir=built.facility_dir,
        served=served,
        content_dir=built.content_dir,
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
    from osprey.facility.records import FacilityRecords
    from osprey.facility.served import resolve_served
    from osprey.facility.views.facts import facts_document

    records = FacilityRecords(built.document)
    served = resolve_served(built.rendered_config, records)
    project_name = built.rendered_config.get("project_name")
    facts = facts_document(records, served, str(project_name) if project_name else None)
    return {
        "identity": facts["identity"],
        "counts": _counts(records),
        "models": facts["models"],
        "views": _views(built, served),
    }


def _record(document: dict[str, Any], record_id: str) -> list[dict[str, Any]] | str:
    """The ``facility show ID`` records, in ``RECORD_KINDS`` order, or the line that says why
    there are none."""
    from osprey.facility.records import FacilityRecords

    found = FacilityRecords(document).find(record_id)
    if not found:
        return f"facility show: no record {record_id}"
    entries = []
    for each in found:
        provenance = dict(each.record.get("provenance") or {})
        entries.append(
            {
                "record": {key: value for key, value in each.record.items() if key != "provenance"},
                "kind": each.kind,
                "provenance": provenance,
                "fixes_applied": list(provenance.get("fixes") or []),
            }
        )
    return entries


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
    prints every record of that id, each under its kind, with its provenance and
    the fixes applied to it; an ID that names no record exits 1. Under ``--json``
    stdout holds one document and every other line goes to stderr.
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
            document = {"records": found}
    if as_json:
        click.echo(json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False))
    elif record_id is None:
        _print_overview(document, _shown(built.primary_config, built.repo_root))
    else:
        for index, entry in enumerate(document["records"]):
            if index:
                report("")
            _print_record(entry)


@facility.group("import")
def import_group() -> None:
    """Write an export as sources under data/facility/imported/.

    A profile that states machine: imports into that machine root instead.
    """


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


def _facility_roots(resolved: LoadedProfile, repo_root: Path) -> FacilityRoots:
    """Where a resolved profile's machine facts and deployment content are read.

    Raises:
        FacilityBuildError: When a stated ``machine:`` root is missing or
            overlaps the data root.
    """
    roots = resolved.profile.resolved_facility_roots(repo_root)
    if roots is None:
        raise RuntimeError("a resolved profile names no data root")
    return roots


def _facility_dir(resolved: LoadedProfile, repo_root: Path) -> Path:
    """The machine root: ``machine:`` when the profile states it, else ``data/facility``."""
    return _facility_roots(resolved, repo_root).machine


def _shown(path: Path, repo_root: Path) -> str:
    """A path as the verb prints it: relative to the repo when it is inside it."""
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


#: The line an import prints when it seeds the blank knowledge index.
KNOWLEDGE_PLACEHOLDER_LINE = "Put your facility's documentation here: one page per topic."


def _writes() -> dict[str, tuple[tuple[str, str], ...]]:
    """Per import verb, what it writes under ``data/facility/``: (file or directory, kind)."""
    from osprey.facility.layers.mml.seed import (
        IDENTITY_FILE,
        LIMITS_FILE,
        MEASUREMENT_DIR,
        SEEDS_FILE,
    )

    channels = ("records/channels.yaml", "channel")
    devices = ("records/devices.yaml", "device")
    return {
        "mml": (
            channels,
            devices,
            ("records/groups.yaml", "group"),
            ("models.yaml", "model"),
            ("decks", "deck"),
            (SEEDS_FILE, "seed"),
            (LIMITS_FILE, "limit"),
            (IDENTITY_FILE, "identity"),
            (MEASUREMENT_DIR, "measurement"),
        ),
        "list": (channels, devices),
    }


#: The kinds an import seeds once as whole files, refused by file; every other
#: kind is refused by record.
_SEED_ONCE = frozenset({"seed", "limit", "identity"})

#: The kinds whose file is one record, named for its model.
_MODEL_FILES = frozenset({"deck", "measurement"})

#: How many colliding ids a refusal names per file.
_IDS_NAMED = 5


def _stated(path: Path, kind: str) -> set[str]:
    """What a hand-written file states of a kind: its records' ids, or its model's name.

    A file that does not parse states nothing here; the build stops on it.
    """
    import yaml

    from osprey.facility.sources import read_yaml

    if kind in _MODEL_FILES:
        return {path.name.split(".", 1)[0]}
    try:
        rows = read_yaml(path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, UnicodeDecodeError):
        return set()
    key = "name" if kind == "model" else "id"
    if not isinstance(rows, list):
        return set()
    return {row[key] for row in rows if isinstance(row, dict) and isinstance(row.get(key), str)}


def _named_ids(ids: list[str]) -> str:
    if len(ids) <= _IDS_NAMED:
        return ", ".join(ids)
    return ", ".join(ids[:_IDS_NAMED]) + f" and {len(ids) - _IDS_NAMED} more"


def _named_together(names: list[str]) -> str:
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + f" and {names[-1]}"


def _root_of(roots: FacilityRoots, relative: str) -> Path:
    """The root a path of the import's view lives under.

    The deployment's content directories are the content root's; everything
    else is the machine root's. For a single tree both are the same directory.
    """
    from osprey.facility.roots import DEPLOYMENT_CONTENT_DIRS

    if relative.split("/", 1)[0] in DEPLOYMENT_CONTENT_DIRS:
        return roots.content
    return roots.machine


def _classify(roots: FacilityRoots, template_root: Path) -> Classification:
    """Classify what an import sees: one tree, or a mounted machine with its content.

    While mounted, the import sees the machine root's entries and the content
    root's content directories, as the facility digest does, and classifies
    them together so the bundled facility is named from both. ``imported/``
    is left out of that union: it never names a bundled facility and the
    import does not act on its class.
    """
    import shutil
    import tempfile

    from osprey.facility.bundled import classify
    from osprey.facility.roots import DEPLOYMENT_CONTENT_DIRS, is_machine_entry

    if not roots.mounted:
        return classify(roots.machine, template_root)
    with tempfile.TemporaryDirectory(prefix="osprey-facility-import-") as scratch:
        view = Path(scratch)
        sources = [
            (path, path.relative_to(roots.machine))
            for path in roots.machine.rglob("*")
            if is_machine_entry(roots, path.relative_to(roots.machine))
        ]
        sources += [
            (path, path.relative_to(roots.content))
            for name in DEPLOYMENT_CONTENT_DIRS
            for path in (roots.content / name).rglob("*")
        ]
        for path, relative in sources:
            if path.is_file() and relative.parts[0] != "imported":
                (view / relative).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, view / relative)
        return classify(view, template_root)


def _remove_bundled(roots: FacilityRoots, bundled: tuple[str, ...]) -> None:
    """Remove the bundled files, each under its own root.

    Their directories stay until :func:`_prune_empty`.
    """
    for relative in bundled:
        (_root_of(roots, relative) / relative).unlink()


def _prune_empty(facility_dir: Path) -> None:
    """Remove every empty directory under *facility_dir*, deepest first.

    Version-control metadata is left as it is: a machine root that is a clone
    keeps its ``.git``.
    """
    from osprey.facility.roots import is_vcs_metadata

    directories = (
        path
        for path in facility_dir.rglob("*")
        if path.is_dir() and not is_vcs_metadata(path.relative_to(facility_dir))
    )
    for directory in sorted(directories, key=lambda path: -len(path.parts)):
        if not any(directory.iterdir()):
            directory.rmdir()


def _replace_or_refuse(
    ctx: click.Context,
    verb: str,
    roots: FacilityRoots,
    repo_root: Path,
    planned: Callable[[], dict[str, frozenset[str]]],
) -> None:
    """Make the facility tree ready for a format; every format under the group calls it first.

    Refuses only on overlap: a hand-written record this import writes, or a
    hand-written file of a kind it seeds once, exits 1 with nothing changed, and
    files seeded under the former header are named apart with the line that
    fixes them. A record overlaps by its id, and a deck and a measurement file by
    the model they are named for; *planned* returns the ids the import writes per
    kind and is called only when the tree holds hand-written records of those
    kinds, before anything is replaced. Otherwise bundled files are replaced, and the
    line that says so prints at once, so a later stop of the format never hides
    the deletion. Everything else stays; the knowledge indexes are settled, and a
    blank index is left when no page remains.

    While mounted, the machine root and the content root's content directories
    are classified together; each bundled file is removed under its own root,
    and the knowledge indexes settled are the content root's.
    """
    from osprey.facility.bundled import under_former_header
    from osprey.facility.layers import HEADER
    from osprey.services.facility_knowledge.okf.index import _settle_bundle

    from .templates.manager import TemplateManager

    facility_dir = roots.machine
    writes = _writes()[verb]
    kinds = _classify(roots, Path(TemplateManager().template_root))
    overlap: list[tuple[str, str, str]] = []
    for path in kinds.authored:
        for entry, kind in writes:
            if path != entry and not path.startswith(f"{entry}/"):
                continue
            if kind in _SEED_ONCE:
                overlap.append((path, kind, ""))
                continue
            written_ids = planned()["model" if kind in _MODEL_FILES else kind]
            ids = sorted(_stated(facility_dir / path, kind) & written_ids)
            if ids:
                overlap.append((path, kind, "" if kind in _MODEL_FILES else _named_ids(ids)))
    if overlap:
        former = [path for path, _kind, _ids in overlap if under_former_header(facility_dir / path)]
        written = [(path, kind, ids) for path, kind, ids in overlap if path not in former]
        lines = []
        if written:
            found = {kind for _path, kind, _ids in written}
            named = _named_together([kind for _entry, kind in writes if kind in found])
            lines += [
                f"import {verb}: {_shown(facility_dir, repo_root)}/ holds hand-written {named} "
                "records, which this import writes:",
                *(
                    _shown(facility_dir / path, repo_root) + (f": {ids}" if ids else "")
                    for path, _kind, ids in written
                ),
                "move corrections into fixes.yaml and remove these records, "
                "or start a new repo: osprey init <name> --preset demo",
            ]
        if former:
            lines += [
                f"import {verb}: {_shown(facility_dir, repo_root)}/ holds files seeded under "
                "the old header, which this import writes:",
                *(_shown(facility_dir / path, repo_root) for path in former),
                f"set line 1 of each to: {HEADER}",
            ]
        click.echo("\n".join(lines), err=True)
        ctx.exit(1)
    if kinds.bundled:
        _remove_bundled(roots, kinds.bundled)
        click.echo(
            f"import {verb}: replaced the bundled {kinds.name} facility "
            f"({len(kinds.bundled)} files)",
            err=True,
        )
    seeded = _settle_bundle(roots.content / "knowledge")
    if kinds.bundled:
        _prune_empty(roots.machine)
        if roots.mounted:
            _prune_empty(roots.content)
    if seeded is not None:
        report(f"wrote {_shown(seeded, repo_root)}")
        report(KNOWLEDGE_PLACEHOLDER_LINE)


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

    A profile that states machine: imports into that machine root instead:
    imported/mml/ and every seeded file go there, and the knowledge index is
    settled under data/facility/knowledge/.

    EXPORT is an export's <stem>.ao.json file; its sibling files are read from
    beside it. First replaces a bundled facility, saying so in one line, and
    refuses a hand-written record this import writes; every other hand-written
    record stays. Exits 1 and
    writes no record while the mapping is a draft, has an undecided slot, has
    the wrong structure or fails its check, or while the profile does not
    resolve.

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
    from functools import cache

    from osprey.facility.layers.mml.importer import MappingProblems, plan_mml
    from osprey.facility.layers.mml.mapping import MAPPING_FILE, MappingError
    from osprey_connectors.errors import BuildProfileError

    repo_root = find_repo_root(repo)
    try:
        roots = _facility_roots(_main_profile(repo_root)[0], repo_root)
    except (BuildProfileError, ValueError, RuntimeError) as error:
        fail("The profile does not resolve.", str(error))
        ctx.exit(1)

    facility_dir = roots.machine
    plan = cache(lambda: plan_mml(list(exports), facility_dir))

    try:
        _replace_or_refuse(ctx, "mml", roots, repo_root, lambda: plan().writes)
        for path in plan().write():
            report(f"wrote {_shown(path, repo_root)}")
    except MappingProblems as stop:
        raise MappingProblems(Path(_shown(stop.path, repo_root)), stop.problems) from None
    except MappingError as error:
        mapping = _shown(facility_dir / MAPPING_FILE, repo_root)
        fail(f"{mapping} is not a valid mapping document.", str(error))
        ctx.exit(1)


@import_group.command("list")
@click.argument(
    "listing",
    metavar="FILE",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=Path),
)
@repo_option
@click.pass_context
def import_list(ctx: click.Context, listing: Path, repo: Path | None) -> None:
    """Write a CSV channel list as sources under data/facility/imported/list/.

    A profile that states machine: imports into that machine root instead,
    and the knowledge stub is seeded under data/facility/knowledge/.

    \b
    FILE is UTF-8 CSV. A header row names its columns, in any order:
      address      the channel's full address; required
      role         setpoint, readback or none; empty means readback
      pair         a setpoint's readback address
      device       the device the channel belongs to
      place        the place the channel belongs to
      unit         free text
      description  free text
      tags         separated by ;
      s            the device's position in metres along model
      model        the model whose deck s is measured in
    A file without an address column holds one address per line.

    Each device a row states s for is written with its s and model to
    devices.yaml beside the channels. First replaces a bundled facility, saying
    so in one line, refuses a hand-written channel or device record this
    import writes, and seeds
    the knowledge stub. Exits 1 and writes no layer file when a row names both a device and a place, states an
    unknown role, has no address or repeats one, states s or model with no
    device or an s that is not a number, when two rows state different
    positions for one device, when the header names an unknown column, or
    while the profile does not resolve.
    """
    from functools import cache

    from osprey.facility.layers.list import plan_list
    from osprey_connectors.errors import BuildProfileError

    repo_root = find_repo_root(repo)
    try:
        roots = _facility_roots(_main_profile(repo_root)[0], repo_root)
    except (BuildProfileError, ValueError, RuntimeError) as error:
        fail("The profile does not resolve.", str(error))
        ctx.exit(1)

    plan = cache(lambda: plan_list(listing))
    _replace_or_refuse(ctx, "list", roots, repo_root, lambda: plan().writes)

    for path in plan().write(roots.machine):
        report(f"wrote {_shown(path, repo_root)}")
