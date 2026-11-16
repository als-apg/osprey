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
settle-tolerance warning on stderr; it does not change the exit code. A facility
file that holds relation claims closes stderr with one ``relations:`` summary
line counting the claims by basis and by standing; a claim-free file prints
none. Render and image renders are checked by ``osprey build`` alone.

A profile that states ``machine:`` mounts a machine root: every verb reads and
writes machine facts there, through
:meth:`~osprey.cli.build_profile_model.BuildProfile.resolved_facility_roots`, and
keeps the deployment's ``knowledge/``, ``ariel/`` and ``benchmarks/`` under
``data/facility/``. ``validate`` then stops on any other entry of
``data/facility/`` (``machine-restated``).

``osprey facility show [--json] [TEXT]`` builds in memory as ``validate`` does
and prints what the build holds: the identity, the records per kind and the
wiring records per model, each model's engine, served flag and solve setting,
and each view with its path and whether the main render, the ``config.yml``
``osprey build`` writes into ``build/``, carries it, with the reason when it
does not. With TEXT it prints the records of that id, else the relation claim
of that id, else the places and devices TEXT names, each marked with how it
matched, with the claims a place or device is an end of, its provenance and the
fixes applied to it. ``--json`` prints one document on stdout and every other line on
stderr; an error leaves stdout empty.

``osprey facility import`` runs one step before the ``mml`` and ``list``
formats read their source, :func:`_replace_or_refuse`, and that step is the one place a format's verb
inspects the facility tree (while mounted, the machine root with the
deployment's content directories; each file is removed under its own root):

==========================================  ==========================================
what the tree holds                         what the step does
==========================================  ==========================================
hand-written records of a kind the format   refuses, naming them; nothing changes
writes, in a tree with no imported layer
hand-written records beside a layer         keeps them; the build combines them
bundled files                               replaces them and says so in one line
anything else of the deployment's own       keeps it
==========================================  ==========================================

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

``osprey facility import relations TABLE --layer NAME`` writes a relations
table, and the ``--names``, ``--places`` and ``--members`` tables beside it, as
the sources of ``data/facility/imported/<NAME>/``. It never runs
:func:`_replace_or_refuse`: bundled and hand-written records stay, the tables
resolve against them, and no file outside the layer's directory changes. Every
refusal prints one ``source-invalid`` line on stderr, and the verb then writes
no layer file and exits 1.

Note: the facility package and the build's render are imported inside the
command body, so ``osprey --help`` does not load them.
"""

from __future__ import annotations

from collections.abc import Mapping
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
    not change the exit code. When the facility file holds relation claims,
    one summary line closes stderr:
    ``relations: N claims (S stated, I inferred); A accepted, C contested``,
    with ``, X superseded`` appended when any claim is superseded; a facility
    without claims prints no such line.
    """
    built = _build_in_memory(ctx, repo)
    line = _relation_summary(built.document)
    if line is not None:
        click.echo(line, err=True)


def _relation_summary(document: dict[str, Any]) -> str | None:
    """The ``validate`` summary of the facility file's relation claims.

    Args:
        document: The built facility file.

    Returns:
        ``relations: N claims (S stated, I inferred); A accepted, C contested``
        with ``, X superseded`` appended when X is nonzero, or ``None`` when
        the file holds no claims. A claim without ``basis`` counts as stated.
    """
    from osprey.facility.relations import ACCEPTED, CONTESTED, SUPERSEDED

    claims = document.get("relations") or []
    if not claims:
        return None
    inferred = sum(1 for claim in claims if claim.get("basis", "stated") == "inferred")
    standing = [claim.get("standing") for claim in claims]
    superseded = standing.count(SUPERSEDED)
    line = (
        f"relations: {len(claims)} claims ({len(claims) - inferred} stated, "
        f"{inferred} inferred); {standing.count(ACCEPTED)} accepted, "
        f"{standing.count(CONTESTED)} contested"
    )
    return f"{line}, {superseded} superseded" if superseded else line


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
        _judges_kinds,
        _listed_kinds,
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
        facility_dir,
        project_name=name,
        later=LATER_STAGES,
        content_dir=content_dir,
        judges_kinds=_judges_kinds(build_profile, content_dir),
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
        # No wheel is built here, so a listed package's kind is found in its
        # project file.
        project_deps=list(
            _plan_project_venv(
                build_profile,
                install=False,
                listed_kinds=_listed_kinds(build_profile, repo_root),
            ).extra_deps
        ),
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
    """The records per kind, the relation claims, and the wiring records per model."""
    from osprey.facility.records import RECORD_KINDS

    counts: dict[str, Any] = {
        key: len(records.of_kind(kind)) for kind, key, _id in RECORD_KINDS[:4]
    }
    counts["relations"] = len(records.document.get("relations") or [])
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


def _entry(record: Mapping[str, Any], kind: str, match: str) -> dict[str, Any]:
    """One ``facility show TEXT`` entry: the record without its provenance, and how it matched."""
    provenance = dict(record.get("provenance") or {})
    return {
        "record": {key: value for key, value in record.items() if key != "provenance"},
        "kind": kind,
        "match": match,
        "provenance": provenance,
        "fixes_applied": list(provenance.get("fixes") or []),
    }


def _claims_of(claims: list[Mapping[str, Any]], kind: str, id_: str) -> list[dict[str, Any]]:
    """The claims with an end naming ``kind`` ``id_``, one row each.

    A row holds the claim's id, type, standing, the ``other`` end's id and the
    ``role`` the record plays: ``subject``, ``object``, or ``both`` when the
    claim names it at both ends (``other`` is then the record itself). ``unit``
    is the unit given on the claim's place end, the object's first.
    """
    from osprey.facility.relations import end_identity

    listed = []
    for claim in claims:
        ends: list[Mapping[str, Any]] = [
            end if isinstance(end := claim.get(side), Mapping) else {}
            for side in ("subject", "object")
        ]
        identities = [end_identity(end) for end in ends]
        named = [identity == (kind, id_) for identity in identities]
        if not any(named):
            continue
        role = "both" if all(named) else "subject" if named[0] else "object"
        other = identities[1] if role == "subject" else identities[0]
        row: dict[str, Any] = {
            "id": claim.get("id"),
            "type": claim.get("type"),
            "other": other[1] if other else None,
            "standing": claim.get("standing"),
            "role": role,
        }
        unit = next(
            (
                end["unit"]
                for end in reversed(ends)
                if end.get("unit") is not None and end.get("place") is not None
            ),
            None,
        )
        if unit is not None:
            row["unit"] = unit
        listed.append(row)
    return listed


def _record(document: dict[str, Any], text: str) -> list[dict[str, Any]] | str:
    """The ``facility show TEXT`` entries, or the line that says why there are none.

    ``text`` is tried as a record id (every kind, ``RECORD_KINDS`` order), then
    as a relation claim id, then as a place or device name
    (:meth:`~osprey.facility.records.FacilityRecords.lookup`); the first that
    finds anything answers. Each entry's ``match`` says how it matched (``id``,
    ``claim``, ``name``, ``segment``); a place or device entry lists in
    ``claims`` the claims it is an end of.
    """
    from osprey.facility.records import FacilityRecords

    records = FacilityRecords(document)
    claims = [claim for claim in document.get("relations") or [] if isinstance(claim, Mapping)]
    hits = [("id", each) for each in records.find(text)]
    if not hits:
        claimed = [claim for claim in claims if claim.get("id") == text]
        if claimed:
            return [_entry(claim, "relation", "claim") for claim in claimed]
        hits = list(records.lookup(text))
    if not hits:
        return f"facility show: no record {text}"
    entries = []
    for match, each in hits:
        entry = _entry(each.record, each.kind, match)
        if each.kind in ("place", "device"):
            entry["claims"] = _claims_of(claims, each.kind, each.id)
        entries.append(entry)
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


def _claim_direction(claim: Mapping[str, Any]) -> str:
    """A claim row read subject first, the shown record written ``(this)``.

    A subject-end row keeps the short ``type other`` form.
    """
    role, other = claim.get("role"), claim["other"]
    if role == "object":
        return f"{other} {claim['type']} (this)"
    if role == "both":
        return f"(this) {claim['type']} (this)"
    return f"{claim['type']} {other}"


def _print_record(document: dict[str, Any]) -> None:
    record = document["record"]
    id_field = "name" if document["kind"] == "model" else "id"
    fields = {key: value for key, value in record.items() if key != id_field}
    provenance = {key: value for key, value in document["provenance"].items() if key != "fixes"}
    fixes = [f"  {fix['op']}: {fix['why']}" for fix in document["fixes_applied"]]
    header = f"{document['kind']} {record[id_field]}"
    if document["match"] != "id":
        header += f" (match: {document['match']})"
    claims = [
        f"  {claim['id']}  {_claim_direction(claim)}, {claim['standing']}"
        + (f", unit {claim['unit']}" if "unit" in claim else "")
        for claim in document.get("claims") or []
    ]
    lines = [
        header,
        *(f"  {line}" for line in _yaml_lines(fields)),
        *(["claims", *claims] if claims else []),
        "provenance",
        *(f"  {line}" for line in _yaml_lines(provenance)),
        "fixes applied",
        *(fixes or ["  none"]),
    ]
    for line in lines:
        report(line)


@facility.command("show")
@click.argument("record_id", metavar="[TEXT]", required=False)
@click.option("--json", "as_json", is_flag=True, help="Print one JSON document on stdout.")
@repo_option
@click.pass_context
def show(ctx: click.Context, record_id: str | None, as_json: bool, repo: Path | None) -> None:
    """Print the facility the repo builds, or the records TEXT names.

    Builds in memory as ``osprey facility validate`` does and exits 1 like it
    on an error. Without TEXT it prints the identity, the records per kind, the
    wiring per model, the relation claims, each model's engine, served flag and
    solve setting, and each view with its path and whether the main render
    carries it. With TEXT it prints every record of that id; failing that, the
    relation claim of that id; failing that, every place and device TEXT names.
    Each entry shows its kind, how it matched, the claims a place or device is
    an end of (each read subject first, the record itself as ``(this)`` at an
    object end), its provenance and the fixes applied to it; TEXT that matches
    nothing exits 1. Under ``--json`` stdout holds one document and every other
    line goes to stderr.
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
        # Writes only under imported/<layer>/ and never replaces or refuses
        # another source (FR10), so it states no file of the tree.
        "relations": (),
    }


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


def _holds_a_layer(facility_dir: Path) -> bool:
    """Whether an import has written a layer under *facility_dir*.

    Every format's layer holds a ``channels.yaml``. A mapping or a source file
    waiting in the layer directory for a first import is no layer.
    """
    from osprey.facility.bundled import LAYER_ROOT

    return any((facility_dir / LAYER_ROOT).glob("*/channels.yaml"))


def _replace_or_refuse(
    ctx: click.Context, verb: str, roots: FacilityRoots, repo_root: Path
) -> None:
    """Make the facility tree ready for a format; every format under the group calls it first.

    Refuses only on overlap: hand-written records of the kinds this verb writes
    exit 1 with nothing changed, and files seeded under the former header are
    named apart with the line that fixes them. A tree that already holds an
    imported layer is past the first: records written by hand beside a layer
    are the facility's own additions, which the build combines with the layers,
    and no later import refuses them. Otherwise bundled files are replaced, and the
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
    overlap = [
        (path, kind)
        for path in kinds.authored
        for entry, kind in writes
        if path == entry or path.startswith(f"{entry}/")
    ]
    if overlap:
        former = [path for path, _kind in overlap if under_former_header(facility_dir / path)]
        written = [(path, kind) for path, kind in overlap if path not in former]
        if _holds_a_layer(facility_dir):
            written = []
        lines = []
        if written:
            found = {kind for _path, kind in written}
            named = _named_together([kind for _entry, kind in writes if kind in found])
            lines += [
                f"import {verb}: {_shown(facility_dir, repo_root)}/ holds hand-written {named} "
                "records, which this import writes:",
                *(_shown(facility_dir / path, repo_root) for path, _kind in written),
                "move corrections into fixes.yaml and remove the record sources, "
                "or start a new repo: osprey init <name> --preset demo",
            ]
        if former:
            lines += [
                f"import {verb}: {_shown(facility_dir, repo_root)}/ holds files seeded under "
                "the old header, which this import writes:",
                *(_shown(facility_dir / path, repo_root) for path in former),
                f"set line 1 of each to: {HEADER}",
            ]
        if lines:
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
    refuses hand-written records of the kinds this import writes. Exits 1 and
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
    from osprey.facility.layers.mml.importer import MappingProblems
    from osprey.facility.layers.mml.importer import import_mml as run_import
    from osprey.facility.layers.mml.mapping import MAPPING_FILE, MappingError
    from osprey_connectors.errors import BuildProfileError

    repo_root = find_repo_root(repo)
    try:
        roots = _facility_roots(_main_profile(repo_root)[0], repo_root)
    except (BuildProfileError, ValueError, RuntimeError) as error:
        fail("The profile does not resolve.", str(error))
        ctx.exit(1)

    _replace_or_refuse(ctx, "mml", roots, repo_root)
    facility_dir = roots.machine

    try:
        for path in run_import(list(exports), facility_dir):
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
    so in one line, refuses hand-written channel and device records, and seeds
    the knowledge stub. Exits 1 and writes no layer file when a row names both a device and a place, states an
    unknown role, has no address or repeats one, states s or model with no
    device or an s that is not a number, when two rows state different
    positions for one device, when the header names an unknown column, or
    while the profile does not resolve.
    """
    from osprey.facility.layers.list import import_list as run_import
    from osprey_connectors.errors import BuildProfileError

    repo_root = find_repo_root(repo)
    try:
        roots = _facility_roots(_main_profile(repo_root)[0], repo_root)
    except (BuildProfileError, ValueError, RuntimeError) as error:
        fail("The profile does not resolve.", str(error))
        ctx.exit(1)

    _replace_or_refuse(ctx, "list", roots, repo_root)

    for path in run_import(listing, roots.machine):
        report(f"wrote {_shown(path, repo_root)}")


_TABLE_PATH = click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=Path)


@import_group.command("relations")
@click.argument("table", metavar="TABLE", type=_TABLE_PATH)
@click.option("--layer", required=True, help="The layer's name; written to imported/<layer>/.")
@click.option("--names", type=_TABLE_PATH, help="A names table.")
@click.option("--places", type=_TABLE_PATH, help="A places table.")
@click.option("--members", type=_TABLE_PATH, help="A members table.")
@click.option(
    "--stub-unknown",
    is_flag=True,
    help="Stub a class-less device end the facility lacks as Unidentified.",
)
@repo_option
@click.pass_context
def import_relations(
    ctx: click.Context,
    table: Path,
    layer: str,
    names: Path | None,
    places: Path | None,
    members: Path | None,
    stub_unknown: bool,
    repo: Path | None,
) -> None:
    """Write relation tables as sources under data/facility/imported/<layer>/.

    A profile that states machine: imports into that machine root instead.

    \b
    Each table is UTF-8 CSV. A header row names its columns, in any order;
    empty cells are absent. Required columns come first:
      TABLE      type, subject, object; subject_kind, object_kind,
                 subject_port, object_port, object_unit, attrs, basis,
                 rule, subject_class, object_class
      --names    thing, name, scheme; basis, rule
      --places   place; within, basis, rule
      --members  group, member; member_kind, group_kind
    Every table also takes the citation columns document, version, state,
    locator and imported.

    A device end the facility does not have is stubbed only when its row
    states subject_class or object_class, or with --stub-unknown, which stubs
    it as Unidentified; place and group ends are never stubbed. Every end and
    name resolves against the whole facility except the layer being replaced,
    bundled and hand-written records included, and no file outside
    imported/<layer>/ changes. Exits 1 and writes no layer file when the layer
    name is refused, when any table has a row the layer refuses, or while the
    profile does not resolve.
    """
    from osprey.facility.layers.relations import import_relations as run_import
    from osprey_connectors.errors import BuildProfileError

    repo_root = find_repo_root(repo)
    try:
        roots = _facility_roots(_main_profile(repo_root)[0], repo_root)
    except (BuildProfileError, ValueError, RuntimeError) as error:
        fail("The profile does not resolve.", str(error))
        ctx.exit(1)

    written = run_import(
        table,
        roots.machine,
        layer,
        names=names,
        places=places,
        members=members,
        stub_unknown=stub_unknown,
    )
    for path in written:
        report(f"wrote {_shown(path, repo_root)}")
