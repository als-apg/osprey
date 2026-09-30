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

Note: the facility package and the build's render are imported inside the
command body, so ``osprey --help`` does not load them.
"""

from __future__ import annotations

from pathlib import Path

import click

from .output import fail
from .phase_reporter import current_reporter
from .repo_resolver import PROFILE_FILENAME, find_repo_root, repo_option


@click.group()
def facility() -> None:
    """Check the facility description under data/facility/."""


@facility.command("validate")
@repo_option
@click.pass_context
def validate(ctx: click.Context, repo: Path | None) -> None:
    """Check data/facility/ and render every facility view in memory; writes nothing.

    Exits 0 and prints nothing when the tree builds and every view renders;
    otherwise prints each error line to stderr and exits 1.
    """
    import tempfile

    from osprey.errors import BuildProfileError
    from osprey.facility.build import LATER_STAGES
    from osprey.facility.render import facility_digest
    from osprey.facility.validate import report, run_stages

    from .build_cmd import _render_project, _render_zones, _SharedRenderInputs
    from .build_profile_resolve import resolve_build_document
    from .templates.manager import TemplateManager
    from .variant_selection import resolve_variant_selection

    repo_root = find_repo_root(repo)
    name = repo_root.name
    profile_path = repo_root / PROFILE_FILENAME

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
            )
    except (BuildProfileError, ValueError) as error:
        fail("The profile does not render.", str(error))
        ctx.exit(1)
