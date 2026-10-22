"""The one spelling of a render's name and directory, and the projection of every render.

A deployment builds one project per tier file in ``tiers/`` and per app file in
``apps/``. Each render is named after the deployment's ``project_name`` and the
file's stem, and the same pair names its directory under ``build/``, its image
tag (``<project>:local``) and the ``project``/``project_path`` of its entry in
the render projection — ``modules.web_terminals.renders.<name>`` — the build
writes into ``build/config.yml``. Every producer of those names — the build
that renders a tier or app, the projection, the profile lint that runs before
any render, and the refusal that tells an operator where a render should be —
derives them here, so ``project == basename(project_path)`` holds by
construction.

A tier and a shared app are both rendered under that one spelling, so a name
that is both would put two renders in one directory. The naming refuses it:
:func:`render_kind_conflict` states the refusal and :func:`render_project`
raises it when told which names are tiers and which are apps.
"""

from __future__ import annotations

import copy
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

import yaml

from osprey_connectors.workspace import BUILD_DIR_NAME

__all__ = [
    "RENDERS_KEY",
    "RENDER_FILE_KNOBS",
    "RENDER_KIND_APP",
    "RENDER_KIND_TIER",
    "derived_renders",
    "render_project",
    "render_kind_conflict",
    "render_projection",
]

#: The kind of a render built from a tier (a person's access level).
RENDER_KIND_TIER = "tier"

#: The kind of a render built from a shared app.
RENDER_KIND_APP = "app"


def render_kind_conflict(
    name: str,
    kind: str,
    *,
    tiers: Collection[str] = (),
    apps: Collection[str] = (),
) -> str | None:
    """Why *name* cannot be rendered as *kind*, or ``None`` when it can.

    A name is a tier or an app, never both: the two would share the render
    ``build/<project>-<name>``, and a shared app card carries no access level
    of its own, so there is no tier-of-an-app to build either.

    Args:
        name: The render's name — a tier's or an app's file stem.
        kind: :data:`RENDER_KIND_TIER` or :data:`RENDER_KIND_APP`.
        tiers: The names the deployment uses as tiers.
        apps: The names the deployment lists under
            ``modules.web_terminals.shared``.

    Returns:
        The refusal, naming both uses, or ``None``.
    """
    if kind == RENDER_KIND_TIER and name in apps:
        return (
            f"{name!r} is used as a tier but is also a shared app in "
            f"modules.web_terminals.shared; a name is a tier or an app, never both "
            f"(both would render to build/<project>-{name}). Point the tier reference "
            f"at a file in tiers/, or open the app from its shared card"
        )
    if kind == RENDER_KIND_APP and name in tiers:
        return (
            f"shared app {name!r} in modules.web_terminals.shared is also used as a "
            f"tier; a name is a tier or an app, never both (both would render to "
            f"build/<project>-{name}). Rename the app or the tier"
        )
    return None


def render_project(
    project_name: str,
    render: str,
    *,
    kind: str | None = None,
    tiers: Collection[str] = (),
    apps: Collection[str] = (),
) -> tuple[str, str]:
    """The render name and repo-relative render directory of one tier or app.

    Args:
        project_name: The deployment's ``project_name`` (already
            compose-normalized — the profile refuses any other spelling).
        tier: The render's name: a tier's or an app's file stem.
        kind: :data:`RENDER_KIND_TIER` or :data:`RENDER_KIND_APP` to have the
            name checked against *tiers* and *apps*; ``None`` names the render
            without the check.
        tiers: The names the deployment uses as tiers.
        apps: The names the deployment lists as shared apps.

    Returns:
        ``(project, project_path)`` — ``<project_name>-<render>`` and
        ``build/<project_name>-<render>``.

    Raises:
        ValueError: When *kind* is given and :func:`render_kind_conflict`
            refuses the name.
    """
    if kind is not None:
        problem = render_kind_conflict(render, kind, tiers=tiers, apps=apps)
        if problem is not None:
            raise ValueError(problem)
    project = f"{project_name}-{render}"
    return project, f"{BUILD_DIR_NAME}/{project}"


#: The rendered-config key the build projects every render under.
RENDERS_KEY = "modules.web_terminals.renders"

#: Top-level keys of a tier or app file the projection carries, when set.
RENDER_FILE_KNOBS: tuple[str, ...] = ("extra_mounts", "seed_base", "image")


def render_projection(repo_root: Path, project_name: str) -> dict[str, dict[str, Any]]:
    """Every render the build makes of *repo_root*, as the rendered config states it.

    One entry per tier and app file
    (:func:`~osprey.cli.profile_root.layer_files`): ``kind`` from the file's
    directory, ``project`` and ``project_path`` from :func:`render_project`,
    and ``extra_mounts``, ``seed_base`` and ``image`` copied from the file's
    top level when it sets them. Values are copied verbatim; their
    well-formedness is lint's to judge. A file that does not parse as a mapping
    contributes its name only — the build that renders it names the problem.

    Args:
        repo_root: The deployment repo, the directory holding ``profile.yml``.
        project_name: The deployment's ``project_name``.

    Returns:
        ``{name: {kind, project, project_path, ...}}`` in enumeration order;
        empty when the repo has no tier or app.

    Raises:
        BuildProfileError: When two layer files share a stem — a tier and an
            app (:func:`render_kind_conflict`), or the same tier in ``tiers/``
            and the retired layer directory — since both would render to one
            directory.
    """
    # Imported in-function: the deployment layer loads none of the CLI at
    # import time.
    from osprey.cli.profile_root import ProfileKind, layer_files
    from osprey_connectors.errors import BuildProfileError

    projection: dict[str, dict[str, Any]] = {}
    sources: dict[str, Path] = {}
    for kind, path in layer_files(repo_root):
        name = path.stem
        render_kind = RENDER_KIND_APP if kind is ProfileKind.APP else RENDER_KIND_TIER
        if name in projection:
            raise BuildProfileError(
                _duplicate_render_problem(
                    name, render_kind, path, projection[name]["kind"], sources[name]
                )
            )
        sources[name] = path
        project, project_path = render_project(project_name, name)
        entry: dict[str, Any] = {
            "kind": render_kind,
            "project": project,
            "project_path": project_path,
        }
        top = _layer_top_level(path)
        for knob in RENDER_FILE_KNOBS:
            if knob in top:
                entry[knob] = copy.deepcopy(top[knob])
        projection[name] = entry
    return projection


def _duplicate_render_problem(
    name: str, kind: str, path: Path, first_kind: str, first_path: Path
) -> str:
    """Why two layer files with the stem *name* cannot both be rendered."""
    if kind != first_kind:
        tiers, apps = ((name,), ()) if first_kind == RENDER_KIND_TIER else ((), (name,))
        reason = render_kind_conflict(name, kind, tiers=tiers, apps=apps)
        return f"{first_path} and {path}: {reason}"
    return (
        f"{first_path} and {path} are both the tier {name!r}; a render is named by its "
        f"file, so both would render to build/<project>-{name}. Keep one of them"
    )


def _layer_top_level(path: Path) -> Mapping[str, Any]:
    """The top-level mapping of one tier or app file; empty when it has none."""
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return {}
    return document if isinstance(document, Mapping) else {}


def derived_renders(
    config: Mapping[str, Any],
    project_name: str,
    repo_root: Path | None,
    *,
    dispatch_tier: str | None = None,
) -> dict[str, dict[str, Any]]:
    """The ``renders`` keys the build writes into the root render of *repo_root*.

    One dotted ``modules.web_terminals.renders.<name>`` key per render
    (:func:`render_projection`), laid over the profile's own ``config:``
    entries — the profile never spells one; validation refuses it. Written when
    *config* stands up a web-terminal stack of its own, or when the deployment's
    dispatch workers run at a tier, whose image is named from this projection
    whether or not a terminal is deployed beside them. Empty otherwise — a tier
    or app render disables the module and is handed no dispatch tier, so a
    render never projects renders-of-a-render — and empty when there is no repo
    to enumerate.

    Args:
        config: A profile's ``config:`` block, in any spelling the profile
            accepts (dotted, nested or mixed).
        project_name: The deployment's ``project_name``.
        repo_root: The deployment repo, or ``None`` when there is none.
        dispatch_tier: The deployment's ``dispatch.tier``, or ``None`` when it
            sets none or *config* is a tier's or app's render.
    """
    from osprey.cli.build_profile_emit import effective_web_terminals

    serves_renders = bool(effective_web_terminals(config).get("enabled")) or bool(dispatch_tier)
    if repo_root is None or not serves_renders:
        return {}
    return {
        f"{RENDERS_KEY}.{name}": entry
        for name, entry in render_projection(Path(repo_root), project_name).items()
    }
