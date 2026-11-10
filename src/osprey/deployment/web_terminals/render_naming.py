"""The one spelling of a render's name and directory, and the projection of every render.

A deployment builds one project per login file in ``tiers/``. Each render is
named after the deployment's ``project_name`` and the file's stem, and the same
pair names its directory under ``build/``, its image tag (``<project>:local``)
and the ``project``/``project_path`` of its entry in the render projection —
``modules.web_terminals.renders.<name>`` — the build writes into
``build/config.yml``. Every producer of those names — the build that renders a
login, the projection, the profile lint that runs before any render, and the
refusal that tells an operator where a render should be — derives them here, so
``project == basename(project_path)`` holds by construction.

A login is used one of two ways: as a person's tier, or as a shared card that
``modules.web_terminals.shared`` names. One file has one audience, so a name
used both ways is refused: :func:`render_kind_conflict` states the refusal and
:func:`render_project` raises it when told which names are used which way.
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
    "RENDER_KIND_SHARED",
    "RENDER_KIND_TIER",
    "derived_renders",
    "render_project",
    "render_kind_conflict",
    "render_projection",
]

#: The kind of every render: one built from a login file in ``tiers/``, and the
#: use of a render as a person's access level.
RENDER_KIND_TIER = "tier"

#: The use of a render as a shared card: ``modules.web_terminals.shared``
#: names it.
RENDER_KIND_SHARED = "shared"


def render_kind_conflict(
    name: str,
    kind: str,
    *,
    tiers: Collection[str] = (),
    shared: Collection[str] = (),
) -> str | None:
    """Why *name* cannot be used as *kind*, or ``None`` when it can.

    A login is a person's tier or a shared card, never both: the two uses
    would share the render ``build/<project>-<name>``, and a shared card is
    sealed read-only for everyone on the roster, so there is no person's
    access level to build from the same file.

    Args:
        name: The render's name — a login file's stem.
        kind: :data:`RENDER_KIND_TIER` or :data:`RENDER_KIND_SHARED`.
        tiers: The names the deployment uses as tiers.
        shared: The names the deployment lists under
            ``modules.web_terminals.shared``.

    Returns:
        The refusal, naming both uses, or ``None``.
    """
    if kind == RENDER_KIND_TIER and name in shared:
        return (
            f"{name!r} is used as a tier but is also a shared login in "
            f"modules.web_terminals.shared; a login is a tier or a shared card, never "
            f"both (both would render to build/<project>-{name}). Point the tier "
            f"reference at another file in tiers/, or open the login from its shared card"
        )
    if kind == RENDER_KIND_SHARED and name in tiers:
        return (
            f"shared login {name!r} in modules.web_terminals.shared is also used as a "
            f"tier; a login is a tier or a shared card, never both (both would render to "
            f"build/<project>-{name}). Rename one of the two files"
        )
    return None


def render_project(
    project_name: str,
    render: str,
    *,
    kind: str | None = None,
    tiers: Collection[str] = (),
    shared: Collection[str] = (),
) -> tuple[str, str]:
    """The render name and repo-relative render directory of one login.

    Args:
        project_name: The deployment's ``project_name`` (already
            compose-normalized — the profile refuses any other spelling).
        render: The render's name: a login file's stem.
        kind: :data:`RENDER_KIND_TIER` or :data:`RENDER_KIND_SHARED` to have
            the name checked against *tiers* and *shared*; ``None`` names the
            render without the check.
        tiers: The names the deployment uses as tiers.
        shared: The names the deployment lists as shared logins.

    Returns:
        ``(project, project_path)`` — ``<project_name>-<render>`` and
        ``build/<project_name>-<render>``.

    Raises:
        ValueError: When *kind* is given and :func:`render_kind_conflict`
            refuses the name.
    """
    if kind is not None:
        problem = render_kind_conflict(render, kind, tiers=tiers, shared=shared)
        if problem is not None:
            raise ValueError(problem)
    project = f"{project_name}-{render}"
    return project, f"{BUILD_DIR_NAME}/{project}"


#: The rendered-config key the build projects every render under.
RENDERS_KEY = "modules.web_terminals.renders"

#: Top-level keys of a login file the projection carries, when set.
RENDER_FILE_KNOBS: tuple[str, ...] = ("extra_mounts", "seed_base", "image")


def render_projection(repo_root: Path, project_name: str) -> dict[str, dict[str, Any]]:
    """Every render the build makes of *repo_root*, as the rendered config states it.

    One entry per login file
    (:func:`~osprey.cli.profile_root.layer_files`): ``kind``
    (:data:`RENDER_KIND_TIER`), ``project`` and ``project_path`` from :func:`render_project`,
    and ``extra_mounts``, ``seed_base`` and ``image`` copied from the file's
    top level when it sets them. Values are copied verbatim; their
    well-formedness is lint's to judge. A file that does not parse as a mapping
    contributes its name only — the build that renders it names the problem.

    Args:
        repo_root: The deployment repo, the directory holding ``profile.yml``.
        project_name: The deployment's ``project_name``.

    Returns:
        ``{name: {kind, project, project_path, ...}}`` in enumeration order;
        empty when the repo has no login.

    Raises:
        BuildProfileError: When two layer files share a stem — the same login
            in ``tiers/`` and the retired layer directory — since both would
            render to one directory.
    """
    # Imported in-function: the deployment layer loads none of the CLI at
    # import time.
    from osprey.cli.profile_root import layer_files
    from osprey_connectors.errors import BuildProfileError

    projection: dict[str, dict[str, Any]] = {}
    sources: dict[str, Path] = {}
    for _kind, path in layer_files(repo_root):
        name = path.stem
        if name in projection:
            raise BuildProfileError(
                f"{sources[name]} and {path} are both the login {name!r}; a render is "
                f"named by its file, so both would render to build/<project>-{name}. "
                "Keep one of them"
            )
        sources[name] = path
        project, project_path = render_project(project_name, name)
        entry: dict[str, Any] = {
            "kind": RENDER_KIND_TIER,
            "project": project,
            "project_path": project_path,
        }
        top = _layer_top_level(path)
        for knob in RENDER_FILE_KNOBS:
            if knob in top:
                entry[knob] = copy.deepcopy(top[knob])
        projection[name] = entry
    return projection


def _layer_top_level(path: Path) -> Mapping[str, Any]:
    """The top-level mapping of one login file; empty when it has none."""
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
    whether or not a terminal is deployed beside them. Empty otherwise — a login
    render disables the module and is handed no dispatch tier, so a
    render never projects renders-of-a-render — and empty when there is no repo
    to enumerate.

    Args:
        config: A profile's ``config:`` block, in any spelling the profile
            accepts (dotted, nested or mixed).
        project_name: The deployment's ``project_name``.
        repo_root: The deployment repo, or ``None`` when there is none.
        dispatch_tier: The deployment's ``dispatch.tier``, or ``None`` when it
            sets none or *config* is a login's render.
    """
    from osprey.cli.build_profile_emit import effective_web_terminals

    serves_renders = bool(effective_web_terminals(config).get("enabled")) or bool(dispatch_tier)
    if repo_root is None or not serves_renders:
        return {}
    return {
        f"{RENDERS_KEY}.{name}": entry
        for name, entry in render_projection(Path(repo_root), project_name).items()
    }
