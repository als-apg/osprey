"""The one spelling of a persona render's name and directory.

A deployment that serves per-persona terminals builds one project per delta in
``personas/``. Each render is named after the deployment's ``project_name`` and
the persona, and the same pair names its directory under ``build/``, its image
tag (``<project>:local``) and the catalog's ``project``/``project_path``. Every
producer of those names — the build that renders the persona, the catalog view
the build writes into ``build/config.yml``, the profile lint that runs before any
render, and the refusal that tells an operator where a render should be — derives
them here, so ``project == basename(project_path)`` holds by construction.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from osprey.utils.workspace import BUILD_DIR_NAME

__all__ = ["derived_persona_catalog", "persona_project"]


def persona_project(project_name: str, persona: str) -> tuple[str, str]:
    """The render name and repo-relative render directory of one persona.

    Args:
        project_name: The deployment's ``project_name`` (already
            compose-normalized — the profile refuses any other spelling).
        persona: The persona's name, which is also its delta's file stem.

    Returns:
        ``(project, project_path)`` — ``<project_name>-<persona>`` and
        ``build/<project_name>-<persona>``.
    """
    project = f"{project_name}-{persona}"
    return project, f"{BUILD_DIR_NAME}/{project}"


def derived_persona_catalog(config: Mapping[str, Any], project_name: str) -> dict[str, str]:
    """The catalog keys the build writes for every persona it renders.

    One ``project`` and one ``project_path`` per catalog entry that names a
    ``build_profile`` — the personas ``osprey build`` renders from a delta. An
    entry without one points at a render the build does not make, and keeps the
    ``project_path`` the profile states. Empty when *config* does not stand up a
    persona stack of its own (a persona render, whose module is disabled), so a
    render never derives personas-of-a-persona.

    Args:
        config: A profile's ``config:`` block, in any spelling the profile
            accepts (dotted, nested or mixed).
        project_name: The deployment's ``project_name``.

    Returns:
        Dotted ``modules.web_terminals.personas.<name>.project`` /
        ``.project_path`` keys mapped to their derived values, laid over the
        profile's own ``config:`` entries.
    """
    # Imported in-function: the catalog reader lives with the CLI's profile
    # emitter, and the deployment layer loads none of the CLI at import time.
    from osprey.cli.build_profile_emit import persona_catalog

    overlay: dict[str, str] = {}
    for persona, entry in persona_catalog(config).items():
        build_profile = entry.get("build_profile")
        if not isinstance(build_profile, str) or not build_profile:
            continue
        project, project_path = persona_project(project_name, persona)
        prefix = f"modules.web_terminals.personas.{persona}"
        overlay[f"{prefix}.project"] = project
        overlay[f"{prefix}.project_path"] = project_path
    return overlay
