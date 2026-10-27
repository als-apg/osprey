"""Template renders that stop at a different step than ``create_project_as_built``.

:func:`tests._preset_data.create_project_as_built` takes a render through every
step a build takes after it. These two are for tests that need a render stopped
short of that, or with a step in between, and so cannot use it.
"""

from __future__ import annotations

from pathlib import Path

from osprey.cli.templates.manager import TemplateManager
from tests._preset_data import bundle_data_root


def create_project_with_manifest(manager: TemplateManager, **kwargs) -> Path:
    """``create_project`` plus the manifest stamp a real build writes next.

    A build stamps ``.osprey-manifest.json`` immediately after the render, and
    the Claude Code regen reads the preset back out of it to learn which
    artifacts the project selected. A fixture that renders without stamping is
    not a project any build produces: it regenerates with an empty artifact
    selection, and the write-gate lint rightly refuses the result.
    """
    project = manager.create_project(**kwargs)
    bundle = kwargs.setdefault("data_bundle", "control_assistant")
    manager.generate_manifest(
        project,
        kwargs["project_name"],
        bundle.replace("_", "-"),
        {},
        # The selection the render was made with, exactly as a build stamps it:
        # a site that hands `create_project` its own `artifacts=` must get that
        # selection back on the regen, not the bundle's full list.
        artifacts=manager._effective_artifacts(bundle, kwargs.get("artifacts")),
    )
    return project


def create_project_with_facility_views(manager: TemplateManager, **kwargs) -> Path:
    """``create_project_as_built`` with the facility views written before the last render.

    A build renders the framework template, overlays the resolved profile's
    ``config:`` block onto the result, stamps ``.osprey-manifest.json``, writes
    the facility views, and regenerates ``.claude/`` from the finished config.
    The facility views land before that last render, so the agent facts name
    the models the render serves.
    """
    from osprey.cli.build_profile import resolve_build_profile
    from osprey.utils.config_writer import config_update_fields
    from tests._facility_file import write_facility_views

    bundle = kwargs.setdefault("data_bundle", "control_assistant")
    preset = bundle.replace("_", "-")
    kwargs.setdefault("data_root", bundle_data_root(bundle))
    project = manager.create_project(**kwargs)
    profile, _preset_dir = resolve_build_profile(None, preset=preset)
    config_update_fields(project / "config.yml", profile.config)
    manager.generate_manifest(
        project, kwargs["project_name"], preset, {}, artifacts=kwargs.get("artifacts")
    )
    write_facility_views(project, bundle)
    manager.regenerate_claude_code(project)
    return project
