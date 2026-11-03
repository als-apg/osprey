"""Template renders that stop at a different step than ``create_project_as_built``.

:func:`tests._preset_data.create_project_as_built` takes a render through every
step a build takes after it. :func:`create_project_with_manifest` is for tests
that need one stopped short of that, and so cannot use it.
:func:`graph_permission_entries` is what any render should grant the graph
server.
"""

from __future__ import annotations

from pathlib import Path

from osprey.cli.templates.manager import TemplateManager
from osprey.registry.mcp import FRAMEWORK_SERVERS


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


def graph_permission_entries() -> list[str]:
    """The four rendered permission strings, derived from the registry.

    ``ServerDefinition.permissions_allow`` holds BARE tool names — the settings
    template splices the ``mcp__<server>__`` prefix onto the whole list — so the
    qualification happens here rather than being restated where it could drift
    from what the render emits.
    """
    return sorted(f"mcp__graph__{tool}" for tool in FRAMEWORK_SERVERS["graph"].permissions_allow)
