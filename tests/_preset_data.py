"""A preset's packaged data, composed once on disk, for tests that render from it.

A preset's source trees are no single packaged directory: they are the app
template's ``web/`` plus the bundled facility at ``facility/``
(:class:`osprey.cli.templates.preset_data.PresetData`). A test that hands a
render a ``data_root`` without going through ``osprey init`` needs the tree
the render's ``data/`` is assembled from, so it is written here by the very
composition ``osprey init`` copies, each declared source then laid at its
render path (:func:`osprey.repo_layout.rendered_sources`), once per process
and app template, under a temporary directory.

Callers read the tree and must not write into it: it is shared by every test of
the process.
"""

from __future__ import annotations

import atexit
import shutil
import tempfile
from functools import cache
from pathlib import Path


@cache
def _scratch_root() -> Path:
    root = Path(tempfile.mkdtemp(prefix="osprey-preset-data-"))
    atexit.register(shutil.rmtree, root, ignore_errors=True)
    return root


def preset_data(bundle: str = "demo"):
    """The packaged composition the preset named after *bundle* copies.

    Args:
        bundle: An app template name; the preset of the same name (hyphenated)
            says which facility goes with it.
    """
    from osprey.cli.profile_cmd import _preset_data
    from osprey.cli.templates.manager import TemplateManager

    return _preset_data(TemplateManager(), bundle.replace("_", "-"))


def bundle_data_root(bundle: str = "demo") -> Path:
    """The composed render-side ``data/`` tree of the preset named after *bundle*.

    Every declared source a profile materialized from that preset holds, laid
    at its render path: the facility at ``data/facility/``, the web content at
    ``data/canopy/...`` and ``data/landing/``.

    Args:
        bundle: An app template name (``demo``, ``hello_world``).

    Returns:
        A directory shared by every caller in this process; read it, never
        write into it.
    """
    return _composed(bundle)


def bundle_source_root(bundle: str = "demo") -> Path:
    """A deployment root holding the sources a profile materialized from *bundle*'s preset holds.

    The facility at ``facility/`` and the web content at ``web/``, as
    ``osprey init`` lays them; the root a render assembles its ``data/`` from.

    Returns:
        A directory shared by every caller in this process; read it, never
        write into it.
    """
    return _source_root(bundle)


@cache
def _source_root(bundle: str) -> Path:
    source = preset_data(bundle)
    root = _scratch_root() / bundle
    root.mkdir(parents=True)
    source.copy_into(root)
    return root


@cache
def _composed(bundle: str) -> Path:
    from osprey import repo_layout

    root = bundle_source_root(bundle)
    render = _scratch_root() / f"{bundle}-render"
    render.mkdir(parents=True)
    for rendered in repo_layout.rendered_sources():
        tree = root.joinpath(*rendered.source_rel.split("/"))
        if tree.is_dir():
            shutil.copytree(tree, render.joinpath(*rendered.render_rel.split("/")))
    data = render / "data"
    data.mkdir(exist_ok=True)
    return data


def copy_bundle_data(destination: Path, bundle: str = "demo") -> Path:
    """Copy the composed render-side ``data/`` tree of *bundle*'s preset to *destination*.

    For a test that edits its copy. *destination* may exist already; files in
    it are overwritten.

    Returns:
        *destination*.
    """
    shutil.copytree(bundle_data_root(bundle), destination, dirs_exist_ok=True)
    return destination


def packaged_facility_dir(bundle: str = "demo") -> Path:
    """The packaged facility tree the preset named after *bundle* lands at ``facility/``.

    The bundled facility the preset names, or — for a preset naming none — the
    ``facility/`` its app template ships, if any.
    """
    from osprey import repo_layout

    source = preset_data(bundle)
    if source.facility_root is not None:
        return source.facility_root
    return source.app_root / repo_layout.entry("facility").path


def create_project_as_built(manager, *, facility_views: bool = False, **kwargs) -> Path:
    """``manager.create_project`` plus the three steps a real build takes next.

    A build renders the framework template, overlays the resolved profile's
    ``config:`` block onto the result, stamps ``.osprey-manifest.json``, and
    regenerates ``.claude/`` from the finished config. The template carries
    only derived and profile-field-derived keys, so a fixture that stops after
    the render holds half a config — the declarative half is the preset's, and
    the artifacts rendered before it landed do not know about the deployment's
    control system, services or servers. These fixtures render from a bundle
    rather than from a profile, so they overlay the preset ``osprey init``
    pairs with that bundle.

    Args:
        manager: A :class:`~osprey.cli.templates.manager.TemplateManager`.
        facility_views: Also write the facility views before the last render,
            as a build does, so the agent facts name the models the render
            serves.
        **kwargs: Passed to ``manager.create_project``; ``data_bundle`` defaults
            to ``demo`` and ``repo_root`` to that bundle's source root.

    Returns:
        The rendered project directory.
    """
    from osprey.cli.build_profile import resolve_build_profile
    from osprey.utils.config_writer import config_update_fields

    bundle = kwargs.setdefault("data_bundle", "demo")
    preset = bundle.replace("_", "-")
    kwargs.setdefault("repo_root", bundle_source_root(bundle))
    project = manager.create_project(**kwargs)
    profile, _preset_dir = resolve_build_profile(None, preset=preset)
    config_update_fields(project / "config.yml", profile.config)
    manager.generate_manifest(
        project, kwargs["project_name"], preset, {}, artifacts=kwargs.get("artifacts")
    )
    if facility_views:
        from tests._facility_file import write_facility_views

        write_facility_views(project, bundle)
    # The build's last render, and the one that ships: `create_project` wrote
    # `.claude/` from a config.yml that did not yet carry the preset's block.
    manager.regenerate_claude_code(project)
    return project
