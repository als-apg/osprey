"""A preset's packaged data, composed once on disk, for tests that render from it.

A preset's ``data/`` tree is no single packaged directory: it is the app
template's ``data/`` plus the bundled facility at ``data/facility/``
(:class:`osprey.cli.templates.preset_data.PresetData`). A test that hands a
render a ``data_root`` without going through ``osprey init`` needs that same
composed tree, so it is written here by the very composition ``osprey init``
copies, once per process and app template, under a temporary directory.

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


def preset_data(bundle: str = "control_assistant"):
    """The packaged composition the preset named after *bundle* copies.

    Args:
        bundle: An app template name; the preset of the same name (hyphenated)
            says which facility goes with it.
    """
    from osprey.cli.profile_cmd import _preset_data
    from osprey.cli.templates.manager import TemplateManager

    return _preset_data(TemplateManager(), bundle.replace("_", "-"))


def bundle_data_root(bundle: str = "control_assistant") -> Path:
    """The composed ``data/`` tree of the preset named after *bundle*.

    The tree a profile materialized from that preset holds under ``data/``:
    the app template's ``data/`` plus its facility at ``data/facility/``.

    Args:
        bundle: An app template name (``control_assistant``, ``hello_world``).

    Returns:
        A directory shared by every caller in this process; read it, never
        write into it.
    """
    return _composed(bundle)


@cache
def _composed(bundle: str) -> Path:
    from osprey.cli.profile_cmd import _data_copy_ignore

    source = preset_data(bundle)
    destination = _scratch_root() / bundle / "data"
    destination.parent.mkdir(parents=True)
    source.copy_into(destination, ignore=_data_copy_ignore(source.app_data))
    return destination


def copy_bundle_data(destination: Path, bundle: str = "control_assistant") -> Path:
    """Copy the composed ``data/`` tree of *bundle*'s preset to *destination*.

    For a test that edits its copy. *destination* may exist already; files in
    it are overwritten.

    Returns:
        *destination*.
    """
    shutil.copytree(bundle_data_root(bundle), destination, dirs_exist_ok=True)
    return destination


def packaged_facility_dir(bundle: str = "control_assistant") -> Path:
    """The packaged facility tree the preset named after *bundle* lands at ``data/facility/``.

    The bundled facility the preset names, or — for a preset naming none — the
    ``data/facility/`` its app template ships, if any.
    """
    source = preset_data(bundle)
    if source.facility_root is not None:
        return source.facility_root
    return source.app_data / "facility"
