"""The render's facility file and facts document, as an application reads them.

A measurement kind's own view files are
``facts["measurement_models"][<model>]["kinds"][<kind>]["view_files"]``, each
resolved by name in ``facts["measurement_models"][<model>]["files"]``; the facts
its writer wrote for the model are
``facts["measurement_models"][<model>]["kinds"][<kind>].get("facts", {})``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from osprey.facility import FACILITY_FILE


def _render_root() -> Path:
    from osprey_connectors.workspace import resolve_config_path

    return Path(resolve_config_path()).parent


def facility_file() -> Path:
    """The path of the render's facility file, beside its ``config.yml``.

    The file is a JSON object whose ``schema`` key is its header, naming the
    document format it was built in. A reader uses none of a document whose
    ``schema`` it does not declare, a missing one included: it refuses the
    document, naming the rebuild (``osprey build``), and never reads the rest of
    it as if the format matched.

    Stable: part of the application SDK's public surface.
    """
    return _render_root() / FACILITY_FILE


def read_facts(
    render_root: str | Path | None = None, *, project_name: str | None = None
) -> dict[str, Any]:
    """Read the render's facts document.

    Stable: part of the application SDK's public surface.

    Args:
        render_root: The directory holding the rendered ``config.yml``; the
            render this process runs from when omitted.
        project_name: The identity of a render with neither a facts file nor
            a facility file; the render directory's name when omitted.

    Returns:
        The facts document, or the zero-source facts when the render holds none.

    Raises:
        ValueError: The render holds no usable facts document and its facility
            file cannot be read or carries a header this reader does not read;
            the message names the rebuild (``osprey build``).
    """
    from osprey.facility.views.facts import read_facts as read

    root = _render_root() if render_root is None else Path(render_root)
    return read(root, project_name or root.name)


def tool_keys(facts: Mapping[str, Any], model: str, kind: str) -> list[str]:
    """The keys the measure tool takes for *kind* on *model*, from a facts document.

    Stable: part of the application SDK's public surface.

    Raises:
        KeyError: The document holds no *kind* block for *model*.
    """
    return list(facts["measurement_models"][model]["kinds"][kind]["tool_keys"])
