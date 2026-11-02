"""What a ``NotebookEdit`` asks the reviewer to read.

Only code cells are reviewed, on insert and on replace. The reviewer sees the new source, the code
cells already on disk without their outputs, and where the new source lands. A notebook file that
cannot be read gives an empty notebook context: the new source is still reviewed.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from osprey.services.code_review.contract import SURFACE_NOTEBOOK, ReviewRequest

_EDIT_INSERT = "insert"
_EDIT_DELETE = "delete"
_EDIT_REPLACE = "replace"

#: The cell types no review applies to. A type outside these is reviewed, so an unfamiliar one
#: cannot carry code past the review.
_UNREVIEWED_CELL_TYPES = ("markdown", "raw")

#: A cell id that is safe to print bare in the reviewer's message.
_PLAIN_ID = re.compile(r"[\w.-]+")

#: The spellings of a cell's position, for a notebook whose cells have no ids.
_INDEX_ID = re.compile(r"(?:cell-)?(\d+)")


def _read_cells(notebook_file: Path) -> list[dict[str, Any]]:
    """The cells on disk; none when the file is missing, unreadable or not a notebook."""
    try:
        document = json.loads(notebook_file.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        return []
    cells = document.get("cells") if isinstance(document, dict) else None
    if not isinstance(cells, list):
        return []
    return [cell if isinstance(cell, dict) else {} for cell in cells]


def _find_cell(cells: list[dict[str, Any]], cell_id: object) -> int | None:
    """The index of the cell *cell_id* names: by its id, else by a position spelled as an id."""
    if cell_id is None:
        return None
    wanted = str(cell_id)
    for index, cell in enumerate(cells):
        if str(cell.get("id")) == wanted:
            return index
    position = _INDEX_ID.fullmatch(wanted)
    if position is not None and int(position[1]) < len(cells):
        return int(position[1])
    return None


def _id_label(cell_id: object) -> str:
    """Print an id on one line: it comes from the tool call or the file, so it is not trusted."""
    text = str(cell_id)
    return text if _PLAIN_ID.fullmatch(text) else json.dumps(text)


def _source(cell: Mapping[str, Any]) -> str:
    """A cell's source, which nbformat stores as one string or a list of lines."""
    source = cell.get("source")
    if isinstance(source, str):
        return source
    if isinstance(source, list):
        return "".join(line for line in source if isinstance(line, str))
    return ""


def _notebook_text(cells: list[dict[str, Any]]) -> str:
    """The code cells, each under a line naming it, with outputs and other cells dropped."""
    return "\n\n".join(
        f"# cell id={_id_label(cell.get('id'))} index={index}\n{_source(cell)}"
        for index, cell in enumerate(cells)
        if cell.get("cell_type") == "code"
    )


def _describe_target(edit_mode: object, cell_id: object, found: int | None) -> str:
    """Say where the new source lands, as ``id=abc index=3 edit_mode=replace``."""
    if cell_id is None:
        return f"new cell at start index=0 edit_mode={edit_mode}"
    label = _id_label(cell_id)
    if edit_mode == _EDIT_INSERT:
        index = "unknown" if found is None else found + 1
        return f"new cell after id={label} index={index} edit_mode={edit_mode}"
    index = "unknown" if found is None else found
    return f"id={label} index={index} edit_mode={edit_mode}"


def notebook_review_request(
    tool_input: Mapping[str, Any],
    *,
    notebook_file: Path,
    subject: str,
) -> ReviewRequest | None:
    """Build the review of one ``NotebookEdit``, if it writes code.

    Args:
        tool_input: The tool call's input.
        notebook_file: The notebook on disk; the only thing read.
        subject: The audit subject.

    Returns:
        The request, or ``None`` for a delete or a cell that does not hold code. A replace that
        names no cell type takes the type of the cell on disk.
    """
    edit_mode = tool_input.get("edit_mode") or _EDIT_REPLACE
    if edit_mode == _EDIT_DELETE:
        return None

    cells = _read_cells(notebook_file)
    cell_id = tool_input.get("cell_id")
    found = _find_cell(cells, cell_id)

    cell_type = tool_input.get("cell_type")
    if not cell_type and edit_mode != _EDIT_INSERT and found is not None:
        cell_type = cells[found].get("cell_type")
    if cell_type in _UNREVIEWED_CELL_TYPES:
        return None

    new_source = tool_input.get("new_source")
    return ReviewRequest(
        surface=SURFACE_NOTEBOOK,
        code=new_source if isinstance(new_source, str) else "",
        subject=subject,
        notebook=_notebook_text(cells),
        cell=_describe_target(edit_mode, cell_id, found),
    )
