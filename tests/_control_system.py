"""The ``control_system:`` section of a deployment serving the simulator in process.

A test that needs a section dialling nothing states it here once: the simulator
type, served in process. :func:`in_process_section` returns a fresh mapping, so
a test that edits its section never edits another test's.
"""

from __future__ import annotations

import copy
from typing import Any

__all__ = [
    "IN_PROCESS",
    "IN_PROCESS_SECTION",
    "IN_PROCESS_YAML",
    "in_process_section",
    "section_for",
]

#: The name a test passes where it would pass a connector type, to mean the
#: simulator served in process.
IN_PROCESS = "in_process"

#: The section itself. Read it, never mutate it; :func:`in_process_section`
#: hands out copies.
IN_PROCESS_SECTION: dict[str, Any] = {
    "type": "virtual_accelerator",
    "connector": {"virtual_accelerator": {"serving": "in_process"}},
}

#: The same section as a rendered ``config.yml`` writes it.
IN_PROCESS_YAML = (
    "control_system:\n"
    "  type: virtual_accelerator\n"
    "  connector:\n"
    "    virtual_accelerator:\n"
    "      serving: in_process\n"
)


def in_process_section(block: dict[str, Any] | None = None, **fields: Any) -> dict[str, Any]:
    """A fresh in-process section.

    Args:
        block: Further settings of the ``virtual_accelerator`` connector block.
        fields: Further top-level keys of the section.
    """
    section = copy.deepcopy(IN_PROCESS_SECTION)
    section["connector"]["virtual_accelerator"].update(block or {})
    section.update(fields)
    return section


def section_for(value: str) -> dict[str, Any]:
    """The section a test names by one word: a connector type, or :data:`IN_PROCESS`."""
    return in_process_section() if value == IN_PROCESS else {"type": value}
