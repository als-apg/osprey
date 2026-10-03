"""Knowledge pages linked to a device the facility file does not hold.

A page under ``data/facility/knowledge`` links itself to a device with the
``device_id`` key of its frontmatter. A page whose id is no device id of the
facility file is a dangling link: the build names it and goes on. A page
without the key, or one whose frontmatter does not parse, is unlinked.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["KNOWLEDGE_DIR", "LINK_KEY", "DanglingLink", "dangling_links"]

#: Where the knowledge pages live, relative to ``data/facility/``.
KNOWLEDGE_DIR = "knowledge"

#: The frontmatter key that links a page to a device.
LINK_KEY = "device_id"


@dataclass(frozen=True)
class DanglingLink:
    """One page linked to a device id the facility file does not hold.

    Attributes:
        page: The page, relative to ``data/facility/``.
        device_id: The id its frontmatter names.
    """

    page: str
    device_id: str


def _linked_device(path: Path) -> str | None:
    """The device id a page's frontmatter names, or ``None`` for an unlinked page."""
    from osprey.services.facility_knowledge.okf.document import OKFDocument, OKFDocumentError

    try:
        document = OKFDocument.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, OKFDocumentError):
        return None
    value = document.frontmatter.get(LINK_KEY)
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def dangling_links(facility_dir: Path, doc: Mapping[str, Any]) -> list[DanglingLink]:
    """Name every knowledge page linked to a device the facility file does not hold.

    Args:
        facility_dir: The ``data/facility`` directory; its ``knowledge``
            directory may be missing.
        doc: The facility file.

    Returns:
        One entry per dangling page, sorted by page path.
    """
    knowledge = facility_dir / KNOWLEDGE_DIR
    if not knowledge.is_dir():
        return []
    held = {device.get("id") for device in doc.get("devices") or () if isinstance(device, Mapping)}
    links: list[DanglingLink] = []
    for path in sorted(knowledge.rglob("*.md")):
        if not path.is_file():
            continue
        device_id = _linked_device(path)
        if device_id is None or device_id in held:
            continue
        links.append(DanglingLink(path.relative_to(facility_dir).as_posix(), device_id))
    return sorted(links, key=lambda link: link.page)
