"""Knowledge pages that link what the facility file does not hold.

A page under ``data/facility/knowledge`` links the facility two ways. The
``device_id`` key of its frontmatter names a device, and a
``[label](facility:ADDRESS)`` link in its body names a channel id or a device
id. Every link resolves through :class:`~osprey.facility.records.FacilityRecords`;
a frontmatter id that is no device id, or a body link that resolves to no
record, is a dangling link: the build names it and goes on. A page whose
frontmatter does not parse links nothing.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from osprey.facility.errors import FacilityBuildError, FacilityBuildWarning
from osprey.facility.records import FacilityRecords
from osprey.facility.references import LINK_SCHEME, FacilityReference, find_facility_links
from osprey.facility.validate import Validated, need

__all__ = ["KNOWLEDGE_DIR", "LINK_KEY", "DanglingLink", "check_knowledge_links", "dangling_links"]

#: Where the knowledge pages live, relative to ``data/facility/``.
KNOWLEDGE_DIR = "knowledge"

#: The frontmatter key that links a page to a device.
LINK_KEY = "device_id"


@dataclass(frozen=True)
class DanglingLink:
    """One link of a page that the facility file does not resolve.

    Attributes:
        page: The page, relative to ``data/facility/``.
        target: The id the link names.
        where: ``frontmatter`` for the ``device_id`` key, ``body`` for a
            facility link in the page's text.
    """

    page: str
    target: str
    where: Literal["frontmatter", "body"]


def _page_links(path: Path, records: FacilityRecords) -> tuple[str | None, list[FacilityReference]]:
    """The device id a page's frontmatter names and the references its body links.

    Returns:
        The ``device_id`` (``None`` when the key is absent or empty) and the
        body's facility links, looked up in ``records``, in page order;
        ``(None, [])`` for a page that does not parse.
    """
    from osprey.services.facility_knowledge.okf.document import OKFDocument, OKFDocumentError

    try:
        document = OKFDocument.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, OKFDocumentError):
        return None, []
    value = document.frontmatter.get(LINK_KEY)
    device_id = value.strip() if isinstance(value, str) and value.strip() else None
    return device_id, find_facility_links(document.body, records)


def dangling_links(facility_dir: Path, doc: Mapping[str, Any]) -> list[DanglingLink]:
    """Name every knowledge-page link the facility file does not resolve.

    Args:
        facility_dir: The ``data/facility`` directory; its ``knowledge``
            directory may be missing.
        doc: The facility file.

    Returns:
        One entry per dangling link, sorted by page path; within a page the
        frontmatter link comes first, then the body links in page order.
    """
    knowledge = facility_dir / KNOWLEDGE_DIR
    if not knowledge.is_dir():
        return []
    records = FacilityRecords(doc)
    links: list[DanglingLink] = []
    for path in sorted(knowledge.rglob("*.md")):
        if not path.is_file():
            continue
        page = path.relative_to(facility_dir).as_posix()
        device_id, references = _page_links(path, records)
        if device_id is not None and not any(
            found.kind == "device" for found in records.resolve(device_id)
        ):
            links.append(DanglingLink(page, device_id, "frontmatter"))
        links.extend(
            DanglingLink(page, found.target, "body") for found in references if not found.kinds
        )
    return sorted(links, key=lambda link: link.page)


def check_knowledge_links(validated: Validated) -> list[FacilityBuildError]:
    """Warn once per knowledge-page link the facility file does not resolve.

    Both link forms warn the same kind: a frontmatter ``device_id`` that is no
    device id, and a body ``[label](facility:ADDRESS)`` link whose target is no
    channel id or device id. The stage warns and never stops: each dangling
    link appends one ``link-dangling`` warning to ``validated.warnings``, in
    page order.

    Args:
        validated: What the earlier stages produced.

    Returns:
        No stops, ever.

    Raises:
        RuntimeError: The stage ran before S2 produced the document.
    """
    document: dict[str, Any] = need(validated.document)
    for link in dangling_links(validated.facility_dir, document):
        if link.where == "frontmatter":
            named = f"device {link.target}"
            remedy = f"set {LINK_KEY} to a device id of the facility file, or remove the key"
        else:
            named = f"{LINK_SCHEME}:{link.target}"
            remedy = "link a channel id or a device id of the facility file, or remove the link"
        validated.warnings.append(
            FacilityBuildWarning(
                kind="link-dangling",
                record_kind="page",
                record_id=link.page,
                detail=f"links {named}, which the facility file does not hold",
                remedy=remedy,
            )
        )
    return []
