"""Read API over an OKF knowledge bundle directory.

A bundle is a directory tree of ``.md`` files whose structure follows the
OSPREY Knowledge Framework specification (OKF §§2–6).  This module exposes
three operations:

* :py:meth:`OKFBundle.list_concepts` — lightweight listing from ``index.md``
  without reading every document body (OKF §6, progressive disclosure).
* :py:meth:`OKFBundle.read_concept` — load a single concept by ID; broken
  cross-links in the body are tolerated (OKF §9).
* :py:meth:`OKFBundle.search` — substring match over frontmatter values and
  body text.

Concept IDs follow OKF §2: the file path relative to the bundle root, with
the ``.md`` suffix removed (e.g. ``tables/users``).

Example usage::

    bundle = OKFBundle(Path("/data/my-bundle"))
    for entry in bundle.list_concepts():
        print(entry.concept_id, entry.title)

    doc = bundle.read_concept("tables/users")
    results = bundle.search("synchrotron")
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osprey.services.facility_knowledge.okf.document import OKFDocument, OKFDocumentError

# Reserved filenames that are never concepts (OKF §3.1).
_RESERVED_NAMES: frozenset[str] = frozenset({"index.md", "log.md"})

# index.md link pattern: Markdown links of the form [title](path)
_INDEX_LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")


class OKFBundleError(OSError):
    """Raised when the bundle directory is missing or unreadable.

    Args:
        message: Human-readable description of the problem.
    """


@dataclass(frozen=True)
class ConceptEntry:
    """A lightweight summary of one concept, drawn from ``index.md``.

    Args:
        concept_id: OKF §2 path-minus-extension identifier (e.g. ``tables/users``).
        title: Display name from the index listing or derived from the filename.
        description: One-line summary from the index listing, or empty string.
        type: Concept type string from the index listing, or empty string.
    """

    concept_id: str
    title: str
    description: str = ""
    type: str = ""


@dataclass
class OKFBundle:
    """Read-only accessor for an OKF knowledge bundle directory.

    Args:
        root: Path to the bundle root directory.

    Raises:
        OKFBundleError: If *root* does not exist.
    """

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        if not self.root.exists():
            raise OKFBundleError(f"Bundle root does not exist: {self.root}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_concepts(self) -> list[ConceptEntry]:
        """List all concepts via the bundle's ``index.md`` files.

        Reads only ``index.md`` files (progressive disclosure — no full doc
        reads), then falls back to a filesystem scan for any ``.md`` files
        not covered by an index.

        Returns:
            Entries in no guaranteed order; callers should sort as needed.
        """
        seen: set[str] = set()
        entries: list[ConceptEntry] = []

        # Primary: walk every index.md in the tree for fast enumeration.
        for idx in self.root.rglob("index.md"):
            dir_entries = _parse_index(self.root, idx)
            for e in dir_entries:
                if e.concept_id not in seen:
                    seen.add(e.concept_id)
                    entries.append(e)

        # Fallback: any .md file not covered by the index scan above.
        for md in self.root.rglob("*.md"):
            if md.name in _RESERVED_NAMES:
                continue
            cid = _path_to_concept_id(self.root, md)
            if cid not in seen:
                seen.add(cid)
                entries.append(ConceptEntry(concept_id=cid, title=md.stem))

        return entries

    def read_concept(self, concept_id: str) -> OKFDocument:
        """Read and parse a concept document by its OKF §2 ID.

        Cross-links embedded in the body may point to documents that do not
        exist in this bundle; this is legal (OKF §9) and the document is
        returned as-is without resolving links.

        Args:
            concept_id: Path-minus-extension identifier, e.g. ``tables/users``
                or a single segment like ``facility_overview``.

        Returns:
            Parsed :class:`~osprey.services.facility_knowledge.okf.document.OKFDocument`.

        Raises:
            OKFBundleError: If the concept file does not exist.
            OKFDocumentError: If the file is present but contains malformed
                frontmatter.
        """
        path = self.resolve_concept_path(concept_id)
        if not path.exists():
            raise OKFBundleError(f"Concept not found: {concept_id!r} (expected {path})")
        text = path.read_text(encoding="utf-8")
        return OKFDocument.parse(text)

    def resolve_concept_path(self, concept_id: str) -> Path:
        """Return the filesystem path for *concept_id* within this bundle.

        The result is guaranteed to live inside the bundle root; a
        *concept_id* that escapes the root (path traversal) raises
        :class:`OKFBundleError`.  This is the supported way to map an OKF §2
        concept ID to a file path — both :meth:`read_concept` and the
        ``draft_concept`` MCP tool call it, so the containment check has a
        single source of truth.

        Args:
            concept_id: OKF §2 identifier, e.g. ``tables/users`` or
                ``/tables/users`` (leading slashes are stripped).

        Returns:
            Resolved absolute path to the ``.md`` file.  The file need not
            exist yet — callers writing a new concept rely on this.

        Raises:
            OKFBundleError: If the resolved path escapes the bundle root.
        """
        # Strip leading slashes; OKF §2 IDs are always root-relative.
        clean = concept_id.lstrip("/")
        candidate = (self.root / f"{clean}.md").resolve()
        resolved_root = self.root.resolve()
        if not candidate.is_relative_to(resolved_root):
            raise OKFBundleError(f"concept_id escapes bundle root: {concept_id!r}")
        return candidate

    def search(self, query: str) -> list[tuple[str, OKFDocument]]:
        """Search the bundle for a query string.

        Performs a case-insensitive substring match against each concept's
        frontmatter values (concatenated as strings) and body text.

        ``index.md`` and ``log.md`` files are never returned.

        Args:
            query: Substring to search for (case-insensitive).

        Returns:
            List of ``(concept_id, document)`` tuples for all matching docs,
            in filesystem traversal order.
        """
        needle = query.casefold()
        results: list[tuple[str, OKFDocument]] = []

        for md in sorted(self.root.rglob("*.md")):
            if md.name in _RESERVED_NAMES:
                continue
            try:
                text = md.read_text(encoding="utf-8")
                doc = OKFDocument.parse(text)
            except (OKFDocumentError, OSError):
                continue

            haystack = _doc_haystack(doc)
            if needle in haystack:
                cid = _path_to_concept_id(self.root, md)
                results.append((cid, doc))

        return results


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _path_to_concept_id(root: Path, path: Path) -> str:
    """Return the OKF §2 concept ID for *path* relative to *root*."""
    return path.relative_to(root).with_suffix("").as_posix()


def _parse_index(root: Path, index_path: Path) -> list[ConceptEntry]:
    """Extract concept entries from a single ``index.md`` file.

    The index format used by the OKF index generator is a Markdown list of
    links (``* [title](path)``).  We parse those links and build lightweight
    entries without reading the linked files.

    Args:
        root: Bundle root, used for ID normalisation.
        index_path: Path to the ``index.md`` to parse.

    Returns:
        List of :class:`ConceptEntry` objects for each concept link found.
    """
    try:
        text = index_path.read_text(encoding="utf-8")
    except OSError:
        return []

    entries: list[ConceptEntry] = []
    for match in _INDEX_LINK_RE.finditer(text):
        title = match.group(1).strip()
        href = match.group(2).strip()

        # Skip external links and directory links (trailing /).
        if href.startswith("http") or href.endswith("/"):
            continue

        # Normalise: strip leading slash, strip .md suffix.
        clean = href.lstrip("/")
        if clean.endswith(".md"):
            clean = clean[:-3]
        if not clean:
            continue

        # Resolve relative to the index's directory for nested indexes.
        # OKF index hrefs are bundle-root-relative (leading slash stripped).
        # We use them as-is since the generator writes root-relative paths.
        concept_id = clean

        # Extract optional description: text after " - " on the same line.
        line_start = text.rfind("\n", 0, match.start()) + 1
        line_end = text.find("\n", match.end())
        line = text[line_start:line_end] if line_end != -1 else text[line_start:]
        desc_match = re.search(r"\)\s*-\s*(.+)$", line)
        description = desc_match.group(1).strip() if desc_match else ""

        entries.append(ConceptEntry(concept_id=concept_id, title=title, description=description))

    return entries


def _doc_haystack(doc: OKFDocument) -> str:
    """Return a single casefolded string covering all searchable content."""
    fm_values = " ".join(str(v) for v in _flatten_values(doc.frontmatter))
    return (fm_values + " " + doc.body).casefold()


def _flatten_values(obj: Any) -> list[str]:
    """Recursively collect string representations of all leaf values."""
    if isinstance(obj, dict):
        result: list[str] = []
        for v in obj.values():
            result.extend(_flatten_values(v))
        return result
    if isinstance(obj, list):
        result = []
        for item in obj:
            result.extend(_flatten_values(item))
        return result
    return [str(obj)]
