"""The file and link rules every reader of an OKF bundle shares.

Two questions come up wherever a bundle is read, and each has one answer here:

* **Which files are part of the bundle, and which of them are concepts?**
  (OKF §3.2.) A concept is a non-reserved ``.md`` file whose frontmatter holds
  a non-empty ``type``. A ``README.md`` or a ``SKILL.md`` is a file in the
  bundle, not a concept; a path starting with ``.`` is not in the bundle at
  all; a symbolic link out of the root is never followed.
* **What does a link or a path field point at?** (OKF §6.) A target with a URI
  scheme is external, one starting with ``/`` resolves from the bundle root,
  and any other resolves from the directory of the file that holds it.
"""

from __future__ import annotations

import posixpath
import re
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from urllib.parse import unquote

from .document import OKFDocument

__all__ = [
    "RESERVED_NAMES",
    "bundle_markdown_files",
    "concept_id_of_link",
    "load_concept",
    "resolve_in_bundle",
]

#: Filenames that are never concepts, at any depth (OKF §3.1).
RESERVED_NAMES: frozenset[str] = frozenset({"index.md", "log.md"})

# RFC 3986 scheme: what makes a target an absolute URL rather than a path.
_SCHEME_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*:")


def bundle_markdown_files(root: Path) -> Iterator[Path]:
    """Yield every ``.md`` file that is part of the bundle at *root*, sorted.

    Reserved files are included: whether a file is a concept is
    :func:`load_concept`'s question, and callers that validate ``index.md`` and
    ``log.md`` need to see them.

    Args:
        root: Bundle root.

    Yields:
        Paths under *root*, skipping any below a name that starts with ``.``
        and any symbolic link that resolves outside the root.
    """
    resolved_root: Path | None = None
    for path in sorted(root.rglob("*.md")):
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        if not path.is_file():
            continue
        if path.is_symlink():
            resolved_root = resolved_root or root.resolve()
            if not path.resolve().is_relative_to(resolved_root):
                continue
        yield path


def load_concept(path: Path) -> OKFDocument | None:
    """Read the concept at *path*, or ``None`` when the file is not one.

    Args:
        path: A ``.md`` file in a bundle.

    Returns:
        The parsed document when its frontmatter holds a non-empty ``type`` and
        its name is not reserved; ``None`` otherwise.

    Raises:
        OKFDocumentError: The frontmatter does not parse. That is a broken
            page, which is not the same as a file that is not a concept, so
            the caller decides what it means.
        OSError: The file cannot be read.
    """
    if path.name in RESERVED_NAMES:
        return None
    doc = OKFDocument.parse(path.read_text(encoding="utf-8"))
    return doc if doc.frontmatter.get("type") else None


def resolve_in_bundle(target: str, *, source_dir: str) -> str | None:
    """Resolve a link target or path-field value to a path inside the bundle.

    Args:
        target: The value as written, e.g. ``/tables/users.md``,
            ``../tables/users.md#schema`` or ``https://example.org``.
        source_dir: Bundle-relative POSIX directory of the file that holds
            *target*; ``""`` for a file at the bundle root.

    Returns:
        The bundle-relative POSIX path the target names, percent-decoded and
        without its fragment or query; a directory target keeps its trailing
        ``/``. ``None`` when the target is external: it carries a URI scheme,
        is only a fragment, or resolves outside the bundle root.
    """
    raw = target.strip()
    if not raw or _SCHEME_RE.match(raw):
        return None
    raw = unquote(raw.split("#", 1)[0].split("?", 1)[0])
    if not raw:
        return None
    joined = raw.lstrip("/") if raw.startswith("/") else posixpath.join(source_dir, raw)
    resolved = posixpath.normpath(joined)
    if resolved == ".." or resolved.startswith("../") or resolved == ".":
        return None
    return resolved + "/" if raw.endswith("/") else resolved


def concept_id_of_link(target: str, *, source_dir: str) -> str | None:
    """Return the ID of the concept a link would name, when it names one.

    Whether that concept exists is not checked: a link to a page nobody has
    written yet is legal (OKF §6.1).

    Args:
        target: The link target as written.
        source_dir: Bundle-relative POSIX directory of the linking file.

    Returns:
        The concept ID — the bundle-relative path without ``.md`` — or ``None``
        for an external target, a directory, an asset or a reserved file.
    """
    resolved = resolve_in_bundle(target, source_dir=source_dir)
    if resolved is None or not resolved.endswith(".md"):
        return None
    path = PurePosixPath(resolved)
    if path.name in RESERVED_NAMES:
        return None
    return path.with_suffix("").as_posix()
