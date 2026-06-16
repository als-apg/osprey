"""OKF index regenerator: writes one index.md per non-empty directory in a bundle.

Index files follow OKF §6: a flat Markdown document with section headings that
group concepts by type.  No YAML frontmatter is written — except in the
bundle-root ``index.md``, where ``okf_version`` MAY be included per OKF §11.

The description synthesizer is intentionally deterministic: it concatenates
child titles (or the first child's description when there is only one child
with a description).  No network or model call is made.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .document import OKFDocument

_INDEX_FILE = "index.md"
_OKF_VERSION = "0.1"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_doc(path: Path) -> OKFDocument | None:
    try:
        return OKFDocument.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _synthesize_description(children: list[tuple[str, str]]) -> str:
    """Return a deterministic one-line description for a subdirectory.

    Strategy (no model call):
    - One child with a non-empty description → use that description directly.
    - Otherwise → "Contains N entries: Title1, Title2, …"

    Args:
        children: List of (title, description) pairs for the directory's
            immediate children (excluding ``index.md``).

    Returns:
        A short descriptive string, never empty.
    """
    if not children:
        return ""
    if len(children) == 1 and children[0][1]:
        return children[0][1]
    titles = ", ".join(t for t, _ in children if t) or "no titled entries"
    return f"Contains {len(children)} entries: {titles}."


def _build_index_text(
    entries: list[tuple[str, str, str, str]],
    *,
    root_index: bool = False,
) -> str:
    """Render the Markdown body for an index file.

    Args:
        entries: Each entry is ``(type, title, bundle_relative_link, description)``.
            ``type`` is used as the section heading; an empty string falls back to
            ``"Other"``.
        root_index: When ``True``, prepend an ``okf_version`` frontmatter block
            (OKF §11 — permitted only in the bundle-root ``index.md``).

    Returns:
        Full index file text, ready to write.
    """
    grouped: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for typ, title, link, desc in entries:
        grouped[typ or "Other"].append((title, link, desc))

    sections: list[str] = []
    for typ in sorted(grouped):
        lines = [f"# {typ}", ""]
        for title, link, desc in sorted(grouped[typ], key=lambda e: e[0].lower()):
            suffix = f" - {desc}" if desc else ""
            lines.append(f"* [{title}]({link}){suffix}")
        sections.append("\n".join(lines))

    body = "\n\n".join(sections) + "\n"

    if root_index:
        return f'---\nokf_version: "{_OKF_VERSION}"\n---\n\n{body}'
    return body


def _directories_to_index(bundle_root: Path) -> list[Path]:
    """Return every directory (including root) that contains at least one .md file."""
    dirs: set[Path] = set()
    for md in bundle_root.rglob("*.md"):
        cur = md.parent
        while cur != bundle_root.parent:
            dirs.add(cur)
            if cur == bundle_root:
                break
            cur = cur.parent
    return sorted(dirs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def regenerate_indexes(bundle_root: Path) -> list[Path]:
    """Write one ``index.md`` per non-empty directory in *bundle_root*.

    Directories are processed deepest-first so that child-directory
    descriptions are available when the parent index is written.

    The bundle-root ``index.md`` includes an ``okf_version`` frontmatter block
    (OKF §11).  All other ``index.md`` files have no frontmatter (OKF §6).

    No network or model calls are made.  Sub-directory descriptions are
    synthesised deterministically from child titles.

    Args:
        bundle_root: Path to the root directory of an OKF bundle.

    Returns:
        List of ``index.md`` paths written (in processing order,
        deepest first).
    """
    bundle_root = Path(bundle_root)
    written: list[Path] = []
    if not bundle_root.exists():
        return written

    # Process deepest directories first so child descriptions bubble up.
    directories = sorted(
        _directories_to_index(bundle_root),
        key=lambda p: (-len(p.relative_to(bundle_root).parts), str(p)),
    )

    dir_descriptions: dict[Path, str] = {}

    for directory in directories:
        entries: list[tuple[str, str, str, str]] = []

        for child in sorted(directory.iterdir()):
            if child.name == _INDEX_FILE:
                continue
            rel = child.relative_to(bundle_root).as_posix()
            if child.is_file() and child.suffix == ".md":
                doc = _load_doc(child)
                if doc is None:
                    continue
                fm = doc.frontmatter
                title = str(fm.get("title") or child.stem)
                desc = str(fm.get("description") or "")
                typ = str(fm.get("type") or "")
                entries.append((typ, title, f"/{rel}", desc))
            elif child.is_dir():
                desc = dir_descriptions.get(child, "")
                entries.append(("Subdirectories", child.name, f"/{rel}/", desc))

        if not entries:
            continue

        is_root = directory == bundle_root
        index_path = directory / _INDEX_FILE
        index_path.write_text(_build_index_text(entries, root_index=is_root), encoding="utf-8")
        written.append(index_path)

        if is_root:
            continue

        # Synthesise a description for this directory to use in its parent's index.
        child_pairs = [(title, desc) for _, title, _, desc in entries]
        dir_descriptions[directory] = _synthesize_description(child_pairs)

    return written


# ---------------------------------------------------------------------------
# Index validation helpers
# ---------------------------------------------------------------------------


class OKFIndexError(ValueError):
    """Raised when an ``index.md`` violates OKF index-file rules."""


def validate_index(path: Path, *, bundle_root: Path) -> None:
    """Validate an ``index.md`` file against OKF §6 and §11.

    Rules enforced:

    * Frontmatter is **forbidden** in any ``index.md`` that is not the
      bundle-root ``index.md``.
    * In the bundle-root ``index.md``, only ``okf_version`` is permitted in
      frontmatter; any other key raises an error.

    Args:
        path: Absolute path to an ``index.md`` file.
        bundle_root: Absolute path to the bundle root directory.

    Raises:
        OKFIndexError: If the file violates OKF index-file frontmatter rules.
    """
    text = path.read_text(encoding="utf-8")
    # Check whether there is a frontmatter block at all.
    lines = text.splitlines()
    has_frontmatter = bool(lines) and lines[0].strip() == "---"

    is_root_index = path.parent.resolve() == bundle_root.resolve()

    if not has_frontmatter:
        return  # No frontmatter — always valid.

    if not is_root_index:
        raise OKFIndexError(
            f"Frontmatter is not permitted in sub-directory index files (OKF §6): {path}"
        )

    # Root index.md: only okf_version is allowed.
    doc = OKFDocument.parse(text)
    forbidden = [k for k in doc.frontmatter if k != "okf_version"]
    if forbidden:
        raise OKFIndexError(
            f"Only 'okf_version' is permitted in root index.md frontmatter "
            f"(OKF §11); found: {', '.join(sorted(forbidden))}"
        )
