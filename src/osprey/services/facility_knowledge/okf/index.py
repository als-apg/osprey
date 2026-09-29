"""OKF index producer: renders, writes and checks one index.md per non-empty directory.

Index files follow OKF §6: a flat Markdown document with section headings that
group concepts by type.  No YAML frontmatter is written — except in the
bundle-root ``index.md``, where ``okf_version`` MAY be included per OKF §11.

The description synthesizer is intentionally deterministic: it concatenates
child titles (or the first child's description when there is only one child
with a description).  No network or model call is made.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
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
    titles = ", ".join(sorted((t for t, _ in children if t), key=str.lower)) or "no titled entries"
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


def _holds_pages(directory: Path, dropped: set[Path]) -> bool:
    """Return whether *directory* holds any ``.md`` file at any depth not in *dropped*."""
    return any(md not in dropped for md in directory.rglob("*.md"))


def _is_stale_index(directory: Path, dropped: set[Path]) -> bool:
    """Return whether *directory*'s non-blank ``index.md`` is the only live ``.md`` in it.

    Such an index lists pages that no longer exist. A blank ``index.md`` lists
    nothing and is not stale: it is the placeholder that holds a skeleton
    directory in place until the facility adds its own pages. Child directories
    are processed first, so a child that held only a stale index already has it
    in *dropped* and no longer counts as content here.
    """
    index_path = directory / _INDEX_FILE
    if not index_path.is_file() or not index_path.read_text(encoding="utf-8").strip():
        return False
    return all(md == index_path or md in dropped for md in directory.rglob("*.md"))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RenderedIndexes:
    """Every index a bundle should hold, computed without touching the disk.

    Attributes:
        texts: Each ``index.md`` path mapped to its full file text, in
            processing order (deepest first).
        stale: The non-blank ``index.md`` files whose directory no longer holds
            any page; the regenerator deletes them.
    """

    texts: dict[Path, str]
    stale: tuple[Path, ...]


def render_indexes(bundle_root: Path) -> RenderedIndexes:
    """Compute one ``index.md`` per non-empty directory in *bundle_root*.

    Directories are processed deepest-first so that child-directory
    descriptions are available when the parent index is rendered.

    A directory whose only ``.md`` file is its own non-blank ``index.md`` has
    that stale index listed in ``stale`` (a blank placeholder index is neither
    rendered nor stale), and a parent never lists a child directory that holds
    no live ``.md`` file, so an emptied directory advertises no removed page.

    The bundle-root ``index.md`` includes an ``okf_version`` frontmatter block
    (OKF §11).  All other ``index.md`` files have no frontmatter (OKF §6).

    No network or model calls are made.  Sub-directory descriptions are
    synthesised deterministically from child titles.  Nothing is written.

    Args:
        bundle_root: Path to the root directory of an OKF bundle.

    Returns:
        The rendered index texts and the stale indexes.
    """
    bundle_root = Path(bundle_root)
    texts: dict[Path, str] = {}
    stale: list[Path] = []
    if not bundle_root.exists():
        return RenderedIndexes(texts=texts, stale=())

    # Process deepest directories first so child descriptions bubble up.
    directories = sorted(
        _directories_to_index(bundle_root),
        key=lambda p: (-len(p.relative_to(bundle_root).parts), str(p)),
    )

    dir_descriptions: dict[Path, str] = {}
    dropped: set[Path] = set()

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
            elif child.is_dir() and _holds_pages(child, dropped):
                desc = dir_descriptions.get(child, "")
                entries.append(("Subdirectories", child.name, f"/{rel}/", desc))

        if not entries:
            if _is_stale_index(directory, dropped):
                index_path = directory / _INDEX_FILE
                dropped.add(index_path)
                stale.append(index_path)
            continue

        is_root = directory == bundle_root
        texts[directory / _INDEX_FILE] = _build_index_text(entries, root_index=is_root)

        if is_root:
            continue

        # Synthesise a description for this directory to use in its parent's index.
        child_pairs = [(title, desc) for _, title, _, desc in entries]
        dir_descriptions[directory] = _synthesize_description(child_pairs)

    return RenderedIndexes(texts=texts, stale=tuple(stale))


def regenerate_indexes(bundle_root: Path) -> list[Path]:
    """Write what :func:`render_indexes` computes for *bundle_root*.

    Every rendered ``index.md`` is written, then every stale index is deleted.

    Args:
        bundle_root: Path to the root directory of an OKF bundle.

    Returns:
        List of ``index.md`` paths written (in processing order,
        deepest first).
    """
    rendered = render_indexes(Path(bundle_root))
    for path, text in rendered.texts.items():
        path.write_text(text, encoding="utf-8")
    for path in rendered.stale:
        path.unlink()
    return list(rendered.texts)


# ---------------------------------------------------------------------------
# Index validation helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IndexDrift:
    """One ``index.md`` that differs from what the regenerator would leave there.

    Attributes:
        path: The ``index.md`` path, present on disk or not.
        message: What differs, phrased to follow ``index.md``.
    """

    path: Path
    message: str


def _first_difference(expected: str, found: str) -> str:
    """Describe the first line where *found* differs from *expected*."""
    exp_lines = expected.splitlines(keepends=True)
    got_lines = found.splitlines(keepends=True)

    def _line(lines: list[str], i: int) -> str:
        if i >= len(lines):
            return "<end of file>"
        line = lines[i]
        return line[:-1] if line.endswith("\n") else line

    i = 0
    while i < len(exp_lines) and i < len(got_lines) and exp_lines[i] == got_lines[i]:
        i += 1
    exp, got = _line(exp_lines, i), _line(got_lines, i)
    return f"does not match its directory at line {i + 1}: expected {exp!r}, found {got!r}"


def check_indexes(bundle_root: Path) -> list[IndexDrift]:
    """Compare every ``index.md`` in *bundle_root* with what :func:`render_indexes` renders.

    The check is exact because the regenerator is the one producer of index
    files: a bundle passes when :func:`regenerate_indexes` would change nothing
    in it.  Nothing is written.

    Args:
        bundle_root: Path to the root directory of an OKF bundle.

    Returns:
        Missing indexes first, then indexes whose text differs, then stale
        indexes the regenerator would delete.  Empty when the bundle passes.
    """
    rendered = render_indexes(Path(bundle_root))
    drifts: list[IndexDrift] = []
    for path in rendered.texts:
        if not path.is_file():
            drifts.append(IndexDrift(path, "missing; this directory holds pages"))
    for path, text in rendered.texts.items():
        if not path.is_file():
            continue
        found = path.read_text(encoding="utf-8")
        if found != text:
            drifts.append(IndexDrift(path, _first_difference(text, found)))
    for path in rendered.stale:
        drifts.append(IndexDrift(path, "lists pages this directory no longer holds"))
    return drifts


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
