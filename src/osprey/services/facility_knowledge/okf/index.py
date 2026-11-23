"""OKF index producer: renders, writes and checks one index.md per non-empty directory.

Index files follow OKF §8: a flat Markdown document with section headings that
group concepts by type.  An index lists concepts only (OKF §3.2): a ``.md``
file without a ``type``, a ``log.md`` and anything below a name starting with
``.`` are left out.  No YAML frontmatter is written — except in the bundle-root
``index.md``, which holds the bundle's metadata (OKF §8.1): ``okf_version``
when the index is new, and whatever the root index already declares when it is
regenerated.

The description synthesizer is intentionally deterministic: it concatenates
child titles (or the first child's description when there is only one child
with a description).  No network or model call is made.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from osprey_facility.frontmatter import FrontmatterError, split_frontmatter

from .document import OKFDocument, OKFDocumentError
from .paths import bundle_markdown_files, load_concept

_INDEX_FILE = "index.md"

#: The OKF version a bundle-root index declares when this producer creates it.
_OKF_VERSION = "0.3"

_DELIM = "---"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _BundleFiles:
    """The Markdown files of a bundle, sorted by what an index does with them.

    Attributes:
        concepts: Each concept's path mapped to its parsed document; these are
            the files an index lists.
        live: Every file that keeps its directory in the tree: the concepts,
            the ``index.md`` files, and pages whose frontmatter does not parse
            (a broken page is not an absent one).  A ``log.md`` and a ``.md``
            file without a ``type`` are in neither.
    """

    concepts: dict[Path, OKFDocument]
    live: frozenset[Path]


def _scan(bundle_root: Path) -> _BundleFiles:
    """Read *bundle_root* once and sort its Markdown files."""
    concepts: dict[Path, OKFDocument] = {}
    live: set[Path] = set()
    for path in bundle_markdown_files(bundle_root):
        if path.name == _INDEX_FILE:
            live.add(path)
            continue
        try:
            doc = load_concept(path)
        except (OKFDocumentError, OSError):
            live.add(path)
            continue
        if doc is not None:
            concepts[path] = doc
            live.add(path)
    return _BundleFiles(concepts=concepts, live=frozenset(live))


def _root_frontmatter(index_path: Path) -> str:
    """Return the frontmatter block the bundle-root index is rendered with.

    A producer that regenerates ``index.md`` keeps the root's frontmatter (OKF
    §8.1), and does not re-serialize values it did not change (OKF §4), so an
    existing block is returned as written.  A root index with no frontmatter,
    an empty one or one that does not parse gets the block a new index is
    given: ``okf_version`` alone.

    Args:
        index_path: The bundle-root ``index.md``, present or not.

    Returns:
        The block from its opening ``---`` line through its closing one, with
        a trailing newline.
    """
    default = f'{_DELIM}\nokf_version: "{_OKF_VERSION}"\n{_DELIM}\n'
    try:
        lines = index_path.read_text(encoding="utf-8").splitlines(keepends=True)
    except OSError:
        return default
    if not lines or lines[0].strip() != _DELIM:
        return default
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == _DELIM), None)
    if end is None:
        return default
    block = "".join(lines[: end + 1])
    try:
        frontmatter, _ = split_frontmatter(block)
    except FrontmatterError:
        return default
    if not frontmatter:
        return default
    return block if block.endswith("\n") else block + "\n"


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
    frontmatter: str = "",
) -> str:
    """Render the Markdown body for an index file.

    Args:
        entries: Each entry is ``(type, title, bundle_relative_link, description)``.
            ``type`` is used as the section heading; an empty string falls back to
            ``"Other"``.
        frontmatter: The frontmatter block to open the file with, from
            :func:`_root_frontmatter` (OKF §8.1 — permitted only in the
            bundle-root ``index.md``).  Empty for every other index.

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

    if frontmatter:
        return f"{frontmatter}\n{body}"
    return body


def _directories_to_index(bundle_root: Path, live: frozenset[Path]) -> list[Path]:
    """Return every directory (including root) that holds at least one live file."""
    dirs: set[Path] = set()
    for md in live:
        cur = md.parent
        while cur != bundle_root.parent:
            dirs.add(cur)
            if cur == bundle_root:
                break
            cur = cur.parent
    return sorted(dirs)


def _live_below(directory: Path, live: frozenset[Path]) -> list[Path]:
    """Return the live files *directory* holds at any depth."""
    return [md for md in live if directory in md.parents]


def _holds_pages(directory: Path, dropped: set[Path], live: frozenset[Path]) -> bool:
    """Return whether *directory* holds any live file at any depth not in *dropped*."""
    return any(md not in dropped for md in _live_below(directory, live))


def _is_stale_index(directory: Path, dropped: set[Path], live: frozenset[Path]) -> bool:
    """Return whether *directory*'s non-blank ``index.md`` is the only live file in it.

    Such an index lists pages that no longer exist. A blank ``index.md`` lists
    nothing and is not stale: it is the placeholder that holds a skeleton
    directory in place until the facility adds its own pages. Child directories
    are processed first, so a child that held only a stale index already has it
    in *dropped* and no longer counts as content here.
    """
    index_path = directory / _INDEX_FILE
    if not index_path.is_file() or not index_path.read_text(encoding="utf-8").strip():
        return False
    return all(md == index_path or md in dropped for md in _live_below(directory, live))


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

    Only concepts are listed (OKF §3.2).  A directory whose only live file is
    its own non-blank ``index.md`` has that stale index listed in ``stale`` (a
    blank placeholder index is neither rendered nor stale), and a parent never
    lists a child directory that holds no live file, so an emptied directory
    advertises no removed page.

    The bundle-root ``index.md`` opens with the bundle's metadata (OKF §8.1):
    the frontmatter it already holds, or ``okf_version`` when it holds none.
    All other ``index.md`` files have no frontmatter (OKF §8).

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
    files = _scan(bundle_root)
    directories = sorted(
        _directories_to_index(bundle_root, files.live),
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
            if child in files.concepts:
                fm = files.concepts[child].frontmatter
                title = str(fm.get("title") or child.stem)
                desc = str(fm.get("description") or "")
                typ = str(fm.get("type") or "")
                entries.append((typ, title, f"/{rel}", desc))
            elif child.is_dir() and _holds_pages(child, dropped, files.live):
                desc = dir_descriptions.get(child, "")
                entries.append(("Subdirectories", child.name, f"/{rel}/", desc))

        if not entries:
            if _is_stale_index(directory, dropped, files.live):
                index_path = directory / _INDEX_FILE
                dropped.add(index_path)
                stale.append(index_path)
            continue

        is_root = directory == bundle_root
        index_path = directory / _INDEX_FILE
        texts[index_path] = _build_index_text(
            entries, frontmatter=_root_frontmatter(index_path) if is_root else ""
        )

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


def _settle_bundle(bundle_root: Path) -> Path | None:
    """Regenerate *bundle_root*'s indexes and leave the root holding an index.

    After it the bundle passes :func:`check_indexes`, and the root holds an
    index: the rendered one, or the blank placeholder when the bundle holds no
    page. The root is created when missing.

    Args:
        bundle_root: Path to the root directory of an OKF bundle.

    Returns:
        The blank placeholder's path when it was written, else ``None``.
    """
    bundle_root = Path(bundle_root)
    bundle_root.mkdir(parents=True, exist_ok=True)
    regenerate_indexes(bundle_root)
    index = bundle_root / _INDEX_FILE
    if index.exists():
        return None
    index.write_text("", encoding="utf-8")
    return index


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
        return line.removesuffix("\n")

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
    drifts: list[IndexDrift] = [
        IndexDrift(path, "missing; this directory holds pages")
        for path in rendered.texts
        if not path.is_file()
    ]
    for path, text in rendered.texts.items():
        if not path.is_file():
            continue
        found = path.read_text(encoding="utf-8")
        if found != text:
            drifts.append(IndexDrift(path, _first_difference(text, found)))
    drifts.extend(
        IndexDrift(path, "lists pages this directory no longer holds") for path in rendered.stale
    )
    return drifts


class OKFIndexError(ValueError):
    """Raised when an ``index.md`` violates OKF index-file rules."""


def validate_index(path: Path, *, bundle_root: Path) -> None:
    """Validate an ``index.md`` file against OKF §8.

    Rules enforced:

    * Frontmatter is **forbidden** in any ``index.md`` that is not the
      bundle-root ``index.md``.
    * The bundle-root ``index.md`` holds the bundle's metadata (OKF §8.1):
      ``okf_version``, ``title``, ``description``, ``license`` and extension
      keys are all permitted.  ``okf_version`` must be a quoted string — an
      unquoted ``0.10`` is the number ``0.1``.

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
            f"Frontmatter is not permitted in sub-directory index files (OKF §8): {path}"
        )

    version = OKFDocument.parse(text).frontmatter.get("okf_version")
    if version is not None and not isinstance(version, str):
        raise OKFIndexError(
            f'okf_version must be a quoted string such as "{_OKF_VERSION}" (OKF §8.1); '
            f"found the unquoted value {version!r}"
        )
