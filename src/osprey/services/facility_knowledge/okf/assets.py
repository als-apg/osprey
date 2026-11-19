"""The rules for a bundle's image files: which file a target names, and what it is.

Four questions, each answered once here for the panel, the agent tools and the
validators:

* **Which directory does a concept's image resolve from?**
  :func:`concept_dir` — the parent of the concept's id, so a symlinked concept
  resolves its images from where its id says it is.
* **Which bundle path does an image target name?** :func:`resolve_image_target`
  applies OKF §6 lexically. It is the only place that decides what is
  external: a target with a URI scheme (``data:`` included) or a leading
  ``//``.
* **What is the file at that path?** :func:`lookup_asset` enforces the
  containment rules concepts already obey (:mod:`.paths`) on the resolved path,
  and takes the type from the file's leading bytes. No image decoder runs.
* **Which images does a concept body embed?** :func:`scan_image_embeds` finds
  what the panel's Markdown renderer draws as an image, with its source span,
  and resolves each target with :func:`resolve_image_target`.

From those answers, :func:`concept_image_findings` reports what is wrong with a
concept's images, and :func:`bundle_image_findings` does so for a whole bundle.

Every refusal is an :class:`AssetError` carrying one of :data:`ASSET_ERROR_CODES`.
"""

from __future__ import annotations

import html
import posixpath
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import unquote

from osprey.imaging.formats import SVG_MIME, sniff

from .document import OKFDocumentError
from .paths import _SCHEME_RE, bundle_markdown_files, load_concept

__all__ = [
    "AGENT_IMAGE_TYPES",
    "ASSET_ERROR_CODES",
    "EMBED_OUTCOMES",
    "MAX_IMAGE_BYTES",
    "PANEL_IMAGE_TYPES",
    "Asset",
    "AssetError",
    "ImageEmbed",
    "concept_dir",
    "lookup_asset",
    "read_asset_bytes",
    "resolve_image_target",
    "scan_image_embeds",
]

#: Largest image file the panel serves and the agent may view.
MAX_IMAGE_BYTES = 10 * 1024 * 1024

#: Types the knowledge panel displays.
PANEL_IMAGE_TYPES: frozenset[str] = frozenset({"png", "jpeg", "gif", "webp", "svg"})

#: Types the agent may view: the panel's, without SVG.
AGENT_IMAGE_TYPES: frozenset[str] = PANEL_IMAGE_TYPES - {"svg"}

#: Every code an :class:`AssetError` carries.
ASSET_ERROR_CODES: frozenset[str] = frozenset(
    {
        "outside_bundle",
        "dot_path",
        "symlink_escape",
        "not_found",
        "not_an_image",
        "disallowed_type",
        "too_large",
    }
)

_RASTER_TYPES = AGENT_IMAGE_TYPES

# Bytes read for the type decision: room for an XML declaration, a few comments
# and a doctype ahead of the root element of an exported figure.
_HEAD_BYTES = 4096

_UTF8_BOM = b"\xef\xbb\xbf"
_XML_WHITESPACE = b" \t\r\n"
_DOCTYPE = b"<!doctype"
_ELEMENT_NAME_RE = re.compile(rb"<([A-Za-z_][A-Za-z0-9_.:-]*)")
_NAME_END = frozenset(b" \t\r\n/>")


class AssetError(Exception):
    """A target or file the asset rules refuse.

    Attributes:
        code: One of :data:`ASSET_ERROR_CODES`.
    """

    def __init__(self, code: str, message: str = "") -> None:
        if code not in ASSET_ERROR_CODES:
            raise ValueError(f"unknown asset error code: {code!r}")
        super().__init__(message or code)
        self.code = code


@dataclass(frozen=True)
class Asset:
    """An image file in a bundle that passed :func:`lookup_asset`.

    Attributes:
        path: The resolved filesystem path, inside the resolved bundle root.
        type: One of :data:`PANEL_IMAGE_TYPES`.
        mime: The MIME type for *type*.
        size: Byte size, at most :data:`MAX_IMAGE_BYTES`.
    """

    path: Path
    type: str
    mime: str
    size: int


def _has_dot_part(parts: tuple[str, ...]) -> bool:
    return any(part.startswith(".") for part in parts)


def _escapes(normalised: str) -> bool:
    return normalised == ".." or normalised.startswith("../")


def concept_dir(root: Path, concept_id: str) -> str:
    """Return the bundle directory a concept's relative targets resolve from.

    Args:
        root: Bundle root.
        concept_id: OKF §2 id, with or without a leading ``/``; ``""`` for the
            bundle root (the overview).

    Returns:
        The parent of the normalised id as a bundle-relative POSIX path, ``""``
        at the root. A symlinked concept resolves from its id, not its target.

    Raises:
        AssetError: ``outside_bundle`` when the id does not name an existing
            concept file inside the root (a missing file, an id that leaves the
            root or lies under a dot-path, a symlink out of the root).
    """
    clean = concept_id.strip().lstrip("/")
    if not clean:
        return ""
    normalised = posixpath.normpath(clean)
    if _escapes(normalised) or normalised == "." or _has_dot_part(PurePosixPath(normalised).parts):
        raise AssetError("outside_bundle", f"concept id leaves the bundle: {concept_id!r}")
    path = root / f"{normalised}.md"
    resolved_root = root.resolve()
    resolved = path.resolve()
    if (
        not path.is_file()
        or not resolved.is_relative_to(resolved_root)
        or _has_dot_part(resolved.relative_to(resolved_root).parts)
    ):
        raise AssetError("outside_bundle", f"no concept with id {concept_id!r}")
    return posixpath.dirname(normalised)


def resolve_image_target(target: str, source_dir: str) -> str | None:
    """Resolve an image target as written in Markdown to a bundle path (OKF §6).

    Lexical only: no file is touched. A ``/`` target resolves from the root, any
    other from *source_dir*. The target is percent-decoded and loses its
    fragment and query.

    Args:
        target: The raw target, e.g. ``figures/plot.png`` or ``/img/a.png``.
        source_dir: Bundle-relative POSIX directory of the embedding concept
            (from :func:`concept_dir`); ``""`` for the root.

    Returns:
        The bundle-relative POSIX path, ``""`` for the root itself, or ``None``
        when the target is external: it carries a URI scheme (``data:``
        included), starts with ``//``, or is empty or only a fragment or query.

    Raises:
        AssetError: ``outside_bundle`` when the path leaves the bundle root.
    """
    raw = target.strip()
    if not raw or raw.startswith("//") or _SCHEME_RE.match(raw):
        return None
    raw = unquote(raw.split("#", 1)[0].split("?", 1)[0])
    if not raw:
        return None
    joined = raw.lstrip("/") if raw.startswith("/") else posixpath.join(source_dir, raw)
    normalised = posixpath.normpath(joined) if joined else "."
    if _escapes(normalised):
        raise AssetError("outside_bundle", f"image target leaves the bundle: {target!r}")
    return "" if normalised == "." else normalised


def _skip_to(head: bytes, start: int, end_marker: bytes) -> int | None:
    found = head.find(end_marker, start)
    return None if found < 0 else found + len(end_marker)


def _skip_doctype(head: bytes, pos: int) -> int | None:
    """Position after the doctype at *pos*, internal subset included."""
    i = pos + len(_DOCTYPE)
    while i < len(head):
        byte = head[i : i + 1]
        if byte == b">":
            return i + 1
        if byte == b"[":
            close = head.find(b"]", i)
            if close < 0:
                return None
            i = close
        i += 1
    return None


def _is_svg(head: bytes) -> bool:
    """Whether the first element in *head* is ``svg``.

    A bounded prefix scan that skips a byte-order mark, whitespace, processing
    instructions (the XML declaration), comments and a doctype. No XML parser.
    """
    head = head.removeprefix(_UTF8_BOM)
    pos: int | None = 0
    while pos is not None and pos < len(head):
        if head[pos] in _XML_WHITESPACE:
            pos += 1
        elif head.startswith(b"<?", pos):
            pos = _skip_to(head, pos + 2, b"?>")
        elif head.startswith(b"<!--", pos):
            pos = _skip_to(head, pos + 4, b"-->")
        elif head[pos : pos + len(_DOCTYPE)].lower() == _DOCTYPE:
            pos = _skip_doctype(head, pos)
        else:
            match = _ELEMENT_NAME_RE.match(head, pos)
            if match is None or match.group(1) != b"svg":
                return False
            after = match.end()
            return after < len(head) and head[after] in _NAME_END
    return False


def _image_type(head: bytes) -> tuple[str, str]:
    """The (type, MIME) of a file from its leading bytes, or raise."""
    found = sniff(head)
    if found.row in _RASTER_TYPES:
        return found.row, found.mime
    if _is_svg(head):
        return "svg", SVG_MIME
    if found.row != "svg" and found.mime.startswith("image/"):
        raise AssetError("disallowed_type", f"image type {found.mime} is not allowed")
    raise AssetError("not_an_image", "the file is not an image")


def lookup_asset(root: Path, bundle_path: str) -> Asset:
    """Find the image file at a root-relative bundle path and decide its type.

    Containment follows the rules concepts obey (:mod:`.paths`) and is checked
    on the resolved path: it stays in the root, holds no part starting with
    ``.``, and no symbolic link leads out of the root. The type comes from the
    leading bytes, never the name; no image decoder is called.

    Args:
        root: Bundle root.
        bundle_path: Root-relative POSIX path, taken literally (not
            percent-decoded); a leading ``/`` is allowed.

    Returns:
        The :class:`Asset`.

    Raises:
        AssetError: With the first code that applies, in this order:
            ``outside_bundle`` (the path leaves the root), ``dot_path``,
            ``symlink_escape``, ``not_found`` (also a directory or other
            non-regular file), ``not_an_image``, ``disallowed_type``,
            ``too_large``.
    """
    normalised = posixpath.normpath(bundle_path.lstrip("/") or ".")
    if _escapes(normalised):
        raise AssetError("outside_bundle", f"path leaves the bundle: {bundle_path!r}")
    if _has_dot_part(PurePosixPath(normalised).parts):
        raise AssetError("dot_path", f"path lies under a dot-path: {bundle_path!r}")

    resolved_root = root.resolve()
    resolved = (root / normalised).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise AssetError("symlink_escape", f"path leads out of the bundle: {bundle_path!r}")
    if _has_dot_part(resolved.relative_to(resolved_root).parts):
        raise AssetError("dot_path", f"path leads to a dot-path: {bundle_path!r}")
    if not resolved.is_file():
        raise AssetError("not_found", f"no file at {bundle_path!r}")

    try:
        with resolved.open("rb") as handle:
            head = handle.read(_HEAD_BYTES)
            size = resolved.stat().st_size
    except FileNotFoundError as exc:
        raise AssetError("not_found", f"no file at {bundle_path!r}") from exc

    kind, mime = _image_type(head)
    if size > MAX_IMAGE_BYTES:
        raise AssetError("too_large", f"{size} bytes exceeds {MAX_IMAGE_BYTES}")
    return Asset(path=resolved, type=kind, mime=mime, size=size)


def read_asset_bytes(asset: Asset) -> bytes:
    """Read the bytes of a looked-up asset, never more than :data:`MAX_IMAGE_BYTES`.

    The file is read again on every call; one that grew past the limit since
    :func:`lookup_asset` is refused.

    Raises:
        AssetError: ``too_large`` when the file now holds more than the limit.
        OSError: The file can no longer be read.
    """
    with asset.path.open("rb") as handle:
        data = handle.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise AssetError("too_large", f"the file grew past {MAX_IMAGE_BYTES} bytes")
    return data


# --------------------------------------------------------------------------- #
# Embed scan
# --------------------------------------------------------------------------- #

#: What the resolver made of an embed's target.
EMBED_OUTCOMES: frozenset[str] = frozenset({"bundle", "external", "outside_bundle"})


@dataclass(frozen=True)
class ImageEmbed:
    """An image a concept body embeds.

    Attributes:
        target: The target as the renderer reads it: angle brackets removed,
            backslash escapes and character references decoded, not
            percent-decoded.
        alt: The alt text, stripped; ``""`` when there is none.
        span: ``(start, end)`` character offsets of the whole embed in the body.
        outcome: One of :data:`EMBED_OUTCOMES`.
        path: The bundle path when *outcome* is ``bundle``, else ``None``.
    """

    target: str
    alt: str
    span: tuple[int, int]
    outcome: str
    path: str | None


_ASCII_PUNCT = frozenset("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
# Paren nesting a bare link destination may have (the CommonMark reference limit).
_MAX_PAREN_DEPTH = 32
_ESCAPE_RE = re.compile(r"\\([!-/:-@\[-`{-~])")
_FENCE_RE = re.compile(r"(`{3,}|~{3,})(.*)$")
_LIST_RE = re.compile(r"(?:[-+*]|\d{1,9}[.)])(?:[ \t]+|$)")
_DEFINITION_RE = re.compile(
    r"\[((?:[^\[\]\\]|\\.)+)\]:[ \t]*(<[^<>\n]*>|\S+)"
    r"(?:[ \t]+(?:\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'|\((?:[^()\\]|\\.)*\)))?[ \t]*$"
)
_ATTRIBUTE = (
    r"[ \t\r\n]+[A-Za-z_:][A-Za-z0-9_.:-]*"
    r"(?:[ \t\r\n]*=[ \t\r\n]*(?:[^ \t\r\n\"'=<>`]+|'[^']*'|\"[^\"]*\"))?"
)
_IMG_TAG_RE = re.compile(rf"<img(?P<attrs>(?:{_ATTRIBUTE})*)[ \t\r\n]*/?>", re.IGNORECASE)
_TAG_RE = re.compile(rf"</?[A-Za-z][A-Za-z0-9-]*(?:{_ATTRIBUTE})*[ \t\r\n]*/?>")
_ATTRIBUTE_ITEM_RE = re.compile(
    r"([A-Za-z_:][A-Za-z0-9_.:-]*)"
    r"(?:[ \t\r\n]*=[ \t\r\n]*(?:\"([^\"]*)\"|'([^']*)'|([^ \t\r\n\"'=<>`]+)))?"
)


def _decode(text: str) -> str:
    return html.unescape(_ESCAPE_RE.sub(r"\1", text))


def _label_key(label: str) -> str:
    return " ".join(label.split()).casefold()


def _indent(line: str) -> tuple[int, int]:
    """(column, index) of the first non-blank character, tabs to multiples of 4."""
    column = 0
    for index, char in enumerate(line):
        if char == " ":
            column += 1
        elif char == "\t":
            column += 4 - column % 4
        else:
            return column, index
    return column, len(line)


def _blocks(body: str) -> tuple[list[tuple[int, int]], dict[str, str]]:
    """Split *body* into rendered paragraphs and collect reference definitions.

    Returns the ``(start, end)`` offsets of each run of paragraph lines (fenced
    and indented code, HTML comment blocks and definitions excluded) and the
    definitions by normalised label, first one winning.
    """
    paragraphs: list[tuple[int, int]] = []
    definitions: dict[str, str] = {}
    paragraph: list[int] | None = None
    fence: tuple[str, int] | None = None
    in_comment = False
    list_indent = 0
    pos = 0

    def close() -> None:
        nonlocal paragraph
        if paragraph is not None:
            paragraphs.append((paragraph[0], paragraph[1]))
            paragraph = None

    for raw in body.splitlines(keepends=True):
        start, pos = pos, pos + len(raw)
        line = raw.rstrip("\r\n")
        if fence is not None:
            column, index = _indent(line)
            closer = line[index:].rstrip()
            if column <= 3 + list_indent and closer and set(closer) == {fence[0]}:
                fence = None if len(closer) >= fence[1] else fence
            continue
        if in_comment:
            in_comment = "-->" not in line
            continue
        if not line.strip():
            close()
            continue

        column, index = _indent(line)
        content = line[index:]
        if (
            list_indent
            and column < list_indent
            and paragraph is None
            and not (column <= 3 and _LIST_RE.match(content))
        ):
            list_indent = 0
        base = list_indent if list_indent and column >= list_indent else 0
        relative = column - base

        if relative >= 4:
            if paragraph is None:
                continue  # indented code
            paragraph[1] = pos
            continue
        fence_match = _FENCE_RE.match(content)
        if fence_match and not (fence_match.group(1)[0] == "`" and "`" in fence_match.group(2)):
            close()
            fence = (fence_match.group(1)[0], len(fence_match.group(1)))
            continue
        if content.startswith("<!--"):
            close()
            in_comment = "-->" not in content[4:]
            continue
        if paragraph is None:
            definition = _DEFINITION_RE.match(content)
            if definition:
                destination = definition.group(2)
                if destination.startswith("<"):
                    destination = destination[1:-1]
                definitions.setdefault(_label_key(definition.group(1)), _decode(destination))
                continue
        list_item = _LIST_RE.match(content)
        if list_item:
            close()
            list_indent = column + max(list_item.end(), len(list_item.group(0).rstrip()) + 1)
        if paragraph is None:
            paragraph = [start, pos]
        else:
            paragraph[1] = pos
    close()
    return paragraphs, definitions


class _Paragraph:
    """One paragraph of a body, with the lookups its scan repeats precomputed.

    Code spans and bracket pairs are found once, so a body full of unclosed
    ``![`` or backtick runs still scans in linear time.
    """

    def __init__(self, body: str, start: int, end: int) -> None:
        self.body = body
        self.start = start
        self.end = end
        # Run width -> a position after which no run of that width closes.
        self._unclosed: dict[int, int] = {}
        self.brackets: dict[int, int] = {}
        stack: list[int] = []
        i = start
        while i < end:
            char = body[i]
            if char == "\\":
                i += 2
                continue
            if char == "`":
                i = self.after_backticks(i)
                continue
            if char == "[":
                stack.append(i)
            elif char == "]" and stack:
                self.brackets[stack.pop()] = i
            i += 1

    def code_span_end(self, pos: int) -> int | None:
        """End of the code span opening at *pos*, or ``None`` when it never closes."""
        body, end = self.body, self.end
        run = pos
        while run < end and body[run] == "`":
            run += 1
        width = run - pos
        if self._unclosed.get(width, end) <= pos:
            return None
        search = run
        while search < end:
            found = body.find("`" * width, search, end)
            if found < 0:
                break
            after = found + width
            while after < end and body[after] == "`":
                after += 1
            if after - found == width:
                return after
            search = after
        self._unclosed[width] = pos
        return None

    def after_backticks(self, pos: int) -> int:
        """Position after the code span at *pos*, or after its literal backticks."""
        span_end = self.code_span_end(pos)
        if span_end is not None:
            return span_end
        while pos < self.end and self.body[pos] == "`":
            pos += 1
        return pos


def _skip_space(body: str, i: int, end: int) -> int:
    newline = False
    while i < end and body[i] in " \t\r\n":
        if body[i] == "\n":
            if newline:
                break
            newline = True
        i += 1
    return i


def _inline_tail(body: str, i: int, end: int) -> tuple[str, int] | None:
    """Parse ``target "title")`` after an image's ``(``: (target, end of embed)."""
    i = _skip_space(body, i, end)
    if i < end and body[i] == "<":
        j = i + 1
        while j < end and body[j] not in "<>\n":
            j += 2 if body[j] == "\\" else 1
        if j >= end or body[j] != ">":
            return None
        target, i = body[i + 1 : j], j + 1
    else:
        j, depth = i, 0
        while j < end:
            char = body[j]
            if char == "\\" and j + 1 < end and body[j + 1] in _ASCII_PUNCT:
                j += 2
                continue
            if char in " \t\r\n" or ord(char) < 0x20:
                break
            if char == "(":
                depth += 1
                if depth > _MAX_PAREN_DEPTH:
                    return None
            elif char == ")":
                if depth == 0:
                    break
                depth -= 1
            j += 1
        if depth:
            return None
        target, i = body[i:j], j
    after_target = i
    i = _skip_space(body, i, end)
    if i < end and body[i] in "\"'(" and (i > after_target or not target):
        closer = ")" if body[i] == "(" else body[i]
        j = i + 1
        while j < end and body[j] != closer and body[j] != "\n":
            j += 2 if body[j] == "\\" else 1
        if j >= end or body[j] != closer:
            return None
        i = _skip_space(body, j + 1, end)
    if i < end and body[i] == ")":
        return _decode(target), i + 1
    return None


def _image_at(
    paragraph: _Paragraph, pos: int, definitions: dict[str, str]
) -> tuple[str, str, int] | None:
    """Parse the Markdown image at *pos* (``![``): (target, alt, end) or ``None``."""
    body, end = paragraph.body, paragraph.end
    close = paragraph.brackets.get(pos + 1)
    if close is None:
        return None
    alt = body[pos + 2 : close]
    after = close + 1
    if after < end and body[after] == "(":
        inline = _inline_tail(body, after + 1, end)
        if inline is not None:
            return inline[0], alt, inline[1]
    if after < end and body[after] == "[":
        label_close = paragraph.brackets.get(after)
        if label_close is not None:
            label = body[after + 1 : label_close] or alt
            target = definitions.get(_label_key(label))
            return None if target is None else (target, alt, label_close + 1)
    target = definitions.get(_label_key(alt))
    return None if target is None else (target, alt, after)


def _img_tag(match: re.Match[str]) -> tuple[str | None, str]:
    """(src, alt) of a raw ``<img>`` tag; src ``None`` when absent."""
    values: dict[str, str] = {}
    for item in _ATTRIBUTE_ITEM_RE.finditer(match.group("attrs")):
        name = item.group(1).lower()
        value = next((g for g in item.group(2, 3, 4) if g is not None), "")
        values.setdefault(name, html.unescape(value))
    return values.get("src"), values.get("alt", "")


def _scan_paragraph(
    body: str, start: int, end: int, definitions: dict[str, str]
) -> list[tuple[str, str, int, int]]:
    paragraph = _Paragraph(body, start, end)
    found: list[tuple[str, str, int, int]] = []
    comment_unclosed = end  # no "-->" follows an unclosed "<!--" from here on
    i = start
    while i < end:
        char = body[i]
        if char == "\\":
            i += 2 if i + 1 < end and body[i + 1] in _ASCII_PUNCT else 1
        elif char == "`":
            i = paragraph.after_backticks(i)
        elif char == "<":
            if body.startswith("<!--", i):
                close = body.find("-->", i + 4, end) if i < comment_unclosed else -1
                if close < 0:
                    comment_unclosed = min(comment_unclosed, i)
                i = close + 3 if close >= 0 else i + 1
                continue
            tag = _IMG_TAG_RE.match(body, i, end)
            if tag:
                src, alt = _img_tag(tag)
                if src is not None:
                    found.append((src, alt, i, tag.end()))
                i = tag.end()
                continue
            other = _TAG_RE.match(body, i, end)
            i = other.end() if other else i + 1
        elif char == "!" and body.startswith("[", i + 1):
            image = _image_at(paragraph, i, definitions)
            if image is None:
                i += 1
            else:
                target, alt, image_end = image
                found.append((target, _decode(alt), i, image_end))
                i = image_end
        else:
            i += 1
    return found


def scan_image_embeds(body: str, source_dir: str) -> list[ImageEmbed]:
    """Find the images a concept body embeds, in source order.

    Recognises inline ``![alt](target "title")``, reference-style images (full,
    collapsed and shortcut, against the body's definitions) and raw HTML
    ``<img>``. Reports nothing the panel's renderer would not draw as an image:
    fenced, indented and inline code, a backslash-escaped ``\\![``, HTML
    comments, and a reference without a definition. Markdown inside a raw HTML
    block is still scanned. Each target goes through
    :func:`resolve_image_target`; this function has no rule of its own about
    what is external.

    Args:
        body: The concept body; spans are offsets into this exact string.
        source_dir: Bundle directory of the concept (from :func:`concept_dir`).

    Returns:
        One :class:`ImageEmbed` per embed, ordered by span start.
    """
    paragraphs, definitions = _blocks(body)
    embeds: list[ImageEmbed] = []
    for start, end in paragraphs:
        for target, alt, span_start, span_end in _scan_paragraph(body, start, end, definitions):
            try:
                path = resolve_image_target(target, source_dir)
            except AssetError:
                outcome, path = "outside_bundle", None
            else:
                outcome = "external" if path is None else "bundle"
            embeds.append(
                ImageEmbed(
                    target=target,
                    alt=alt.strip(),
                    span=(span_start, span_end),
                    outcome=outcome,
                    path=path,
                )
            )
    return embeds


#: The kinds of :class:`ImageFinding`.
IMAGE_FINDING_KINDS: frozenset[str] = frozenset(
    {"image-missing", "image-type", "image-too-large", "image-alt-empty"}
)

# The one mapping from an AssetError code to a finding kind.
_FINDING_KIND_BY_CODE: dict[str, str] = {
    "not_found": "image-missing",
    "dot_path": "image-missing",
    "symlink_escape": "image-missing",
    "outside_bundle": "image-missing",
    "not_an_image": "image-type",
    "disallowed_type": "image-type",
    "too_large": "image-too-large",
}


@dataclass(frozen=True)
class ImageFinding:
    """A problem with an image a concept embeds.

    Attributes:
        concept_id: Id of the embedding concept.
        target: The embed's target, as :class:`ImageEmbed` reports it.
        kind: One of :data:`IMAGE_FINDING_KINDS`.
        detail: Human-readable reason; for a refused file it carries the
            :class:`AssetError` code.
    """

    concept_id: str
    target: str
    kind: str
    detail: str


def _refusal_detail(error: AssetError) -> str:
    message = str(error)
    return error.code if message == error.code else f"{error.code}: {message}"


def concept_image_findings(root: Path, concept_id: str, body: str) -> list[ImageFinding]:
    """Report what is wrong with the images one concept body embeds.

    A refused in-bundle image gives ``image-missing``, ``image-type`` or
    ``image-too-large`` by its :class:`AssetError` code; empty alt text on an
    embed that is not external gives ``image-alt-empty`` as well. External
    images give nothing, and so does SVG: the panel displays it.

    Args:
        root: Bundle root.
        concept_id: The concept's id (see :func:`concept_dir`).
        body: The concept body.

    Returns:
        Findings in source order; for one embed, the file finding comes first.

    Raises:
        AssetError: ``outside_bundle`` when *concept_id* names no concept.
    """
    findings: list[ImageFinding] = []
    for embed in scan_image_embeds(body, concept_dir(root, concept_id)):
        if embed.outcome == "external":
            continue
        if embed.outcome == "outside_bundle":
            refusal: AssetError | None = AssetError(
                "outside_bundle", f"target leaves the bundle: {embed.target!r}"
            )
        else:
            try:
                lookup_asset(root, embed.path or "")
            except AssetError as exc:
                refusal = exc
            else:
                refusal = None
        if refusal is not None:
            findings.append(
                ImageFinding(
                    concept_id=concept_id,
                    target=embed.target,
                    kind=_FINDING_KIND_BY_CODE[refusal.code],
                    detail=_refusal_detail(refusal),
                )
            )
        if not embed.alt:
            findings.append(
                ImageFinding(
                    concept_id=concept_id,
                    target=embed.target,
                    kind="image-alt-empty",
                    detail="the image has no alt text",
                )
            )
    return findings


def bundle_image_findings(root: Path) -> list[ImageFinding]:
    """Run :func:`concept_image_findings` over every concept in a bundle.

    A concept that cannot be read or parsed is skipped: the bundle validators
    report it.

    Args:
        root: Bundle root.

    Returns:
        Findings in bundle-walk order, then source order within a concept.
    """
    findings: list[ImageFinding] = []
    for path in bundle_markdown_files(root):
        try:
            doc = load_concept(path)
        except (OKFDocumentError, OSError):
            continue
        if doc is None:
            continue
        concept_id = path.relative_to(root).with_suffix("").as_posix()
        findings.extend(concept_image_findings(root, concept_id, doc.body))
    return findings
