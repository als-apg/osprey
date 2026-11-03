"""The push grammar (CANOPY's, copied) and the terminal-safe text every printed line passes through."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from osprey.integrations.canopy.library import (
    DEPLOYMENT_IGNORED_NAMES,
    IMAGE_SUFFIXES,
    SCREEN_SUFFIX,
)

from .refusals import GRAMMAR_EXIT, Refusal

# --- the grammar (CANOPY's, copied: this module cannot import canopy) ------------------------

ID_PATTERN = re.compile(r"[A-Za-z0-9_-]+(/[A-Za-z0-9_-]+)*")
ASSET_SEGMENT = re.compile(r"[A-Za-z0-9 ._-]+")
PUSH_IMAGE_SUFFIXES = frozenset(IMAGE_SUFFIXES - {".svg"})
RESERVED_SEGMENTS = frozenset({"__pycache__", ".git", ".venv", ".pytest_cache"})
RESERVED_PREFIXES = (".", "-")
RESERVED_ENDINGS = (".swp", ".swo", ".pyc", ".pyo", ".pyd", ".log", ".pid")
_FORBIDDEN_CATEGORIES = frozenset({"Cc", "Cf", "Cs"})
_ALLOWED_CONTROLS = frozenset("\t\n\r")

MAX_ASSET_BYTES = 5 * 2**20
MAX_FILES = 256
MAX_TOTAL_BYTES = 32 * 2**20
MAX_REQUEST_BYTES = 64 * 2**10
MAX_NOTE_CHARS = 2000
#: How many bundle directories ``list`` reads per volume at most.
MAX_LIST_BUNDLES = 256
#: What the host reads of a container's stdout at most (the two reports can be large).
MAX_VERDICT_BYTES = 256 * 2**20


def is_reserved_segment(segment: str) -> bool:
    """Whether ``segment`` is one the push grammar reserves (compared on ``casefold()``)."""
    folded = segment.casefold()
    return (
        folded in RESERVED_SEGMENTS
        or folded in DEPLOYMENT_IGNORED_NAMES
        or folded.startswith(RESERVED_PREFIXES)
        or folded.endswith(RESERVED_ENDINGS)
    )


def is_forbidden_char(char: str) -> bool:
    """Whether the single character ``char`` may not appear in a pushed string.

    C0 controls other than tab, LF and CR, DEL, C1 controls, the bidi controls
    and every other format character, and lone surrogates.
    """
    return char not in _ALLOWED_CONTROLS and unicodedata.category(char) in _FORBIDDEN_CATEGORIES


def has_forbidden_char(text: str) -> bool:
    """Whether any character of ``text`` is one :func:`is_forbidden_char` names."""
    return any(is_forbidden_char(char) for char in text)


def first_forbidden_pointer(doc: Any) -> str | None:
    """The RFC 6901 pointer of the first string in ``doc`` holding a forbidden character, else ``None``.

    Depth-first in document order, each key before its value; an offending key
    reports the pointer of the object that holds it. The root is ``""``.
    """
    if isinstance(doc, str):
        return "" if has_forbidden_char(doc) else None
    stack: list[tuple[str, Any]] = []
    if isinstance(doc, dict):
        stack.append(("", iter(doc.items())))
    elif isinstance(doc, list):
        stack.append(("", iter(enumerate(doc))))
    while stack:
        pointer, members = stack[-1]
        entry = next(members, None)
        if entry is None:
            stack.pop()
            continue
        key, value = entry
        if isinstance(key, str):
            if has_forbidden_char(key):
                return pointer
            token = key.replace("~", "~0").replace("/", "~1")
        else:
            token = str(key)
        child = f"{pointer}/{token}"
        if isinstance(value, str):
            if has_forbidden_char(value):
                return child
        elif isinstance(value, dict):
            stack.append((child, iter(value.items())))
        elif isinstance(value, list):
            stack.append((child, iter(enumerate(value))))
    return None


def casefold_collisions(paths: Iterable[str]) -> list[list[str]]:
    """The names among ``paths`` and their directory prefixes that are not unique under ``casefold()``."""
    members = set(paths)
    directories = {path[:at] for path in members for at, char in enumerate(path) if char == "/"}
    groups: dict[str, list[str]] = {}
    for name in (*members, *directories):
        groups.setdefault(name.casefold(), []).append(name)
    return sorted(sorted(set(names)) for names in groups.values() if len(names) > 1)


# --- output -------------------------------------------------------------------------------


def safe_text(value: Any) -> str:
    """``value`` as one terminal-safe line.

    A string is printed as itself; anything else as sorted ASCII JSON. CR, LF,
    TAB and backslash become two-character escapes, every forbidden character
    is dropped, and every other non-ASCII character is written as ``\\uXXXX``
    (one per UTF-16 unit).
    """
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=True)
    out: list[str] = []
    for char in text:
        if char == "\\":
            out.append("\\\\")
        elif char == "\r":
            out.append("\\r")
        elif char == "\n":
            out.append("\\n")
        elif char == "\t":
            out.append("\\t")
        elif is_forbidden_char(char):
            continue
        elif 0x20 <= ord(char) < 0x7F:
            out.append(char)
        else:
            units = char.encode("utf-16-be")
            out.extend(f"\\u{units[at] << 8 | units[at + 1]:04x}" for at in range(0, len(units), 2))
    return "".join(out)


_safe_text = safe_text


def _invalid_text(data: bytes) -> tuple[str, str] | None:
    """``(pointer, what)`` when a screen's bytes hold text a push may not carry, else ``None``."""
    if data.startswith(b"\xef\xbb\xbf"):
        return "", "starts with a byte-order mark"
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return "", "is not UTF-8"
    try:
        doc = json.loads(text)
    except ValueError:
        return "", "is not JSON"
    pointer = first_forbidden_pointer(doc)
    if pointer is not None:
        return pointer, "holds a forbidden character"
    if has_forbidden_char(text):
        return "", "holds a forbidden character outside the parsed document"
    if not isinstance(doc, dict):
        return "", "is not a JSON object"
    return None


def _grammar(files: Mapping[str, bytes]) -> list[dict[str, Any]]:
    """Every way ``files`` (path below the source directory -> bytes) breaks the push grammar."""
    found: list[dict[str, Any]] = []

    def refuse(code: str, path: str, message: str, **extra: Any) -> None:
        found.append({"code": code, "path": path, "message": message, **extra})

    for path, data in sorted(files.items()):
        is_screen = path.endswith(SCREEN_SUFFIX)
        named = path[: -len(SCREEN_SUFFIX)] if is_screen else path
        segments = named.split("/")
        pattern = ID_PATTERN if is_screen else ASSET_SEGMENT
        if not all(pattern.fullmatch(segment) for segment in segments):
            kind = "screen" if is_screen else "image"
            refuse("bad_segment", path, f"{path!r} is not a name a pushed {kind} may have")
        for segment in segments:
            if is_reserved_segment(segment):
                refuse("reserved_segment", path, f"{path!r} holds the reserved name {segment!r}")
        if not is_screen and Path(path.lower()).suffix not in PUSH_IMAGE_SUFFIXES:
            refuse("suffix", path, f"{path!r} is not an image a push may carry")
        if len(data) > MAX_ASSET_BYTES:
            refuse(
                "file_too_large",
                path,
                f"{path!r} is {len(data)} bytes; at most {MAX_ASSET_BYTES} per file",
            )
        if is_screen:
            invalid = _invalid_text(data)
            if invalid is not None:
                refuse("invalid_text", path, f"{path!r} {invalid[1]}", pointer=invalid[0])
    if len(files) > MAX_FILES:
        refuse("too_many_files", "", f"the push holds {len(files)} files; at most {MAX_FILES}")
    total = sum(len(data) for data in files.values())
    if total > MAX_TOTAL_BYTES:
        refuse("bundle_too_large", "", f"the push holds {total} bytes; at most {MAX_TOTAL_BYTES}")
    for spellings in casefold_collisions(files):
        message = f"{', '.join(repr(s) for s in spellings)} differ only in case, or name both a file and a directory"
        refuse("case_collision", spellings[0], message, paths=spellings)
    return found


def _check_grammar(files: Mapping[str, bytes]) -> None:
    found = _grammar(files)
    if found:
        lines = [
            f"{item['code']}: {item['message']}"
            + (f" at {item['pointer']!r}" if "pointer" in item else "")
            for item in found
        ]
        raise Refusal(GRAMMAR_EXIT[found[0]["code"]], *lines)
