"""The files of a deployment's CANOPY screen library, and whether its report still covers them.

A screen library (``web/canopy/library`` in a deployment) holds ``screens/``
and, beside it, the report CANOPY generates (``LIBRARY-REPORT.json`` and
``LIBRARY-REPORT.md``) and, when the deployment writes one, ``navigation.json``:
the library's own ordering of its screens for the viewer's sidebar. This module
answers three questions about one, with the standard library alone: it runs under the
interpreter that runs ``osprey build`` and on the host that reviews a push,
where CANOPY is not installed.

**Which files are the library** — the listing. CANOPY implements the same
definition (``canopy_mcp.listing``), and ``listing-vectors.json``, shared byte
for byte between the two repositories, pins both:

- :func:`scan` reads the filesystem and says what is there, nothing more: one
  entry ``{path, kind, mode, sha256}`` per directory entry, with ``kind`` one
  of ``file``, ``dir``, ``symlink`` or ``other`` (from ``lstat``, so a symlink
  is never followed) and ``sha256`` on regular files only.
- :func:`judge` is pure. From entries alone it gives the list, its digest and
  the refusals. It never asks the filesystem, so its answer is the same on a
  host that cannot hold two names differing only in case; the shared vectors
  are entry lists run through it.

The list is one line ``<sha256 hex>  screens/<relpath>\\n`` (two spaces) per
listed file, sorted by the UTF-8 bytes of ``relpath``, spaces kept and nothing
escaped, encoded as UTF-8; :func:`library_list` returns it for a directory.
The digest (:func:`library_digest`) is ``sha256:`` and the hex sha256 of those
bytes. A file is listed when it is a regular file under ``screens/`` named
``*.screen.json`` (as spelled) or carrying an image suffix (in any case), and
no segment of its path is refused.

Every other entry is refused by path and code, and the text of a refusal
(:func:`refusal_text`) says what to do with it:

- ``outside_screens``: an entry of the root other than ``screens/``, the two
  report files and ``navigation.json`` (reported once, by its top-level name).
  Each of the three files is admitted as a regular file only: a symlink of
  that name is a ``symlink`` refusal and a directory of that name is refused
  with this code. None of them is on the list, so none moves the digest.
- ``symlink``, ``not_regular``: a symlink anywhere; a socket, pipe or device.
- ``bad_name``: a name holding a control, format or surrogate character, which
  the list could not carry on one line.
- ``leftover_tmp``: a ``.*.tmp`` name, what an interrupted atomic write
  leaves, under ``screens/`` or in the root itself.
- ``dotfile``: any other name starting with ``.``.
- ``ignored_name``: a name the image build context or the deployment's
  ``.gitignore`` prunes (:func:`is_ignored_name`), so the image would not hold
  it.
- ``suffix``: a regular file that is neither a screen nor an image.
- ``case_collision``: entries, or directory names under one parent, that are
  equal under ``str.casefold()``.

A name rule applies to every segment below ``screens/`` and is reported at the
shortest path that breaks it, so a directory named ``img.log`` is one refusal,
whatever it holds. One exception: a regular file named exactly ``.DS_Store``
is skipped, neither listed nor refused, since git ignores it and the image
context drops it.

A CANOPY server keeps its listing of the screens between runs in a ``.cache``
directory, by default inside the store it writes to (``CANOPY_CACHE_DIR``
moves it). The panel's server writes to the user's personal store, so a
deployed library never holds one. A server started on a library as its own
store does write it there, and the directory is then refused like any other
entry beside ``screens/`` (``outside_screens``) until it is deleted.

**Whether the navigation is one the viewer will draw** —
:func:`navigation_faults`. ``navigation.json`` is a JSON list of entries. A
leaf is ``{"screen": "library/<id>", "label": "..."}``, the label optional. A
group is ``{"label": "...", "children": [...]}`` and may name a ``screen`` of
its own. CANOPY reads the file when it serves the library and hands a valid one
to the viewer; an invalid one it ignores with a line in the server's log, and
the viewer leaves out an entry whose screen the library does not hold. Neither
is seen by whoever builds the deployment, so the check here refuses both
before anything is built: the shape and the limits CANOPY states (the file's
size, the nesting depth, the number of entries, the length of a label and of
a screen name, no other key), and every ``screen`` naming a screen on the
library's list. CANOPY's listing admits the file as this one does, and its
report does not read it: the file is in no row and no digest.

**Whether the report is stale** — :func:`library_is_fresh`. The report names
the digest of the files it was computed over (``files.digest``), so a library
whose files changed without the report being regenerated is told apart by one
comparison. The check is about staleness only: that the report's rows are
*right* for those files is recomputed by CANOPY itself in the image build
(``python -m canopy_mcp.report … --check``).
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import stat
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NotRequired, TypedDict

__all__ = [
    "DEPLOYMENT_IGNORED_NAMES",
    "HOME_SCREEN_KEY",
    "IGNORED_NAMES_ENV",
    "IMAGE_SUFFIXES",
    "KNOWN_SCHEMAS",
    "MAX_NAVIGATION_BYTES",
    "MAX_NAVIGATION_DEPTH",
    "MAX_NAVIGATION_ENTRIES",
    "MAX_NAVIGATION_LABEL",
    "MAX_NAVIGATION_SCREEN",
    "NAVIGATION_NAME",
    "REPORT_JSON",
    "REPORT_MD",
    "REPORT_NAMES",
    "SCREENS_DIRNAME",
    "SCREEN_SUFFIX",
    "Entry",
    "Freshness",
    "HomeScreen",
    "Judged",
    "Refusal",
    "chmod_command",
    "find_home_screen",
    "is_ignored_name",
    "judge",
    "library_digest",
    "library_is_fresh",
    "library_list",
    "navigation_faults",
    "refusal_text",
    "regeneration_command",
    "scan",
    "screen_ids",
]

SCREENS_DIRNAME = "screens"
SCREEN_SUFFIX = ".screen.json"
#: The suffixes of an image the viewer serves, compared on the lower-cased name.
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".bmp", ".ico"})
REPORT_JSON = "LIBRARY-REPORT.json"
REPORT_MD = "LIBRARY-REPORT.md"
#: The report files a library root holds beside ``screens/``.
REPORT_NAMES = (REPORT_JSON, REPORT_MD)
#: The one other file a library root may hold: the library's own ordering of its
#: screens for the viewer's sidebar. Optional; a library without it is drawn as
#: its folder tree.
NAVIGATION_NAME = "navigation.json"
#: Every file a library root may hold beside ``screens/``, each as a regular file.
_ROOT_FILE_NAMES = (*REPORT_NAMES, NAVIGATION_NAME)
#: The limits CANOPY states for ``navigation.json``; a file past one is ignored by
#: the server, so it is refused here.
MAX_NAVIGATION_BYTES = 256 * 2**10
MAX_NAVIGATION_DEPTH = 8
MAX_NAVIGATION_ENTRIES = 2000
MAX_NAVIGATION_LABEL = 120
MAX_NAVIGATION_SCREEN = 512
#: The keys an entry of ``navigation.json`` may carry.
_NAVIGATION_KEYS = frozenset({"screen", "label", "children"})
#: The one name that is skipped: neither listed nor refused.
SKIPPED_NAME = ".DS_Store"
#: The ``generator.schema`` values this release can tell stale from fresh.
KNOWN_SCHEMAS = frozenset({"canopy-library-report/1"})

#: The names an OSPREY deployment's image build context and .gitignore prune, beyond
#: CANOPY's own. CANOPY reads them as ``CANOPY_LIBRARY_IGNORED_NAMES``, casefolded here.
DEPLOYMENT_IGNORED_NAMES = ("claude.local.md", "dockerfile", "prefetched-models", "var")
#: The variable CANOPY reads them from, whitespace-separated.
IGNORED_NAMES_ENV = "CANOPY_LIBRARY_IGNORED_NAMES"
# CANOPY's own ignored names, in casefolded form. A leading "." is the third rule;
# see is_ignored_name.
_IGNORED_NAMES = frozenset({"__pycache__", ".git", ".venv", ".pytest_cache"})
_IGNORED_ENDINGS = (".swp", ".swo", ".pyc", ".pyo", ".pyd", ".log", ".pid")

_NAME_CATEGORIES = frozenset({"Cc", "Cf", "Cs"})
_REASONS = {
    "outside_screens": (
        "a library root holds only screens/, the LIBRARY-REPORT files and navigation.json; "
        "delete it"
    ),
    "symlink": "a symlink; delete it, or replace it with the file it points at",
    "not_regular": "not a regular file; delete it",
    "bad_name": "its name holds a control or format character; delete or rename it",
    "leftover_tmp": "a leftover of an interrupted write; delete it",
    "dotfile": "a dotfile; delete it",
    "ignored_name": "a name the image build leaves out; delete or rename it",
    "suffix": "neither a *.screen.json file nor an image; delete it",
    "case_collision": "differs from another entry only in case; delete or rename one of them",
}


class Entry(TypedDict):
    """One directory entry as :func:`scan` saw it; ``path`` is relative to the root, with ``/``."""

    path: str
    kind: str
    mode: int
    sha256: NotRequired[str]


class Refusal(TypedDict):
    """One entry that does not belong in a library, and why (a code of the module text's list)."""

    path: str
    code: str


@dataclass(frozen=True)
class Judged:
    """What :func:`judge` makes of a root's entries.

    ``files`` holds ``(store path, sha256)`` per listed file in list order;
    ``refusals`` is sorted by path and code.
    """

    files: tuple[tuple[str, str], ...]
    refusals: tuple[Refusal, ...]

    @property
    def list_bytes(self) -> bytes:
        """The list: one line ``<sha256>  screens/<relpath>`` per file."""
        return "".join(f"{sha}  {path}\n" for path, sha in self.files).encode("utf-8")

    @property
    def digest(self) -> str:
        """The digest of :attr:`list_bytes`."""
        return library_digest(self.list_bytes)

    @property
    def paths(self) -> frozenset[str]:
        """The listed store paths, each exactly as spelled on disk."""
        return frozenset(path for path, _sha in self.files)


def library_digest(listing: bytes) -> str:
    """``sha256:`` + the hex sha256 of the list's bytes."""
    return f"sha256:{hashlib.sha256(listing).hexdigest()}"


def library_list(library: Path | str) -> bytes:
    """The list of the library at ``library``: what :func:`judge` lists of what :func:`scan` finds.

    A refused entry is not on it. A directory that does not exist has the empty
    list. Raises :class:`OSError` when an entry cannot be read.
    """
    try:
        entries = scan(library)
    except FileNotFoundError:
        return b""
    return judge(entries).list_bytes


def is_ignored_name(segment: str) -> bool:
    """Whether a file or directory called ``segment`` is left out of the image build context.

    True for a name starting with ``.``, for ``var``, ``__pycache__``,
    ``prefetched-models``, ``CLAUDE.local.md`` and ``Dockerfile``, and for a
    name ending in ``.swp``, ``.swo``, ``.pyc``, ``.pyo``, ``.pyd``, ``.log``
    or ``.pid``, each compared on ``segment.casefold()``. The image context is
    built from the project ``dockerignore`` with every pattern matched at any
    depth, so a name it prunes is pruned below ``screens/`` too.
    """
    folded = segment.casefold()
    return (
        folded.startswith(".")
        or folded in _IGNORED_NAMES
        or folded in DEPLOYMENT_IGNORED_NAMES
        or folded.endswith(_IGNORED_ENDINGS)
    )


def _printable(text: str) -> str:
    """``text`` as one line of ASCII: every other character as a backslash escape."""
    return text.encode("unicode_escape").decode("ascii")


def refusal_text(refusal: Refusal) -> str:
    """One printable line for a refusal: its code, its path (escaped to ASCII) and what to do."""
    return f"{refusal['code']}: {_printable(refusal['path'])}: {_REASONS[refusal['code']]}"


# --- scan -----------------------------------------------------------------------------


def _kind(mode: int) -> str:
    if stat.S_ISLNK(mode):
        return "symlink"
    if stat.S_ISDIR(mode):
        return "dir"
    return "file" if stat.S_ISREG(mode) else "other"


def scan(root: Path | str) -> list[Entry]:
    """Every entry of the library root ``root``, in a stable order (each directory sorted by name).

    The root's own entries are all reported; of its directories only
    ``screens/`` is descended into, so a stray directory beside it is one entry
    however much it holds. Symlinks are reported, never followed. ``mode`` is
    the permission bits. Raises :class:`OSError` when ``root`` or a file under
    it cannot be read.
    """
    entries: list[Entry] = []

    def visit(directory: Path, prefix: str) -> None:
        with os.scandir(directory) as found:
            items = sorted(found, key=lambda item: item.name)
        for item in items:
            status = item.stat(follow_symlinks=False)
            entry: Entry = {
                "path": f"{prefix}{item.name}",
                "kind": _kind(status.st_mode),
                "mode": stat.S_IMODE(status.st_mode),
            }
            if entry["kind"] == "file":
                entry["sha256"] = hashlib.sha256(Path(item.path).read_bytes()).hexdigest()
            entries.append(entry)
            if entry["kind"] == "dir" and (prefix or item.name == SCREENS_DIRNAME):
                visit(Path(item.path), f"{entry['path']}/")

    visit(Path(root), "")
    return entries


# --- judge ----------------------------------------------------------------------------


def _bad_name(segment: str) -> bool:
    return any(unicodedata.category(char) in _NAME_CATEGORIES for char in segment)


def _name_code(segment: str) -> str | None:
    """The refusal code the name ``segment`` earns on its own, or ``None``."""
    if _bad_name(segment):
        return "bad_name"
    if segment.startswith(".") and segment.endswith(".tmp"):
        return "leftover_tmp"
    if segment.startswith("."):
        return "dotfile"
    return "ignored_name" if is_ignored_name(segment) else None


def _is_listed_name(name: str) -> bool:
    return name.endswith(SCREEN_SUFFIX) or os.path.splitext(name.lower())[1] in IMAGE_SUFFIXES


def _sort_key(text: str) -> bytes:
    return text.encode("utf-8", "surrogatepass")


def _case_collisions(paths: Iterable[str]) -> set[str]:
    """The spellings among ``paths`` and their directory prefixes sharing a casefold with another."""
    names: set[str] = set()
    for path in paths:
        parts = path.split("/")
        names.update("/".join(parts[:end]) for end in range(2, len(parts) + 1))
    groups: dict[str, set[str]] = {}
    for name in names:
        groups.setdefault(name.casefold(), set()).add(name)
    return {name for spellings in groups.values() if len(spellings) > 1 for name in spellings}


def judge(entries: Iterable[Entry]) -> Judged:
    """The list, the digest and the refusals for a root holding ``entries`` (see the module text)."""
    files: dict[str, str] = {}
    refused: set[tuple[str, str]] = set()
    below: list[str] = []
    for entry in entries:
        path, kind = entry["path"], entry["kind"]
        parts = path.split("/")
        if parts[-1] == SKIPPED_NAME and kind == "file":
            continue
        if parts[0] != SCREENS_DIRNAME or len(parts) == 1:
            if path == SCREENS_DIRNAME:
                if kind != "dir":
                    refused.add((path, "symlink" if kind == "symlink" else "outside_screens"))
            elif path in _ROOT_FILE_NAMES and kind in ("file", "symlink"):
                if kind == "symlink":
                    refused.add((path, "symlink"))
            elif len(parts) == 1 and _name_code(path) == "leftover_tmp":
                refused.add((path, "leftover_tmp"))
            else:
                refused.add((parts[0], "outside_screens"))
            continue
        below.append(path)
        named = next(
            (
                ("/".join(parts[: at + 1]), code)
                for at in range(1, len(parts))
                if (code := _name_code(parts[at]))
            ),
            None,
        )
        if named is not None:
            refused.add(named)
        elif kind == "symlink":
            refused.add((path, "symlink"))
        elif kind == "file":
            if _is_listed_name(parts[-1]):
                files[path] = entry.get("sha256", "")
            else:
                refused.add((path, "suffix"))
        elif kind != "dir":
            refused.add((path, "not_regular"))
    refused.update((name, "case_collision") for name in _case_collisions(below))
    prefix = len(SCREENS_DIRNAME) + 1
    return Judged(
        files=tuple(sorted(files.items(), key=lambda item: _sort_key(item[0][prefix:]))),
        refusals=tuple(
            {"path": path, "code": code}
            for path, code in sorted(refused, key=lambda r: (_sort_key(r[0]), r[1]))
        ),
    )


# --- home screen ----------------------------------------------------------------------

#: The profile key that names the screen CANOPY opens first, by its id.
HOME_SCREEN_KEY = "web.panels.canopy.home_screen"
#: The prefix an id copied from a collection member carries, which a screen id does not.
_LIBRARY_PREFIX = "library/"


@dataclass(frozen=True)
class HomeScreen:
    """The answer of :func:`find_home_screen`.

    ``found`` holds when ``path`` is a listed screen. Otherwise ``problem`` says
    what is wrong, as a clause, and ``fix`` what to do, as a sentence.
    ``store_path`` is ``screens/<id>.screen.json`` and ``path`` the file it
    names under the root, or ``None`` when the id does not map to one.
    ``screens`` holds the ids of every screen the library lists, in list order,
    and ``root`` is the library looked in.
    """

    screen_id: Any
    store_path: str | None
    path: Path | None
    problem: str | None
    fix: str | None
    screens: tuple[str, ...]
    root: Path

    @property
    def found(self) -> bool:
        return self.problem is None

    @property
    def looked_in(self) -> Path:
        """Where a refusal points: the file the id names, else the library."""
        return self.path if self.path is not None else self.root


def screen_ids(store_paths: Iterable[str]) -> tuple[str, ...]:
    """The ids of the screens among ``store_paths``, in their order.

    An id is the path below ``screens/`` without the ``.screen.json`` suffix,
    the spelling :data:`HOME_SCREEN_KEY` takes; a path that is no screen has none.
    """
    prefix = f"{SCREENS_DIRNAME}/"
    return tuple(
        path[len(prefix) : -len(SCREEN_SUFFIX)]
        for path in store_paths
        if path.startswith(prefix) and path.endswith(SCREEN_SUFFIX)
    )


def _refusal_of(judged: Judged, store_path: str) -> Refusal | None:
    """The refusal covering ``store_path`` itself or a directory above it, if any."""
    parts = store_path.split("/")
    covering = {"/".join(parts[:end]) for end in range(1, len(parts) + 1)}
    return next((r for r in judged.refusals if r["path"] in covering), None)


def find_home_screen(library: Path | str, screen_id: Any) -> HomeScreen:
    """Whether the library at ``library`` holds the screen ``screen_id`` names.

    The id maps to ``screens/<id>.screen.json``. It is accepted only when that
    store path is on the library's list (so a symlink, a dotfile, an ignored
    name or a case collision is refused as the listing refuses it) and the file
    is a JSON object whose ``kind`` is ``screen``. An id spelled with the
    ``library/`` prefix of a collection member, or with the ``.screen.json``
    suffix, is refused with a fix naming the plain id. Never raises.
    """
    root = Path(library)
    try:
        judged: Judged | None = judge(scan(root))
        unreadable = None
    except FileNotFoundError:
        judged, unreadable = None, None
    except OSError as error:
        judged, unreadable = None, f"{error.strerror or error}"
    screens = screen_ids(path for path, _sha in judged.files) if judged is not None else ()

    def answer(problem: str | None, fix: str | None, store_path: str | None = None) -> HomeScreen:
        path = root / store_path if store_path is not None else None
        return HomeScreen(screen_id, store_path, path, problem, fix, screens, root)

    if not isinstance(screen_id, str) or not screen_id.strip():
        return answer(
            "that is not a screen id",
            "Set it to a screen's path under screens/ without the `.screen.json` suffix, "
            "or remove it so CANOPY opens its first screen.",
        )
    plain = screen_id.removeprefix(_LIBRARY_PREFIX).removesuffix(SCREEN_SUFFIX)
    if plain != screen_id:
        spelled = []
        if screen_id.startswith(_LIBRARY_PREFIX):
            spelled.append(f"the `{_LIBRARY_PREFIX}` prefix of a collection member")
        if screen_id.endswith(SCREEN_SUFFIX):
            spelled.append(f"the `{SCREEN_SUFFIX}` suffix of the file")
        return answer(
            f"a screen id is written without {' and without '.join(spelled)}",
            f"Write it as {plain!r}.",
            f"{SCREENS_DIRNAME}/{plain}{SCREEN_SUFFIX}",
        )
    segments = screen_id.split("/")
    if any(segment in ("", ".", "..") for segment in segments) or "\\" in screen_id:
        return answer(
            "a screen id is a relative path under screens/, with no leading `/`, "
            "no empty, `.` or `..` segment and no backslash",
            "Name the screen by its path under screens/, for example 'facility/overview'.",
        )
    store_path = f"{SCREENS_DIRNAME}/{screen_id}{SCREEN_SUFFIX}"
    add = (
        "Add the screen to the library, name one the library holds, "
        "or remove the key so CANOPY opens its first screen."
    )
    if unreadable is not None:
        return answer(f"the screen library cannot be read ({unreadable})", add, store_path)
    if judged is None:
        return answer("the project has no screen library", add, store_path)
    if store_path not in judged.paths:
        refusal = _refusal_of(judged, store_path)
        if refusal is not None:
            return answer(
                f"the library's listing refuses it ({refusal_text(refusal)})",
                "Make it a regular screen file the listing accepts, or name another screen.",
                store_path,
            )
        return answer("the library holds no such screen", add, store_path)
    try:
        document = json.loads((root / store_path).read_bytes())
    except (OSError, ValueError):
        document = None
    if not isinstance(document, dict) or document.get("kind") != "screen":
        kind = document.get("kind") if isinstance(document, dict) else None
        what = (
            f"its `kind` is {kind!r}" if isinstance(document, dict) else "it is not a JSON object"
        )
        return answer(
            f"that file is not a screen ({what})",
            "Name a file whose `kind` is 'screen'.",
            store_path,
        )
    return answer(None, None, store_path)


# --- navigation -----------------------------------------------------------------------

#: The prefix a screen of the library carries where the viewer names it.
_NAVIGATION_SCREEN_PREFIX = "library/"


class _NavigationFault(ValueError):
    """The first place a navigation breaks the shape or a limit."""


def _navigation_text(entry: dict[str, Any], key: str, longest: int, here: str) -> None:
    if key not in entry:
        return
    value = entry[key]
    if not isinstance(value, str) or not value.strip():
        raise _NavigationFault(f"{here}.{key}: expected non-empty text")
    if len(value) > longest:
        raise _NavigationFault(f"{here}.{key}: longer than {longest} characters")


def _navigation_screens(document: Any) -> list[tuple[str, str]]:
    """``(where, screen)`` for every ``screen`` of a well-formed navigation, in file order.

    :raises _NavigationFault: naming the first entry that breaks the shape or a limit.
    """
    named: list[tuple[str, str]] = []
    count = 0

    def walk(items: Any, where: str, depth: int) -> None:
        nonlocal count
        if not isinstance(items, list):
            raise _NavigationFault(f"{where}: expected a list of entries")
        if depth > MAX_NAVIGATION_DEPTH:
            raise _NavigationFault(f"{where}: nested deeper than {MAX_NAVIGATION_DEPTH} levels")
        for index, entry in enumerate(items):
            here = f"{where}[{index}]"
            count += 1
            if count > MAX_NAVIGATION_ENTRIES:
                raise _NavigationFault(f"more than {MAX_NAVIGATION_ENTRIES} entries")
            if not isinstance(entry, dict):
                raise _NavigationFault(f"{here}: expected an object")
            unknown = sorted(set(entry) - _NAVIGATION_KEYS)
            if unknown:
                raise _NavigationFault(f"{here}: unknown key {unknown[0]!r}")
            _navigation_text(entry, "screen", MAX_NAVIGATION_SCREEN, here)
            _navigation_text(entry, "label", MAX_NAVIGATION_LABEL, here)
            if "screen" in entry:
                named.append((f"{here}.screen", entry["screen"]))
            if "children" in entry:
                if "label" not in entry:
                    raise _NavigationFault(f"{here}: a group needs a label")
                walk(entry["children"], f"{here}.children", depth + 1)
            elif "screen" not in entry:
                raise _NavigationFault(f"{here}: needs a screen or children")

    walk(document, "navigation", 1)
    return named


def navigation_faults(document: Any, screens: Iterable[str]) -> list[str]:
    """What is wrong with the parsed ``navigation.json`` *document*; empty when nothing is.

    *screens* are the ids of the library's screens (:func:`screen_ids`). The
    shape and the limits are judged as CANOPY judges them, and the first fault
    is the answer: the server ignores the whole file for it. A well-formed
    document is then held to the library: every ``screen`` has to be
    ``library/<id>`` for an id among *screens*, and each one that is not is a
    fault of its own, since the viewer would leave that entry out. Each fault
    starts with where it is, as ``navigation[1].children[0]``.
    """
    try:
        named = _navigation_screens(document)
    except _NavigationFault as fault:
        return [str(fault)]
    held = frozenset(screens)
    faults = []
    for where, screen in named:
        if not screen.startswith(_NAVIGATION_SCREEN_PREFIX):
            faults.append(
                f"{where}: {screen!r} is not a screen of the library; "
                f"write it as {_NAVIGATION_SCREEN_PREFIX}<id>"
            )
        elif screen[len(_NAVIGATION_SCREEN_PREFIX) :] not in held:
            faults.append(f"{where}: the library holds no screen {screen!r}")
    return faults


def _navigation_refusals(root: Path, entries: Iterable[Entry], judged: Judged) -> list[str]:
    """One line per fault of the root's ``navigation.json``; empty without the file.

    A name the listing refuses (a symlink, a directory) is left to that refusal.
    """
    if not any(e["path"] == NAVIGATION_NAME and e["kind"] == "file" for e in entries):
        return []
    with (root / NAVIGATION_NAME).open("rb") as handle:
        data = handle.read(MAX_NAVIGATION_BYTES + 1)
    if len(data) > MAX_NAVIGATION_BYTES:
        faults = [f"larger than the {MAX_NAVIGATION_BYTES}-byte limit"]
    else:
        try:
            document = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, ValueError, RecursionError) as exc:
            faults = [f"not JSON ({exc})"]
        else:
            faults = navigation_faults(document, screen_ids(path for path, _sha in judged.files))
    return [_printable(f"navigation: {NAVIGATION_NAME}: {fault}") for fault in faults]


# --- freshness ------------------------------------------------------------------------


@dataclass(frozen=True)
class Freshness:
    """The answer of :func:`library_is_fresh`: true when fresh, and the lines to print.

    For a fresh library ``lines`` is one line saying what was found. For a
    refused one it is every refusal, each on its own ASCII line, followed by
    the command that repairs it.
    """

    fresh: bool
    lines: tuple[str, ...]

    def __bool__(self) -> bool:
        return self.fresh


def regeneration_command(library: Path | str) -> str:
    """The command that rewrites the report of ``library`` from its files."""
    return f"python -m canopy_mcp.report {shlex.quote(str(library))} --served-at library --write"


def chmod_command(library: Path | str) -> str:
    """The command that gives every entry of ``library`` the modes the image needs."""
    return f"chmod -R o+rX,o-w {shlex.quote(str(library))}"


def _mode_refusal(path: str, kind: str, mode: int) -> str | None:
    """What is wrong with the "other" bits of one file or directory, or ``None``."""
    wrong = []
    if not mode & stat.S_IROTH:
        wrong.append("not readable by others")
    if kind == "dir" and not mode & stat.S_IXOTH:
        wrong.append("not searchable by others")
    if mode & stat.S_IWOTH:
        wrong.append("writable by others")
    if not wrong:
        return None
    return f"mode: {_printable(path)}: {' and '.join(wrong)} (mode {mode:04o})"


def _mode_refusals(root_mode: int, entries: Iterable[Entry], judged: Judged) -> list[str]:
    """One line per entry the image would hold whose "other" bits are wrong.

    The image copies host modes as root-owned files, so only the "other" bits
    reach the runtime user: it must be able to read every file and enter every
    directory, and to write to none. An entry the listing refuses by name or
    kind is left to that refusal.
    """
    refused = {r["path"] for r in judged.refusals if r["code"] != "case_collision"}
    lines = [_mode_refusal(".", "dir", root_mode)]
    for entry in entries:
        path, kind = entry["path"], entry["kind"]
        parts = path.split("/")
        if kind not in ("file", "dir") or (kind == "file" and parts[-1] == SKIPPED_NAME):
            continue
        if any("/".join(parts[:end]) in refused for end in range(1, len(parts) + 1)):
            continue
        lines.append(_mode_refusal(path, kind, entry["mode"]))
    return [line for line in lines if line is not None]


def _report_files(report: dict[str, Any]) -> dict[str, str] | None:
    """``files.list`` of a report as ``{path: sha256}``, or ``None`` when it has no usable one."""
    files = report.get("files")
    rows = files.get("list") if isinstance(files, dict) else None
    if not isinstance(rows, list):
        return None
    listed: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            return None
        path, sha = row.get("path"), row.get("sha256")
        if not isinstance(path, str) or not isinstance(sha, str):
            return None
        listed[path] = sha
    return listed


def _differences(reported: dict[str, str] | None, judged: Judged) -> list[str]:
    """The paths that differ between a report's ``files.list`` and the library, one per line."""
    if reported is None:
        return ["the report has no files.list, so no comparison with the library is possible"]
    current = dict(judged.files)
    lines = []
    for path in sorted(set(current) | set(reported), key=_sort_key):
        if path not in reported:
            lines.append(f"added: {_printable(path)} (in the library, not in the report)")
        elif path not in current:
            lines.append(f"missing: {_printable(path)} (in the report, not in the library)")
        elif current[path] != reported[path]:
            lines.append(f"changed: {_printable(path)}")
    if not lines:
        lines.append("the report's files.list matches the library, but its files.digest does not")
    return lines


def _report_refusals(root: Path, entries: Iterable[Entry], judged: Judged) -> list[str]:
    """The lines that say the report does not cover the library's files; empty when it does."""
    kinds = {entry["path"]: entry["kind"] for entry in entries}
    if kinds.get(REPORT_JSON) != "file":
        if not judged.files:
            return []
        return [
            f"no report: the library holds {len(judged.files)} files and no {REPORT_JSON}, "
            "so no comparison is possible"
        ]
    try:
        report = json.loads((root / REPORT_JSON).read_bytes())
    except ValueError as exc:
        return [f"no report: {REPORT_JSON} is not JSON ({_printable(str(exc))})"]
    if not isinstance(report, dict):
        return [f"no report: {REPORT_JSON} is not a JSON object"]
    generator = report.get("generator")
    schema = generator.get("schema") if isinstance(generator, dict) else None
    if schema not in KNOWN_SCHEMAS:
        named = _printable(schema) if isinstance(schema, str) else "none"
        return [
            f"unknown schema: {REPORT_JSON} has generator.schema {named}; "
            f"this release reads {', '.join(sorted(KNOWN_SCHEMAS))}"
        ]
    files = report.get("files")
    digest = files.get("digest") if isinstance(files, dict) else None
    if not isinstance(digest, str):
        return [f"no digest: {REPORT_JSON} has no files.digest, so no comparison is possible"]
    if digest == judged.digest:
        return []
    return [
        f"digest mismatch: {REPORT_JSON} was written for {_printable(digest)}, "
        f"the library is {judged.digest}",
        *_differences(_report_files(report), judged),
    ]


def library_is_fresh(library: Path | str) -> Freshness:
    """Whether the report of the library at ``library`` still covers the files that are there.

    A library that does not exist is fresh, and so is one with no files and no
    report. Otherwise it is refused when:

    - it holds files but no ``LIBRARY-REPORT.json``, or that report has no
      ``files.digest``;
    - ``files.digest`` is not :func:`library_digest` of the current list. The
      refusal names every path that differs from the report's ``files.list``
      (added, changed, missing), or says that no comparison is possible;
    - the report's ``generator.schema`` is not one of :data:`KNOWN_SCHEMAS`;
    - the listing has a refusal (:func:`refusal_text`);
    - any ``.svg`` is present: markup a browser runs;
    - ``navigation.json`` is there and :func:`navigation_faults` finds a fault
      in it, or it is not JSON or is larger than CANOPY reads;
    - an entry lacks the other-read bit (other-execute for a directory) or
      carries the other-write bit.

    Every refusal is collected, not only the first. A refusal about the report
    ends with the command that regenerates it, a mode refusal with the
    ``chmod`` command, and a navigation refusal with what to do about the
    file. Raises :class:`OSError` when an entry cannot be read.
    """
    root = Path(library)
    try:
        root_status = os.lstat(root)
    except FileNotFoundError:
        return Freshness(True, ("no library",))
    if not stat.S_ISDIR(root_status.st_mode):
        kind = "a symlink" if stat.S_ISLNK(root_status.st_mode) else "not a directory"
        return Freshness(
            False, (f"not_a_library: {_printable(str(root))}: {kind}; replace it with a directory",)
        )
    entries = scan(root)

    judged = judge(entries)
    lines = [refusal_text(refusal) for refusal in judged.refusals]
    lines += [
        f"svg: {_printable(path)}: SVG is markup a browser runs; "
        "delete it or convert it to a raster image"
        for path, _sha in judged.files
        if path.lower().endswith(".svg")
    ]
    navigation = _navigation_refusals(root, entries, judged)
    if navigation:
        lines += [
            *navigation,
            f"correct {NAVIGATION_NAME}, or delete it and the sidebar shows the folder tree",
        ]
    modes = _mode_refusals(stat.S_IMODE(root_status.st_mode), entries, judged)
    if modes:
        lines += [*modes, f"fix the modes with: {chmod_command(root)}"]
    report = _report_refusals(root, entries, judged)
    if report:
        lines += [
            *report,
            "review the tree, then regenerate the report from a venv built from the staged "
            f"payload's wheels: {regeneration_command(root)}",
        ]
    if lines:
        return Freshness(False, tuple(lines))
    if not judged.files and not any(entry["path"] == REPORT_JSON for entry in entries):
        return Freshness(True, ("empty library, no report",))
    return Freshness(True, (f"report is current ({len(judged.files)} files)",))
