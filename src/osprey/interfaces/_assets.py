"""How OSPREY's interfaces serve their own static assets: cached, compressed, never stale.

Three rules, one owner each:

* **A file is identified by its content.** Its ``ETag`` is a digest of its
  bytes, so a browser revalidates with ``If-None-Match`` and is answered
  ``304`` until the bytes change. The digest never depends on a timestamp: a
  reproducible build stamps every file with one fixed time, and an edit that
  keeps a file's size would then be invisible to a time-and-size tag.
* **A page names its assets by the fingerprint of the whole set.** A URL of
  the shape ``<mount>/_v/<fingerprint>/<path>`` (:meth:`AssetSet.url`) is
  answered ``immutable`` while the fingerprint is the current one, so a repeat
  visit asks for nothing. Any edit to any file changes the fingerprint, the
  page then names new URLs, and no browser can pair a new module with a cached
  old one. A stale fingerprint still resolves, to the current file, uncached.
* **Text is sent compressed.** A compressible file is gzipped once and the
  result kept; a client that does not accept gzip, or asks for a byte range,
  gets the file itself.

* **A page's style sheets are one request.** A named, ordered list of the
  set's sheets (:meth:`AssetSet.define_bundle`) is served concatenated at
  ``<mount>/_v/<fingerprint>/_bundle/<name>.css``, under the same rules as a
  file: fingerprinted, validated by content, gzipped once.

The plain URL (``<mount>/<path>``) keeps working for every page that does not
name the fingerprint: it is answered ``no-cache``, which stores the file and
revalidates it on every use.
"""

from __future__ import annotations

import gzip
import hashlib
import os
import posixpath
import re
import threading
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "CACHE_IMMUTABLE",
    "CACHE_REVALIDATE",
    "VERSION_SEGMENT",
    "BUNDLE_SEGMENT",
    "AssetBundle",
    "AssetEntry",
    "AssetSet",
    "etag_matches",
    "split_version",
]

#: A fingerprinted URL's answer: the browser never asks again.
CACHE_IMMUTABLE = "public, max-age=31536000, immutable"

#: A plain URL's answer: stored, and revalidated on every use.
CACHE_REVALIDATE = "no-cache"

#: The path segment that introduces a fingerprint, directly under a mount.
VERSION_SEGMENT = "_v"

#: The path segment, under a mount, that a bundle of style sheets is served at.
BUNDLE_SEGMENT = "_bundle"

_BUNDLE_PATH = re.compile(rf"^{BUNDLE_SEGMENT}/([A-Za-z0-9][A-Za-z0-9_-]*)\.css$")

#: A ``url(...)`` in a style sheet: a quoted string, or a bare token.
_CSS_URL = re.compile(
    r"""url\(\s*(?:"((?:[^"\\\n]|\\.)*)"|'((?:[^'\\\n]|\\.)*)'|([^\s"'()]+))\s*\)""",
    re.IGNORECASE,
)

#: A reference a concatenated sheet may not move: it names its own place.
_CSS_ABSOLUTE = re.compile(r"^(?:[a-z][a-z0-9+.-]*:|//|/|#)", re.IGNORECASE)

#: Rules that are only valid before every other rule of a sheet. A sheet that
#: carries one cannot keep its meaning in the middle of a concatenation.
_CSS_LEADING_RULE = re.compile(r"@(?:import|namespace)\b", re.IGNORECASE)

#: ``@charset`` is dropped: the bundle is UTF-8, as every source is read.
_CSS_CHARSET = re.compile(r"""@charset\s+(?:"[^"]*"|'[^']*')\s*;""", re.IGNORECASE)

_CSS_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)

#: How many hex digits of the set's digest a fingerprint carries.
_FINGERPRINT_LENGTH = 16

_VERSIONED_PATH = re.compile(rf"^{VERSION_SEGMENT}/([0-9a-f]{{{_FINGERPRINT_LENGTH}}})/(.+)$")

#: A static ``import``/``export … from`` statement, or a bare ``import '…'``.
#: Anchored to the start of a line, which is where the formatter puts them and
#: where prose in a comment block (``* import …``) is not.
_STATIC_IMPORT = re.compile(
    r"""^(?:import|export)\s[^'"`;]*?\sfrom\s*['"]([^'"]+)['"]|^import\s*['"]([^'"]+)['"]""",
    re.MULTILINE,
)

#: Suffixes worth compressing. Fonts are TrueType here, which gzip halves.
_COMPRESSIBLE = frozenset(
    {".css", ".js", ".mjs", ".html", ".json", ".map", ".svg", ".ttf", ".txt", ".mts"}
)

#: Below this size the gzip framing costs more than it saves.
_MIN_COMPRESS_BYTES = 1024

#: A compressed copy is kept only when it is at most this fraction of the file.
_MIN_SAVING = 0.9

#: How often, at most, the set is walked to notice an edited file.
_RESCAN_INTERVAL_S = 1.0


@dataclass(frozen=True, slots=True)
class AssetEntry:
    """What is known about one file's current bytes.

    Attributes:
        signature: The ``(mtime_ns, size)`` the entry was read at; a different
            stat result means the file was edited and the entry is rebuilt.
        digest: Hex SHA-256 of the file's bytes.
        compressed: The gzipped bytes, or ``None`` when the file is not worth
            compressing. Filled on first request, not at scan time.
    """

    signature: tuple[int, int]
    digest: str
    compressed: bytes | None = None
    compression_tried: bool = False

    @property
    def etag(self) -> str:
        """The validator of the file itself."""
        return f'"{self.digest[:32]}"'

    @property
    def compressed_etag(self) -> str:
        """The validator of the gzipped representation."""
        return f'"{self.digest[:32]}-gz"'


@dataclass(frozen=True, slots=True)
class AssetBundle:
    """Several style sheets of the set, concatenated in a stated order.

    Attributes:
        body: The concatenated sheets, UTF-8.
        digest: Hex SHA-256 of ``body``.
        compressed: ``body`` gzipped, or ``None`` when that saves too little.
    """

    body: bytes
    digest: str
    compressed: bytes | None

    @property
    def etag(self) -> str:
        """The validator of the bundle itself."""
        return f'"{self.digest[:32]}"'

    @property
    def compressed_etag(self) -> str:
        """The validator of the gzipped representation."""
        return f'"{self.digest[:32]}-gz"'


def split_version(path: str) -> tuple[str | None, str]:
    """Split a mount-relative path into its fingerprint and the file's path.

    Args:
        path: The path under a mount, without a leading slash.

    Returns:
        ``(fingerprint, file_path)`` for ``_v/<fingerprint>/<file_path>``, and
        ``(None, path)`` for a plain path.
    """
    match = _VERSIONED_PATH.match(path)
    if match is None:
        return None, path
    return match.group(1), match.group(2)


def etag_matches(if_none_match: str | None, *etags: str) -> bool:
    """Whether an ``If-None-Match`` header names one of *etags*.

    Weak comparison, as conditional GET requires: a ``W/`` prefix on the
    client's tag is ignored.
    """
    if not if_none_match:
        return False
    offered = {tag.strip().removeprefix("W/") for tag in if_none_match.split(",")}
    return "*" in offered or any(tag in offered for tag in etags)


def _signature(stat_result: os.stat_result) -> tuple[int, int]:
    return stat_result.st_mtime_ns, stat_result.st_size


class AssetSet:
    """The static files one interface app serves, and the fingerprint of them all.

    Built once per app from its mounts. Thread-safe: requests are served from
    the event loop and from Starlette's threadpool.

    Args:
        mounts: URL mount path (``/static``) to the directory served there.
            A request is attributed to the longest mount that prefixes it.
    """

    def __init__(self, mounts: Mapping[str, Path | str]) -> None:
        # Longest first, so /static/fonts wins over /static.
        self._mounts: tuple[tuple[str, Path], ...] = tuple(
            sorted(
                (
                    (mount.rstrip("/"), Path(directory).resolve())
                    for mount, directory in mounts.items()
                ),
                key=lambda item: len(item[0]),
                reverse=True,
            )
        )
        self._entries: dict[str, AssetEntry] = {}
        self._lock = threading.Lock()
        self._version: str | None = None
        self._scanned_at = 0.0
        self._graphs: dict[tuple[str, tuple[str, ...]], tuple[str, ...]] = {}
        self._bundles: dict[str, tuple[str, tuple[str, ...]]] = {}
        self._built: dict[str, tuple[str, AssetBundle | None]] = {}

    # -- the fingerprint ---------------------------------------------------

    def version(self) -> str:
        """The fingerprint of every file under every mount, as it is now.

        The set is walked at most once per :data:`_RESCAN_INTERVAL_S`; only a
        file whose stat changed is read again. In a deployment nothing changes
        and the walk is a few hundred ``stat`` calls; under a source checkout
        an edit shows up on the next page load.
        """
        with self._lock:
            now = time.monotonic()
            if self._version is None or now - self._scanned_at >= _RESCAN_INTERVAL_S:
                self._version = self._scan()
                self._scanned_at = time.monotonic()
            return self._version

    def _scan(self) -> str:
        digest = hashlib.sha256()
        seen: set[str] = set()
        for mount, directory in self._mounts:
            if not directory.is_dir():
                continue
            for root, dirs, files in os.walk(directory):
                dirs.sort()
                for name in sorted(files):
                    full_path = os.path.join(root, name)
                    try:
                        stat_result = os.stat(full_path)
                    except OSError:
                        continue
                    entry = self._entry_locked(full_path, stat_result)
                    if entry is None:
                        continue
                    seen.add(full_path)
                    relative = os.path.relpath(full_path, directory).replace(os.sep, "/")
                    digest.update(f"{mount}/{relative}\0{entry.digest}\n".encode())
        for stale in self._entries.keys() - seen:
            del self._entries[stale]
        return digest.hexdigest()[:_FINGERPRINT_LENGTH]

    # -- one file ----------------------------------------------------------

    def entry(self, full_path: str | os.PathLike[str], stat_result: os.stat_result) -> AssetEntry:
        """The entry for the file at *full_path*, rebuilt if the file changed.

        Raises:
            OSError: When the file cannot be read.
        """
        with self._lock:
            entry = self._entry_locked(os.fspath(full_path), stat_result)
        if entry is None:
            raise FileNotFoundError(os.fspath(full_path))
        return entry

    def _entry_locked(self, full_path: str, stat_result: os.stat_result) -> AssetEntry | None:
        signature = _signature(stat_result)
        entry = self._entries.get(full_path)
        if entry is not None and entry.signature == signature:
            return entry
        try:
            with open(full_path, "rb") as handle:
                content = handle.read()
        except OSError:
            return None
        entry = AssetEntry(signature=signature, digest=hashlib.sha256(content).hexdigest())
        self._entries[full_path] = entry
        return entry

    def compressed(
        self, full_path: str | os.PathLike[str], stat_result: os.stat_result
    ) -> AssetEntry:
        """The entry for *full_path* with its gzipped bytes decided.

        The first call for a file compresses it; later calls return the kept
        result. ``entry.compressed`` is ``None`` when the suffix is not
        compressible, the file is small, or gzip saves too little.

        Raises:
            OSError: When the file cannot be read.
        """
        path = os.fspath(full_path)
        entry = self.entry(path, stat_result)
        if entry.compression_tried:
            return entry
        body: bytes | None = None
        if (
            Path(path).suffix.lower() in _COMPRESSIBLE
            and stat_result.st_size >= _MIN_COMPRESS_BYTES
        ):
            with open(path, "rb") as handle:
                content = handle.read()
            # mtime=0 keeps the bytes a pure function of the content.
            packed = gzip.compress(content, compresslevel=9, mtime=0)
            if len(packed) <= len(content) * _MIN_SAVING:
                body = packed
        decided = AssetEntry(
            signature=entry.signature,
            digest=entry.digest,
            compressed=body,
            compression_tried=True,
        )
        with self._lock:
            # An edit between the read above and here leaves a newer entry in
            # place; keep that one.
            if self._entries.get(path) is entry:
                self._entries[path] = decided
        return decided

    # -- URLs --------------------------------------------------------------

    def url(self, path: str) -> str:
        """The fingerprinted form of a root-relative asset URL.

        Args:
            path: A root-relative URL under one of the mounts, such as
                ``/static/js/app.js``; a query string or fragment is kept.

        Returns:
            ``<mount>/_v/<fingerprint>/<rest>``. A path under no mount, or one
            that already names a fingerprint, is returned unchanged.
        """
        mount = self._mount_of(path)
        if mount is None:
            return path
        rest = path[len(mount) + 1 :]
        if rest.startswith(f"{VERSION_SEGMENT}/"):
            return path
        return f"{mount}/{VERSION_SEGMENT}/{self.version()}/{rest}"

    def url_of_file(self, path: str) -> str | None:
        """The fingerprinted URL of *path* when it names a file of the set, else ``None``.

        What a page served from somewhere else uses to name one of these files
        at the address this app's own pages name it by, so a browser that has
        the file keeps using it. Answered from the last scan: no file is read.

        Args:
            path: A root-relative URL path under one of the mounts, without a
                query string or fragment.
        """
        mount = self._mount_of(path)
        if mount is None:
            return None
        rest = path[len(mount) + 1 :]
        parts = rest.split("/")
        if not rest or rest.startswith(f"{VERSION_SEGMENT}/") or ".." in parts or "" in parts:
            return None
        version = self.version()
        full_path = os.path.join(dict(self._mounts)[mount], *parts)
        with self._lock:
            if full_path not in self._entries:
                return None
        return f"{mount}/{VERSION_SEGMENT}/{version}/{rest}"

    def root(self, mount: str) -> str:
        """The fingerprinted root of *mount*, with a trailing slash.

        What an import map points a mount's specifiers at.
        """
        return f"{mount.rstrip('/')}/{VERSION_SEGMENT}/{self.version()}/"

    # -- the module graph --------------------------------------------------

    def module_graph(self, entries: Sequence[str]) -> tuple[str, ...]:
        """Every module the *entries* import statically, as plain root-relative URLs.

        A browser discovers an unbundled module graph one level per round
        trip. A page that lists the graph as ``<link rel="modulepreload">``
        lets it fetch every level at once.

        The scan reads ``import … from '…'`` and ``export … from '…'``
        statements, follows relative specifiers and root-absolute ones under a
        mount, and ignores ``import()`` calls: what is loaded on demand is not
        part of the first load. A specifier that names no file is skipped, so
        the list never asks for something that would 404.

        Args:
            entries: Root-relative URLs of the page's module entry points.

        Returns:
            The entries and everything they reach, breadth first, each once.
            Cached per fingerprint.
        """
        key = (self.version(), tuple(entries))
        with self._lock:
            cached = self._graphs.get(key)
        if cached is not None:
            return cached
        ordered: list[str] = []
        seen: set[str] = set()
        queue = deque(entries)
        while queue:
            url = queue.popleft()
            if url in seen:
                continue
            seen.add(url)
            source_path = self._file_of(url)
            if source_path is None:
                continue
            try:
                source = source_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            ordered.append(url)
            for match in _STATIC_IMPORT.finditer(source):
                specifier = match.group(1) or match.group(2)
                if specifier.startswith("."):
                    queue.append(
                        posixpath.normpath(posixpath.join(posixpath.dirname(url), specifier))
                    )
                elif specifier.startswith("/"):
                    queue.append(specifier)
        graph = tuple(ordered)
        with self._lock:
            # One page's graphs per fingerprint; an older fingerprint's are dropped.
            self._graphs = {k: v for k, v in self._graphs.items() if k[0] == key[0]}
            self._graphs[key] = graph
        return graph

    # -- bundles of style sheets ---------------------------------------------

    def define_bundle(self, name: str, mount: str, sheets: Sequence[str]) -> None:
        """Name an ordered list of the set's style sheets to be served as one.

        Defined when the app is built, never per request, so every process
        that serves the app answers the same bundle URL.

        Args:
            name: The bundle's name, the stem of its file name.
            mount: The mount the bundle is served under.
            sheets: Plain root-relative URLs of the sheets, in cascade order.

        Raises:
            ValueError: *name* is not a plain file stem, or *mount* is not a
                mount of the set.
        """
        if not _BUNDLE_PATH.match(f"{BUNDLE_SEGMENT}/{name}.css"):
            raise ValueError(f"{name!r} is not a bundle name")
        mount = mount.rstrip("/")
        if mount not in dict(self._mounts):
            raise ValueError(f"{mount!r} is not a mount of this asset set")
        with self._lock:
            self._bundles[name] = (mount, tuple(sheets))
            self._built.pop(name, None)

    def bundle_sheets(self, name: str) -> tuple[str, ...]:
        """The sheets of the bundle *name*, in order; empty for an unknown name."""
        with self._lock:
            return self._bundles.get(name, ("", ()))[1]

    def bundle_url(self, name: str) -> str | None:
        """The fingerprinted URL of the bundle *name*, or ``None`` when it cannot be built.

        ``None`` tells the page to link the sheets one by one: an unknown
        name, a sheet that is missing, or one that carries a rule only valid
        at the top of a sheet.
        """
        if self.bundle(name) is None:
            return None
        with self._lock:
            mount = self._bundles[name][0]
        return f"{mount}/{VERSION_SEGMENT}/{self.version()}/{BUNDLE_SEGMENT}/{name}.css"

    def bundle_named(self, path: str) -> AssetBundle | None:
        """The bundle a mount-relative *path* (``_bundle/<name>.css``) names, or ``None``."""
        match = _BUNDLE_PATH.match(path)
        return self.bundle(match.group(1)) if match else None

    def bundle(self, name: str) -> AssetBundle | None:
        """The bundle *name* as it is now; built once per fingerprint.

        Each sheet is preceded by a comment naming it. A relative ``url(...)``
        is rewritten to reach the same file from the bundle's address, at that
        file's fingerprinted URL, so a font or image is the one request the
        rest of the page makes for it.
        """
        version = self.version()
        with self._lock:
            defined = self._bundles.get(name)
            built = self._built.get(name)
        if defined is None:
            return None
        if built is not None and built[0] == version:
            return built[1]
        bundle = self._build_bundle(defined[0], defined[1], version)
        with self._lock:
            self._built[name] = (version, bundle)
        return bundle

    def _build_bundle(self, mount: str, sheets: Sequence[str], version: str) -> AssetBundle | None:
        # Where the bundle is served, which every rewritten reference starts from.
        served_in = f"{mount}/{VERSION_SEGMENT}/{version}/{BUNDLE_SEGMENT}"
        parts: list[str] = []
        for sheet in sheets:
            source_path = self._file_of(sheet)
            if source_path is None:
                return None
            try:
                text = source_path.read_text(encoding="utf-8-sig")
            except (OSError, UnicodeDecodeError):
                return None
            if _CSS_LEADING_RULE.search(_CSS_COMMENT.sub("", text)):
                return None
            text = _CSS_CHARSET.sub("", text)

            def moved(match: re.Match[str], sheet: str = sheet) -> str:
                reference = match.group(1) or match.group(2) or match.group(3) or ""
                if not reference or _CSS_ABSOLUTE.match(reference):
                    return match.group(0)
                target, mark, rest = reference.partition("?")
                if not mark:
                    target, mark, rest = reference.partition("#")
                plain = posixpath.normpath(posixpath.join(posixpath.dirname(sheet), target))
                relative = posixpath.relpath(self._versioned(plain, version), served_in)
                return f'url("{relative}{mark}{rest}")'

            parts.append(f"/* {sheet} */\n{_CSS_URL.sub(moved, text).rstrip()}\n")
        body = "\n".join(parts).encode("utf-8")
        packed = gzip.compress(body, compresslevel=9, mtime=0)
        return AssetBundle(
            body=body,
            digest=hashlib.sha256(body).hexdigest(),
            compressed=packed if len(packed) <= len(body) * _MIN_SAVING else None,
        )

    def _versioned(self, path: str, version: str) -> str:
        """*path* under its mount's fingerprint segment; unchanged under no mount."""
        mount = self._mount_of(path)
        if mount is None:
            return path
        return f"{mount}/{VERSION_SEGMENT}/{version}/{path[len(mount) + 1 :]}"

    def _file_of(self, url: str) -> Path | None:
        """The file a plain root-relative URL names, or ``None``."""
        mount = self._mount_of(url)
        if mount is None:
            return None
        directory = dict(self._mounts)[mount]
        candidate = (directory / url[len(mount) + 1 :]).resolve()
        if not candidate.is_relative_to(directory) or not candidate.is_file():
            return None
        return candidate

    def _mount_of(self, path: str) -> str | None:
        for mount, _ in self._mounts:
            if path.startswith(f"{mount}/"):
                return mount
        return None

    def cache_control(self, fingerprint: str | None) -> str:
        """The ``Cache-Control`` for a request that named *fingerprint* (or none)."""
        if fingerprint is not None and fingerprint == self.version():
            return CACHE_IMMUTABLE
        return CACHE_REVALIDATE
