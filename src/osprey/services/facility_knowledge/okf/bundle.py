"""Read API over an OKF knowledge bundle directory.

A bundle is a directory tree of ``.md`` files whose structure follows the
Open Knowledge Format specification (OKF §§2–6).  This module exposes
three operations:

* :py:meth:`OKFBundle.list_concepts` — lightweight listing from ``index.md``
  without reading every document body (OKF §6, progressive disclosure).
* :py:meth:`OKFBundle.read_concept` — load a single concept by ID; broken
  cross-links in the body are tolerated (OKF §9).
* :py:meth:`OKFBundle.search` — ranked retrieval through the qmd sidecar when
  one is deployed and answering, and a case-insensitive substring scan when it
  is not.

Concept IDs follow OKF §2: the file path relative to the bundle root, with
the ``.md`` suffix removed (e.g. ``tables/users``).

Search has two backends and one return type. Which backend answered is visible
in :attr:`OKFSearchResult.score`: a number when qmd ranked the hits, ``None``
when the substring scan produced them. A deployment with no sidecar is a
supported configuration, not a degradation, so that fallback is silent; a
deployment that *has* a sidecar which is not answering gets one warning.

Example usage::

    bundle = OKFBundle(Path("/data/my-bundle"))
    for entry in bundle.list_concepts():
        print(entry.concept_id, entry.title)

    doc = bundle.read_concept("tables/users")
    results = bundle.search("synchrotron")

Wiring the ranked backend takes the two resolvers its inputs already have::

    from osprey.deployment.qmd_service import resolve_qmd_service_config
    from osprey.services.qmd import QMDClient

    bundle = OKFBundle(
        bundle_root,
        qmd_client=QMDClient(resolve_qmd_service_config(config)),
        search_settings=OKFSearchSettings.from_config(config),
    )

``QMDClient(None)`` — what an unconfigured deployment resolves to — never opens
a socket, so passing the client unconditionally is correct.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote

from osprey.services.facility_knowledge.okf.document import OKFDocument, OKFDocumentError
from osprey.services.qmd import QMDClient, QMDClientError

logger = logging.getLogger(__name__)

# Reserved filenames that are never concepts (OKF §3.1).
_RESERVED_NAMES: frozenset[str] = frozenset({"index.md", "log.md"})

# index.md link pattern: Markdown links of the form [title](path)
_INDEX_LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")

#: qmd collection the OKF bundle is indexed under. One sidecar serves several
#: corpora side by side, and the collection filter is what keeps them apart.
OKF_COLLECTION = "okf"

#: Maximum hits :meth:`OKFBundle.search` returns when the caller names no limit.
#: Matches qmd's own default so the two backends are bounded alike.
DEFAULT_SEARCH_LIMIT = 10

#: Config block the search knobs are read from.
_SEARCH_CONFIG_PREFIX = "facility_knowledge.search"


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


@dataclass(frozen=True)
class OKFSearchSettings:
    """Query knobs for ranked OKF search, read from ``facility_knowledge.search``.

    Args:
        rerank: Whether qmd reranks candidates with its LLM reranker. qmd's own
            default is ``True``; OKF's is ``False``, because reranking is the
            dominant latency term — roughly 4x — and its cost is near-constant
            in corpus size, so no bundle is small enough to outrun it. The OKF
            surfaces are interactive and hold a sub-second budget that
            reranking does not fit in. A deployment that values ranking quality
            over latency sets ``facility_knowledge.search.rerank: true``.
        candidate_limit: qmd's ``candidateLimit`` — how many candidates the
            reranker considers (qmd's default is 40). Lowering it trades recall
            for latency. ``None`` leaves qmd's default in place.
        collection: qmd collection the bundle is indexed under. Not a config
            key: it is fixed by the sidecar's rendered index, not chosen per
            deployment.
    """

    rerank: bool = False
    candidate_limit: int | None = None
    collection: str = OKF_COLLECTION

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> OKFSearchSettings:
        """Read the ``facility_knowledge.search`` block, defaults filled in.

        An absent block is the normal case and yields the defaults. A *present*
        key of the wrong type is refused rather than defaulted: silently
        ignoring ``rerank: "false"`` would leave the deployment on the slow path
        it explicitly asked to leave, which is far harder to diagnose than a
        refusal at startup.

        Args:
            config: A loaded project config mapping, or ``None``.

        Returns:
            The resolved settings.

        Raises:
            ValueError: If ``rerank`` is present but not a boolean, or
                ``candidate_limit`` is present but not a positive integer.
        """
        block: Any = (config or {}).get("facility_knowledge") or {}
        search: Any = block.get("search") if isinstance(block, Mapping) else None
        if not isinstance(search, Mapping):
            return cls()

        rerank = search.get("rerank", cls.rerank)
        if not isinstance(rerank, bool):
            raise ValueError(f"{_SEARCH_CONFIG_PREFIX}.rerank must be a boolean, got {rerank!r}")

        candidate_limit = search.get("candidate_limit")
        if candidate_limit is not None and (
            not isinstance(candidate_limit, int)
            or isinstance(candidate_limit, bool)
            or candidate_limit < 1
        ):
            raise ValueError(
                f"{_SEARCH_CONFIG_PREFIX}.candidate_limit must be a positive integer, "
                f"got {candidate_limit!r}"
            )

        return cls(rerank=rerank, candidate_limit=candidate_limit)


@dataclass(frozen=True)
class OKFSearchResult:
    """One hit from :meth:`OKFBundle.search`, from either backend.

    Args:
        concept_id: OKF §2 identifier of the matching concept.
        document: The parsed concept document.
        score: Relevance in 0-1 when qmd ranked this hit, ``None`` when it came
            from the substring fallback, which does not rank. The number is an
            **ordering signal, not a calibrated relevance probability**: with
            ``rerank=False`` qmd derives it from rank alone (1.0, 0.5, 0.33 …)
            and it carries nothing the result order does not already say. Do
            not threshold it against a fixed number and do not compare it
            across queries — render it, or ignore it.
        snippet: qmd's line-numbered, match-centered excerpt. Empty on the
            substring fallback, which produces no excerpt; a caller that shows
            snippets should fall back to the document's own description when
            this is empty.
    """

    concept_id: str
    document: OKFDocument
    score: float | None = None
    snippet: str = ""


@dataclass
class OKFBundle:
    """Read-only accessor for an OKF knowledge bundle directory.

    Args:
        root: Path to the bundle root directory.
        qmd_client: Client for the qmd sidecar that indexes this bundle, or
            ``None`` for substring-only search. An unconfigured client is
            equivalent to ``None`` and costs nothing, so callers with a project
            config can pass one unconditionally.
        search_settings: Query knobs for the ranked backend.

    Raises:
        OKFBundleError: If *root* does not exist.
    """

    root: Path
    qmd_client: QMDClient | None = None
    search_settings: OKFSearchSettings = field(default_factory=OKFSearchSettings)

    # One warning per outage, not one per query: an open panel polls search,
    # and a stopped sidecar must not be able to fill the log.
    _fallback_warned: bool = field(default=False, init=False, repr=False, compare=False)

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

    def search(self, query: str, *, limit: int = DEFAULT_SEARCH_LIMIT) -> list[OKFSearchResult]:
        """Search the bundle, ranked through qmd when the sidecar can answer.

        With a reachable sidecar the query is a hybrid keyword-plus-semantic
        one, so a paraphrase finds the concept that answers it. Without one it
        degrades to a case-insensitive substring match over each concept's
        frontmatter values (concatenated as strings) and body text, which only
        finds literal occurrences. :attr:`OKFSearchResult.score` tells the two
        apart.

        Falling back is silent when the deployment has no sidecar, and logs one
        warning — not one per query — when it has one that is not answering.

        ``index.md`` and ``log.md`` are never returned by either backend, and
        neither is a path that resolves outside the bundle root.

        Args:
            query: Query text. Natural language when qmd answers; a literal
                substring when the fallback does.
            limit: Maximum number of results.

        Returns:
            At most *limit* results: in descending relevance order from qmd, or
            in filesystem traversal order from the fallback.
        """
        client = self.qmd_client
        if client is not None and client.is_available():
            try:
                ranked = self._ranked_search(client, query, limit)
            except QMDClientError as exc:
                # Reached but could not answer — same user-visible outcome as a
                # stopped sidecar, so it takes the same fallback and warning.
                self._warn_degraded(f"query failed: {exc}")
            else:
                self._fallback_warned = False
                return ranked
        elif client is not None and client.is_configured:
            self._warn_degraded("sidecar is configured but not answering")

        return self._substring_search(query, limit)

    # ------------------------------------------------------------------
    # Search backends
    # ------------------------------------------------------------------

    def _ranked_search(self, client: QMDClient, query: str, limit: int) -> list[OKFSearchResult]:
        """Query the sidecar and map its hits back onto bundle concepts.

        A hit is dropped rather than raised on when it does not name a concept
        this bundle can serve — a reserved file, a path outside the root, a
        document deleted since it was indexed, or one that no longer parses.
        The index is a copy of the filesystem and is allowed to lag it.

        The daemon normalises the paths it reports, so mapping one back is not
        a string operation: see :func:`_qmd_file_to_concept_id`, which is given
        this bundle's root so it can recover an ID that normalisation changed.

        Args:
            client: An available qmd client.
            query: Query text.
            limit: Maximum number of results.

        Returns:
            Results in the order qmd ranked them.

        Raises:
            QMDClientError: If the sidecar could not answer the query.
        """
        settings = self.search_settings
        hits = client.query(
            settings.collection,
            query,
            limit=limit,
            rerank=settings.rerank,
            candidate_limit=settings.candidate_limit,
        )

        # One fold map for the whole result set, built at most once and only if
        # some hit actually needs it. Hoisted out of the loop because every hit
        # of an underscore-convention bundle needs recovery, so a per-hit build
        # would repeat one filesystem sweep per result — on the panel's
        # interactive path. Kept lazy because a hyphen-convention bundle needs
        # none of them and should pay nothing. Per SEARCH, never cached across
        # searches: see _normalised_concept_index for why that is the design.
        index_for_this_search = _lazy_normalised_index(self.root)

        results: list[OKFSearchResult] = []
        seen: set[str] = set()
        for hit in hits:
            concept_id = _qmd_file_to_concept_id(hit.file, self.root, index_for_this_search)
            if concept_id is None or concept_id in seen:
                continue
            try:
                path = self.resolve_concept_path(concept_id)
            except OKFBundleError:
                logger.warning("qmd returned a path outside the bundle root: %r", hit.file)
                continue
            try:
                doc = OKFDocument.parse(path.read_text(encoding="utf-8"))
            except (OKFDocumentError, OSError):
                continue
            seen.add(concept_id)
            results.append(
                OKFSearchResult(
                    concept_id=concept_id,
                    document=doc,
                    score=hit.score,
                    snippet=hit.snippet,
                )
            )
        return results

    def _substring_search(self, query: str, limit: int) -> list[OKFSearchResult]:
        """Scan every concept for a case-insensitive substring match.

        Args:
            query: Substring to search for.
            limit: Maximum number of results.

        Returns:
            Matching concepts in filesystem traversal order, scored ``None``
            because this backend does not rank.
        """
        needle = query.casefold()
        results: list[OKFSearchResult] = []

        for md in sorted(self.root.rglob("*.md")):
            if len(results) >= limit:
                break
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
                results.append(OKFSearchResult(concept_id=cid, document=doc))

        return results

    def _warn_degraded(self, reason: str) -> None:
        """Log the first fallback of an outage and stay quiet for the rest.

        Args:
            reason: What went wrong, appended to the warning.
        """
        if self._fallback_warned:
            return
        self._fallback_warned = True
        logger.warning(
            "OKF ranked search is unavailable (%s); falling back to substring search "
            "until the sidecar answers again",
            reason,
        )


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _path_to_concept_id(root: Path, path: Path) -> str:
    """Return the OKF §2 concept ID for *path* relative to *root*."""
    return path.relative_to(root).with_suffix("").as_posix()


def _lazy_normalised_index(root: Path) -> Callable[[], dict[str, list[str]]]:
    """Return a zero-argument builder that sweeps the bundle at most once.

    The laziness and the memoisation each pay for a different case, and one
    search can meet both:

    * a bundle whose concept IDs survive normalisation unchanged never calls
      the builder at all, so it pays nothing;
    * a bundle whose IDs do not needs recovery for *every* hit, and pays for
      exactly one sweep instead of one per result.

    Deliberately scoped to one call. Returning a module-level cache instead
    would make a concept drafted between two searches permanently unfindable —
    see :func:`_normalised_concept_index`.

    Args:
        root: Bundle root.

    Returns:
        A callable returning the index, computed on first use.
    """
    built: list[dict[str, list[str]]] = []

    def get() -> dict[str, list[str]]:
        if not built:
            built.append(_normalised_concept_index(root))
        return built[0]

    return get


def _qmd_file_to_concept_id(
    file: str,
    root: Path | None = None,
    index: Callable[[], dict[str, list[str]]] | None = None,
) -> str | None:
    """Map a qmd result path onto an OKF §2 concept ID.

    ``QMDSearchResult.file`` is already stripped of its collection prefix, so
    what arrives here is bundle-root-relative. It is percent-encoded and
    suffixed, and it may name a file that is not a concept at all.

    **The daemon does not echo the filename it indexed.** It normalises it —
    measured against qmd 2.5.3, every ``_`` in a reported path comes back as
    ``-``, so a concept stored as ``magnets/corrector_magnets.md`` is reported
    as ``magnets/corrector-magnets.md``. Taken at face value that names no file,
    the caller's read fails, and the hit is dropped with no error anywhere: the
    search simply returns fewer results, which is indistinguishable from a
    bundle that does not cover the topic. Passing *root* enables recovery
    (:func:`_recover_normalised_concept_id`); omitting it keeps the pure
    string mapping, which is all a caller without a bundle can do.

    Containment is *not* checked here — :meth:`OKFBundle.resolve_concept_path`
    owns that check for every path in this module, and the caller runs the ID
    through it.

    Args:
        file: A ``QMDSearchResult.file`` value.
        root: Bundle root, when the caller has one. Used only to recover an ID
            the daemon normalised; ``None`` skips recovery.
        index: A shared :func:`_lazy_normalised_index` builder for *root*.
            Callers mapping several hits from one search MUST pass one — every
            hit of an underscore-convention bundle needs recovery, so leaving
            this to be rebuilt per hit repeats the same filesystem sweep once
            per result. ``None`` builds a single-use one, which is right for a
            caller mapping one path.

    Returns:
        The concept ID, or ``None`` when the path names no concept — empty, a
        reserved OKF §3.1 filename, or a normalised name that no concept (or
        more than one) can be recovered from.
    """
    raw = unquote(file).strip().lstrip("/")
    if not raw:
        return None
    posix = PurePosixPath(raw)
    if not posix.name or posix.name in _RESERVED_NAMES:
        return None
    concept_id = posix.with_suffix("").as_posix() if posix.suffix else posix.as_posix()
    if not concept_id:
        return None
    if root is None or (root / f"{concept_id}.md").is_file():
        return concept_id
    if index is None:
        index = _lazy_normalised_index(root)
    return _recover_normalised_concept_id(index(), concept_id)


#: Everything qmd's path normalisation collapses, per path component. Folding
#: case as well as punctuation is deliberate: it costs nothing when the daemon
#: preserves case, and a bundle where two concepts differ only in case is
#: already ambiguous to the recovery below — which drops rather than guesses.
_NON_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _normalised_key(concept_id: str) -> str:
    """Fold a concept ID to what survives qmd's path normalisation.

    Per component, so the directory structure still distinguishes concepts:
    ``a/b_c`` and ``a_b/c`` fold to different keys rather than colliding.

    Args:
        concept_id: An OKF §2 concept ID.

    Returns:
        A key equal for any two IDs the daemon would report identically.
    """
    return "/".join(
        _NON_SLUG_RE.sub("-", part.casefold()).strip("-")
        for part in PurePosixPath(concept_id).parts
    )


def _normalised_concept_index(root: Path) -> dict[str, list[str]]:
    """Group every concept ID in the bundle by its normalisation key.

    One filesystem sweep, built **once per search and never cached across
    searches**. Both halves of that matter and they answer different questions:

    * *Why not cache it across searches?* A bundle gains concepts at runtime —
      that is exactly what ``draft_concept`` plus the ``.qmd-touch`` marker
      exist to do. A cache would need an invalidation signal, and the only
      honest one is a directory scan, which is the very thing the cache would
      have been avoiding. So it buys nothing and costs a freshly drafted
      concept being permanently unfindable.
    * *Why build it once per search rather than per hit?* Within a single
      search the bundle cannot change, and for an underscore-convention bundle
      **every** hit needs recovery — so a per-hit sweep repeats identical work
      once per result. Measured on host CPU over synthetic bundles, for a
      default ``limit=10`` search:

      ==========  ==================  ====================
      concepts    per hit (was)       per search (now)
      ==========  ==================  ====================
      100         14.5 ms             1.5 ms
      1000        122.0 ms            12.0 ms
      5000        609.3 ms            62.5 ms
      ==========  ==================  ====================

      Roughly 10x either way, and it lands on top of daemon latency against the
      OKF panel's sub-second interactive budget — which the 5000-concept figure
      alone would have eaten.

    Values are lists because two concept IDs can share a key — the daemon would
    report both identically — and the caller must be able to tell that apart
    from a unique match rather than pick one.

    Args:
        root: Bundle root.

    Returns:
        Normalisation key to the sorted concept IDs that fold onto it.
    """
    index: dict[str, list[str]] = {}
    for path in root.rglob("*.md"):
        if path.name in _RESERVED_NAMES:
            continue
        concept_id = _path_to_concept_id(root, path)
        index.setdefault(_normalised_key(concept_id), []).append(concept_id)
    for matches in index.values():
        matches.sort()
    return index


def _recover_normalised_concept_id(index: dict[str, list[str]], reported: str) -> str | None:
    """Find the concept whose ID the daemon would have reported as *reported*.

    Args:
        index: A :func:`_normalised_concept_index` for the bundle.
        reported: Concept ID derived from the daemon's reported path.

    Returns:
        The single concept ID that matches, or ``None`` when none does or more
        than one does.
    """
    matches = index.get(_normalised_key(reported), [])
    if len(matches) == 1:
        return matches[0]
    if matches:
        # Returning either would be a coin flip that presents as a correct
        # answer, so the hit is dropped — but the collision is a property of the
        # bundle the operator can fix, so it must be discoverable.
        logger.warning(
            "qmd reported %r, which %d concepts in this bundle could be: %s. "
            "Dropping the hit rather than guessing; rename one of them so their "
            "IDs differ by more than punctuation or case.",
            reported,
            len(matches),
            ", ".join(matches),
        )
    return None


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
