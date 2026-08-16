"""Pure helper functions for the OKF knowledge panel.

Small, dependency-free helpers used by the panel backend (grouping, structure
overview, search snippets). Functions here stay self-contained so they can be
unit-tested without importing the rest of the package.

Ported from the ALS ``mcp_servers/okf_panel`` service during its promotion to a
native OSPREY builtin. The only intentional change from the ALS original is that
:func:`build_structure_markdown` renders a facility-neutral title (this panel now
serves any profile's bundle, not just ALS).
"""

from __future__ import annotations

import re

#: Leading ``"<n>: "`` line-number prefix qmd puts on every snippet line.
_QMD_LINE_NUMBER_RE = re.compile(r"^\s*\d+:\s?")

#: Diff-hunk header qmd emits as the first line of a snippet (``@@ -1,3 @@``).
#: Anchored: a ``@@...@@`` span mid-line is document content, and deleting it
#: would silently drop words out of an operator-facing excerpt. The leading
#: ``\s*`` is free because the line-number prefix is stripped before this runs.
_QMD_HUNK_RE = re.compile(r"^\s*@@[^@]*@@")


def format_qmd_snippet(snippet: str) -> str:
    """Flatten a qmd excerpt into one line of readable prose.

    qmd hands back a match-centered excerpt that is line-numbered and prefixed
    with a diff-hunk header — ``"1: @@ -1,3 @@ ...\\n2: # RF system"``. That
    markup is addressing information for a tool that will fetch more context;
    rendered verbatim in a reading panel it is noise an operator has to look
    past. The panel shows a single-line excerpt, so strip the markers and let
    the document's own words through.

    Args:
        snippet: The raw ``OKFSearchResult.snippet``. Empty on the substring
            fallback, which produces no excerpt.

    Returns:
        The excerpt as one whitespace-collapsed line, or "" when *snippet* is
        empty or holds nothing but markers (the caller then falls back to its
        own snippet/description chain, exactly as on the fallback path).
    """
    if not snippet:
        return ""

    parts = []
    for raw_line in snippet.splitlines():
        line = _QMD_LINE_NUMBER_RE.sub("", raw_line)
        line = _QMD_HUNK_RE.sub("", line)
        line = line.strip()
        # A hunk-header line collapses to a bare "..." continuation marker.
        if line and line.strip(". "):
            parts.append(line)

    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def make_snippet(body: str, query: str, before: int = 60, after: int = 120) -> str:
    """Build a short context snippet around the first match of ``query`` in ``body``.

    The snippet lets the search endpoint avoid shipping full document bodies.

    Args:
        body: The full document text to extract a snippet from.
        query: The search term to locate (case-insensitive).
        before: Number of characters of context to include before the match.
        after: Number of characters of context to include after the match.

    Returns:
        A whitespace-collapsed snippet with leading/trailing "…" truncation
        indicators where the snippet does not reach the start/end of ``body``.
        Returns "" when ``query`` is empty or not found (the caller falls back
        to the frontmatter description).
    """
    if not query:
        return ""

    pos = body.lower().find(query.lower())
    if pos == -1:
        return ""

    start = max(0, pos - before)
    end = pos + len(query) + after
    snippet = body[start:end]

    # Collapse all runs of whitespace/newlines to single spaces and strip.
    snippet = re.sub(r"\s+", " ", snippet).strip()

    if start > 0:
        snippet = "…" + snippet
    if end < len(body):
        snippet = snippet + "…"

    return snippet


def group_concepts(entries) -> dict:
    """Group concept entries by their top-level path segment for /api/concepts.

    Builds the payload consumed by the knowledge panel's concept browser. Each
    concept is placed in the group named by the FIRST segment of its
    ``concept_id`` (the path component before the first ``/``). The grouping is
    derived purely from ``concept_id`` and never relies on
    :attr:`ConceptEntry.type`, which is empty for index-derived entries.

    Args:
        entries: Iterable of ConceptEntry objects (e.g. from
            ``OKFBundle.list_concepts()``). Each must expose ``concept_id``,
            ``title``, and ``description`` attributes.

    Returns:
        A dict of the form::

            {"groups": [
                {"id": <segment>,
                 "label": <Title Cased segment>,
                 "concepts": [{"id": concept_id,
                               "title": title,
                               "description": description}, ...]},
                ...
            ]}

        Groups are sorted alphabetically by ``id``; concepts within each group
        are sorted case-insensitively by ``title``. Per-concept dicts contain
        only ``id``, ``title``, and ``description`` (never ``type``).
    """
    grouped: dict[str, list] = {}
    for entry in entries:
        segment = entry.concept_id.split("/")[0]
        grouped.setdefault(segment, []).append(entry)

    groups = []
    for segment in sorted(grouped):
        label = segment.replace("-", " ").replace("_", " ").title()
        concepts = [
            {
                "id": e.concept_id,
                "title": e.title,
                "description": e.description,
            }
            for e in sorted(grouped[segment], key=lambda e: e.title.lower())
        ]
        groups.append({"id": segment, "label": label, "concepts": concepts})

    return {"groups": groups}


def build_structure_markdown(grouped: dict) -> str:
    """Render a knowledge-base structure overview as a markdown document.

    Produces a single markdown document summarizing the whole bundle: a title,
    a totals line, and one ``##`` section per group listing every concept as a
    bullet linking to that concept.

    Args:
        grouped: The payload returned by :func:`group_concepts` — a dict of the
            form ``{"groups": [{"id", "label", "concepts": [{"id", "title",
            "description"}, ...]}, ...]}``.

    Returns:
        Markdown text. Each concept link target is the ROOT-ABSOLUTE path
        ``/<concept_id>.md`` so the panel's existing cross-link navigation hook
        intercepts the click and opens the concept in-panel. The
        " — <description>" suffix is omitted for concepts with empty
        descriptions. Groups and concepts appear in the order given.
    """
    groups = grouped.get("groups", [])
    total_concepts = sum(len(g.get("concepts", [])) for g in groups)
    group_count = len(groups)

    lines = ["# Facility Knowledge Base", ""]
    lines.append(f"_{total_concepts} concepts across {group_count} groups._")

    for group in groups:
        lines.append("")
        lines.append(f"## {group.get('label') or group.get('id') or ''}")
        for concept in group.get("concepts", []):
            cid = concept.get("id", "")
            title = concept.get("title") or cid
            description = (concept.get("description") or "").strip()
            line = f"- [{title}](/{cid}.md)"
            if description:
                line += f" — {description}"
            lines.append(line)

    return "\n".join(lines) + "\n"
