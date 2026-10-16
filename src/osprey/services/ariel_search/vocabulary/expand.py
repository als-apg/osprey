"""Deterministic query expansion against a loaded facility vocabulary.

:func:`expand_query` is pure dictionary matching over normalized text: no
stemming, no LLM, microseconds. It is **text-agnostic** — it expands whatever
string it is handed and knows nothing about keyword-search syntax. Callers that
want field filters (``author:ts``), pattern spans (``/SR01C___BPM[0-9]+/``) or
quoted phrases treated specially must strip or split the query themselves
before calling; the keyword module does exactly that, while the semantic and
hybrid modules pass the whole query because the whole query is what gets
embedded.

Matching rules (all pinned by tests):

* The text is split into tokens by
  :func:`~osprey.services.ariel_search.vocabulary.model.match_tokens`: the
  whitespace-and-hyphen split of
  :func:`~osprey.services.ariel_search.vocabulary.model.normalize`, lowercased,
  with edge characters stripped from both ends of each token and recorded as
  clause breaks.
* The scan runs left-to-right. At each position it tries n-gram widths from
  ``min(vocabulary.max_form_tokens, remaining)`` down to ``1`` — **longest form
  first** — and the first width that matches wins. A match consumes its tokens,
  so spans never overlap and a strict prefix form (``beam``) cannot fire inside
  a longer match (``beam position monitor``).
* A candidate span matches only when the clause breaks between its tokens are
  a pattern one of its vocabulary spellings carries
  (``Vocabulary.interior_patterns``): ``beam. position monitor`` does not match
  ``beam position monitor``, while a form written ``vert. orbit`` matches
  ``vert. orbit`` and not ``vert orbit``. Breaks before the first or after the
  last token never block a span, so ``(BPM),`` matches ``bpm``.
* Because matching happens over normalized tokens, an
  :class:`~osprey.services.ariel_search.search.base.ExpansionGroup`'s
  ``original`` is the matched span **as its normalized tokens**, space-joined:
  ``t/s`` stays ``t/s`` (interior ``/`` is kept) but ``Beam-Position`` becomes
  ``beam position``. ``flattened_text`` is the only value built from the
  caller's untouched input.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import NamedTuple, assert_never

from osprey.services.ariel_search.search.base import ExpansionGroup, QueryExpansion
from osprey.services.ariel_search.vocabulary.model import (
    ConceptKind,
    Token,
    Vocabulary,
    interior_pattern,
    match_tokens,
    token_key,
)

__all__ = [
    "EntryExpansion",
    "ExpansionGroup",
    "Insertion",
    "QueryExpansion",
    "Token",
    "expand_entry_text",
    "expand_query",
    "match_tokens",
]


class _Gates(NamedTuple):
    """The two canonical-to-form direction gates.

    Attributes:
        canonical_to_acronym: Gate for ``acronym`` concepts.
        canonical_to_shorthand: Gate for ``shorthand`` concepts. Neither gate
            applies to ``synonym`` concepts, which are always bidirectional.
    """

    canonical_to_acronym: bool
    canonical_to_shorthand: bool


@dataclass(frozen=True)
class _Span:
    """One accepted match of the scan.

    Attributes:
        start: Index of the span's first token.
        stop: Index just after the span's last token (exclusive).
        alternatives: The deduplicated terms the span expands to, never
            containing the span itself; possibly empty.
        concepts: The canonicals the span binds — every concept binding it as a
            form, then the span itself when it matched as a canonical — in file
            order, deduplicated.
    """

    start: int
    stop: int
    alternatives: tuple[str, ...]
    concepts: tuple[str, ...]


def expand_query(
    text: str,
    vocabulary: Vocabulary,
    *,
    canonical_to_acronym: bool,
    canonical_to_shorthand: bool,
) -> QueryExpansion:
    """Expand query text against a vocabulary.

    Directions (FR3):

    * **form to canonical** is always on. A form bound to more than one concept
      (an *ambiguous* form) yields ONE group whose ``alternatives`` are the
      canonicals of every concept binding it, in file order.
    * **canonical to forms** is on only when the gate for the concept's kind is
      on: ``acronym`` concepts follow ``canonical_to_acronym``, ``shorthand``
      concepts follow ``canonical_to_shorthand``. A ``synonym`` concept is
      ungated — its canonical always expands into its forms.
    * **form to sibling forms** fires for ``synonym`` concepts only, and is
      likewise ungated. Together with the two directions above this makes a
      ``synonym`` concept bidirectional: every member (canonical or form)
      reaches every other member.

    A span that is both a form of one concept and the canonical of another
    contributes alternatives from both directions. A span never appears in its
    own alternatives, and alternatives are deduplicated keeping first
    occurrence. A span whose alternatives are empty after that (possible when
    the only alternative *is* the span) still consumes its tokens but produces
    no group.

    Args:
        text: The text to expand, exactly as the caller wants it matched. Not
            parsed in any way — see the module docstring.
        vocabulary: The loaded vocabulary to match against.
        canonical_to_acronym: Enable canonical-to-form expansion for ``acronym``
            concepts. Does not affect ``synonym`` concepts.
        canonical_to_shorthand: Enable canonical-to-form expansion for
            ``shorthand`` concepts. Does not affect ``synonym`` concepts.

    Returns:
        A :class:`~osprey.services.ariel_search.search.base.QueryExpansion`.
        ``groups`` holds one group per matched span in text order, with
        ``original`` spelled as its match tokens' norms (edge punctuation
        stripped, so ``bpm?`` gives ``bpm``). ``flattened_text`` is
        the caller's original ``text`` followed by every new term once, in group
        order, skipping any term the query's match tokens already contain
        (``bpm,`` contains ``bpm``) — so it equals
        ``text`` exactly when nothing new was added.
    """
    tokens = match_tokens(text)
    if not tokens or vocabulary.max_form_tokens < 1:
        return QueryExpansion(groups=(), flattened_text=text)

    gates = _Gates(
        canonical_to_acronym=canonical_to_acronym,
        canonical_to_shorthand=canonical_to_shorthand,
    )
    groups = [
        ExpansionGroup(
            original=token_key(tokens[span.start : span.stop]),
            alternatives=span.alternatives,
        )
        for span in _scan(tokens, vocabulary, gates)
        if span.alternatives
    ]

    return QueryExpansion(
        groups=tuple(groups),
        flattened_text=_flatten(text, [token.norm for token in tokens], groups),
    )


def _scan(
    tokens: list[Token],
    vocabulary: Vocabulary,
    gates: _Gates,
    *,
    limit: int | None = None,
) -> Iterator[_Span]:
    """Scan ``tokens`` longest-form-first, yielding every accepted span.

    At each position the widths ``min(vocabulary.max_form_tokens, remaining)``
    down to ``1`` are tried; the first width whose span is eligible, carries an
    accepted interior pattern and resolves to a match wins and consumes its
    tokens. A position where no width wins is skipped.

    Args:
        tokens: The text's tokens, from :func:`match_tokens`.
        vocabulary: The vocabulary to match against.
        gates: The canonical-to-form direction gates.
        limit: Code-point offset into the original text. When set, a span is
            eligible only when its last token's ``end`` is at most ``limit``.

    Yields:
        The accepted spans in text order, never overlapping.
    """
    if vocabulary.max_form_tokens < 1:
        return
    index = 0
    while index < len(tokens):
        widest = min(vocabulary.max_form_tokens, len(tokens) - index)
        for width in range(widest, 0, -1):
            stop = index + width
            if limit is not None and tokens[stop - 1].end > limit:
                continue
            window = tokens[index:stop]
            key = token_key(window)
            if not _interior_accepted(key, window, vocabulary):
                continue
            span = _resolve(key, vocabulary, gates)
            if span is None:
                continue
            alternatives, concepts = span
            yield _Span(
                start=index,
                stop=stop,
                alternatives=tuple(alternatives),
                concepts=tuple(concepts),
            )
            index = stop
            break
        else:
            index += 1


def _interior_accepted(key: str, window: list[Token], vocabulary: Vocabulary) -> bool:
    """Return whether the clause breaks inside ``window`` are allowed for ``key``.

    The pattern is :func:`interior_pattern`, exactly as the loader computes a
    spelling's.

    Args:
        key: The window's lookup key, its norms space-joined.
        window: The candidate span's tokens.
        vocabulary: The vocabulary holding the accepted patterns.

    Returns:
        ``True`` when the pattern is one of ``key``'s spellings', or — for a key
        absent from ``vocabulary.interior_patterns`` — when it has no breaks.
    """
    pattern = interior_pattern(window)
    accepted = vocabulary.interior_patterns.get(key)
    if accepted is None:
        return not any(left or right for left, right in pattern)
    return pattern in accepted


def _resolve(
    span: str, vocabulary: Vocabulary, gates: _Gates
) -> tuple[list[str], list[str]] | None:
    """Resolve a candidate span to its alternatives and bound canonicals.

    Args:
        span: A candidate lookup key.
        vocabulary: The vocabulary to look the span up in.
        gates: The canonical-to-form direction gates.

    Returns:
        ``None`` when the span matches nothing, otherwise
        ``(alternatives, concepts)`` as described on :class:`_Span`.
    """
    alternatives = _alternatives_for(
        span,
        vocabulary,
        canonical_to_acronym=gates.canonical_to_acronym,
        canonical_to_shorthand=gates.canonical_to_shorthand,
    )
    if alternatives is None:
        return None
    concepts = [concept.canonical for concept in vocabulary.concepts_for_form(span)]
    canonical = vocabulary.concept_for_canonical(span)
    if canonical is not None and _direction_enabled(
        canonical.kind,
        canonical_to_acronym=gates.canonical_to_acronym,
        canonical_to_shorthand=gates.canonical_to_shorthand,
    ):
        concepts.append(canonical.canonical)
    return alternatives, _dedupe(concepts, exclude=set())


def _alternatives_for(
    span: str,
    vocabulary: Vocabulary,
    *,
    canonical_to_acronym: bool,
    canonical_to_shorthand: bool,
) -> list[str] | None:
    """Resolve one candidate span.

    Args:
        span: A normalized n-gram from the query.
        vocabulary: The vocabulary to look the span up in.
        canonical_to_acronym: Gate for ``acronym`` concepts.
        canonical_to_shorthand: Gate for ``shorthand`` concepts. Neither gate
            applies to ``synonym`` concepts, which are always bidirectional.

    Returns:
        ``None`` when the span matches nothing (the scan should try a narrower
        width), otherwise the deduplicated alternatives — possibly empty, which
        still counts as a match and consumes the span.
    """
    matched = False
    alternatives: list[str] = []

    bound = vocabulary.concepts_for_form(span)
    if bound:
        matched = True
        alternatives.extend(concept.canonical for concept in bound)
        # A synonym concept's members are interchangeable, so a matched form
        # also reaches its siblings. The span itself is dropped by `_dedupe`.
        for bound_concept in bound:
            if bound_concept.kind == "synonym":
                alternatives.extend(bound_concept.forms)

    concept = vocabulary.concept_for_canonical(span)
    if concept is not None and _direction_enabled(
        concept.kind,
        canonical_to_acronym=canonical_to_acronym,
        canonical_to_shorthand=canonical_to_shorthand,
    ):
        matched = True
        alternatives.extend(concept.forms)

    if not matched:
        return None
    return _dedupe(alternatives, exclude={span})


def _direction_enabled(
    kind: ConceptKind,
    *,
    canonical_to_acronym: bool,
    canonical_to_shorthand: bool,
) -> bool:
    """Return whether canonical-to-form expansion is enabled for ``kind``.

    ``synonym`` is ungated: every member of a synonym concept reaches every
    other member, so its canonical always expands into its forms whatever the
    two gates say.

    Args:
        kind: The concept kind, ``"acronym"``, ``"shorthand"`` or ``"synonym"``.
        canonical_to_acronym: Gate for ``acronym`` concepts.
        canonical_to_shorthand: Gate for ``shorthand`` concepts.

    Returns:
        ``True`` when a canonical of that kind may expand into its forms.
    """
    if kind == "synonym":
        return True
    if kind == "acronym":
        return canonical_to_acronym
    if kind == "shorthand":
        return canonical_to_shorthand
    assert_never(kind)


def _dedupe(values: list[str], *, exclude: set[str]) -> list[str]:
    """Deduplicate ``values`` keeping first occurrence, dropping ``exclude``.

    Args:
        values: Terms in the order they should be kept.
        exclude: Terms to drop entirely.

    Returns:
        The filtered, deduplicated list.
    """
    seen = set(exclude)
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _flatten(text: str, tokens: list[str], groups: list[ExpansionGroup]) -> str:
    """Build ``flattened_text`` from the original input and the matched groups.

    Args:
        text: The caller's untouched input — semantic and hybrid embed this, so
            it is never replaced by its normalized form.
        tokens: The norms of ``text``'s match tokens, edge punctuation
            stripped, used to detect terms the query already contains — so
            ``bpm,`` already contains ``bpm``.
        groups: The matched groups, in text order.

    Returns:
        ``text`` plus each new term once, space-joined, or ``text`` unchanged
        when every alternative was already present in the query.
    """
    new_terms: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for alternative in group.alternatives:
            if alternative in seen:
                continue
            seen.add(alternative)
            if _contains_tokens(tokens, alternative.split()):
                continue
            new_terms.append(alternative)

    if not new_terms:
        return text
    return f"{text} {' '.join(new_terms)}"


def _contains_tokens(tokens: list[str], needle: list[str]) -> bool:
    """Return whether ``needle`` occurs as a whole token run inside ``tokens``.

    Args:
        tokens: The normalized query tokens.
        needle: The normalized tokens of a candidate new term.

    Returns:
        ``True`` when the query already contains the term, so appending it
        would only repeat a query term.
    """
    if not needle:
        return True
    span = len(needle)
    return any(tokens[start : start + span] == needle for start in range(len(tokens) - span + 1))


@dataclass(frozen=True)
class Insertion:
    """One span's alternatives, inserted into an entry after the span's word.

    Attributes:
        anchor: Code-point offset in the original text after the
            whitespace-delimited word containing the span's last token, before
            that word's trailing edge characters.
        text: The span's alternatives, space-joined, without the single
            separating space the renderer places before it.
        concepts: The canonicals the span binds, in file order — several for an
            ambiguous form.
    """

    anchor: int
    text: str
    concepts: tuple[str, ...]


@dataclass(frozen=True)
class EntryExpansion:
    """The result of :func:`expand_entry_text`.

    Attributes:
        text: The rendered entry: the original text with one space plus each
            insertion's ``text`` placed at its ``anchor``, in span order. It is
            the input object itself when nothing was inserted.
        insertions: The insertions in span (text) order.
    """

    text: str
    insertions: tuple[Insertion, ...]


def expand_entry_text(
    text: str,
    vocabulary: Vocabulary,
    *,
    canonical_to_acronym: bool,
    canonical_to_shorthand: bool,
    limit: int | None = None,
    tokens: list[Token] | None = None,
) -> EntryExpansion:
    """Expand logbook-entry text in place against a vocabulary.

    The spans and their alternatives are exactly those of :func:`expand_query`
    — the same scan, gates and directions. Instead of appending terms at the
    end, each span's alternatives are inserted after the word holding its last
    token (``RF-trip`` becomes ``RF-trip radio frequency``); insertions at one
    word follow each other in span order.

    A span's insertion is dropped whole when the token run directly before or
    directly after it spells the canonical or a form of any concept the span
    binds, whatever the gates say: ``BPM beam position monitor`` and
    ``quad (quadrupole)`` gain nothing. Edge-only tokens are not tokens, so the
    comparison looks across them.

    Args:
        text: The entry text, matched as-is.
        vocabulary: The loaded vocabulary to match against.
        canonical_to_acronym: Enable canonical-to-form expansion for ``acronym``
            concepts.
        canonical_to_shorthand: Enable canonical-to-form expansion for
            ``shorthand`` concepts.
        limit: Code-point offset into ``text``. When set, only spans whose last
            token ends at or before it are expanded, and only tokens ending at
            or before it count as an after-side duplicate.
        tokens: ``match_tokens(text)``, when the caller already has it.

    Returns:
        An :class:`EntryExpansion`; its ``text`` is ``text`` itself when no
        insertion survives.
    """
    if tokens is None:
        tokens = match_tokens(text)
    if not tokens or vocabulary.max_form_tokens < 1:
        return EntryExpansion(text=text, insertions=())

    gates = _Gates(
        canonical_to_acronym=canonical_to_acronym,
        canonical_to_shorthand=canonical_to_shorthand,
    )
    norms = [token.norm for token in tokens]
    visible = len(tokens)
    if limit is not None:
        visible = sum(1 for token in tokens if token.end <= limit)

    insertions = [
        Insertion(
            anchor=tokens[span.stop - 1].word_end,
            text=" ".join(span.alternatives),
            concepts=span.concepts,
        )
        for span in _scan(tokens, vocabulary, gates, limit=limit)
        if span.alternatives and not _spelled_out_nearby(span, norms[:visible], vocabulary)
    ]
    if not insertions:
        return EntryExpansion(text=text, insertions=())

    pieces: list[str] = []
    position = 0
    for insertion in insertions:
        pieces.append(text[position : insertion.anchor])
        pieces.append(f" {insertion.text}")
        position = insertion.anchor
    pieces.append(text[position:])
    return EntryExpansion(text="".join(pieces), insertions=tuple(insertions))


def _spelled_out_nearby(span: _Span, norms: list[str], vocabulary: Vocabulary) -> bool:
    """Return whether a member of the span's concepts sits right next to it.

    Args:
        span: An accepted span.
        norms: The norms of the tokens that may serve as neighbours; tokens past
            a cut are already left out.
        vocabulary: The vocabulary holding the span's concepts.

    Returns:
        ``True`` when the run directly before ``span.start`` or directly after
        ``span.stop`` equals a bound concept's canonical or one of its forms.
    """
    for canonical in span.concepts:
        concept = vocabulary.canonical_to_concept.get(canonical)
        members = [canonical, *(concept.forms if concept is not None else ())]
        for member in members:
            needle = member.split()
            width = len(needle)
            if span.start >= width and norms[span.start - width : span.start] == needle:
                return True
            if norms[span.stop : span.stop + width] == needle:
                return True
    return False
