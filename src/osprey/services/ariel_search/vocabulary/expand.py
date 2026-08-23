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

* The text is normalized once with
  :func:`~osprey.services.ariel_search.vocabulary.model.normalize` and split on
  whitespace into tokens.
* The scan runs left-to-right. At each position it tries n-gram widths from
  ``min(vocabulary.max_form_tokens, remaining)`` down to ``1`` — **longest form
  first** — and the first width that matches wins. A match consumes its tokens,
  so spans never overlap and a strict prefix form (``beam``) cannot fire inside
  a longer match (``beam position monitor``).
* Because matching happens over normalized text, an
  :class:`~osprey.services.ariel_search.search.base.ExpansionGroup`'s
  ``original`` is the matched span **as it appears in the normalized text**:
  ``t/s`` stays ``t/s`` (``normalize`` keeps ``/``) but ``Beam-Position``
  becomes ``beam position``. ``flattened_text`` is the only value built from
  the caller's untouched input.
"""

from __future__ import annotations

from osprey.services.ariel_search.search.base import ExpansionGroup, QueryExpansion
from osprey.services.ariel_search.vocabulary.model import Vocabulary, normalize

__all__ = ["ExpansionGroup", "QueryExpansion", "expand_query"]


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
      concepts follow ``canonical_to_shorthand``.

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
            concepts.
        canonical_to_shorthand: Enable canonical-to-form expansion for
            ``shorthand`` concepts.

    Returns:
        A :class:`~osprey.services.ariel_search.search.base.QueryExpansion`.
        ``groups`` holds one group per matched span in text order, with
        ``original`` spelled as in the *normalized* text. ``flattened_text`` is
        the caller's original ``text`` followed by every new term once, in group
        order, skipping any term already present in the query — so it equals
        ``text`` exactly when nothing new was added.
    """
    normalized = normalize(text)
    tokens = normalized.split()
    if not tokens or vocabulary.max_form_tokens < 1:
        return QueryExpansion(groups=(), flattened_text=text)

    groups: list[ExpansionGroup] = []
    index = 0
    while index < len(tokens):
        widest = min(vocabulary.max_form_tokens, len(tokens) - index)
        for width in range(widest, 0, -1):
            span = " ".join(tokens[index : index + width])
            alternatives = _alternatives_for(
                span,
                vocabulary,
                canonical_to_acronym=canonical_to_acronym,
                canonical_to_shorthand=canonical_to_shorthand,
            )
            if alternatives is None:
                continue
            if alternatives:
                groups.append(ExpansionGroup(original=span, alternatives=tuple(alternatives)))
            index += width
            break
        else:
            index += 1

    return QueryExpansion(
        groups=tuple(groups),
        flattened_text=_flatten(text, tokens, groups),
    )


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
        canonical_to_shorthand: Gate for ``shorthand`` concepts.

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
    kind: str,
    *,
    canonical_to_acronym: bool,
    canonical_to_shorthand: bool,
) -> bool:
    """Return whether canonical-to-form expansion is enabled for ``kind``.

    Args:
        kind: The concept kind, ``"acronym"`` or ``"shorthand"``.
        canonical_to_acronym: Gate for ``acronym`` concepts.
        canonical_to_shorthand: Gate for ``shorthand`` concepts.

    Returns:
        ``True`` when a canonical of that kind may expand into its forms.
    """
    if kind == "acronym":
        return canonical_to_acronym
    return canonical_to_shorthand


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
        tokens: The normalized tokens of ``text``, used to detect terms the
            query already contains.
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
