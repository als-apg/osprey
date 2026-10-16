"""Vocabulary data model and the shared normalization rule.

A vocabulary is a facility-authored list of *concepts*. Each concept has one
``canonical`` phrase (the words that actually appear in logbook prose), a
``kind`` (``acronym`` or ``shorthand``) and one or more ``forms`` (the shorthand
operators type). Query expansion is pure dictionary lookup over these strings.

Every key in :class:`Vocabulary`'s lookup tables, and every ``canonical`` and
``forms`` string on :class:`Concept`, is the space-joined norms of
:func:`match_tokens` over the raw spelling — :func:`normalize` with edge
punctuation stripped. Callers matching query text must tokenize it the same way
before looking anything up — that identity is what makes the matching
deterministic. The raw spellings are kept beside the keys for the clause-break
patterns in :attr:`Vocabulary.interior_patterns`.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

#: The three concept kinds. For ``acronym`` and ``shorthand``, ``kind`` selects
#: which direction gate (``canonical_to_acronym`` / ``canonical_to_shorthand``)
#: enables expanding a canonical *into* this concept's forms; form-to-canonical
#: is always enabled. Kind ``synonym`` is bidirectional and ungated: every
#: member reaches every other member, independent of the direction gates.
ConceptKind = Literal["acronym", "shorthand", "synonym"]

#: Kinds accepted by the loader, in the order they are named in error messages.
CONCEPT_KINDS: tuple[ConceptKind, ...] = ("acronym", "shorthand", "synonym")

_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Normalize text for vocabulary matching.

    The single rule applied identically to vocabulary entries at load time and
    to query text at expansion time:

    * lowercase
    * ``-`` becomes a space (so ``Beam-Position Monitor`` and
      ``beam position monitor`` are the same phrase)
    * ``/`` is **kept** (so facility shorthand like ``t/s`` and ``i/o`` stays a
      single token)
    * runs of whitespace collapse to one space, and the result is stripped

    No stemming and no punctuation stripping beyond the above: expansion is
    exact dictionary matching, and PostgreSQL's stemmer runs afterwards on the
    text this produces.

    Args:
        text: Raw text from a vocabulary file or a user query.

    Returns:
        The normalized string. Tokens are the whitespace-separated pieces of it.
    """
    return _WHITESPACE.sub(" ", text.lower().replace("-", " ")).strip()


#: Edge characters stripped from both ends of every token core by
#: :func:`match_tokens`. Interior occurrences (``a.b``, ``t/s``) are kept.
_EDGE_CHARS = ",.:;!?()[]\"'"

#: :data:`_EDGE_CHARS` rendered space-joined, for user-facing messages.
_EDGE_TEXT = " ".join(_EDGE_CHARS)

_WORD = re.compile(r"\S+")


@dataclass(frozen=True)
class Token:
    """One matchable token of a text, located in the original string.

    Attributes:
        norm: The lowercased token core with edge characters stripped. Never
            empty.
        start: Code-point offset in the original text of the core's first
            character.
        end: Code-point offset just after the core's last character (exclusive);
            stripped edge characters lie outside ``start:end``.
        opens: A clause break precedes this token — a leading edge character was
            stripped from it, or an edge-only token before it was dropped.
        closes: A clause break follows this token — a trailing edge character was
            stripped from it, or an edge-only token after it was dropped.
        word_end: Offset just after the whitespace-delimited word containing
            this token, before that word's trailing edge characters (both
            tokens of ``RF-trip,`` end their word after ``p``).
    """

    norm: str
    start: int
    end: int
    opens: bool
    closes: bool
    word_end: int


def match_tokens(text: str) -> list[Token]:
    """Split ``text`` into located, normalized tokens.

    Splits on whitespace and ``-`` exactly as :func:`normalize` does and
    lowercases each token, so ``" ".join(t.norm for t in match_tokens(s))``
    equals ``normalize(s)`` whenever ``s`` carries no edge characters. Edge
    characters (:data:`_EDGE_TEXT`) are stripped from both ends of each token
    and recorded as ``opens``/``closes``. A token that is edge characters only
    (a spaced ``(``, ``.``, ``...``) is dropped; it sets ``closes`` on the
    previous token and ``opens`` on the next, so the break still separates them.

    Args:
        text: Raw query or entry text.

    Returns:
        The tokens in text order, each with a non-empty ``norm``.
    """
    # Mutable rows [norm, start, end, opens, closes, word_end], frozen at the end.
    rows: list[list] = []
    pending_open = False
    for word in _WORD.finditer(text):
        word_rows: list[list] = []
        offset = word.start()
        for piece in word.group().split("-"):
            piece_start = offset
            offset += len(piece) + 1
            if not piece:
                continue
            core = piece.strip(_EDGE_CHARS)
            if not core:
                if rows:
                    rows[-1][4] = True
                pending_open = True
                continue
            lead = len(piece) - len(piece.lstrip(_EDGE_CHARS))
            start = piece_start + lead
            end = start + len(core)
            row = [
                core.lower(),
                start,
                end,
                pending_open or lead > 0,
                end < piece_start + len(piece),
                0,
            ]
            pending_open = False
            rows.append(row)
            word_rows.append(row)
        if word_rows:
            word_end = word_rows[-1][2]
            for row in word_rows:
                row[5] = word_end
    return [Token(*row) for row in rows]


def token_key(tokens: Sequence[Token]) -> str:
    """Return the lookup key of a token run: its norms, space-joined."""
    return " ".join(token.norm for token in tokens)


def interior_pattern(tokens: Sequence[Token]) -> tuple[tuple[bool, bool], ...]:
    """Return the clause breaks inside a token run.

    Args:
        tokens: Consecutive tokens from :func:`match_tokens`.

    Returns:
        One ``(left closes, right opens)`` pair per interior token boundary;
        ``()`` for a single token.
    """
    return tuple(
        (left.closes, right.opens) for left, right in zip(tokens, tokens[1:], strict=False)
    )


@dataclass(frozen=True)
class Concept:
    """One vocabulary concept.

    Attributes:
        canonical: The canonical phrase as a lookup key. This is what a matched form
            expands to, and what reaches ``plainto_tsquery``/the embedder.
        kind: ``acronym`` or ``shorthand`` — selects the direction gate that
            enables canonical-to-form expansion for this concept.
        forms: The shorthand spellings as lookup keys, in file order. Never empty,
            never contains the canonical, and never contains duplicates.
        raw_canonical: The canonical exactly as written in the vocabulary file,
            or ``""`` for a concept not built by the loader. Excluded from
            equality.
        raw_forms: Every accepted form exactly as written, in file order —
            spellings that share one normalized form each appear. Empty for a
            concept not built by the loader. Excluded from equality.
    """

    canonical: str
    kind: ConceptKind
    forms: tuple[str, ...]
    raw_canonical: str = field(default="", compare=False)
    raw_forms: tuple[str, ...] = field(default=(), compare=False)


@dataclass(frozen=True)
class Vocabulary:
    """A loaded, validated vocabulary with its matching lookup tables.

    Built only by :func:`osprey.services.ariel_search.vocabulary.load_vocabulary`
    — a ``Vocabulary`` that exists is a vocabulary that passed validation.

    Attributes:
        concepts: Every concept, in file order.
        forms_to_concepts: Normalized form to the concepts binding it, in file
            order. A form bound by more than one concept is legal (it is an
            *ambiguous form*: the loader warns and expansion targets every
            binding), which is why the value is a list.
        canonical_to_concept: Normalized canonical to its concept. Canonicals
            are unique after normalization — the loader rejects duplicates.
        max_form_tokens: The largest token count over **every matchable
            string** — all forms *and* all canonicals. An n-gram scan may stop
            at this width and be sure it cannot miss a match.
        interior_patterns: Lookup key to the clause-break patterns its raw
            spellings carry between tokens. A pattern holds one
            ``(left closes, right opens)`` pair per interior token boundary, as
            :func:`match_tokens` reports them, so ``vert. orbit`` contributes
            ``((True, False),)`` and ``beam position monitor`` contributes
            ``((False, False), (False, False))``; the set is the union over every
            raw spelling of every concept binding the key. A key absent from the
            table accepts exactly the no-break pattern — ``()`` for one token,
            ``((False, False),) * (n - 1)`` for ``n`` — so a vocabulary built by
            hand without the table still matches punctuation-free text.
    """

    concepts: tuple[Concept, ...]
    forms_to_concepts: dict[str, list[Concept]]
    canonical_to_concept: dict[str, Concept]
    max_form_tokens: int
    interior_patterns: dict[str, frozenset[tuple[tuple[bool, bool], ...]]] = field(
        default_factory=dict
    )

    @property
    def concept_count(self) -> int:
        """Number of concepts in the vocabulary."""
        return len(self.concepts)

    def concepts_for_form(self, normalized_form: str) -> list[Concept]:
        """Return every concept binding ``normalized_form``.

        Args:
            normalized_form: A form already passed through :func:`normalize`.

        Returns:
            The binding concepts in file order — empty when nothing binds the
            form, and longer than one for an ambiguous form.
        """
        return self.forms_to_concepts.get(normalized_form, [])

    def concept_for_canonical(self, normalized_canonical: str) -> Concept | None:
        """Return the concept whose canonical is ``normalized_canonical``.

        Args:
            normalized_canonical: A canonical already passed through
                :func:`normalize`.

        Returns:
            The concept, or ``None`` when no concept has that canonical.
        """
        return self.canonical_to_concept.get(normalized_canonical)


__all__ = [
    "CONCEPT_KINDS",
    "Concept",
    "ConceptKind",
    "Token",
    "Vocabulary",
    "interior_pattern",
    "match_tokens",
    "normalize",
    "token_key",
]
