"""Vocabulary data model and the shared normalization rule.

A vocabulary is a facility-authored list of *concepts*. Each concept has one
``canonical`` phrase (the words that actually appear in logbook prose), a
``kind`` (``acronym`` or ``shorthand``) and one or more ``forms`` (the shorthand
operators type). Query expansion is pure dictionary lookup over these strings.

Every string stored on :class:`Concept` and used as a key in
:class:`Vocabulary`'s lookup tables is **already normalized** by
:func:`normalize`. Callers matching query text must normalize it the same way
before looking anything up — that identity is what makes the matching
deterministic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

#: The two concept kinds. ``kind`` selects which direction gate
#: (``canonical_to_acronym`` / ``canonical_to_shorthand``) enables expanding a
#: canonical *into* this concept's forms. Form-to-canonical is always enabled.
ConceptKind = Literal["acronym", "shorthand"]

#: Kinds accepted by the loader, in the order they are named in error messages.
CONCEPT_KINDS: tuple[ConceptKind, ...] = ("acronym", "shorthand")

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


@dataclass(frozen=True)
class Concept:
    """One vocabulary concept.

    Attributes:
        canonical: The canonical phrase, normalized. This is what a matched form
            expands to, and what reaches ``plainto_tsquery``/the embedder.
        kind: ``acronym`` or ``shorthand`` — selects the direction gate that
            enables canonical-to-form expansion for this concept.
        forms: The shorthand spellings, normalized, in file order. Never empty,
            never contains the canonical, and never contains duplicates.
    """

    canonical: str
    kind: ConceptKind
    forms: tuple[str, ...]


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
    """

    concepts: tuple[Concept, ...]
    forms_to_concepts: dict[str, list[Concept]]
    canonical_to_concept: dict[str, Concept]
    max_form_tokens: int

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


__all__ = ["CONCEPT_KINDS", "Concept", "ConceptKind", "Vocabulary", "normalize"]
