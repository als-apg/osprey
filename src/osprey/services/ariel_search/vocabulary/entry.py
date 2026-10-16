"""Entry-side expander: a vocabulary bound to its direction gates.

:class:`EntryExpander` is the single object the embedding pipeline holds to
expand logbook entry text. It pairs a loaded :class:`Vocabulary` with the two
direction gates and derives a content fingerprint, so every embedding written
with it can record exactly which vocabulary and rule produced its input.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import cached_property

from osprey.services.ariel_search.vocabulary.expand import (
    EntryExpansion,
    expand_entry_text,
)
from osprey.services.ariel_search.vocabulary.model import Token, Vocabulary

__all__ = ["CAUSE_PHRASES", "EXPANSION_RULE", "EntryExpander", "classify_build_keys"]

#: Version of the entry-expansion rule. It covers the output of
#: ``expand_entry_text`` and the expander branch of ``build_embedding_input``;
#: bump it whenever either produces different text for the same vocabulary and
#: gates, so stored embedding input keys stop matching and entries re-embed.
EXPANSION_RULE = 1


def _collapse(raw: str) -> str:
    """Collapse whitespace runs to one space and trim the ends; case is kept."""
    return " ".join(raw.split())


@dataclass(frozen=True)
class EntryExpander:
    """A vocabulary and its direction gates, ready to expand entry text.

    Attributes:
        vocabulary: The loaded facility vocabulary.
        canonical_to_acronym: Gate letting an acronym concept's canonical
            expand into its forms.
        canonical_to_shorthand: Gate letting a shorthand concept's canonical
            expand into its forms.
    """

    vocabulary: Vocabulary
    canonical_to_acronym: bool
    canonical_to_shorthand: bool

    @cached_property
    def fingerprint(self) -> str:
        """SHA-256 hex digest of everything that shapes expanded text.

        The digest covers ``EXPANSION_RULE``, both gates, and every concept in
        file order as ``[raw_canonical, kind, raw_forms]``. Raw spellings are
        hashed with whitespace runs collapsed but case and punctuation kept, so
        YAML comments and layout never change the fingerprint while any edit to
        a spelling does. A hand-built concept without raw spellings contributes
        its normalized ``canonical`` and ``forms`` instead.
        """
        concepts = [
            [
                _collapse(concept.raw_canonical or concept.canonical),
                concept.kind,
                [_collapse(form) for form in (concept.raw_forms or concept.forms)],
            ]
            for concept in self.vocabulary.concepts
        ]
        payload = {
            "rule": EXPANSION_RULE,
            "gates": {
                "canonical_to_acronym": bool(self.canonical_to_acronym),
                "canonical_to_shorthand": bool(self.canonical_to_shorthand),
            },
            "concepts": concepts,
        }
        canonical_json = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    @cached_property
    def key(self) -> str:
        """Short build key ``e<rule>:<first 16 hex digits of the fingerprint>``."""
        return f"e{EXPANSION_RULE}:{self.fingerprint[:16]}"

    def expand(
        self,
        text: str,
        *,
        limit: int | None = None,
        tokens: list[Token] | None = None,
    ) -> EntryExpansion:
        """Expand ``text`` with this vocabulary and gates.

        Args:
            text: The entry text to expand.
            limit: Passed through to ``expand_entry_text``.
            tokens: Pre-computed ``match_tokens(text)``, passed through.

        Returns:
            The :class:`EntryExpansion` from ``expand_entry_text``.
        """
        return expand_entry_text(
            text,
            self.vocabulary,
            canonical_to_acronym=self.canonical_to_acronym,
            canonical_to_shorthand=self.canonical_to_shorthand,
            limit=limit,
            tokens=tokens,
        )

    def __call__(self, text: str) -> str:
        """Return ``text`` with every vocabulary insertion applied."""
        return self.expand(text).text


#: Why a stored build key differs from the running one, keyed by stale code.
#: Codes appear in classification precedence order.
CAUSE_PHRASES: dict[str, str] = {
    "input_rule": "with a different input rule",
    "unexpanded": "without entry expansion",
    "expanded": "with entry expansion",
    "expansion_rule": "with a different expansion rule",
    "vocabulary": "with a different vocabulary",
    "unrecognized": "with an unrecognized build key",
}

_BUILD_KEY = re.compile(r"r([1-9][0-9]*)(?::e([1-9][0-9]*):([0-9a-f]+))?")


def _parse_build_key(key: str) -> tuple[int, int | None, str | None] | None:
    """Split a build key into ``(input_rule, expansion_rule, fingerprint)``.

    Args:
        key: ``r<rule>`` or ``r<rule>:e<rule>:<hex fingerprint>``.

    Returns:
        The three components, with the expansion parts ``None`` for an
        unexpanded key, or ``None`` when the key does not follow the grammar.
    """
    match = _BUILD_KEY.fullmatch(key)
    if match is None:
        return None
    input_rule, expansion_rule, fingerprint = match.groups()
    return (
        int(input_rule),
        int(expansion_rule) if expansion_rule is not None else None,
        fingerprint,
    )


def classify_build_keys(counts: dict[str, int], running_key: str) -> dict[str, int]:
    """Sum stored entry counts per reason their build key is stale.

    Each stored key different from ``running_key`` is classified by the first
    component that differs, in order: input rule (``input_rule``), presence of
    entry expansion (``unexpanded`` when only the running key expands,
    ``expanded`` when only the stored key does), expansion rule
    (``expansion_rule``), then vocabulary fingerprint (``vocabulary``). A
    stored key outside the grammar is ``unrecognized``. Keys equal to the
    running key are current and contribute nothing.

    Args:
        counts: Entry count per stored build key.
        running_key: The build key the current settings would write.

    Returns:
        Entry count per stale code; codes with no stale entries are absent.
        Every code is a key of :data:`CAUSE_PHRASES`.

    Raises:
        ValueError: If ``running_key`` does not follow the build-key grammar.
    """
    running = _parse_build_key(running_key)
    if running is None:
        raise ValueError(f"unrecognized running build key: {running_key!r}")
    run_rule, run_expansion, run_fingerprint = running
    stale: dict[str, int] = {}
    for key, count in counts.items():
        if key == running_key:
            continue
        stored = _parse_build_key(key)
        if stored is None:
            code = "unrecognized"
        else:
            rule, expansion, fingerprint = stored
            if rule != run_rule:
                code = "input_rule"
            elif expansion is None and run_expansion is not None:
                code = "unexpanded"
            elif expansion is not None and run_expansion is None:
                code = "expanded"
            elif expansion != run_expansion:
                code = "expansion_rule"
            elif fingerprint != run_fingerprint:
                code = "vocabulary"
            else:
                continue
        stale[code] = stale.get(code, 0) + count
    return stale
