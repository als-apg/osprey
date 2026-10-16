"""Facility vocabulary: model, strict loader, and query expansion.

A facility authors a ``vocabulary.yml`` listing the shorthand its operators
type ("t/s the BPM offset") beside the words its logbook prose actually uses
("troubleshoot", "beam position monitor"). ARIEL loads that file once at config
parse and expands queries against it deterministically — dictionary matching,
no LLM, microseconds.

Public surface of the model and loader half of this package:

``normalize(text: str) -> str``
    The one normalization rule, applied identically to vocabulary entries at
    load time and to query text at expansion time: lowercase, ``-`` to space,
    ``/`` kept, whitespace collapsed. Tokens are the whitespace-separated
    pieces of the result.

``Concept(canonical: str, kind: ConceptKind, forms: tuple[str, ...])``
    A frozen dataclass. ``kind`` is ``"acronym"``, ``"shorthand"`` or
    ``"synonym"``. The first two pick which direction gate
    (``canonical_to_acronym`` / ``canonical_to_shorthand``) lets a canonical
    expand into this concept's forms; form-to-canonical is always on.
    ``"synonym"`` is ungated and bidirectional: every member reaches every
    other member, canonical and forms alike, whatever the gates say. Use it for
    different phrasings of one thing ("beam loss" / "lost beam" / "beam dump"),
    never for a clipped word (that is ``"shorthand"``) or an initialism (that
    is ``"acronym"``). ``canonical`` and ``forms`` are lookup keys: the
    space-joined ``match_tokens`` norms of the raw spelling, so edge
    punctuation is stripped (``vert. orbit`` keys as ``vert orbit``). The raw
    spellings survive as ``raw_canonical: str`` and ``raw_forms: tuple[str, ...]``
    (file order, every accepted spelling), both excluded from equality and
    empty on a hand-built concept.

``Vocabulary``
    A frozen dataclass holding ``concepts: tuple[Concept, ...]`` (file order),
    ``forms_to_concepts: dict[str, list[Concept]]`` (normalized form to every
    concept binding it — a list because an ambiguous form is legal),
    ``canonical_to_concept: dict[str, Concept]``, and ``max_form_tokens: int``
    (the largest token count over every matchable string, forms *and*
    canonicals, so an n-gram scan knows how wide to go), and
    ``interior_patterns: dict[str, frozenset[tuple[tuple[bool, bool], ...]]]``
    — per key, the clause breaks its raw spellings carry, one
    ``(left closes, right opens)`` pair per interior token boundary, unioned
    over every raw spelling of every concept binding the key (``vert. orbit``
    gives ``((True, False),)``). A key absent from the table accepts exactly
    the no-break pattern, ``()`` for one token and ``((False, False),) * (n-1)``
    for ``n``, so a hand-built vocabulary without the table still matches
    punctuation-free text. Convenience accessors:
    ``concept_count`` (property), ``concepts_for_form(normalized)`` and
    ``concept_for_canonical(normalized)``.

``load_vocabulary(path, *, canonical_to_acronym=True, canonical_to_shorthand=False)``
    Returns ``(Vocabulary | None, errors, warnings)``. Never raises for a
    content problem: every error in the file is collected, and a non-empty
    error list means the first element is ``None``. Warnings (ambiguous forms,
    strict-prefix form collisions, stopword-valued canonicals and
    direction-enabled forms) never block loading.

``_ENGLISH_FTS_STOPWORDS``
    The 127 words PostgreSQL's ``english`` text-search configuration discards,
    vendored so validation needs no database.

And the expansion half:

``expand_query(text, vocabulary, *, canonical_to_acronym, canonical_to_shorthand)``
    Returns a ``QueryExpansion(groups, flattened_text)``. Matching is
    longest-form-first, left-to-right and non-overlapping over normalized text,
    so each group's ``original`` is the matched span *as normalized*. The
    function is text-agnostic: callers strip field filters, pattern spans and
    quotes themselves before calling. ``ExpansionGroup`` and ``QueryExpansion``
    are re-exported here for convenience; they are defined in
    :mod:`osprey.services.ariel_search.search.base`, which is where third-party
    search modules find them.

And the entry half:

``EntryExpander(vocabulary, canonical_to_acronym, canonical_to_shorthand)``
    A frozen dataclass binding a vocabulary to its direction gates for logbook
    entry text. ``expand(text, *, limit=None, tokens=None)`` returns the
    ``EntryExpansion`` of ``expand_entry_text``; calling the expander returns
    just the expanded text. ``fingerprint`` is the SHA-256 hex digest of the
    rule version, both gates and every concept's whitespace-collapsed raw
    spellings and kind in file order; ``key`` is ``e<rule>:<fingerprint[:16]>``.

``EXPANSION_RULE``
    The integer version of the entry-expansion rule, folded into every
    fingerprint so a rule change re-keys every stored embedding input.
"""

# --- model & loader ---
from osprey.services.ariel_search.vocabulary.loader import load_vocabulary
from osprey.services.ariel_search.vocabulary.model import (
    CONCEPT_KINDS,
    Concept,
    ConceptKind,
    Vocabulary,
    normalize,
)
from osprey.services.ariel_search.vocabulary.stopwords import _ENGLISH_FTS_STOPWORDS

# isort: split
# --- expansion ---
from osprey.services.ariel_search.vocabulary.expand import (
    ExpansionGroup,
    QueryExpansion,
    expand_query,
)

# isort: split
# --- entry ---
from osprey.services.ariel_search.vocabulary.entry import EXPANSION_RULE, EntryExpander

__all__ = [
    # --- model & loader ---
    "CONCEPT_KINDS",
    "Concept",
    "ConceptKind",
    "Vocabulary",
    "_ENGLISH_FTS_STOPWORDS",
    "load_vocabulary",
    "normalize",
    # --- expansion ---
    "ExpansionGroup",
    "QueryExpansion",
    "expand_query",
    # --- entry ---
    "EXPANSION_RULE",
    "EntryExpander",
]
