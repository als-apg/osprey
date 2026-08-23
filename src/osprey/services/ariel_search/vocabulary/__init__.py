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
    A frozen dataclass. ``kind`` is ``"acronym"`` or ``"shorthand"``. Every
    string on it is already normalized.

``Vocabulary``
    A frozen dataclass holding ``concepts: tuple[Concept, ...]`` (file order),
    ``forms_to_concepts: dict[str, list[Concept]]`` (normalized form to every
    concept binding it — a list because an ambiguous form is legal),
    ``canonical_to_concept: dict[str, Concept]``, and ``max_form_tokens: int``
    (the largest token count over every matchable string, forms *and*
    canonicals, so an n-gram scan knows how wide to go). Convenience accessors:
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
]
