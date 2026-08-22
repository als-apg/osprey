"""Strict loader and validator for a facility ``vocabulary.yml``.

The loader never raises for a content problem and never partially applies a
broken file: it collects **every** error it can find and returns
``(None, errors, warnings)``, so ``osprey ariel vocab-check`` can print the
whole list in one pass instead of making the operator fix one line per run.
Warnings describe a vocabulary that loads but will not behave as its author
probably expects; they never fail the check.

Only :func:`yaml.safe_load` is used — a vocabulary file is facility data, never
a place from which to construct Python objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from osprey.services.ariel_search.vocabulary.model import (
    CONCEPT_KINDS,
    Concept,
    ConceptKind,
    Vocabulary,
    normalize,
)
from osprey.services.ariel_search.vocabulary.stopwords import _ENGLISH_FTS_STOPWORDS

#: The only key a vocabulary document may carry.
_ALLOWED_TOP_LEVEL_KEYS = frozenset({"concepts"})

#: The only keys a concept entry may carry.
_ALLOWED_CONCEPT_KEYS = frozenset({"canonical", "kind", "forms"})

_KINDS_TEXT = ", ".join(CONCEPT_KINDS)
_CONCEPT_KEYS_TEXT = ", ".join(sorted(_ALLOWED_CONCEPT_KEYS))

_STOPWORD_SUFFIX = (
    "consists only of English full-text-search stopwords — plainto_tsquery "
    "discards it, so expanding to it matches nothing"
)


def load_vocabulary(
    path: Path,
    *,
    canonical_to_acronym: bool = True,
    canonical_to_shorthand: bool = False,
) -> tuple[Vocabulary | None, list[str], list[str]]:
    """Load and validate a vocabulary file.

    Args:
        path: Path to the ``vocabulary.yml`` to read. Already resolved by the
            caller — the loader does no path resolution of its own.
        canonical_to_acronym: Whether canonical-to-form expansion is enabled for
            ``acronym`` concepts. Only affects which forms are checked against
            the stopword list: a form that never reaches ``plainto_tsquery``
            cannot be silently discarded by it.
        canonical_to_shorthand: The same gate for ``shorthand`` concepts.

    Returns:
        ``(vocabulary, errors, warnings)``. On any error the first element is
        ``None`` and ``errors`` is non-empty; otherwise the vocabulary is
        returned with an empty error list. ``warnings`` may be non-empty either
        way and never blocks loading.
    """
    document, read_errors = _read_document(path)
    if read_errors:
        return None, read_errors, []

    concepts, errors = _parse_document(document)
    warnings = _collect_warnings(concepts, canonical_to_acronym, canonical_to_shorthand)
    if errors:
        return None, errors, warnings
    return _build_vocabulary(concepts), [], warnings


def _read_document(path: Path) -> tuple[Any, list[str]]:
    """Read and YAML-parse the file, returning ``(document, errors)``."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None, [f"vocabulary file not found: {path}"]
    except (OSError, UnicodeDecodeError) as exc:
        return None, [f"vocabulary file could not be read: {path} ({exc})"]

    try:
        return yaml.safe_load(text), []
    except yaml.YAMLError as exc:
        detail = str(exc).replace("\n", " ")
        return None, [f"vocabulary file is not valid YAML: {path} ({detail})"]


def _parse_document(document: Any) -> tuple[list[Concept], list[str]]:
    """Validate the document shape and build every well-formed concept."""
    if not isinstance(document, dict):
        got = type(document).__name__
        return [], [f"vocabulary file must contain a YAML mapping with a 'concepts' key, got {got}"]

    errors: list[str] = []
    unknown = set(document) - _ALLOWED_TOP_LEVEL_KEYS
    if unknown:
        listed = ", ".join(sorted(str(key) for key in unknown))
        errors.append(f"unknown top-level key(s): {listed} (only 'concepts' is allowed)")

    if "concepts" not in document:
        errors.append("missing required top-level key 'concepts'")
        return [], errors

    raw_concepts = document["concepts"]
    if not isinstance(raw_concepts, list):
        errors.append(f"'concepts' must be a non-empty list, got {type(raw_concepts).__name__}")
        return [], errors
    if not raw_concepts:
        errors.append("'concepts' must be a non-empty list, got an empty list")
        return [], errors

    concepts: list[Concept] = []
    # Normalized canonical -> the index of the entry that claimed it first, so a
    # duplicate names both entries instead of just complaining about itself.
    claimed: dict[str, int] = {}
    for index, raw in enumerate(raw_concepts):
        concept, concept_errors = _parse_concept(index, raw, claimed)
        errors.extend(concept_errors)
        if concept is not None:
            concepts.append(concept)
    return concepts, errors


def _parse_concept(
    index: int, raw: Any, claimed: dict[str, int]
) -> tuple[Concept | None, list[str]]:
    """Validate one concept entry, collecting every problem it has."""
    label = f"concepts[{index}]"
    if not isinstance(raw, dict):
        return None, [f"{label}: must be a mapping, got {type(raw).__name__}"]

    raw_canonical = raw.get("canonical")
    if isinstance(raw_canonical, str) and raw_canonical.strip():
        label = f"{label} ('{raw_canonical}')"

    errors: list[str] = []
    unknown = set(raw) - _ALLOWED_CONCEPT_KEYS
    if unknown:
        listed = ", ".join(sorted(str(key) for key in unknown))
        errors.append(f"{label}: unknown key(s): {listed} (allowed: {_CONCEPT_KEYS_TEXT})")

    canonical = _validate_canonical(label, raw, claimed, errors)
    if canonical is not None:
        # Claimed even when a sibling key is malformed, so a later duplicate of
        # this canonical is still reported against the entry that claimed it.
        claimed[canonical] = index
    kind = _validate_kind(label, raw, errors)
    forms = _validate_forms(label, raw, canonical, errors)

    if errors or canonical is None or kind is None or forms is None:
        return None, errors
    return Concept(canonical=canonical, kind=kind, forms=forms), errors


def _validate_canonical(
    label: str, raw: dict[str, Any], claimed: dict[str, int], errors: list[str]
) -> str | None:
    """Return the normalized canonical, appending any error it has."""
    if "canonical" not in raw:
        errors.append(f"{label}: missing required key 'canonical'")
        return None
    value = raw["canonical"]
    if not isinstance(value, str) or not normalize(value):
        errors.append(f"{label}: 'canonical' must be a non-empty string")
        return None
    canonical = normalize(value)
    first = claimed.get(canonical)
    if first is not None:
        errors.append(
            f"{label}: duplicate canonical '{canonical}' after normalization "
            f"— already defined by concepts[{first}]"
        )
        return None
    return canonical


def _validate_kind(label: str, raw: dict[str, Any], errors: list[str]) -> ConceptKind | None:
    """Return the concept kind, appending any error it has."""
    if "kind" not in raw:
        errors.append(f"{label}: missing required key 'kind'")
        return None
    value = raw["kind"]
    for kind in CONCEPT_KINDS:
        if value == kind:
            return kind
    errors.append(f"{label}: unknown kind {value!r} (expected one of: {_KINDS_TEXT})")
    return None


def _validate_forms(
    label: str, raw: dict[str, Any], canonical: str | None, errors: list[str]
) -> tuple[str, ...] | None:
    """Return the normalized, de-duplicated forms, appending any errors."""
    if "forms" not in raw:
        errors.append(f"{label}: missing required key 'forms'")
        return None
    value = raw["forms"]
    if not isinstance(value, list):
        errors.append(f"{label}: 'forms' must be a non-empty list, got {type(value).__name__}")
        return None
    if not value:
        errors.append(f"{label}: 'forms' must be a non-empty list, got an empty list")
        return None

    forms: list[str] = []
    for position, item in enumerate(value):
        if not isinstance(item, str):
            errors.append(f"{label}: forms[{position}] must be a string, got {type(item).__name__}")
            continue
        form = normalize(item)
        if not form:
            errors.append(f"{label}: forms[{position}] must be a non-empty string")
            continue
        if canonical is not None and form == canonical:
            errors.append(
                f"{label}: forms[{position}] '{item}' is the same as its canonical "
                f"'{canonical}' after normalization"
            )
            continue
        # A form repeated inside one concept is harmless; keep the first.
        if form not in forms:
            forms.append(form)
    return tuple(forms)


def _collect_warnings(
    concepts: list[Concept], canonical_to_acronym: bool, canonical_to_shorthand: bool
) -> list[str]:
    """Collect every warning for a set of well-formed concepts."""
    warnings = _stopword_warnings(concepts, canonical_to_acronym, canonical_to_shorthand)
    warnings.extend(_ambiguous_form_warnings(concepts))
    warnings.extend(_prefix_collision_warnings(concepts))
    return warnings


def _stopword_warnings(
    concepts: list[Concept], canonical_to_acronym: bool, canonical_to_shorthand: bool
) -> list[str]:
    """Warn about strings reaching ``plainto_tsquery`` that it will discard.

    Scope: every canonical always (a matched form always expands to it), plus
    the forms whose direction gate is on — a form that is only ever matched
    against the query text never reaches the tsquery and so cannot be dropped
    by it.
    """
    enabled_by_kind = {"acronym": canonical_to_acronym, "shorthand": canonical_to_shorthand}
    warnings: list[str] = []
    for concept in concepts:
        if _is_all_stopwords(concept.canonical):
            warnings.append(f'canonical "{concept.canonical}" {_STOPWORD_SUFFIX}')
        if not enabled_by_kind[concept.kind]:
            continue
        for form in concept.forms:
            if _is_all_stopwords(form):
                warnings.append(
                    f'form "{form}" of concept "{concept.canonical}" {_STOPWORD_SUFFIX}'
                )
    return warnings


def _ambiguous_form_warnings(concepts: list[Concept]) -> list[str]:
    """Warn about each form bound by more than one concept.

    Ambiguity is legal: the form expands to every canonical it is bound to. The
    warning exists so the widening is visible at build time.
    """
    bindings = _form_bindings(concepts)
    warnings: list[str] = []
    for form, bound in bindings.items():
        if len(bound) < 2:
            continue
        canonicals = ", ".join(concept.canonical for concept in bound)
        target = "both" if len(bound) == 2 else "all of them"
        warnings.append(
            f'form "{form}" is bound to {len(bound)} concepts: {canonicals} '
            f'— queries containing "{form}" expand to {target}'
        )
    return warnings


def _prefix_collision_warnings(concepts: list[Concept]) -> list[str]:
    """Warn when one concept's form is a strict token prefix of another's.

    Matching is longest-form-first, so the longer form always wins where both
    could match and the shorter one silently never fires on that span.
    """
    bindings = _form_bindings(concepts)
    tokens = {form: tuple(form.split()) for form in bindings}
    warnings: list[str] = []
    for short, short_tokens in tokens.items():
        for long, long_tokens in tokens.items():
            if len(short_tokens) >= len(long_tokens):
                continue
            if long_tokens[: len(short_tokens)] != short_tokens:
                continue
            pair = _differing_concepts(bindings[short], bindings[long])
            if pair is None:
                continue
            shorter, longer = pair
            warnings.append(
                f'form "{short}" (concept "{shorter.canonical}") is a strict prefix of '
                f'form "{long}" (concept "{longer.canonical}") — longest-form-first '
                f'matching means "{long}" wins wherever both could match'
            )
    return warnings


def _differing_concepts(
    short_bindings: list[Concept], long_bindings: list[Concept]
) -> tuple[Concept, Concept] | None:
    """Return the first pair of bindings that belong to different concepts."""
    for shorter in short_bindings:
        for longer in long_bindings:
            if shorter.canonical != longer.canonical:
                return shorter, longer
    return None


def _form_bindings(concepts: list[Concept]) -> dict[str, list[Concept]]:
    """Map each normalized form to the concepts binding it, in file order."""
    bindings: dict[str, list[Concept]] = {}
    for concept in concepts:
        for form in concept.forms:
            bindings.setdefault(form, []).append(concept)
    return bindings


def _is_all_stopwords(text: str) -> bool:
    """Return True when every token of ``text`` is an English FTS stopword."""
    tokens = text.split()
    return bool(tokens) and all(token in _ENGLISH_FTS_STOPWORDS for token in tokens)


def _build_vocabulary(concepts: list[Concept]) -> Vocabulary:
    """Assemble the lookup tables for a validated concept list."""
    matchable = [text for concept in concepts for text in (concept.canonical, *concept.forms)]
    return Vocabulary(
        concepts=tuple(concepts),
        forms_to_concepts=_form_bindings(concepts),
        canonical_to_concept={concept.canonical: concept for concept in concepts},
        max_form_tokens=max(len(text.split()) for text in matchable),
    )


__all__ = ["load_vocabulary"]
