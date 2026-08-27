"""Guard a rendered graph prompt against hard-coded facility vocabulary.

Ruling of 2026-08-27, on issue #739: facility terminology has exactly one
source of truth. For the graph paradigm that source is the store's own
``(c:Class).altLabel``, captured into the *Graph at Hand* block when the corpus
is seeded — so no framework template may ship a facility's device kinds, class
labels or operator synonyms of its own.

Two rendered agents carry the graph terminology partial — the
facility-knowledge-graph subagent and the channel finder in its graph paradigm
— and each is asserted by the test module that owns its render. The token list
and the matching rule live here so the two guards cannot drift into disagreeing
about what the ruling forbids.

Lives in a ``_``-prefixed module rather than ``conftest.py`` because it is a
plain function and a constant, not a fixture. Same convention as
:mod:`tests.cli._lifecycle_build`.
"""

from __future__ import annotations

import re

#: Facility vocabulary that a framework prompt may not spell for itself: the
#: concrete class labels and the operator synonyms the partial used to ship.
HARDCODED_VOCABULARY_TOKENS: tuple[str, ...] = (
    "dcct",
    "bcm",
    "bpm",
    "quad",
    "Quadrupole",
    "BeamPositionMonitor",
    "HCorrector",
    "VCorrector",
    "Corrector",
)


def hardcoded_vocabulary_hits(text: str) -> list[str]:
    """Tokens from the ruling that survive in *text*.

    The CamelCase class names are matched case-sensitively, so English prose
    about "a quadrupole" is not read as the class label ``Quadrupole``. The
    lowercase operator tokens are matched case-insensitively but word-bounded,
    so "quadrupole" does not false-positive on the abbreviation "quad".
    """
    hits: list[str] = []
    for token in HARDCODED_VOCABULARY_TOKENS:
        if token[0].isupper():
            if token in text:
                hits.append(token)
        elif re.search(rf"\b{re.escape(token)}\b", text, re.IGNORECASE):
            hits.append(token)
    return hits
