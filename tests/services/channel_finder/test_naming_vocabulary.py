"""Invariant tests for the canonical channel-naming vocabulary.

``naming`` is a leaf data module: two generators consume the ``*_TOKENS`` and
``*_PHRASES`` dicts, one of them (``generate_from_spec``) looking tokens up
*strictly* so a token without a matching phrase (or vice versa) is a latent
bug. These tests pin the key-parity contract for each token/phrase pair.
"""

from __future__ import annotations

import pytest

from osprey.services.channel_finder import naming

_PAIRS = [
    ("RING_TOKENS", "RING_PHRASES"),
    ("FAMILY_TOKENS", "FAMILY_PHRASES"),
    ("FIELD_TOKENS", "FIELD_PHRASES"),
    ("SUBFIELD_TOKENS", "SUBFIELD_PHRASES"),
]


@pytest.mark.parametrize("tokens_name, phrases_name", _PAIRS)
def test_token_and_phrase_keys_match(tokens_name: str, phrases_name: str):
    tokens = getattr(naming, tokens_name)
    phrases = getattr(naming, phrases_name)
    assert tokens.keys() == phrases.keys(), (
        f"{tokens_name} and {phrases_name} must define identical keys so strict lookups never miss"
    )


@pytest.mark.parametrize("tokens_name, phrases_name", _PAIRS)
def test_no_empty_spellings(tokens_name: str, phrases_name: str):
    for mapping_name in (tokens_name, phrases_name):
        mapping = getattr(naming, mapping_name)
        assert all(v.strip() for v in mapping.values()), f"{mapping_name} has an empty spelling"


def test_tokens_are_pascal_case_no_spaces():
    # Tokens are channel-name components: they must not carry spaces.
    for mapping_name in ("RING_TOKENS", "FAMILY_TOKENS", "FIELD_TOKENS", "SUBFIELD_TOKENS"):
        mapping = getattr(naming, mapping_name)
        assert all(" " not in v for v in mapping.values()), f"{mapping_name} has a spaced token"


def test_golden_field_and_subfield_are_distinct_spellings():
    # The module docstring calls out that FIELD "GOLDEN" (top-level golden
    # orbit) is deliberately spelled differently from the SUBFIELD "GOLDEN".
    assert naming.FIELD_TOKENS["GOLDEN"] == "GoldenOrbit"
    assert naming.SUBFIELD_TOKENS["GOLDEN"] == "Golden"
