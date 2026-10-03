"""Shared fakes for ARIEL repository doubles.

A plain module rather than conftest fixtures, so any test file can import it
without pulling in a fixture namespace.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

from osprey.services.ariel_search.database.repository import SchemaFacts


def attach_fake_fts(repo: Any, *, has_v2: bool, has_copy_state: bool) -> Any:
    """Give a repository double an awaitable ``schema_facts`` reporting both facts.

    Both facts are required keywords, so every caller states the schema it
    pretends to stand on. The double also gets a ``caption_matches`` matching
    nothing (see :func:`attach_fake_caption_matches`), because a keyword search
    with hits asks for it.

    Args:
        repo: The repository double, typically a ``MagicMock``.
        has_v2: The ``has_v2_fts`` fact.
        has_copy_state: The ``has_copy_state`` fact.

    Returns:
        `repo`, for chaining.
    """
    repo.schema_facts = AsyncMock(
        return_value=SchemaFacts(has_v2_fts=has_v2, has_copy_state=has_copy_state)
    )
    attach_fake_caption_matches(repo)
    return repo


def attach_fake_caption_matches(
    repo: Any, matches: dict[str, list[str]] | None = None, *, error: Exception | None = None
) -> AsyncMock:
    """Give a repository double an awaitable ``caption_matches``.

    Args:
        repo: The repository double, typically a ``MagicMock``.
        matches: The ``{entry_id: [attachment_id, ...]}`` it answers; ``{}``
            (no caption matched) when omitted.
        error: Raise this instead of answering.

    Returns:
        The ``caption_matches`` mock, for call assertions.
    """
    if error is not None:
        fake = AsyncMock(side_effect=error)
    else:
        fake = AsyncMock(return_value=dict(matches or {}))
    repo.caption_matches = fake
    return fake
