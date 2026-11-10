"""An application's own module, reached by the fixture's relative imports."""

from __future__ import annotations


def label(value: object) -> str:
    """Spell *value* for a report."""
    return str(value)
