"""Printed text, read the way an assertion about its words needs it."""

from __future__ import annotations


def flat(text: str) -> str:
    """``text`` with its whitespace collapsed, so console line wrapping cannot split a phrase."""
    return " ".join(text.split())
