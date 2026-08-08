"""Domain errors shared across the auth sidecar's modules."""

from __future__ import annotations

__all__ = ["InvalidSessionError"]


class InvalidSessionError(Exception):
    """A session cookie could not be trusted and must be treated as absent.

    Raised for tampering, a malformed or unknown-version payload, and cookies
    older than the permitted age. Callers do not need to distinguish the causes:
    every one of them means "no session", and the message says which it was for
    logs.
    """
