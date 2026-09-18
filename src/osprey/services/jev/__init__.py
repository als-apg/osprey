"""TypeSafe System One ("Jev") integration.

A small, dependency-light client for a decision endpoint that answers typed
questions about a JSON state. Osprey uses it as the opt-in re-ranking stage of
the ARIEL ``jev`` search module; nothing else in the framework depends on it,
and a deployment that never sets an API key never reaches it.
"""

from osprey.services.jev.client import (
    DEFAULT_API_KEY_ENV,
    DEFAULT_ENDPOINT,
    DEFAULT_MAX_IN_FLIGHT,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT_SECONDS,
    JevAnswer,
    JevClient,
    JevError,
    JevResult,
    JevSettings,
    JevUnavailableError,
    choice_question,
    noul_question,
    score_question,
)

__all__ = [
    "DEFAULT_API_KEY_ENV",
    "DEFAULT_ENDPOINT",
    "DEFAULT_MAX_IN_FLIGHT",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_MODEL",
    "DEFAULT_TIMEOUT_SECONDS",
    "JevAnswer",
    "JevClient",
    "JevError",
    "JevResult",
    "JevSettings",
    "JevUnavailableError",
    "choice_question",
    "noul_question",
    "score_question",
]
