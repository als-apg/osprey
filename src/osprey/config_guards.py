"""Shared guards for authored config values that must be positive integers.

A cap, an interval and a port are all the same authoring contract: a whole
number of at least one. Spelling that check at each reading site lets copies
drift, so the same mistake — ``0``, ``true``, ``"8181"`` — is refused in one
deployment surface and quietly replaced in another. The predicate and the two
refusing readers live here so every surface answers the same way, and so the
refusal an operator sees is one sentence rather than one per module.

``bool`` is excluded once, here. It is an ``int`` subclass, so ``port: true``
would otherwise resolve to port 1 and ``max_runs: true`` to a buffer of one —
values no config author ever meant.

Not every caller refuses. A surface whose own lint rule reports the mistake, or
one that must keep serving with a shipped default, reads :func:`is_positive_int`
and falls back; it says so in its docstring and names the surface that does
refuse. Those callers still share this definition of "positive integer".

This module imports nothing from ``osprey``, so the deploy tree, the runtime
tree and the standalone auth sidecar can all reach it.
"""

from __future__ import annotations

from typing import Any

__all__ = ["is_positive_int", "require_positive_int", "require_positive_int_str"]


def _refusal(name: str, value: Any) -> str:
    """The one refusal sentence, naming the config key or variable and the value."""
    return f"{name} must be a positive integer, got {value!r}"


def is_positive_int(value: Any) -> bool:
    """Whether *value* is an integer of at least one.

    Args:
        value: Any value read from config, already parsed.

    Returns:
        ``True`` for an ``int`` that is not a ``bool`` and is at least 1,
        ``False`` for everything else — ``None``, a string, a float, ``0``, a
        negative, and both ``bool`` values.
    """
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def require_positive_int(value: Any, default: int, key: str) -> int:
    """Read an already-parsed config value as a positive integer, or refuse.

    Args:
        value: The raw value read from config. ``None`` is the absent key.
        default: Value to use when the key is absent.
        key: Dotted config key, named in the refusal.

    Returns:
        *default* when *value* is ``None``, otherwise the value itself.

    Raises:
        ValueError: If *value* is present and is not a positive integer. The
            message names *key* and what was found.
    """
    if value is None:
        return default
    if not is_positive_int(value):
        raise ValueError(_refusal(key, value))
    return int(value)


def require_positive_int_str(raw: str | None, default: int, variable: str) -> int:
    """Read a config value that arrives as text as a positive integer, or refuse.

    The text form of :func:`require_positive_int`, for an environment variable
    or an env-file entry — where every value is a string and an unset variable
    is ``None``.

    Args:
        raw: The raw text. ``None`` is the unset variable.
        default: Value to use when the variable is unset.
        variable: Variable name, named in the refusal.

    Returns:
        *default* when *raw* is ``None``, otherwise the parsed value.

    Raises:
        ValueError: If *raw* is set and does not parse to a positive integer.
            The message names *variable* and the raw text, so an operator can
            find the line that set it.
    """
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise ValueError(_refusal(variable, raw)) from None
    if not is_positive_int(value):
        raise ValueError(_refusal(variable, raw))
    return value
