"""Shared resolution of an interface document's ``$extensions.inherits`` opt-out map.

An interface extension document (``interfaces/*.json``) may opt a theme stem
out of authoring its own mode group by mapping it, in its root
``$extensions.inherits`` object, onto a base stem whose group it borrows
instead — e.g. ``{"high-contrast-dark": "dark", "high-contrast-light":
"light"}`` so a purely decorative high-contrast twin doesn't force every
interface to duplicate an identical group.

Both the validator (which fail-closed-checks the map) and the CSS emitter
(which must emit the borrowed group into the opted-out theme's block) have to
agree on where that map lives and what it means. Reading it in exactly one
place is what stops them from drifting — the drift that previously left every
high-contrast block emitting *no* interface tokens because the emitter never
consulted ``inherits`` at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from osprey.interfaces.design_system.generator.model import TokenTree

__all__ = ["raw_inherits", "string_inherits", "observed_modes", "source_mode"]


def raw_inherits(tree: TokenTree, stem: str) -> object:
    """The interface document's raw ``$extensions.inherits`` value (``{}`` if absent).

    Returned untyped so the validator can report a non-object value as an
    error rather than silently coercing it.
    """
    return tree.interface_metadata.get(stem, {}).get("inherits", {})


def string_inherits(tree: TokenTree, stem: str) -> dict[str, str]:
    """The ``str -> str`` entries of an interface's inherits map.

    Non-object maps and non-string entries are dropped — appropriate for the
    emitter, which only ever runs on an already-validated tree where the
    validator has already rejected any malformed entry.
    """
    raw = raw_inherits(tree, stem)
    if not isinstance(raw, dict):
        return {}
    return {
        mode: base for mode, base in raw.items() if isinstance(mode, str) and isinstance(base, str)
    }


def observed_modes(tokens: dict[str, object]) -> set[str]:
    """The mode groups an interface's flattened token dict actually authors.

    Keys are mode-prefixed dot-paths (``"dark.wt-crt.opacity"``); the mode is
    the first segment.
    """
    return {path.partition(".")[0] for path in tokens}


def source_mode(
    tree: TokenTree, stem: str, tokens: dict[str, object], theme_stem: str
) -> str | None:
    """Which of an interface's own mode groups supplies ``theme_stem``'s tokens.

    Returns ``theme_stem`` when the interface authors that group directly, the
    ``inherits`` base when the stem is validly opted out onto a group the
    interface does author, or ``None`` when the interface contributes nothing
    to this theme (never the case for a validated tree, since the validator
    requires every theme to be authored or validly inherited).
    """
    modes = observed_modes(tokens)
    if theme_stem in modes:
        return theme_stem
    base = string_inherits(tree, stem).get(theme_stem)
    if base is not None and base in modes:
        return base
    return None
