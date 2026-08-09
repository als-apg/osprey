"""Resolving a configured theme name into concrete themes and their CSS values.

Two surfaces need to turn a configured ``web.theme`` value into something
concrete, and they must agree:

- The **web terminal** server-renders a theme id onto ``<html data-theme>`` so
  the generated ``theme-boot.js`` first-paints without a flash.
- The **multi-user landing page** is a flat file nginx serves with no app
  behind it, so it cannot link the token stylesheet at all — its renderer bakes
  the resolved theme's values straight into the page's inline ``<style>``.

Both read the same ``tokens/`` source tree the design-system generator builds
from, rather than parsing a generated artifact, so neither can drift from a
stale build.

A configured value may name either a **family** (``"desy"`` — a palette, with
light/dark left to each viewer's OS) or a concrete **theme id**
(``"desy-light"`` — a palette *and* a mode). Keeping that distinction is the
whole reason :func:`resolve_pinned_mode` exists alongside
:func:`resolve_theme_id`: the resolved id alone cannot express it, because a
family resolves to that family's dark id, which is indistinguishable from an
explicitly configured dark id.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

from osprey.interfaces.design_system.generator.emit_js import ThemeManifestEntry

logger = logging.getLogger(__name__)

__all__ = [
    "load_theme_registry",
    "resolve_theme_id",
    "resolve_pinned_mode",
    "family_of",
    "theme_css_variables",
    "MissingThemeVariableError",
]


class MissingThemeVariableError(KeyError):
    """A requested CSS custom property is not defined by the requested theme.

    Raised by :func:`theme_css_variables` rather than returning a partial
    mapping. A consumer that bakes values into a standalone page has no
    stylesheet to fall back to, so a silently-dropped variable would render as
    an unstyled element rather than as an error — the failure would surface as a
    cosmetic oddity in a deployed artifact instead of at build time.
    """


def load_theme_registry() -> tuple[list[ThemeManifestEntry], dict[str, dict[str, str]]]:
    """Load the baked theme manifest + per-family defaults.

    Reads the same ``tokens/`` source tree the design-system generator builds
    from (``generator/build.py::DEFAULT_TOKENS_DIR``) rather than parsing the
    generated ``tokens.js`` — one source, no risk of drifting from a stale
    generated artifact. This intentionally skips ``validate.assert_valid``: the
    checked-in tree is validated by ``build --check`` in CI, and full
    WCAG/completeness validation isn't needed just to read theme identity.

    Returns:
        ``(entries, defaults)`` as produced by
        :func:`~osprey.interfaces.design_system.generator.emit_js.build_theme_manifest`
        and :func:`~osprey.interfaces.design_system.generator.emit_js.build_theme_defaults`.
    """
    from osprey.interfaces.design_system.generator.build import DEFAULT_TOKENS_DIR
    from osprey.interfaces.design_system.generator.emit_js import (
        build_theme_defaults,
        build_theme_manifest,
    )
    from osprey.interfaces.design_system.generator.model import load_token_tree

    tree = load_token_tree(DEFAULT_TOKENS_DIR)
    entries = build_theme_manifest(tree)
    defaults = build_theme_defaults(entries)
    return entries, defaults


def resolve_theme_id(
    configured: str,
    entries: Sequence[ThemeManifestEntry],
    defaults: dict[str, dict[str, str]],
    *,
    config_key: str = "web.theme",
) -> str:
    """Resolve a configured theme value into a concrete baked theme id.

    ``configured`` may be:

    - A concrete theme id (e.g. ``"high-contrast-light"``) — used as-is.
    - A theme *family* name (e.g. ``"main"``, ``"high-contrast"``) — resolved to
      that family's **dark** id, the canonical server-rendered default.
    - Anything else (unknown/misspelled) — logged as a warning and resolved to
      the ``main`` family's dark id.

    Never raises: a broken theme value must not take down a server's startup or
    a deploy render.

    Args:
        configured: The raw configured value.
        entries: The theme manifest.
        defaults: The per-family ``{family: {mode: id}}`` map.
        config_key: The config key named in the warning, so the message points
            at whatever key the caller actually read.

    Returns:
        A concrete theme id present in ``entries`` — never a family name and
        never ``"auto"``, since the pre-paint ``theme-boot.js`` rung only honors
        a server-rendered ``data-theme`` that is a real baked id.
    """
    valid_ids = {entry.id for entry in entries}
    if configured in valid_ids:
        return configured
    if configured in defaults:
        return defaults[configured]["dark"]

    logger.warning(
        "Unknown %s %r (not a theme id or family); falling back to "
        "the main family's dark theme. Valid ids: %s; valid families: %s",
        config_key,
        configured,
        sorted(valid_ids),
        sorted(defaults),
    )
    main_dark = defaults.get("main", {}).get("dark")
    if main_dark is not None:
        return main_dark
    # Degenerate case (no built-in ``main`` family): still return a real baked
    # dark id — ``build_theme_defaults`` guarantees each family has a dark
    # member — rather than an unverified literal, so the boot rung honors it
    # instead of silently dropping to auto (FOUC).
    for family_modes in defaults.values():
        if "dark" in family_modes:
            return family_modes["dark"]
    return next(iter(sorted(valid_ids)), "dark")


def resolve_pinned_mode(configured: str, entries: Sequence[ThemeManifestEntry]) -> str | None:
    """The light/dark mode a configured theme value *pins*, if any.

    A configured **theme id** (``"desy-light"``) states both a palette and a
    mode: the deployment wants light, and light is what every viewer gets until
    they choose otherwise. A configured **family** (``"desy"``) states only a
    palette, leaving light/dark to each viewer's OS preference.

    Args:
        configured: The raw configured value.
        entries: The theme manifest.

    Returns:
        ``"dark"`` or ``"light"`` when ``configured`` names a concrete theme id;
        ``None`` when it names a family, or is unknown (an unknown value is not
        a pin — it falls back, and a fallback must not masquerade as an
        operator's stated intent).
    """
    for entry in entries:
        if entry.id == configured:
            return entry.mode
    return None


def family_of(theme_id: str, entries: Sequence[ThemeManifestEntry]) -> str | None:
    """The family a concrete theme id belongs to, or ``None`` if unknown."""
    for entry in entries:
        if entry.id == theme_id:
            return entry.family
    return None


def theme_css_variables(theme_id: str, names: Iterable[str]) -> dict[str, str]:
    """Read specific CSS custom properties out of one theme, by name.

    For consumers that cannot link ``tokens.css`` and must inline values
    instead. Reading *by name* — rather than copying a whole emitted block — is
    what keeps such a page small; raising on a name the theme does not define is
    what keeps it honest, since a renamed token would otherwise leave the page
    quietly off-palette (see :class:`MissingThemeVariableError`).

    Args:
        theme_id: A concrete baked theme id (e.g. ``"desy-light"``).
        names: CSS custom-property names, with their leading ``--``.

    Returns:
        ``{name: value}`` for exactly the requested names, in the requested
        order.

    Raises:
        MissingThemeVariableError: If ``theme_id`` is not a known theme, or does
            not define one of ``names`` as a resolved literal.
    """
    from osprey.interfaces.design_system.generator.build import DEFAULT_TOKENS_DIR
    from osprey.interfaces.design_system.generator.emit_css import css_variable_name
    from osprey.interfaces.design_system.generator.model import load_token_tree

    tree = load_token_tree(DEFAULT_TOKENS_DIR)

    stem = next(
        (s for s, metadata in tree.theme_metadata.items() if metadata.get("id") == theme_id),
        None,
    )
    if stem is None:
        raise MissingThemeVariableError(f"unknown theme id {theme_id!r}")

    # `has_literal_value` is the pipeline's core invariant: False means the
    # token is an UNRESOLVED alias whose `value` is only the raw "{a.b.c}"
    # reference text. Skipping those here (rather than reading `value` blindly)
    # is what stops that reference text being baked into a page as if it were a
    # colour — the same refusal `emit_css` makes.
    by_name: dict[str, str] = {}
    for path, token in tree.themes[stem].items():
        name = css_variable_name(path)
        if name is not None and token.has_literal_value and isinstance(token.value, str):
            by_name[name] = token.value

    resolved: dict[str, str] = {}
    for name in names:
        if name not in by_name:
            raise MissingThemeVariableError(
                f"theme {theme_id!r} does not define {name!r} as a resolved literal; "
                f"the token may have been renamed, removed, or left a dangling alias"
            )
        resolved[name] = by_name[name]
    return resolved
