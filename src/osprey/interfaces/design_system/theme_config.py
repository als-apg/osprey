"""Resolving a configured theme name into concrete themes and their CSS values.

Three surfaces need to turn a configured ``web.theme`` value into something
concrete, and they must agree:

- The **web terminal** server-renders a theme id onto ``<html data-theme>`` so
  the generated ``theme-boot.js`` first-paints without a flash.
- The **artifact gallery** stamps the same attribute onto the artifact pages it
  serves, so one opened outside the hub honors the deployment's pin too.
- The **multi-user landing page** is a flat file nginx serves with no app
  behind it, so it cannot link the token stylesheet at all — its renderer bakes
  the resolved theme's values straight into the page's inline ``<style>``.

All three read the same ``tokens/`` source tree the design-system generator
builds from, rather than parsing a generated artifact, so none can drift from a
stale build. The first two also read the *deployment's* configured value rather
than being handed one, so :func:`resolve_configured_web_theme` owns that read
(environment before config) as well as the resolution.

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
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from osprey.interfaces.design_system.generator.emit_js import ThemeManifestEntry

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_WEB_THEME",
    "ConfiguredWebTheme",
    "configured_web_theme",
    "resolve_configured_web_theme",
    "load_theme_registry",
    "resolve_theme_id",
    "resolve_pinned_mode",
    "family_of",
    "theme_css_variables",
    "MissingThemeVariableError",
]

#: Used when neither ``OSPREY_WEB_THEME`` nor ``web.theme`` names anything: the
#: built-in family, which pins no mode and so leaves light/dark to each viewer.
DEFAULT_WEB_THEME = "main"


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


@dataclass(frozen=True)
class ConfiguredWebTheme:
    """What a deployment's configured web-theme value resolves to.

    Attributes:
        id: The concrete baked theme id — always a real id, never a family.
        pinned_mode: ``"dark"``/``"light"`` if the configured value named a
            concrete id, ``None`` if it named a family (or was unknown). See
            :func:`resolve_pinned_mode` for why this cannot be read back off
            ``id``.
        family: The family ``id`` belongs to, or ``None`` if unknown.
    """

    id: str
    pinned_mode: str | None
    family: str | None


def configured_web_theme() -> str:
    """The raw web-theme value a deployment configured.

    ``OSPREY_WEB_THEME`` wins over ``web.theme``, so several containers sharing
    one baked config image can each be themed individually through the
    environment — the same shape ``OSPREY_WEB_APP_NAME`` uses, and how
    multi-user deployments theme each user's container from the roster's
    ``theme:`` key. Falls back to :data:`DEFAULT_WEB_THEME`.

    Raises:
        Exception: Whatever reading the config raises — ``FileNotFoundError``
            when none is primed. The servers that call this disagree about what
            that should mean (the terminal must still render *some* theme, the
            gallery simply serves unpinned pages), so the fallback is theirs.
    """
    from_env = os.environ.get("OSPREY_WEB_THEME", "").strip()
    if from_env:
        return from_env

    from osprey.utils.config import get_config_value

    return str(get_config_value("web.theme", DEFAULT_WEB_THEME) or DEFAULT_WEB_THEME)


def resolve_configured_web_theme(configured: str | None = None) -> ConfiguredWebTheme:
    """Resolve the configured web theme into id, pin and family in one read.

    The single interpretation of the environment → config → family/id chain, so
    the surfaces that server-render a theme cannot disagree about what a
    configured value means.

    Args:
        configured: A raw value to resolve instead of reading the deployment's
            own (mainly for tests).

    Returns:
        The resolved :class:`ConfiguredWebTheme`.

    Raises:
        Exception: As :func:`configured_web_theme`, plus anything
            :func:`load_theme_registry` raises. Callers own the fallback.
    """
    if configured is None:
        configured = configured_web_theme()
    entries, defaults = load_theme_registry()
    theme_id = resolve_theme_id(configured, entries, defaults)
    return ConfiguredWebTheme(
        id=theme_id,
        pinned_mode=resolve_pinned_mode(configured, entries),
        family=family_of(theme_id, entries),
    )


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
