"""Semantic validation for a loaded DTCG :class:`~.model.TokenTree`.

``generator/model.py`` only parses and one-hop alias-resolves the token
sources; it performs no semantic checks. This module is where the design
system's actual contract is enforced:

- **Theme completeness** — every theme document (``themes/*.json``) must
  define the identical set of semantic dot-paths, and every interface
  extension document (``interfaces/*.json``) must have one top-level mode
  group per theme (``dark``/``light``/...), each defining the identical
  set of extension dot-paths.
- **Alias resolution** — every alias that ``model.py`` left unresolved
  (dangling, multi-hop, or pointing at a non-primitive) is a hard error.
- **Color syntax** — every ``$type: "color"`` token's value must parse as
  a hex or legacy comma ``rgba()``/``rgb()`` color (see :func:`parse_color`).
- **Terminal-group serialization** — ``terminal.*``/``terminal.ansi.*``
  values must additionally serialize as full-length ``#RRGGBB[AA]`` hex or
  legacy ``rgba()``/``rgb()`` (see :func:`is_terminal_safe_color`), the
  subset xterm.js's ``css.toColor()`` understands.
- **Namespace collisions** — an interface extension token's top-level
  namespace (``wt-crt``, ``art-violet``, ...) must not collide with a
  semantic token group name (``bg``, ``text``, ...).
- **Promoted-primitive collisions** — no theme or interface extension
  token's emitted CSS name may collide with a promoted primitive scale
  (``emit_css.py``'s ``_PROMOTED_PRIMITIVE_GROUPS`` — font, text, weight,
  leading, space, radius, z, duration), which is theme-independent and
  cannot be overridden.
- **Theme metadata** — every theme document's root ``$extensions`` must
  declare ``mode`` as ``"dark"`` or ``"light"`` (plus non-empty string
  ``id``/``label``/``family``). ``family`` groups a ``{light, dark}`` pair
  (e.g. the built-in ``osprey`` family, or a future ``high-contrast``
  family) and selects which :func:`gates_for_family` tuple applies.
- **WCAG contrast gates** — see :data:`WCAG_GATES` (AA, the default/
  ``osprey``-family gates) and :data:`WCAG_GATES_AAA` (the ``high-contrast``
  family's gates), selected per theme by :func:`gates_for_family`; the
  relative-luminance and contrast-ratio functions here are also reused by
  the contract test suite (``tests/interfaces/design_system/test_contract.py``).
- **Interface mode-group inheritance** — an interface extension document
  (``interfaces/*.json``) may opt a theme stem out of authoring its own
  mode group via a root ``$extensions.inherits`` map (see
  :func:`check_interface_mode_completeness`), so a purely decorative theme
  variant (e.g. a ``high-contrast-dark`` twin of ``dark``) doesn't force
  every interface to duplicate an identical group.

:func:`validate_token_tree` runs every check and returns *all* failures
(never stops at the first), each carrying its source file and token
dot-path, per the generator's error-reporting contract.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from osprey.interfaces.design_system.generator.emit_css import (
    _PROMOTED_PRIMITIVE_GROUPS,
    css_variable_name,
)
from osprey.interfaces.design_system.generator.inherits import raw_inherits
from osprey.interfaces.design_system.generator.model import (
    AliasStatus,
    ResolvedToken,
    TokenTree,
)

__all__ = [
    "VALID_THEME_MODES",
    "WCAG_GATES",
    "WCAG_GATES_AAA",
    "gates_for_family",
    "ValidationRule",
    "ValidationError",
    "TokenValidationError",
    "RGBColor",
    "WcagGate",
    "parse_color",
    "is_terminal_safe_color",
    "relative_luminance",
    "contrast_ratio",
    "check_alias_resolution",
    "check_color_syntax",
    "check_terminal_serialization",
    "check_theme_completeness",
    "check_theme_metadata",
    "check_interface_mode_completeness",
    "check_namespace_collisions",
    "check_promoted_primitive_collisions",
    "check_wcag_gates",
    "validate_token_tree",
    "assert_valid",
]

#: The only $extensions.mode values a theme document may declare.
VALID_THEME_MODES = frozenset({"dark", "light"})


class ValidationRule(StrEnum):
    """Machine-readable identifier for each kind of validation failure."""

    #: A theme (or an interface's per-mode group) is missing a token that
    #: a sibling theme/mode defines.
    MISSING_TOKEN = "missing_token"
    #: An alias reference does not resolve to anything in the tree.
    DANGLING_ALIAS = "dangling_alias"
    #: An alias reference resolves to another alias (a second hop).
    MULTI_HOP_ALIAS = "multi_hop_alias"
    #: An alias reference resolves to a real token that isn't a primitive.
    NOT_PRIMITIVE_ALIAS = "not_primitive_alias"
    #: A ``$type: "color"`` token's value isn't a valid hex/legacy-rgba color.
    INVALID_COLOR = "invalid_color"
    #: An extension token's namespace collides with a semantic group name.
    NAMESPACE_COLLISION = "namespace_collision"
    #: A theme or interface extension token's emitted CSS name collides
    #: with a promoted primitive scale (see :data:`_PROMOTED_PRIMITIVE_GROUPS`
    #: in ``emit_css.py``).
    PROMOTED_PRIMITIVE_COLLISION = "promoted_primitive_collision"
    #: A theme document's ``$extensions`` metadata is missing or invalid.
    INVALID_THEME_METADATA = "invalid_theme_metadata"
    #: A ``terminal.*``/``terminal.ansi.*`` value isn't xterm-safe.
    TERMINAL_SERIALIZATION = "terminal_serialization"
    #: An interface extension document is missing a mode group a theme
    #: defines, or defines one that matches no known theme.
    MISSING_MODE_GROUP = "missing_mode_group"
    #: A WCAG contrast gate (see :data:`WCAG_GATES`) is not met.
    WCAG_CONTRAST = "wcag_contrast"


@dataclass(frozen=True)
class ValidationError:
    """A single, located validation failure.

    Attributes:
        rule: Which check produced this error.
        message: Human-readable description of the failure.
        source_file: The token file the offending value came from.
        path: The dot-path of the offending token, or ``""`` for a
            document-level failure (e.g. missing ``$extensions.mode``).
    """

    rule: ValidationRule
    message: str
    source_file: Path
    path: str

    def __str__(self) -> str:
        location = str(self.source_file) + (f" ({self.path})" if self.path else "")
        return f"{location}: {self.message}"


class TokenValidationError(ValueError):
    """Raised by :func:`assert_valid` bundling every :class:`ValidationError`.

    Attributes:
        errors: Every validation failure, in the order they were found.
    """

    def __init__(self, errors: Sequence[ValidationError]) -> None:
        self.errors = list(errors)
        super().__init__("\n".join(str(error) for error in self.errors))


# --- Color parsing ------------------------------------------------------------

_SHORT_OR_FULL_HEX_PATTERN = re.compile(r"^#([0-9a-fA-F]{3,4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$")
_FULL_HEX_PATTERN = re.compile(r"^#([0-9a-fA-F]{6}|[0-9a-fA-F]{8})$")
_LEGACY_RGBA_PATTERN = re.compile(
    r"^rgba?\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*"
    r"(?:,\s*(0(?:\.\d+)?|1(?:\.0+)?|\.\d+)\s*)?\)$"
)


@dataclass(frozen=True)
class RGBColor:
    """A parsed sRGB(A) color.

    Attributes:
        red: Red channel, ``0``-``255``.
        green: Green channel, ``0``-``255``.
        blue: Blue channel, ``0``-``255``.
        alpha: Alpha channel, ``0.0``-``1.0`` (``1.0`` if the source syntax
            had none).
    """

    red: int
    green: int
    blue: int
    alpha: float = 1.0


def parse_color(value: str) -> RGBColor | None:
    """Parse a token value as a hex or legacy comma-syntax color.

    These are the only two color syntaxes DTCG token authors may use in
    this design system: 3/4/6/8-digit ``#hex`` and legacy comma
    ``rgb()``/``rgba()`` (e.g. ``rgba(148, 163, 184, 0.12)``). No
    ``hsl()``, no CSS Color 4 space-separated ``rgb()``/``color()`` syntax.

    Args:
        value: The raw token value to parse.

    Returns:
        The parsed :class:`RGBColor`, or ``None`` if ``value`` is not a
        valid hex or legacy ``rgba()``/``rgb()`` color string.
    """
    hex_match = _SHORT_OR_FULL_HEX_PATTERN.match(value)
    if hex_match:
        return _parse_hex(hex_match.group(1))
    rgba_match = _LEGACY_RGBA_PATTERN.match(value)
    if rgba_match:
        return _parse_legacy_rgba(rgba_match)
    return None


def _parse_hex(digits: str) -> RGBColor | None:
    """Expand matched hex digits (3/4/6/8 of them) into an :class:`RGBColor`."""
    if len(digits) in (3, 4):
        channels = [int(digit * 2, 16) for digit in digits]
    else:
        channels = [int(digits[i : i + 2], 16) for i in range(0, len(digits), 2)]
    alpha = channels[3] / 255 if len(channels) == 4 else 1.0
    return RGBColor(red=channels[0], green=channels[1], blue=channels[2], alpha=alpha)


def _parse_legacy_rgba(match: re.Match[str]) -> RGBColor | None:
    """Build an :class:`RGBColor` from a matched legacy ``rgb()``/``rgba()`` call."""
    red, green, blue = (int(match.group(index)) for index in (1, 2, 3))
    if red > 255 or green > 255 or blue > 255:
        return None
    alpha_text = match.group(4)
    alpha = float(alpha_text) if alpha_text is not None else 1.0
    if alpha > 1.0:
        return None
    return RGBColor(red=red, green=green, blue=blue, alpha=alpha)


def is_terminal_safe_color(value: str) -> bool:
    """Whether ``value`` serializes the way xterm.js's ``css.toColor()`` expects.

    xterm.js only understands full-length ``#RRGGBB``/``#RRGGBBAA`` hex
    (not the 3/4-digit shorthand) and legacy comma ``rgba()``/``rgb()``.
    This is a stricter subset of :func:`parse_color`, applied only to
    ``terminal.*``/``terminal.ansi.*`` tokens.

    Args:
        value: The raw token value to check.

    Returns:
        True if ``value`` is a full-length hex color or a legacy
        ``rgba()``/``rgb()`` color.
    """
    if _FULL_HEX_PATTERN.match(value):
        return True
    return _LEGACY_RGBA_PATTERN.match(value) is not None


# --- WCAG contrast --------------------------------------------------------------


def relative_luminance(color: RGBColor) -> float:
    """Compute WCAG relative luminance for an sRGB color.

    Alpha is ignored — the WCAG gates in this design system (see
    :data:`WCAG_GATES`) only ever pair fully opaque tokens (text/accent
    over a solid background).

    Args:
        color: The color to compute luminance for.

    Returns:
        Relative luminance in ``[0, 1]`` per WCAG 2.x section 1.4.3.
    """

    def channel(value: int) -> float:
        c = value / 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = channel(color.red), channel(color.green), channel(color.blue)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(foreground: RGBColor, background: RGBColor) -> float:
    """Compute the WCAG contrast ratio between two opaque sRGB colors.

    Args:
        foreground: The foreground (e.g. text) color.
        background: The background color.

    Returns:
        The contrast ratio, in ``[1, 21]``. Argument order does not
        matter — the lighter color is always treated as the numerator.
    """
    l1 = relative_luminance(foreground)
    l2 = relative_luminance(background)
    lighter, darker = max(l1, l2), min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


@dataclass(frozen=True)
class WcagGate:
    """One required contrast pair, checked against every theme.

    Attributes:
        foreground: Dot-path of the foreground (semantic) token.
        background: Dot-path of the background (semantic) token.
        minimum: Minimum acceptable contrast ratio.
    """

    foreground: str
    background: str
    minimum: float


#: Required contrast pairs, evaluated against ``bg.primary`` in every
#: theme of the default/``osprey`` family (see :func:`gates_for_family`),
#: per the proposal's WCAG AA gates: text.primary/secondary >= 4.5:1 (body
#: text), text.muted >= 3:1 (large/secondary text), accent.base >= 3:1
#: (non-text UI). Never weakened to fit a value — a failing value gets
#: nudged in the token source instead.
WCAG_GATES: tuple[WcagGate, ...] = (
    WcagGate(foreground="text.primary", background="bg.primary", minimum=4.5),
    WcagGate(foreground="text.secondary", background="bg.primary", minimum=4.5),
    WcagGate(foreground="text.muted", background="bg.primary", minimum=3.0),
    WcagGate(foreground="accent.base", background="bg.primary", minimum=3.0),
    # accent.on is text/icons ON an accent.base fill (primary buttons), so it
    # is gated against accent.base, not bg.primary — the one consumer-color
    # pairing the fleet actually depends on that bg.primary gates never see.
    WcagGate(foreground="accent.on", background="accent.base", minimum=4.5),
)

#: Required contrast pairs for the ``high-contrast`` theme family (WCAG
#: AAA), evaluated the same way as :data:`WCAG_GATES`: text.primary/
#: secondary >= 7:1 (AAA normal/body text — both are treated as body-weight
#: here, not the "large text" AAA exception, which permits 4.5:1),
#: text.muted/accent.base >= 4.5:1 (AAA large-scale text / non-text UI).
WCAG_GATES_AAA: tuple[WcagGate, ...] = (
    WcagGate(foreground="text.primary", background="bg.primary", minimum=7.0),
    WcagGate(foreground="text.secondary", background="bg.primary", minimum=7.0),
    WcagGate(foreground="text.muted", background="bg.primary", minimum=4.5),
    WcagGate(foreground="accent.base", background="bg.primary", minimum=4.5),
    WcagGate(foreground="accent.on", background="accent.base", minimum=7.0),
)

#: Theme ``$extensions.family`` to its required WCAG gate tuple.
_WCAG_GATES_BY_FAMILY: dict[str, tuple[WcagGate, ...]] = {
    "osprey": WCAG_GATES,
    "high-contrast": WCAG_GATES_AAA,
}


def gates_for_family(family: str | None) -> tuple[WcagGate, ...]:
    """Select the WCAG gate tuple required for a theme's ``$extensions.family``.

    Args:
        family: The theme's ``$extensions.family`` value, or ``None``.

    Returns:
        :data:`WCAG_GATES_AAA` for the ``"high-contrast"`` family;
        :data:`WCAG_GATES` (AA) for ``"osprey"`` and for every other value,
        including ``None`` and unrecognized families. Fail-closed: an
        unspecified or unknown family never silently loosens below AA.
    """
    return _WCAG_GATES_BY_FAMILY.get(family, WCAG_GATES) if family else WCAG_GATES


# --- Tree traversal helpers -----------------------------------------------------


def _iter_all_tokens(tree: TokenTree) -> Iterator[ResolvedToken]:
    """Yield every resolved token in the tree: primitives, themes, interfaces."""
    yield from tree.primitives.values()
    for tokens in tree.themes.values():
        yield from tokens.values()
    for tokens in tree.interfaces.values():
        yield from tokens.values()


def _document_source_file(name: str, tokens: dict[str, ResolvedToken]) -> Path:
    """Best-effort source file for an (possibly empty) document's tokens."""
    if tokens:
        return next(iter(tokens.values())).source_file
    return Path(f"<empty document: {name}>")


def _completeness_errors(
    label: str,
    path_sets: dict[str, set[str]],
    source_files: dict[str, Path],
) -> list[ValidationError]:
    """Report tokens missing from some members of a themed/moded family.

    Every member of ``path_sets`` must define the same set of dot-paths as
    the union across all members; anything short of that is reported once
    per missing path, attributed to that member's source file.

    Args:
        label: Human-readable kind of member for the error message (e.g.
            ``"theme"`` or ``"interface 'artifacts' mode"``).
        path_sets: Member name (e.g. a theme stem) to its set of dot-paths.
        source_files: Member name to the file its tokens came from.

    Returns:
        One :class:`ValidationError` (:attr:`ValidationRule.MISSING_TOKEN`)
        per ``(member, missing path)`` pair. Empty if fewer than two
        members are given (nothing to compare).
    """
    if len(path_sets) < 2:
        return []
    reference: set[str] = set()
    for paths in path_sets.values():
        reference |= paths
    errors: list[ValidationError] = []
    for name, paths in path_sets.items():
        for path in sorted(reference - paths):
            errors.append(
                ValidationError(
                    rule=ValidationRule.MISSING_TOKEN,
                    message=f"{label} {name!r} is missing token {path!r} (defined elsewhere)",
                    source_file=source_files[name],
                    path=path,
                )
            )
    return errors


# --- Individual checks -----------------------------------------------------------

_ALIAS_RULE_BY_STATUS: dict[AliasStatus, ValidationRule] = {
    AliasStatus.DANGLING: ValidationRule.DANGLING_ALIAS,
    AliasStatus.MULTI_HOP: ValidationRule.MULTI_HOP_ALIAS,
    AliasStatus.NOT_PRIMITIVE: ValidationRule.NOT_PRIMITIVE_ALIAS,
}
_ALIAS_REASON_BY_STATUS: dict[AliasStatus, str] = {
    AliasStatus.DANGLING: "does not resolve to any known token",
    AliasStatus.MULTI_HOP: "resolves to another alias, which would require a second hop",
    AliasStatus.NOT_PRIMITIVE: "resolves to a real token that is not a primitive",
}


def check_alias_resolution(tree: TokenTree) -> list[ValidationError]:
    """Reject every alias `model.py` left unresolved: dangling, multi-hop, non-primitive.

    Args:
        tree: The loaded token tree.

    Returns:
        One error per unresolved alias reference.
    """
    errors: list[ValidationError] = []
    for token in _iter_all_tokens(tree):
        rule = _ALIAS_RULE_BY_STATUS.get(token.alias_status)
        if rule is None:
            continue
        reason = _ALIAS_REASON_BY_STATUS[token.alias_status]
        errors.append(
            ValidationError(
                rule=rule,
                message=f"alias {token.value!r} referencing {token.alias_target!r} {reason}",
                source_file=token.source_file,
                path=token.path,
            )
        )
    return errors


def check_color_syntax(tree: TokenTree) -> list[ValidationError]:
    """Reject any ``$type: "color"`` token whose value isn't a valid color.

    Tokens with an already-reported unresolved alias are skipped here
    (see :func:`check_alias_resolution`) to avoid duplicate noise.

    Args:
        tree: The loaded token tree.

    Returns:
        One error per invalid color value.
    """
    errors: list[ValidationError] = []
    for token in _iter_all_tokens(tree):
        if token.type != "color" or not token.has_literal_value:
            continue
        if not isinstance(token.value, str) or parse_color(token.value) is None:
            errors.append(
                ValidationError(
                    rule=ValidationRule.INVALID_COLOR,
                    message=(
                        f"$type is 'color' but value {token.value!r} is not a valid "
                        "hex or legacy rgba()/rgb() color"
                    ),
                    source_file=token.source_file,
                    path=token.path,
                )
            )
    return errors


def check_terminal_serialization(tree: TokenTree) -> list[ValidationError]:
    """Require ``terminal.*``/``terminal.ansi.*`` values to be xterm-safe.

    Scoped to theme documents only (the terminal group lives in the
    semantic vocabulary, not primitives or interface extensions).

    Args:
        tree: The loaded token tree.

    Returns:
        One error per terminal-group value that isn't full-hex or legacy
        ``rgba()``/``rgb()``.
    """
    errors: list[ValidationError] = []
    for tokens in tree.themes.values():
        for path, token in tokens.items():
            if path != "terminal" and not path.startswith("terminal."):
                continue
            if not token.has_literal_value:
                continue
            if not isinstance(token.value, str) or not is_terminal_safe_color(token.value):
                errors.append(
                    ValidationError(
                        rule=ValidationRule.TERMINAL_SERIALIZATION,
                        message=(
                            f"terminal-group value {token.value!r} must serialize as "
                            "#RRGGBB[AA] or legacy rgba()/rgb() for xterm.js compatibility"
                        ),
                        source_file=token.source_file,
                        path=path,
                    )
                )
    return errors


def check_theme_completeness(tree: TokenTree) -> list[ValidationError]:
    """Require every theme to define the identical set of semantic dot-paths.

    Args:
        tree: The loaded token tree.

    Returns:
        One error per ``(theme, missing token)`` pair.
    """
    path_sets = {stem: set(tokens) for stem, tokens in tree.themes.items()}
    source_files = {
        stem: _document_source_file(stem, tokens) for stem, tokens in tree.themes.items()
    }
    return _completeness_errors("theme", path_sets, source_files)


def check_theme_metadata(tree: TokenTree) -> list[ValidationError]:
    """Require every theme document's ``$extensions`` to be well-formed.

    ``mode`` must be present and equal to ``"dark"`` or ``"light"``; ``id``,
    ``label``, and ``family`` must be present as non-empty strings.
    ``id``/``label`` are required per the design spec's theme registry
    metadata, consumed by the JS emitters; ``family`` groups a
    ``{light, dark}`` pair (e.g. the built-in ``osprey`` family) and
    selects the theme's WCAG gate tuple (see :func:`gates_for_family`).

    Args:
        tree: The loaded token tree.

    Returns:
        One error per malformed or missing metadata field.
    """
    errors: list[ValidationError] = []
    for stem, metadata in tree.theme_metadata.items():
        source_file = _document_source_file(stem, tree.themes.get(stem, {}))

        mode = metadata.get("mode")
        if mode is None:
            errors.append(
                ValidationError(
                    rule=ValidationRule.INVALID_THEME_METADATA,
                    message="theme document is missing required $extensions.mode",
                    source_file=source_file,
                    path="",
                )
            )
        elif mode not in VALID_THEME_MODES:
            errors.append(
                ValidationError(
                    rule=ValidationRule.INVALID_THEME_METADATA,
                    message=f"$extensions.mode must be 'dark' or 'light', got {mode!r}",
                    source_file=source_file,
                    path="",
                )
            )

        for field in ("id", "label", "family"):
            value = metadata.get(field)
            if not isinstance(value, str) or not value:
                errors.append(
                    ValidationError(
                        rule=ValidationRule.INVALID_THEME_METADATA,
                        message=f"theme document is missing required $extensions.{field}",
                        source_file=source_file,
                        path="",
                    )
                )
    return errors


def _interface_inherits(
    tree: TokenTree,
    stem: str,
    observed_modes: set[str],
    source_file: Path,
    errors: list[ValidationError],
) -> dict[str, str]:
    """Read and fail-closed-validate an interface doc's ``$extensions.inherits`` opt-out map.

    ``$extensions.inherits`` maps an opted-out theme stem (typically a
    purely decorative variant, e.g. a ``high-contrast-dark`` twin of
    ``dark``) to the base theme stem whose extension group it borrows
    instead of requiring its own group to be authored, e.g.
    ``{"$extensions": {"inherits": {"high-contrast-dark": "dark"}}}``. Only
    entries whose base is itself a mode group this same document actually
    defines are honored — a mapping is not itself a substitute for the
    tokens existing somewhere, so a base that isn't observed (e.g. a typo)
    is a validation error rather than a silent no-op, per this module's
    fail-closed convention.

    Args:
        tree: The loaded token tree.
        stem: The interface document's file stem.
        observed_modes: The mode groups this document actually defines
            (the keys of the per-mode token-path partition).
        source_file: The document's source file, for error attribution.
        errors: Appended to in place with any ``$extensions.inherits``
            validation failures.

    Returns:
        A mapping from opted-out mode to the base mode it borrows, for
        every entry that passed validation.
    """
    raw = raw_inherits(tree, stem)
    if not raw:
        return {}
    if not isinstance(raw, dict):
        errors.append(
            ValidationError(
                rule=ValidationRule.MISSING_MODE_GROUP,
                message=f"interface {stem!r} $extensions.inherits must be an object",
                source_file=source_file,
                path="",
            )
        )
        return {}

    valid: dict[str, str] = {}
    for mode, base in raw.items():
        if not isinstance(mode, str) or not isinstance(base, str):
            errors.append(
                ValidationError(
                    rule=ValidationRule.MISSING_MODE_GROUP,
                    message=(
                        f"interface {stem!r} $extensions.inherits entry {mode!r}: {base!r} "
                        "must map a string mode to a string base mode"
                    ),
                    source_file=source_file,
                    path=str(mode),
                )
            )
            continue
        if base not in observed_modes:
            errors.append(
                ValidationError(
                    rule=ValidationRule.MISSING_MODE_GROUP,
                    message=(
                        f"interface {stem!r} $extensions.inherits maps {mode!r} to "
                        f"{base!r}, which this document does not itself define a group for"
                    ),
                    source_file=source_file,
                    path=mode,
                )
            )
            continue
        valid[mode] = base
    return valid


def check_interface_mode_completeness(tree: TokenTree) -> list[ValidationError]:
    """Require each interface extension document to cover every theme's mode.

    Interface extension documents (``interfaces/*.json``) are structured
    as one top-level group per theme (e.g. ``dark``/``light``), each
    defining the extension tokens for that mode. Every document must have
    exactly one such group per theme stem in ``tree.themes``, and every
    group must define the identical set of (mode-prefix-stripped)
    dot-paths.

    A document may opt a stem out of authoring its own group by declaring
    it in its root ``$extensions.inherits`` map (see
    :func:`_interface_inherits`) — the opted-out stem then inherits
    (borrows) the base stem's group instead, e.g. so a purely decorative
    ``high-contrast-dark`` twin of ``dark`` doesn't force every interface
    to duplicate an identical group. A stem that is neither authored nor
    validly opted out is still a hard error — opting out one stem never
    excuses any other missing stem.

    Args:
        tree: The loaded token tree.

    Returns:
        One error per missing/unexpected mode group, plus one per
        ``(mode, missing extension token)`` pair within a document.
    """
    errors: list[ValidationError] = []
    expected_modes = set(tree.themes)

    for stem, tokens in tree.interfaces.items():
        observed: dict[str, set[str]] = {}
        for path in tokens:
            mode, _, rest = path.partition(".")
            observed.setdefault(mode, set()).add(rest)

        source_file = _document_source_file(stem, tokens)
        inherits = _interface_inherits(tree, stem, set(observed), source_file, errors)

        for mode in sorted(expected_modes - observed.keys()):
            if mode in inherits:
                continue
            errors.append(
                ValidationError(
                    rule=ValidationRule.MISSING_MODE_GROUP,
                    message=f"interface {stem!r} has no {mode!r} mode group (every theme needs one)",
                    source_file=source_file,
                    path=mode,
                )
            )
        for mode in sorted(observed.keys() - expected_modes):
            errors.append(
                ValidationError(
                    rule=ValidationRule.MISSING_MODE_GROUP,
                    message=f"interface {stem!r} has mode group {mode!r} matching no known theme",
                    source_file=source_file,
                    path=mode,
                )
            )

        matched = {mode: paths for mode, paths in observed.items() if mode in expected_modes}
        source_files = dict.fromkeys(matched, source_file)
        errors.extend(_completeness_errors(f"interface {stem!r} mode", matched, source_files))

    return errors


def check_namespace_collisions(tree: TokenTree) -> list[ValidationError]:
    """Reject extension tokens whose namespace collides with a semantic group.

    An interface extension token's top-level namespace (the path segment
    immediately after its mode group, e.g. ``wt-crt`` in
    ``dark.wt-crt.scanline-opacity``) must not match a semantic token
    group name (``bg``, ``text``, ``accent``, ...) — both are emitted into
    the same ``tokens.css``.

    Args:
        tree: The loaded token tree.

    Returns:
        One error per colliding extension token.
    """
    semantic_roots: set[str] = set()
    for tokens in tree.themes.values():
        semantic_roots.update(path.split(".", 1)[0] for path in tokens)

    errors: list[ValidationError] = []
    for stem, tokens in tree.interfaces.items():
        for path, token in tokens.items():
            _mode, separator, rest = path.partition(".")
            if not separator or not rest:
                continue
            root = rest.split(".", 1)[0]
            if root in semantic_roots:
                errors.append(
                    ValidationError(
                        rule=ValidationRule.NAMESPACE_COLLISION,
                        message=(
                            f"extension token namespace {root!r} (interface {stem!r}) "
                            "collides with a semantic token group of the same name"
                        ),
                        source_file=token.source_file,
                        path=path,
                    )
                )
    return errors


def check_promoted_primitive_collisions(tree: TokenTree) -> list[ValidationError]:
    """Reject theme/interface tokens whose emitted CSS name collides with a promoted primitive.

    ``emit_css.py`` promotes an ordered tuple of core.json primitive groups
    (``_PROMOTED_PRIMITIVE_GROUPS`` — font, text, weight, leading, space,
    radius, z, duration) directly to root-level CSS custom properties,
    emitted once in the default ``:root`` block and never per-theme. These
    scales are theme-independent by construction, so no theme's semantic
    token and no interface's extension token may emit the same CSS custom
    property name — a theme can never override them.

    Args:
        tree: The loaded token tree.

    Returns:
        One error per theme or interface extension token whose emitted CSS
        name collides with a promoted primitive.
    """
    promoted_names = {
        f"--{path.replace('.', '-')}"
        for path in tree.primitives
        if path.split(".", 1)[0] in _PROMOTED_PRIMITIVE_GROUPS
    }
    if not promoted_names:
        return []

    errors: list[ValidationError] = []
    for tokens in tree.themes.values():
        for path, token in tokens.items():
            name = css_variable_name(path)
            if name in promoted_names:
                errors.append(
                    ValidationError(
                        rule=ValidationRule.PROMOTED_PRIMITIVE_COLLISION,
                        message=(
                            f"theme token {path!r} emits {name!r}, which collides with a "
                            "promoted primitive scale (theme-independent, cannot be overridden)"
                        ),
                        source_file=token.source_file,
                        path=path,
                    )
                )

    for tokens in tree.interfaces.values():
        for path, token in tokens.items():
            _mode, separator, rest = path.partition(".")
            if not separator:
                continue
            name = css_variable_name(rest)
            if name in promoted_names:
                errors.append(
                    ValidationError(
                        rule=ValidationRule.PROMOTED_PRIMITIVE_COLLISION,
                        message=(
                            f"extension token {path!r} emits {name!r}, which collides with a "
                            "promoted primitive scale (theme-independent, cannot be overridden)"
                        ),
                        source_file=token.source_file,
                        path=path,
                    )
                )

    return errors


def _gate_color(token: ResolvedToken | None) -> RGBColor | None:
    """Extract a usable :class:`RGBColor` from a WCAG gate token, if any."""
    if token is None or not token.has_literal_value:
        return None
    if not isinstance(token.value, str):
        return None
    return parse_color(token.value)


def check_wcag_gates(tree: TokenTree) -> list[ValidationError]:
    """Require every applicable WCAG gate to meet its contrast minimum, per theme.

    The gate tuple applied to each theme is selected by its
    ``$extensions.family`` via :func:`gates_for_family` — AAA
    (:data:`WCAG_GATES_AAA`) for the ``high-contrast`` family, AA
    (:data:`WCAG_GATES`) for ``osprey`` and for every other/unspecified
    family (fail-closed: nothing silently loosens below AA).

    Gates whose tokens are missing (already reported by
    :func:`check_theme_completeness`) or whose values don't parse as
    colors (already reported by :func:`check_color_syntax`) are silently
    skipped here to avoid duplicate errors.

    Args:
        tree: The loaded token tree.

    Returns:
        One error per ``(theme, gate)`` pair that fails its minimum.
    """
    errors: list[ValidationError] = []
    for stem, tokens in tree.themes.items():
        family = tree.theme_metadata.get(stem, {}).get("family")
        gates = gates_for_family(family if isinstance(family, str) else None)
        for gate in gates:
            fg_token = tokens.get(gate.foreground)
            bg_token = tokens.get(gate.background)
            fg_color = _gate_color(fg_token)
            bg_color = _gate_color(bg_token)
            if fg_color is None or bg_color is None or fg_token is None:
                continue
            ratio = contrast_ratio(fg_color, bg_color)
            if ratio < gate.minimum:
                errors.append(
                    ValidationError(
                        rule=ValidationRule.WCAG_CONTRAST,
                        message=(
                            f"theme {stem!r}: {gate.foreground} vs {gate.background} "
                            f"contrast {ratio:.2f}:1 is below the required "
                            f"{gate.minimum:.1f}:1"
                        ),
                        source_file=fg_token.source_file,
                        path=gate.foreground,
                    )
                )
    return errors


# --- Orchestration --------------------------------------------------------------


def validate_token_tree(tree: TokenTree) -> list[ValidationError]:
    """Run every validation check and collect every failure.

    Never stops at the first failure — callers (the ``build`` CLI) can
    report the complete set in one pass.

    Args:
        tree: The loaded token tree, as returned by
            :func:`osprey.interfaces.design_system.generator.model.load_token_tree`.

    Returns:
        Every :class:`ValidationError` found, in check-then-discovery
        order. Empty if the tree is fully valid.
    """
    errors: list[ValidationError] = []
    errors.extend(check_alias_resolution(tree))
    errors.extend(check_color_syntax(tree))
    errors.extend(check_terminal_serialization(tree))
    errors.extend(check_theme_completeness(tree))
    errors.extend(check_theme_metadata(tree))
    errors.extend(check_interface_mode_completeness(tree))
    errors.extend(check_namespace_collisions(tree))
    errors.extend(check_promoted_primitive_collisions(tree))
    errors.extend(check_wcag_gates(tree))
    return errors


def assert_valid(tree: TokenTree) -> None:
    """Validate ``tree`` and raise if anything failed.

    Args:
        tree: The loaded token tree.

    Raises:
        TokenValidationError: If :func:`validate_token_tree` found any
            failures; carries all of them, not just the first.
    """
    errors = validate_token_tree(tree)
    if errors:
        raise TokenValidationError(errors)
