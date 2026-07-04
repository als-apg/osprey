"""CSS emitter: render ``tokens.css`` from a validated DTCG :class:`~.model.TokenTree`.

Structure (see the design spec section 3.3): a do-not-edit header, then
one ``{ ... }`` block per theme. The theme whose ``$extensions.mode`` is
``"dark"`` is combined with ``:root`` — so a bare, un-themed page still
gets sane defaults — and is emitted first; every other theme follows in
``tree.themes`` order. Within a block: ``color-scheme``, then (default
theme only) the fleet-wide font primitives, then semantic tokens in
source order, then each interface's extension tokens (interfaces sorted
alphabetically, each interface's own tokens in source order).

Dot-path to CSS custom property name is *not* a uniform kebab-join — see
:func:`css_variable_name` for the exact rules (a handful of legacy names
are preserved verbatim, ``tint.*``/``terminal.ansi.*`` are reshaped, and
``code.*`` is excluded entirely since it selects a highlight.js asset
name consumed elsewhere, not a CSS value).

Output is fully deterministic: no timestamps, source-order property
declarations, no trailing whitespace on any line, exactly one trailing
newline — so the file survives pre-commit's whitespace/end-of-file hooks
unchanged and the freshness gate (a future task) can byte-compare it.
"""

from __future__ import annotations

from osprey.interfaces.design_system.generator.model import ResolvedToken, TokenTree

__all__ = ["css_variable_name", "emit_css"]

_HEADER = (
    "/* AUTO-GENERATED — DO NOT EDIT.\n"
    " * Source: src/osprey/interfaces/design_system/tokens/\n"
    " * Regenerate with: python -m osprey.interfaces.design_system.generator.build\n"
    " */"
)

#: Semantic token groups excluded from CSS emission entirely. ``code.*``
#: selects a highlight.js asset name for server-side href resolution /
#: tokens.js, not a CSS value — see the module docstring.
_EXCLUDED_GROUPS = frozenset({"code"})

#: Dot-path -> CSS custom property name overrides that don't follow the
#: naive kebab-join rule, preserving today's widely-used legacy names
#: (per Task 1.1's naming-reconciliation report).
_NAME_OVERRIDES: dict[str, str] = {
    "accent.base": "--color-accent",
    "accent.light": "--color-accent-light",
    "amber.base": "--color-amber",
    "amber.light": "--color-amber-light",
    "amber.hover": "--color-amber-hover",
    "status.success": "--color-success",
    "status.warning": "--color-warning",
    "status.error": "--color-error",
    "status.error-hover": "--color-error-hover",
}

#: core.json primitive group promoted directly to root-level CSS
#: variables (``--font-display``, ``--font-mono``, ...). Fonts have no
#: semantic theme wrapper group of their own, are theme-independent, and
#: base.css already depends on ``--font-display`` existing.
_PROMOTED_PRIMITIVE_GROUP = "font"


def css_variable_name(path: str) -> str | None:
    """Map a semantic or (mode-prefix-stripped) extension dot-path to a CSS name.

    Args:
        path: A theme-relative dot-path, e.g. ``"bg.primary"``, or — for
            an interface extension token — its dot-path with the leading
            mode segment already stripped, e.g. ``"wt-crt.scanline-opacity"``.

    Returns:
        The ``--kebab-case`` custom property name, or ``None`` if this
        path is not emitted into CSS at all (see :data:`_EXCLUDED_GROUPS`).
    """
    root = path.split(".", 1)[0]
    if root in _EXCLUDED_GROUPS:
        return None
    if path in _NAME_OVERRIDES:
        return _NAME_OVERRIDES[path]
    if path.startswith("tint."):
        _, family, step = path.split(".", 2)
        return f"--{family}-tint-{step}"
    if path.startswith("terminal.ansi."):
        name = path[len("terminal.ansi.") :]
        return f"--ansi-{name}"
    return "--" + path.replace(".", "-")


def _resolved_value(token: ResolvedToken) -> str:
    """Return a token's emittable literal value, or raise if unresolved/non-string."""
    if not token.has_literal_value:
        raise ValueError(
            f"{token.source_file} ({token.path}): cannot emit unresolved alias "
            f"{token.value!r} — the tree must be validated before emission"
        )
    if not isinstance(token.value, str):
        raise ValueError(
            f"{token.source_file} ({token.path}): token value must be a string to "
            f"emit as CSS, got {type(token.value).__name__}"
        )
    return token.value


def _default_theme_stem(tree: TokenTree) -> str:
    """Return the stem of the theme whose ``$extensions.mode`` is ``"dark"``.

    That theme doubles as the ``:root`` fallback (see the module
    docstring) and is always emitted first.

    Args:
        tree: The loaded token tree.

    Returns:
        The theme's stem (e.g. ``"dark"``).

    Raises:
        ValueError: If no theme declares ``mode: "dark"``. In the normal
            pipeline this can't happen — ``validate.py``'s
            ``check_theme_metadata`` already requires every theme's mode
            to be ``"dark"`` or ``"light"`` — but at least one theme must
            actually be dark for a ``:root`` fallback to make sense.
    """
    for stem in tree.themes:
        if tree.theme_metadata.get(stem, {}).get("mode") == "dark":
            return stem
    raise ValueError("no theme declares $extensions.mode == 'dark'; cannot emit tokens.css")


def _theme_declarations(tree: TokenTree, stem: str, *, include_fonts: bool) -> list[str]:
    """Build the indented ``--name: value;`` lines for one theme's CSS block."""
    mode = tree.theme_metadata[stem]["mode"]
    lines = [f"  color-scheme: {mode};"]

    if include_fonts:
        prefix = f"{_PROMOTED_PRIMITIVE_GROUP}."
        for path, token in tree.primitives.items():
            if not path.startswith(prefix):
                continue
            lines.append(f"  --{path.replace('.', '-')}: {_resolved_value(token)};")

    for path, token in tree.themes[stem].items():
        name = css_variable_name(path)
        if name is None:
            continue
        lines.append(f"  {name}: {_resolved_value(token)};")

    mode_prefix = f"{stem}."
    for _interface_stem, tokens in sorted(tree.interfaces.items()):
        for path, token in tokens.items():
            if not path.startswith(mode_prefix):
                continue
            name = css_variable_name(path[len(mode_prefix) :])
            if name is None:
                continue
            lines.append(f"  {name}: {_resolved_value(token)};")

    return lines


def emit_css(tree: TokenTree) -> str:
    """Render ``tokens.css`` for a fully loaded and validated token tree.

    Args:
        tree: The loaded, alias-resolved token tree (see
            ``generator/model.py``). Should already have passed
            ``generator.validate.assert_valid`` — emitting a tree with
            unresolved aliases raises.

    Returns:
        The complete ``tokens.css`` file content: no trailing whitespace
        on any line, exactly one trailing newline, deterministic (no
        timestamps; identical input always produces identical output).

    Raises:
        ValueError: If no theme has ``mode: "dark"`` (see
            :func:`_default_theme_stem`), or if any token to be emitted
            still has an unresolved alias or a non-string value.
    """
    default_stem = _default_theme_stem(tree)

    default_id = tree.theme_metadata[default_stem].get("id", default_stem)
    blocks = [
        f':root, [data-theme="{default_id}"] {{\n'
        + "\n".join(_theme_declarations(tree, default_stem, include_fonts=True))
        + "\n}"
    ]

    for stem in tree.themes:
        if stem == default_stem:
            continue
        theme_id = tree.theme_metadata[stem].get("id", stem)
        blocks.append(
            f'[data-theme="{theme_id}"] {{\n'
            + "\n".join(_theme_declarations(tree, stem, include_fonts=False))
            + "\n}"
        )

    return f"{_HEADER}\n\n" + "\n\n".join(blocks) + "\n"
