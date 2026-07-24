"""CSS emitter: render ``tokens.css`` from a validated DTCG :class:`~.model.TokenTree`.

Structure (see the design spec section 3.3): a do-not-edit header, then
one ``{ ... }`` block per theme. The theme whose ``$extensions.mode`` is
``"dark"`` is combined with ``:root`` — so a bare, un-themed page still
gets sane defaults — and is emitted first; every other theme follows in
``tree.themes`` order. Within a block: ``color-scheme``, then (default
theme only) the fleet-wide promoted-primitive scales (see
:data:`~.naming.PROMOTED_PRIMITIVE_GROUPS` — font, text, weight,
leading, space, radius, z, duration), then semantic tokens in source
order, then each interface's extension tokens (interfaces sorted
alphabetically, each interface's own tokens in source order).

Dot-path to CSS custom property name mapping lives in ``naming.py`` (the
shared naming authority, also consumed by the validator's collision
checks) — see :func:`~.naming.css_variable_name` for the exact rules;
``css_variable_name`` is re-exported here so this module's public API is
unchanged.

Output is fully deterministic: no timestamps, source-order property
declarations, no trailing whitespace on any line, exactly one trailing
newline — so the file survives pre-commit's whitespace/end-of-file hooks
unchanged and the freshness gate (a future task) can byte-compare it.
"""

from __future__ import annotations

from osprey.interfaces.design_system.generator.inherits import source_mode
from osprey.interfaces.design_system.generator.model import (
    ResolvedToken,
    TokenTree,
    default_flagged_stem,
)
from osprey.interfaces.design_system.generator.naming import (
    PROMOTED_PRIMITIVE_GROUPS,
    css_variable_name,
    promoted_css_name,
)

__all__ = ["css_variable_name", "emit_css"]

_HEADER = (
    "/* AUTO-GENERATED — DO NOT EDIT.\n"
    " * Source: src/osprey/interfaces/design_system/tokens/\n"
    " * Regenerate with: python -m osprey.interfaces.design_system.generator.build\n"
    " */"
)


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
    """Return the stem of the dark theme that doubles as the ``:root`` fallback.

    Prefer the theme flagged ``$extensions.default: true`` (resolved by
    the shared :func:`~.model.default_flagged_stem`, the same source the
    JS emitters use for ``DEFAULT_FAMILY`` — so the two artifacts cannot
    disagree). The flag pins the ``:root`` fallback deterministically,
    independent of filename/manifest order, so adding a theme family whose
    files sort before the canonical one can never hijack the product
    default. When no theme is flagged, fall back to the first dark theme
    in manifest order (the historical behavior).

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
    dark_stems = [
        stem for stem in tree.themes if tree.theme_metadata.get(stem, {}).get("mode") == "dark"
    ]
    flagged = default_flagged_stem(tree)
    if flagged is not None and flagged in dark_stems:
        return flagged
    if dark_stems:
        return dark_stems[0]
    raise ValueError("no theme declares $extensions.mode == 'dark'; cannot emit tokens.css")


def _theme_declarations(tree: TokenTree, stem: str, *, include_fonts: bool) -> list[str]:
    """Build the indented ``--name: value;`` lines for one theme's CSS block."""
    mode = tree.theme_metadata[stem]["mode"]
    lines = [f"  color-scheme: {mode};"]

    if include_fonts:
        for group in PROMOTED_PRIMITIVE_GROUPS:
            prefix = f"{group}."
            for path, token in tree.primitives.items():
                if not path.startswith(prefix):
                    continue
                lines.append(f"  {promoted_css_name(path)}: {_resolved_value(token)};")

    for path, token in tree.themes[stem].items():
        name = css_variable_name(path)
        if name is None:
            continue
        lines.append(f"  {name}: {_resolved_value(token)};")

    for interface_stem, tokens in sorted(tree.interfaces.items()):
        # An interface may author this theme's group directly or opt it out via
        # $extensions.inherits onto a base mode it does author; either way the
        # tokens are emitted under this theme's block with the same CSS names.
        src = source_mode(tree, interface_stem, tokens, stem)
        if src is None:
            continue
        src_prefix = f"{src}."
        for path, token in tokens.items():
            if not path.startswith(src_prefix):
                continue
            name = css_variable_name(path[len(src_prefix) :])
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
