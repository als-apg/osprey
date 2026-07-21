"""Single naming authority: dot-path -> CSS custom property name.

Both the CSS emitter (``emit_css.py``) and the validator (``validate.py``)
need the same two naming rules — the semantic/extension mapping
(:func:`css_variable_name`) and the promoted-primitive mapping
(:func:`promoted_css_name`, driven by :data:`PROMOTED_PRIMITIVE_GROUPS`).
Keeping them in this leaf module means the validator's collision checks and
the emitter's output physically cannot disagree about an emitted name, and
the validator no longer depends on the emitter (previously it imported the
emitter's private promoted-groups tuple and re-derived the promoted name
inline — exactly the drift class ``inherits.py`` was created to kill).

Dot-path to CSS custom property name is *not* a uniform kebab-join — see
:func:`css_variable_name` for the exact rules (a handful of legacy names
are preserved verbatim, ``tint.*``/``terminal.ansi.*`` are reshaped, and
``code.*`` is excluded entirely since it selects a highlight.js asset name
consumed elsewhere, not a CSS value).
"""

from __future__ import annotations

__all__ = ["PROMOTED_PRIMITIVE_GROUPS", "css_variable_name", "promoted_css_name"]

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
    "accent.on": "--color-on-accent",
    "amber.base": "--color-amber",
    "amber.light": "--color-amber-light",
    "amber.hover": "--color-amber-hover",
    "status.success": "--color-success",
    "status.warning": "--color-warning",
    "status.error": "--color-error",
    "status.error-hover": "--color-error-hover",
}

#: core.json primitive groups promoted directly to root-level CSS variables
#: (--font-*, --text-*, --space-*, ...). Theme-independent by construction:
#: emitted once in the default (:root) block, never per-theme; a theme
#: cannot override them.
PROMOTED_PRIMITIVE_GROUPS: tuple[str, ...] = (
    "font",
    "text",
    "weight",
    "leading",
    "space",
    "radius",
    "z",
    "duration",
)


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


def promoted_css_name(path: str) -> str:
    """CSS custom property name for a promoted core.json primitive.

    Promoted primitives use the plain kebab-join unconditionally (no
    overrides, no exclusions — those rules apply to semantic/extension
    paths only). Shared by the emitter's ``:root`` block and the
    validator's collision check so the two derive identical names.

    Args:
        path: A core.json primitive dot-path, e.g. ``"space.md"``.

    Returns:
        The ``--kebab-case`` custom property name, e.g. ``"--space-md"``.
    """
    return "--" + path.replace(".", "-")
