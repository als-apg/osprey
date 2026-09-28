"""The facility file: authored and imported sources built into one description.

``PN_LOCAL`` is the shape every identity ``code`` and model name must take, so a
code can name IRIs, status channels and the graph token without escaping.
"""

from __future__ import annotations

import re

__all__ = ["PN_LOCAL", "fold_code"]

PN_LOCAL = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

_OUTSIDE_PN_LOCAL = re.compile(r"[^A-Za-z0-9_]")


def fold_code(name: str) -> str:
    """Fold a free-form name into a ``PN_LOCAL`` identity code.

    Every character outside ``[A-Za-z0-9_]`` becomes ``_``, and ``x`` is
    prefixed when the result is empty or starts with a digit, so the result
    always fully matches ``PN_LOCAL``.

    Args:
        name: The name to fold, such as a project name.

    Returns:
        The folded code, e.g. ``my proj`` -> ``my_proj``, ``1st-lab`` ->
        ``x1st_lab``, ``als.u`` -> ``als_u``.
    """
    code = _OUTSIDE_PN_LOCAL.sub("_", name)
    if not code or code[0].isdigit():
        code = "x" + code
    return code
