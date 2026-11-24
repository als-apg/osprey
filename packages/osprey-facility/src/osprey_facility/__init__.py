"""The OSPREY facility description: facility.json, its schema, and the layers that build it.

The facility file is authored and imported sources built into one description.

``PN_LOCAL`` is the shape every identity ``code`` and model name must take, so a
code can name IRIs, status channels and the graph token without escaping.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version
from typing import Any

__all__ = [
    "DEFAULT_ROLE",
    "FACILITY_FILE",
    "PN_LOCAL",
    "STILL_ALL",
    "TEXTURE",
    "__version__",
    "fold_code",
    "wiring_addresses",
]

# The version comes from the package's own osprey-facility-v* calendar tags (see
# pyproject.toml): a built distribution answers from its own metadata, and a
# bare source tree — no dist installed, no build hook run — answers honestly
# that it is unbuilt.
try:
    __version__ = _dist_version("osprey-facility")
except PackageNotFoundError:  # pragma: no cover — source tree without an install
    __version__ = "0.0.0+unbuilt"

PN_LOCAL = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

#: The facility file's name at the root of every render.
FACILITY_FILE = "facility.json"

#: The built-in model that holds every channel no other model wires.
TEXTURE = "texture"

#: The role of a channel record that states none, as the schema's ``role`` slot defines it.
DEFAULT_ROLE = "readback"

#: The ``still`` value that stills every float reading of the scenario stating it.
STILL_ALL = "all"

_OUTSIDE_PN_LOCAL = re.compile(r"[^A-Za-z0-9_]")


def wiring_addresses(record: Mapping[str, Any]) -> list[str]:
    """Every address a wiring record names: its ``address``, then each of its ``also``.

    Args:
        record: A wiring record, checked against the schema or not.

    Returns:
        The addresses in stated order; an ``address`` or an ``also`` entry
        that is no string is left out.
    """
    also = record.get("also")
    stated = [record.get("address"), *(also if isinstance(also, list | tuple) else ())]
    return [address for address in stated if isinstance(address, str)]


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
