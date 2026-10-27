"""The dock layout rules shared by OSPREY and CANOPY.

The Python side of `@als-apg/dock-engine`: the layout rules themselves
(:mod:`dock_engine.layout`), the case corpus the engine is held to and the digest
of its layout rules. Standard library only, and independent of the framework.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

from ._data import cases_dir, rules_digest
from .layout import (
    LAYOUT_REASONS,
    REASON_TEXT,
    SIZE_TOLERANCE,
    SPLITS,
    LayoutForm,
    LayoutRefused,
    layout_keys,
    layout_refusal,
    normalize_layout,
    old_shape_refusal,
    validate_layout,
)

# The version comes from the framework's v-tags (see pyproject.toml). The module
# reaches an interpreter two ways: inside the framework wheel, which bundles it
# from packages/dock-engine, or as its own distribution installed from git by
# CANOPY. Whichever metadata is present answers; a bare source tree — no dist
# installed, no build hook run — answers honestly that it is unbuilt. Both
# distributions own the same dock_engine/ files, so finding both means one has
# overwritten the other: say so rather than pick one silently.
_versions: dict[str, str] = {}
for _dist in ("osprey-dock-engine", "osprey-framework"):
    try:
        _versions[_dist] = _dist_version(_dist)
    except PackageNotFoundError:
        continue
if len(_versions) == 2:  # pragma: no cover — a misconfigured environment
    import warnings

    warnings.warn(
        "osprey-dock-engine and osprey-framework are both installed; they ship the same "
        "dock_engine package and overwrite each other. Install one or the other.",
        RuntimeWarning,
        stacklevel=2,
    )
__version__ = next(iter(_versions.values()), "0.0.0+unbuilt")
del _dist, _versions

__all__ = [
    "LAYOUT_REASONS",
    "REASON_TEXT",
    "SIZE_TOLERANCE",
    "SPLITS",
    "LayoutForm",
    "LayoutRefused",
    "__version__",
    "cases_dir",
    "layout_keys",
    "layout_refusal",
    "normalize_layout",
    "old_shape_refusal",
    "rules_digest",
    "validate_layout",
]
