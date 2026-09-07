"""Bundled preset and trigger discovery — where the shipped YAMLs live on disk.

Resolves the two packaged resource directories (``osprey.profiles.presets`` and
``osprey.profiles.triggers``), normalizes CLI preset spellings to their on-disk
filenames, and reads a preset YAML into a raw dict. Kept a leaf so the loader,
the build command, and the service injectors can all locate bundled assets
without importing the profile parser in :mod:`osprey.cli.build_profile_load`.
"""

from __future__ import annotations

import importlib.resources
from pathlib import Path
from typing import Any

from osprey.errors import BuildProfileError

from .build_profile_document import _read_profile_document

_PRESETS_PACKAGE = "osprey.profiles.presets"

#: The preset-side key naming the data bundle a preset's ``data/`` tree comes
#: from (``templates/apps/<bundle>/``). Preset-side ONLY: it is not a profile
#: key, so :func:`_load_preset_raw` consumes it the way ``extends:`` is
#: consumed and a repo ``profile.yml`` spelling it is refused by
#: :func:`~osprey.cli.build_profile_load._reject_unknown_keys`. What a
#: deployment builds is stated by its own ``config:`` block and its ``data:``
#: tree; the bundle name only says which packaged tree ``osprey init`` copied.
PRESET_DATA_BUNDLE_KEY = "app_template"

#: The bundle used when no preset is in play, or when the preset names none.
DEFAULT_DATA_BUNDLE = "control_assistant"


def _normalize_preset_name(name: str) -> str:
    """Normalize CLI preset spelling to the on-disk filename form.

    CLI accepts both ``control-assistant`` and ``control_assistant``;
    bundled YAML files are hyphenated.
    """
    return name.replace("_", "-")


def _presets_dir() -> Path:
    """Return the directory containing bundled preset YAMLs."""
    return Path(str(importlib.resources.files(_PRESETS_PACKAGE)))


def is_bundled_preset_dir(path: Path) -> bool:
    """Whether *path* IS the packaged presets directory itself.

    The question a caller holding a resolved profile's directory asks to find
    out whether it is looking at a bundled preset rather than at a facility's
    own profile — a bundled preset has no profile directory of its own, so the
    rules anchored on one (the ``data:`` requirement above all) do not apply to
    it.

    It lives here, beside :func:`_presets_dir`, so the directory is read in one
    place. A caller that imported ``_presets_dir`` by name and compared for
    itself would bind the function at import time and keep answering from the
    real packaged directory after a test replaced this module's, which is a
    difference no reader of the caller can see.
    """
    return path.resolve() == _presets_dir().resolve()


_TRIGGERS_PACKAGE = "osprey.profiles.triggers"


def _triggers_dir() -> Path:
    """Return the directory containing bundled trigger-config YAMLs.

    Distinct from :func:`_presets_dir` — trigger configs are not build presets
    and must not appear in the preset namespace (``--list-presets``).
    """
    return Path(str(importlib.resources.files(_TRIGGERS_PACKAGE)))


def _preset_exists(name: str) -> Path | None:
    """Return the resolved preset path if ``name`` matches a bundled preset, else None.

    Non-raising probe; mirrors :func:`_load_preset_raw`'s lookup so callers
    that need to *try* preset resolution before falling back can do so
    without absorbing an exception. Note that :func:`_normalize_preset_name`
    only translates ``_`` → ``-``; values containing ``.yml`` (e.g. path-style
    ``extends: als-base.yml``) probe as ``als-base.yml.yml`` and correctly miss.
    """
    normalized = _normalize_preset_name(name)
    candidate = _presets_dir() / f"{normalized}.yml"
    return candidate if candidate.is_file() else None


def list_presets() -> list[str]:
    """Return the sorted list of bundled preset names (hyphenated)."""
    return sorted(
        p.name.removesuffix(".yml")
        for p in _presets_dir().iterdir()
        if p.name.endswith(".yml") and not p.name.startswith("_")
    )


def _read_preset_document(name: str) -> tuple[dict[str, Any], Path]:
    """Read a bundled preset YAML verbatim; return (raw_dict, preset_file_path).

    Keeps every preset-side key, including :data:`PRESET_DATA_BUNDLE_KEY`.
    Only :func:`preset_data_bundle` reads that key, and only from here — every
    other caller goes through :func:`_load_preset_raw`, which consumes it.

    Raises ``BuildProfileError`` if the preset is unknown or invalid YAML.
    """
    normalized = _normalize_preset_name(name)
    target = _presets_dir() / f"{normalized}.yml"
    if not target.exists():
        available = ", ".join(list_presets()) or "(none)"
        raise BuildProfileError(f"Unknown preset {name!r}. Available: {available}")
    raw = _read_profile_document(target, source=f"preset {name!r}")
    if not isinstance(raw, dict):
        raise BuildProfileError(f"Preset {name!r} must be a YAML mapping")
    return raw, target


def _load_preset_raw(name: str) -> tuple[dict[str, Any], Path]:
    """Read a bundled preset YAML as a profile layer; return (raw_dict, path).

    ``app_template:`` is popped here, exactly the way ``extends:`` is consumed
    during resolution: it selects the packaged data tree ``osprey init``
    copies, and is not part of what the resolved profile says. Popping it at
    the single read point is what lets it stay out of ``_KNOWN_PROFILE_KEYS``
    while the presets keep spelling it.

    Raises ``BuildProfileError`` if the preset is unknown or invalid YAML.
    """
    raw, target = _read_preset_document(name)
    raw.pop(PRESET_DATA_BUNDLE_KEY, None)
    return raw, target


def preset_data_bundle(name: str | None) -> str:
    """The packaged data bundle *name* resolves to, following ``extends:``.

    The bundle is a preset-side fact, so it is read from the preset files
    rather than from a profile: the nearest ``app_template:`` on the chain
    wins, exactly as the deep merge would have resolved it.

    Args:
        name: Preset name in either CLI spelling, or ``None`` for a profile
            that names no preset.

    Returns:
        The bundle the chain names, or :data:`DEFAULT_DATA_BUNDLE` when the
        name is ``None``, names no bundled preset, or the chain names none.
    """
    if not name:
        return DEFAULT_DATA_BUNDLE
    seen: set[str] = set()
    current: str | None = _normalize_preset_name(name)
    while current and current not in seen and _preset_exists(current) is not None:
        seen.add(current)
        raw, _path = _read_preset_document(current)
        bundle = raw.get(PRESET_DATA_BUNDLE_KEY)
        if isinstance(bundle, str) and bundle:
            return bundle
        parent = raw.get("extends")
        current = _normalize_preset_name(parent) if isinstance(parent, str) and parent else None
    return DEFAULT_DATA_BUNDLE


def _preset_extends_chain_reaches(child: str, ancestor: str) -> bool:
    """Whether preset ``child``'s ``extends`` chain passes through ``ancestor``.

    Walks bundled-preset names only: a chain hop that is missing, empty,
    path-shaped, or otherwise not a bundled preset ends the walk with False —
    such a chain cannot be rebased onto a profile materialized from
    ``ancestor`` without changing what it resolves to. ``child == ancestor``
    is also False: reaching requires at least one ``extends`` hop, because the
    caller's delta emission rewrites an ``extends`` line that must exist.
    A cycle returns False here and is rejected with a proper error by
    :func:`~.build_profile_merge._resolve_extends` when the preset is used.
    """
    target = _normalize_preset_name(ancestor)
    current = _normalize_preset_name(child)
    seen: set[str] = set()
    while current not in seen:
        seen.add(current)
        if _preset_exists(current) is None:
            return False
        raw, _path = _load_preset_raw(current)
        parent = raw.get("extends")
        if not isinstance(parent, str) or not parent or _preset_exists(parent) is None:
            return False
        parent = _normalize_preset_name(parent)
        if parent == target:
            return True
        current = parent
    return False
