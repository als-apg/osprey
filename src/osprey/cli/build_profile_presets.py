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


def _normalize_preset_name(name: str) -> str:
    """Normalize CLI preset spelling to the on-disk filename form.

    CLI accepts both ``control-assistant`` and ``control_assistant``;
    bundled YAML files are hyphenated.
    """
    return name.replace("_", "-")


def _presets_dir() -> Path:
    """Return the directory containing bundled preset YAMLs."""
    return Path(str(importlib.resources.files(_PRESETS_PACKAGE)))


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


def _load_preset_raw(name: str) -> tuple[dict[str, Any], Path]:
    """Read a bundled preset YAML; return (raw_dict, preset_file_path).

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
