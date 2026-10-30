"""Bundled preset and trigger discovery — where the shipped YAMLs live on disk.

Resolves the two packaged resource directories (``osprey.profiles.presets`` and
``osprey.profiles.triggers``), normalizes CLI preset spellings to their on-disk
filenames, and reads a preset YAML into a raw dict. Kept a leaf so the loader,
the build command, and the service injectors can all locate bundled assets
without importing the profile parser in :mod:`osprey.cli.build_profile_load`.
"""

from __future__ import annotations

import importlib.resources
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osprey.utils.workspace import APPS_DIRNAME, TIERS_DIRNAME
from osprey_connectors.errors import BuildProfileError

from .build_profile_document import _read_profile_document
from .profile_root import ProfileKind

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

#: The preset-side key naming the bundled facility a preset's ``data/facility/``
#: tree comes from (``templates/facilities/<name>/``). Preset-side ONLY, handled
#: exactly like :data:`PRESET_DATA_BUNDLE_KEY`: :func:`_load_preset_raw`
#: consumes it, and a repo ``profile.yml`` spelling it is refused. A facility is
#: owned by no app template and no preset; a preset only names which one it shows.
PRESET_FACILITY_KEY = "facility"

#: Every key a preset spells for ``osprey init`` and never for the profile it emits.
PRESET_ONLY_KEYS: tuple[str, ...] = (PRESET_DATA_BUNDLE_KEY, PRESET_FACILITY_KEY)


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


@dataclass(frozen=True)
class TriggersSource:
    """The file a ``dispatch.triggers`` value resolves to.

    Attributes:
        path: The trigger file on disk.
        bundled: True when the file ships with the package rather than living
            beside the profile.
    """

    path: Path
    bundled: bool


def resolve_triggers_path(profile_dir: Path, triggers: str) -> TriggersSource | None:
    """Resolve a ``dispatch.triggers`` value to the file it names.

    A file beside the profile wins over a bundled trigger file of the same
    name; a bundled file is looked up by name in the packaged triggers
    directory. ``bundled`` on the result tells a caller whether the file ships
    with the package.

    Args:
        profile_dir: Directory holding the profile.
        triggers: The ``dispatch.triggers`` value.

    Returns:
        The resolved source, or None when neither location holds the file.
    """
    local = profile_dir / triggers
    if local.is_file():
        return TriggersSource(path=local, bundled=False)
    packaged = _triggers_dir() / triggers
    if packaged.is_file():
        return TriggersSource(path=packaged, bundled=True)
    return None


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


def _yaml_stems(directory: Path) -> list[str]:
    """Sorted stems of the ``*.yml`` files directly in *directory*, ``_``-prefixed skipped."""
    if not directory.is_dir():
        return []
    return sorted(
        p.name.removesuffix(".yml")
        for p in directory.iterdir()
        if p.is_file() and p.name.endswith(".yml") and not p.name.startswith("_")
    )


def list_presets() -> list[str]:
    """Return the sorted list of bundled deployment preset names (hyphenated).

    Deployments only: the bundled tiers and apps live one directory down and
    are listed by :func:`list_tiers` and :func:`list_apps`.
    """
    return _yaml_stems(_presets_dir())


#: The packaged subdirectory each layer kind's bundled files live in.
_LAYER_PRESET_DIRNAMES: dict[ProfileKind, str] = {
    ProfileKind.TIER: TIERS_DIRNAME,
    ProfileKind.APP: APPS_DIRNAME,
}


def _layer_presets_dir(kind: ProfileKind) -> Path:
    """The packaged directory holding the bundled files of layer *kind*."""
    return _presets_dir() / _LAYER_PRESET_DIRNAMES[kind]


def list_tiers() -> list[str]:
    """Return the sorted list of bundled tier names (``presets/tiers/``)."""
    return _yaml_stems(_layer_presets_dir(ProfileKind.TIER))


def list_apps() -> list[str]:
    """Return the sorted list of bundled app names (``presets/apps/``)."""
    return _yaml_stems(_layer_presets_dir(ProfileKind.APP))


def layer_preset_kind(name: str) -> ProfileKind | None:
    """The layer kind a bundled tier or app called *name* has, or ``None``.

    ``None`` for a deployment preset and for a name nothing bundles. Accepts
    either CLI spelling, like every preset lookup.
    """
    normalized = _normalize_preset_name(name)
    for kind in _LAYER_PRESET_DIRNAMES:
        if (_layer_presets_dir(kind) / f"{normalized}.yml").is_file():
            return kind
    return None


def _layer_refusal(subject: str, name: str, kind: ProfileKind) -> str:
    """The one sentence a tier or app gets where a deployment was expected."""
    folder = _LAYER_PRESET_DIRNAMES[kind]
    return (
        f"{subject} is a {kind.value}, not a deployment. It renders inside a repo: "
        f"`osprey init <dir> --preset control-assistant` writes {folder}/{name}.yml, "
        f"and `osprey build` there renders build/<project>-{name}"
    )


def layer_preset_refusal(name: str) -> str | None:
    """Why *name* cannot stand where a deployment preset is expected, or ``None``.

    The refusal ``osprey init --preset``, ``osprey build --preset``,
    ``osprey validate`` and ``osprey set`` share for a bundled tier or app:
    the name is a real file in the package, just not a deployment, so
    "unknown preset" would send the operator looking for a typo.
    """
    kind = layer_preset_kind(name)
    if kind is None:
        return None
    normalized = _normalize_preset_name(name)
    return _layer_refusal(f"`{normalized}`", normalized, kind)


def bundled_layer_file_refusal(path: Path) -> str | None:
    """The kind refusal for *path* when it is a bundled tier or app file, else ``None``.

    A bundled tier or app file is a template for a repo's ``tiers/`` or
    ``apps/``; it has no root profile beside it, so no verb acts on it in place.
    """
    resolved = path.resolve()
    for kind in _LAYER_PRESET_DIRNAMES:
        if resolved.parent == _layer_presets_dir(kind).resolve() and resolved.suffix == ".yml":
            stem = resolved.stem
            return _layer_refusal(f"`{stem}`", stem, kind)
    return None


def bundled_layer_dir_refusal(path: Path) -> str | None:
    """The kind refusal for *path* when it lies in a bundled tier or app directory.

    For a verb that takes a directory, such as ``osprey set --repo``: standing
    in ``presets/tiers/`` or ``presets/apps/`` names every file there at once.
    """
    resolved = path.resolve()
    for kind in _LAYER_PRESET_DIRNAMES:
        layer_dir = _layer_presets_dir(kind).resolve()
        if resolved == layer_dir or layer_dir in resolved.parents:
            names = list_tiers() if kind is ProfileKind.TIER else list_apps()
            listed = ", ".join(f"`{n}`" for n in names) or f"`{layer_dir.name}`"
            subject = f"{layer_dir} holds bundled {kind.value}s; each of {listed}"
            return _layer_refusal(subject, "<name>", kind)
    return None


def _read_preset_document(name: str) -> tuple[dict[str, Any], Path]:
    """Read a bundled preset YAML verbatim; return (raw_dict, preset_file_path).

    Keeps every preset-side key, including :data:`PRESET_ONLY_KEYS`. Only
    :func:`preset_data_bundle` and :func:`preset_facility` read those keys, and
    only from here — every other caller goes through :func:`_load_preset_raw`,
    which consumes them.

    Raises ``BuildProfileError`` if the preset is unknown or invalid YAML.
    """
    normalized = _normalize_preset_name(name)
    target = _presets_dir() / f"{normalized}.yml"
    if not target.exists():
        refusal = layer_preset_refusal(name)
        if refusal is not None:
            raise BuildProfileError(refusal)
        available = ", ".join(list_presets()) or "(none)"
        raise BuildProfileError(f"Unknown preset {name!r}. Available: {available}")
    raw = _read_profile_document(target, source=f"preset {name!r}")
    if not isinstance(raw, dict):
        raise BuildProfileError(f"Preset {name!r} must be a YAML mapping")
    return raw, target


def _load_preset_raw(name: str) -> tuple[dict[str, Any], Path]:
    """Read a bundled preset YAML as a profile layer; return (raw_dict, path).

    ``app_template:`` and ``facility:`` are popped here, exactly the way
    ``extends:`` is consumed during resolution: they select the packaged data
    tree ``osprey init`` copies, and are not part of what the resolved profile
    says. Popping them at the single read point is what lets them stay out of
    ``_KNOWN_PROFILE_KEYS`` while the presets keep spelling them.

    Raises ``BuildProfileError`` if the preset is unknown or invalid YAML.
    """
    raw, target = _read_preset_document(name)
    for key in PRESET_ONLY_KEYS:
        raw.pop(key, None)
    return raw, target


def _nearest_preset_value(name: str, key: str) -> str | None:
    """The nearest non-empty string *key* on preset *name*'s ``extends:`` chain.

    Walks bundled presets only; a hop that names no bundled preset, or a cycle,
    ends the walk. ``None`` when no preset on the chain spells the key.
    """
    seen: set[str] = set()
    current: str | None = _normalize_preset_name(name)
    while current and current not in seen and _preset_exists(current) is not None:
        seen.add(current)
        raw, _path = _read_preset_document(current)
        value = raw.get(key)
        if isinstance(value, str) and value:
            return value
        parent = raw.get("extends")
        current = _normalize_preset_name(parent) if isinstance(parent, str) and parent else None
    return None


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

    Raises:
        BuildProfileError: When *name* is a bundled app. An app has no data
            bundle of its own; its render uses its deployment's, so asking an
            app for one is a caller that lost track of the deployment.
    """
    if not name:
        return DEFAULT_DATA_BUNDLE
    if layer_preset_kind(name) is ProfileKind.APP:
        raise BuildProfileError(
            f"{layer_preset_refusal(name)}. Its data bundle is its deployment's; "
            "ask the deployment preset instead."
        )
    return _nearest_preset_value(name, PRESET_DATA_BUNDLE_KEY) or DEFAULT_DATA_BUNDLE


def preset_facility(name: str | None) -> str | None:
    """The bundled facility *name* resolves to, following ``extends:``.

    Read off the preset files like :func:`preset_data_bundle`: the nearest
    ``facility:`` on the chain wins.

    Args:
        name: Preset name in either CLI spelling, or ``None`` for a profile
            that names no preset.

    Returns:
        The facility the chain names, or ``None`` when the name is ``None``,
        names no bundled preset, or the chain names none.
    """
    if not name:
        return None
    return _nearest_preset_value(name, PRESET_FACILITY_KEY)
