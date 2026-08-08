"""Multi-source profile resolution: preset / file + overlays + ``--set``.

The entry point the ``build`` command actually calls. Picks the base layer
(bundled preset or on-disk file), deep-merges override files and ``--set``
values over it, then hands the assembled raw dict to ``extends`` resolution and
:func:`osprey.cli.build_profile_load._parse_profile`. Also owns the ``--set``
mini-parser and the model-selection shorthand keys whose explicit use is
recorded in the build manifest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from osprey.errors import BuildProfileError

from .build_profile_document import _normalize_profile_aliases, _read_profile_document
from .build_profile_load import _parse_profile
from .build_profile_merge import _deep_merge, _resolve_extends
from .build_profile_model import BuildProfile
from .build_profile_presets import _load_preset_raw


def _parse_set_pairs(pairs: tuple[str, ...]) -> dict[str, Any]:
    """Parse ``--set KEY.PATH=VALUE`` pairs into a nested dict.

    The right-hand side is parsed with ``yaml.safe_load`` so callers get
    type coercion for free: ``true``/``false`` -> bool, ``[a,b]`` -> list,
    bare ints/floats -> numeric, anything else -> string.
    """
    result: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise BuildProfileError(f"--set expects KEY=VALUE (with '='), got: {pair!r}")
        key, _, raw_value = pair.partition("=")
        key = key.strip()
        if not key:
            raise BuildProfileError(f"--set key must be non-empty: {pair!r}")
        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError as e:
            raise BuildProfileError(f"--set value for {key!r} is not valid YAML: {e}") from e
        target: dict[str, Any] = result
        parts = key.split(".")
        for part in parts[:-1]:
            existing = target.get(part)
            if existing is None:
                existing = {}
                target[part] = existing
            elif not isinstance(existing, dict):
                raise BuildProfileError(
                    f"--set key {key!r} conflicts with earlier scalar at {part!r}"
                )
            target = existing
        target[parts[-1]] = value
    # ``--set`` is its own authored layer, so it normalizes like a document —
    # but the value loads above are scalar, not document, reads.
    return _normalize_profile_aliases(result, "--set")


# The model-selection shorthand keys a user can override via `--set` whose
# explicit use is recorded in the project manifest (extract_build_args) and
# re-applied by persona auto-render, so one parent-build override retints
# every derived persona project.
MODEL_SELECTION_OVERRIDE_KEYS = ("provider", "model", "channel_finder_mode")


def explicit_model_override_keys(set_pairs: tuple[str, ...]) -> list[str]:
    """Model-selection keys the user explicitly overrode via bare ``--set``.

    Only top-level shorthand keys count (``--set provider=x``); a dotted path
    into ``config:`` addresses the rendered config directly and carries no
    whole-stack intent, so it is never forwarded to persona renders.

    Returns the matching keys in :data:`MODEL_SELECTION_OVERRIDE_KEYS` order.
    """
    parsed = _parse_set_pairs(set_pairs)
    return [key for key in MODEL_SELECTION_OVERRIDE_KEYS if key in parsed]


def merge_cli_overrides(
    base: dict[str, Any],
    overrides: tuple[Path, ...],
    set_pairs: tuple[str, ...],
) -> dict[str, Any]:
    """Layer ``-O`` override files and ``--set`` pairs over ``base``.

    The shared CLI layering step: override files deep-merge in declaration
    order, then ``--set`` values merge on top. Used by the project-render path
    (:func:`resolve_build_profile`) and by ``osprey profile new``, which bakes
    the merged result into the materialized ``profile.yml``.
    """
    raw = base
    for override_path in overrides:
        if not override_path.exists():
            raise BuildProfileError(f"Override not found: {override_path}")
        override_raw = _read_profile_document(override_path)
        if override_raw is None:
            continue
        if not isinstance(override_raw, dict):
            raise BuildProfileError(f"Override must be a YAML mapping: {override_path}")
        raw = _deep_merge(raw, override_raw)

    if set_pairs:
        raw = _deep_merge(raw, _parse_set_pairs(set_pairs))
    return raw


def resolve_build_profile(
    profile_path: Path | None,
    preset: str | None,
    overrides: tuple[Path, ...] = (),
    set_pairs: tuple[str, ...] = (),
) -> tuple[BuildProfile, Path]:
    """Resolve a build profile from any combination of preset / file / overlays.

    Mode is determined by which of ``profile_path`` and ``preset`` is given;
    they are mutually exclusive and exactly one is required.

    Layers are applied in order: base -> override file(s) -> --set values.
    All layers are merged via :func:`_deep_merge` (string lists union-dedup,
    other lists concatenate) before ``extends:`` is resolved.

    Returns:
        ``(profile, profile_dir)``. ``profile_dir`` anchors overlay/services
        path lookups in :meth:`BuildProfile.validate`. For preset mode it is
        the bundled ``profiles/presets/`` package directory.

    Raises:
        BuildProfileError: For mutual-exclusion violations, missing files,
        invalid YAML, a ``data:`` tree in preset mode, or validation failures.
    """
    if profile_path is not None and preset is not None:
        raise BuildProfileError("Pass either a profile path or --preset, not both.")
    if profile_path is None and preset is None:
        raise BuildProfileError("Either a profile path or --preset is required.")

    if preset is not None:
        raw, base_anchor = _load_preset_raw(preset)
        profile_dir = base_anchor.parent
    else:
        assert profile_path is not None  # narrows for type-checkers
        if not profile_path.exists():
            raise BuildProfileError(f"Profile not found: {profile_path}")
        raw = _read_profile_document(profile_path)
        if not isinstance(raw, dict):
            raise BuildProfileError(f"Profile must be a YAML mapping, got {type(raw).__name__}")
        base_anchor = profile_path.resolve()
        profile_dir = profile_path.parent

    raw = merge_cli_overrides(raw, overrides, set_pairs)

    raw = _resolve_extends(raw, base_anchor)

    # Checked after extends resolution so no injection path escapes: the preset
    # itself, a -O file, a --set pair, or an extends parent. A preset has no
    # profile directory to anchor a data tree against (profile_dir is the
    # bundled package dir), so carrying one is always a mistake.
    if preset is not None and raw.get("data") is not None:
        raise BuildProfileError(
            f"Profile key 'data' is not supported with --preset (got {raw['data']!r}). "
            f"A preset carries no profile directory to resolve the data tree against. "
            f"Materialize the preset first — 'osprey profile new DIR --preset {preset}' — "
            f"then build from that directory."
        )

    profile = _parse_profile(raw)
    profile.validate(profile_dir)
    return profile, profile_dir
