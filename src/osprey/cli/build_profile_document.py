"""Raw profile-YAML document reads — the single parse + alias-normalization point.

Every layer of the profile pipeline reads its YAML through
:func:`_read_profile_document`: the bundled preset, each ``-O`` override file,
the ``extends:`` parent inside :func:`~osprey.cli.build_profile_merge._resolve_extends`,
the positional profile file, and the hashing path. Normalizing there rather than
at the call sites means a YAML-surface spelling is translated to its canonical
field name exactly once per document, so only one spelling ever survives a
layer — which is what lets mixed-spelling ``extends:`` and override chains merge
child-wins through the ordinary deep merge, and what scopes the
"both spellings, different values" error to a single authored document.

Kept a leaf — it imports nothing else from the profile pipeline — so
:mod:`osprey.cli.build_profile_presets` can use it without gaining a dependency
on the parser.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from osprey.errors import BuildProfileError

# YAML-surface spelling -> canonical BuildProfile field name. The rename is
# confined to the YAML surface: the Python identifier, the manifest JSON keys,
# and the Jinja render context all stay on the field name, so normalizing
# towards the field keeps every downstream reader unchanged.
_YAML_TO_FIELD: dict[str, str] = {"app_template": "data_bundle"}


def _normalize_profile_aliases(raw: dict[str, Any], source: str) -> dict[str, Any]:
    """Rewrite YAML-surface key spellings to canonical field names, in place.

    Args:
        raw: One raw profile mapping — a single YAML document or a single
            ``--set`` layer. Never a merged multi-layer dict: the
            double-spelling check below is only meaningful within one
            authored source.
        source: How to name ``raw``'s origin in error messages.

    Returns:
        ``raw``, mutated in place.

    Raises:
        BuildProfileError: If one source spells both the YAML key and its
            canonical field name with differing values.
    """
    for yaml_key, field in _YAML_TO_FIELD.items():
        if yaml_key not in raw:
            continue
        value = raw.pop(yaml_key)
        if field in raw and raw[field] != value:
            raise BuildProfileError(
                f"{source} sets both {yaml_key!r} ({value!r}) and its alias "
                f"{field!r} ({raw[field]!r}) to different values. "
                f"Keep one spelling — {yaml_key!r} is the current one."
            )
        raw[field] = value
    return raw


def _read_profile_document(path: Path, source: str | None = None) -> Any:
    """Read one raw profile-YAML document, normalizing YAML-surface aliases.

    The only ``yaml.safe_load`` of a profile document in the pipeline (pinned
    by a guard test). Mapping-shape checks stay with the callers so each keeps
    naming its own layer ("Override must be a YAML mapping", ...); a
    non-mapping document is returned exactly as parsed.

    Args:
        path: File to read.
        source: How to name the file in error messages (defaults to ``path``).

    Returns:
        The parsed document — normalized in place when it is a mapping.

    Raises:
        BuildProfileError: On invalid YAML, or on a same-document double
            spelling with differing values.
    """
    label = str(path) if source is None else source
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise BuildProfileError(f"Invalid YAML in {label}: {e}") from e
    if isinstance(raw, dict):
        _normalize_profile_aliases(raw, label)
    return raw
