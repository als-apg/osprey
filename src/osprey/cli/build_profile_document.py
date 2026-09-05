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
from osprey_connectors import yaml_loader

# YAML-surface spelling -> canonical BuildProfile field name. The rename is
# confined to the YAML surface: the Python identifier, the manifest JSON keys,
# and the Jinja render context all stay on the field name, so normalizing
# towards the field keeps every downstream reader unchanged.
_YAML_TO_FIELD: dict[str, str] = {"app_template": "data_bundle"}


# Top-level collection keys whose value is a plain selection — a list of names,
# or the panel-layout mapping — and whose empty spelling therefore has exactly
# one reading: "none of these". YAML parses a key written with nothing under it
# (every entry commented out, or a delta stub whose list was cleared) to
# ``None``, which is neither the collection the field's type promises nor the
# value the author wrote, so it is flattened to the empty collection here.
#
# Confined to these fields on purpose. The block-shaped keys (``config:``,
# ``lifecycle:``, ``env:``, the connector blocks) are read by parsers that
# already narrow their own empty spelling, and an empty block there is a
# question about inheritance rather than a selection — so they keep the reading
# their own parser gives them.
_EMPTY_COLLECTION_KEYS: dict[str, type] = {
    "hooks": list,
    "rules": list,
    "skills": list,
    "agents": list,
    "output_styles": list,
    "web_panels": list,
    "dependencies": list,
    "panel_presets": dict,
}


def _normalize_empty_collections(raw: dict[str, Any]) -> dict[str, Any]:
    """Rewrite present-but-empty selection keys to the empty collection, in place.

    Run on every document as it is parsed, and so *before* the ``extends`` and
    persona-delta merges. That placement is the point: it makes an empty
    ``web_panels:`` mean precisely what ``web_panels: []`` means, in the merge
    as well as in the parser. Normalizing only at parse time would leave
    ``None`` to win the merge key-by-key, and a persona delta whose list the
    author had emptied would silently *subtract* the root profile's whole
    selection — a power no list spelling has (that is what ``exclude:`` is
    for).

    A value of the wrong shape is left exactly as written: the schema
    validation downstream names it far better than a silent rewrite would.

    Args:
        raw: One raw profile mapping, mutated in place.

    Returns:
        ``raw``.
    """
    for key, empty in _EMPTY_COLLECTION_KEYS.items():
        if key in raw and raw[key] is None:
            raw[key] = empty()
    return raw


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
                f"Keep one spelling — {yaml_key!r} is the profile-YAML spelling."
            )
        raw[field] = value
    return raw


def _read_profile_document(path: Path, source: str | None = None) -> Any:
    """Read one raw profile-YAML document, normalizing YAML-surface aliases.

    Args:
        path: File to read.
        source: How to name the file in error messages (defaults to ``path``).

    Returns:
        The parsed document — normalized in place when it is a mapping.

    Raises:
        BuildProfileError: On invalid YAML, or on a same-document double
            spelling with differing values.
    """
    return _parse_profile_document(
        path.read_text(encoding="utf-8"), str(path) if source is None else source
    )


def _parse_profile_document(text: str, source: str) -> Any:
    """Parse one raw profile-YAML document, normalizing YAML-surface aliases.

    The only YAML parse of a profile document in the pipeline (pinned
    by a guard test), reached from files through :func:`_read_profile_document`
    and directly for a document that exists only as text — the profile the
    emitter would write. Mapping-shape checks stay with the callers so each
    keeps naming its own layer ("Override must be a YAML mapping", ...); a
    non-mapping document is returned exactly as parsed.

    Args:
        text: The document.
        source: How to name it in error messages.

    Returns:
        The parsed document — normalized in place when it is a mapping.

    Raises:
        BuildProfileError: On invalid YAML, or on a same-document double
            spelling with differing values.
    """
    try:
        raw = yaml_loader.safe_load(text)
    except yaml.YAMLError as e:
        raise BuildProfileError(f"Invalid YAML in {source}: {e}") from e
    if isinstance(raw, dict):
        _normalize_profile_aliases(raw, source)
        _normalize_empty_collections(raw)
    return raw
