"""Standalone profile emission for ``osprey build --emit-profile``.

Materializes a bundled preset into a fully explicit, self-contained
``profile.yml`` — no ``extends:`` — so a facility profile shows every knob
it configures and never depends on upstream preset content at build time.

Content and comments are produced by two cooperating passes:

1. **Content** comes from the exact pipeline the render path uses
   (:func:`~osprey.cli.build_profile_presets._load_preset_raw` +
   :func:`~osprey.cli.build_profile_resolve.merge_cli_overrides` +
   :func:`~osprey.cli.build_profile_merge._resolve_extends`), so the emitted
   profile resolves to byte-identical semantics as building from the preset
   directly (with the same ``-O`` / ``--set`` layers).
2. **Comments** come from ruamel.yaml round-trip documents of the preset
   file(s) — the whole ``extends`` chain, root first, overlaid child-over-base
   so each layer contributes the comments for the keys it introduces. The
   commented document is then synced to the resolved content: keys absent from
   the resolved dict (``extends``, ``exclude``, excluded entries) are dropped,
   and differing values are replaced in place, which keeps the preset's
   comment attached to the key whose value changed.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML, CommentedMap
from ruamel.yaml.error import CommentMark
from ruamel.yaml.tokens import CommentToken

from .build_profile_merge import _resolve_extends
from .build_profile_presets import _load_preset_raw
from .build_profile_resolve import merge_cli_overrides

# Round-trip mode preserves comments, key order, and quoting style (same
# conventions as osprey.utils.config_writer).
_yaml = YAML(typ="rt")
_yaml.preserve_quotes = True
_yaml.width = 4096
# Match the bundled presets' list style (`  - item` under the key).
_yaml.indent(mapping=2, sequence=4, offset=2)


# Appended when the resolved profile has no ``overlay:`` section — the one
# facility-customization section bundled presets never carry.
_OVERLAY_APPENDIX = """
# --- Facility overlay artifacts ----------------------------------------------
# Drop custom artifacts under overlays/ and map each one here. Sources are
# relative to this profile directory, destinations to the rendered project.
# Add the artifact's name to the matching list above (skills:/rules:/agents:)
# when it should appear in the artifact selection.
#
# overlay:
#   overlays/rules/my-facility-rule.md: .claude/rules/my-facility-rule.md
#   overlays/skills/my-custom-skill: .claude/skills/my-custom-skill
"""


def _to_plain(value: Any) -> Any:
    """Recursively convert ruamel container types to plain dict/list.

    Scalars are left as-is: ruamel scalar wrappers subclass their plain
    counterparts, so equality against plain values holds.
    """
    if isinstance(value, dict):
        return {key: _to_plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    return value


def _tokens_to_lines(tokens: list[Any] | None) -> list[str]:
    """Flatten comment tokens to raw text lines, trimming edge blank lines."""
    if not tokens:
        return []
    text = "".join(tok.value for tok in tokens if tok is not None)
    lines = text.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def _entry_texts(entry: list[Any]) -> tuple[str | None, list[str]]:
    """Split a ruamel ca entry into (eol comment, trailing block lines).

    ruamel packs the end-of-line comment on a key's own line *and* every
    comment line up to the next key into one token (index 2 of the entry).
    The first line is the key's eol comment; the rest visually belongs to the
    NEXT key in the file.
    """
    tok = entry[2]
    if tok is None:
        return None, []
    first, _, rest = tok.value.partition("\n")
    eol = first.strip() if first.strip().startswith("#") else None
    lines = rest.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return eol, lines


def _extract_visual(
    cm: CommentedMap, parent_entry: list[Any] | None = None
) -> tuple[dict[Any, list[str]], dict[Any, str]]:
    """Map each key of ``cm`` to the comments that visually belong to it.

    Returns ``(pre, eol)``: ``pre[key]`` is the comment block ABOVE the key's
    line (stored by ruamel on the predecessor key — or, for a nested map's
    first key, on the parent's ca entry), ``eol[key]`` the comment on the
    key's own line.
    """
    pre: dict[Any, list[str]] = {}
    eol: dict[Any, str] = {}
    keys = list(cm.keys())
    if keys and parent_entry is not None and parent_entry[3]:
        lines = _tokens_to_lines(parent_entry[3])
        if lines:
            pre[keys[0]] = lines
    for idx, key in enumerate(keys):
        entry = cm.ca.items.get(key)
        if entry is None:
            continue
        own_pre = _tokens_to_lines(entry[1])
        if own_pre:
            pre[key] = pre.get(key, []) + own_pre
        eol_text, trailing = _entry_texts(entry)
        if eol_text:
            eol[key] = eol_text
        if trailing and idx + 1 < len(keys):
            pre[keys[idx + 1]] = trailing
    return pre, eol


def _set_pre_comment(cm: CommentedMap, key: Any, lines: list[str], indent: int) -> None:
    """Attach comment ``lines`` above ``key``, indented to the key's depth."""
    tokens = [
        CommentToken((line.strip() if line.strip() else "") + "\n", CommentMark(indent), None)
        for line in lines
    ]
    entry = cm.ca.items.setdefault(key, [None, None, None, None])
    entry[1] = (entry[1] or []) + tokens


def _relocate_trailing(cm: CommentedMap, old_last_key: Any, new_last_key: Any) -> None:
    """Move ``old_last_key``'s trailing comment block behind ``new_last_key``.

    A section-header comment for the NEXT sibling section is stored by ruamel
    as the trailing block of a map's last key — on the deepest last leaf when
    that key's value is itself a mapping. Appending new keys to the map would
    render them AFTER that header, visually placing them in the wrong
    section — so the trailing block moves to the appended final key.
    """
    container: CommentedMap = cm
    leaf_key = old_last_key
    while isinstance(container.get(leaf_key), CommentedMap) and len(container[leaf_key]) > 0:
        container = container[leaf_key]
        leaf_key = list(container.keys())[-1]
    entry = container.ca.items.get(leaf_key)
    if not entry or entry[2] is None:
        return
    first, sep, rest = entry[2].value.partition("\n")
    if not sep or not rest.strip():
        return
    entry[2].value = first + "\n"
    new_entry = cm.ca.items.setdefault(new_last_key, [None, None, None, None])
    if new_entry[2] is None:
        new_entry[2] = CommentToken("\n" + rest, CommentMark(0), None)
    else:
        new_entry[2].value += rest


def _merge_commented(
    base: CommentedMap,
    child: CommentedMap,
    child_parent_entry: list[Any] | None = None,
    indent: int = 0,
) -> None:
    """Overlay ``child`` onto ``base`` in place, carrying comments.

    Keys already in ``base`` keep the base file's comments (their values are
    replaced); keys new in ``child`` bring along the comments that visually
    belong to them in the child file (block above + eol), re-anchored to the
    key itself so later deletions of neighboring keys cannot orphan them.
    List semantics deliberately do not mirror the render path's union-dedup —
    :func:`_sync_to_resolved` enforces content afterwards; this pass only
    decides which file's comments win.
    """
    pre, eol = _extract_visual(child, child_parent_entry)
    base_keys = list(base.keys())
    old_last_key = base_keys[-1] if base_keys else None
    appended = False
    for key, child_val in child.items():
        base_val = base.get(key)
        if isinstance(base_val, CommentedMap) and isinstance(child_val, CommentedMap):
            _merge_commented(base_val, child_val, child.ca.items.get(key), indent + 2)
            continue
        is_new = key not in base
        base[key] = child_val
        if is_new:
            appended = True
            if key in pre:
                _set_pre_comment(base, key, pre[key], indent)
            if key in eol:
                base.yaml_add_eol_comment(eol[key], key)
    if appended and old_last_key is not None:
        new_last_key = list(base.keys())[-1]
        if new_last_key != old_last_key:
            _relocate_trailing(base, old_last_key, new_last_key)


def _sync_to_resolved(doc: CommentedMap, resolved: dict[str, Any]) -> None:
    """Make ``doc``'s content exactly match ``resolved``, keeping comments.

    Values that already compare equal are left untouched so their item-level
    comments survive; differing values are replaced (the key-level comment
    stays attached); keys absent from ``resolved`` are deleted together with
    their comments.
    """
    for key in [k for k in doc.keys() if k not in resolved]:
        del doc[key]
        doc.ca.items.pop(key, None)
    for key, value in resolved.items():
        current = doc.get(key)
        if isinstance(current, CommentedMap) and isinstance(value, dict):
            _sync_to_resolved(current, value)
        elif key not in doc or _to_plain(current) != _to_plain(value):
            doc[key] = value


def _replace_header(text: str, header: str) -> str:
    """Swap the leading comment block of ``text`` for ``header``.

    The preset file's own header addresses preset readers ("copy this file
    ..."); the emitted profile gets a header describing what it is and how to
    rebuild from it.
    """
    lines = text.splitlines()
    body_start = 0
    while body_start < len(lines) and (
        lines[body_start].startswith("#") or not lines[body_start].strip()
    ):
        body_start += 1
    return header.rstrip("\n") + "\n\n" + "\n".join(lines[body_start:]).rstrip("\n") + "\n"


def emit_standalone_profile_yaml(
    preset_name: str,
    overrides: tuple[Path, ...],
    set_pairs: tuple[str, ...],
    profile_name: str,
    profile_filename: str,
) -> str:
    """Render the standalone ``profile.yml`` text for ``--emit-profile``.

    Args:
        preset_name: Bundled preset to materialize (any CLI spelling).
        overrides: ``-O`` override files, layered in declaration order.
        set_pairs: ``--set KEY=VALUE`` pairs, layered on top.
        profile_name: Display name written to the profile's ``name:`` key.
        profile_filename: Profile path relative to the user's cwd, for the
            rebuild hint in the generated header.

    Returns:
        Complete ``profile.yml`` content: fully explicit (no ``extends:``),
        preset comments preserved, overrides applied in place.
    """
    # Content authority — identical layering to the render path: preset raw,
    # -O files, --set pairs, then extends resolution (which also applies any
    # exclude: subtractions and consumes the extends/exclude keys).
    raw, base_anchor = _load_preset_raw(preset_name)
    raw = merge_cli_overrides(raw, overrides, set_pairs)
    chain: list[Path] = []
    resolved = _resolve_extends(raw, base_anchor, chain)
    resolved["name"] = profile_name

    # Comment source — the extends chain's files, root first, so the richly
    # commented base contributes most comments and each child layer brings the
    # comments for the keys it introduces.
    doc: CommentedMap | None = None
    for layer_path in reversed(chain):
        with open(layer_path, encoding="utf-8") as f:
            layer_doc = _yaml.load(f)
        if not isinstance(layer_doc, CommentedMap):  # pragma: no cover - presets are mappings
            continue
        if doc is None:
            doc = layer_doc
        else:
            _merge_commented(doc, layer_doc)
    if doc is None:  # pragma: no cover - _load_preset_raw already validated
        doc = CommentedMap()

    _sync_to_resolved(doc, resolved)

    buffer = io.StringIO()
    _yaml.dump(doc, buffer)
    text = buffer.getvalue()

    normalized = base_anchor.stem
    header = (
        f"# {profile_name} — OSPREY build profile\n"
        f"#\n"
        f"# Emitted from the bundled `{normalized}` preset as a fully explicit,\n"
        f"# standalone profile: everything the preset configures is written out\n"
        f"# below and is yours to edit. Nothing is inherited at build time.\n"
        f"#\n"
        f"# Build a project from this profile with:\n"
        f"#   osprey build <PROJECT_NAME> {profile_filename}"
    )
    text = _replace_header(text, header)

    if "overlay" not in resolved:
        text += _OVERLAY_APPENDIX
    return text
