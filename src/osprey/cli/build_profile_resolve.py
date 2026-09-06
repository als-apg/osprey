"""Profile resolution: a preset or a file, host-variant overlays, ``--set`` edits.

The entry point ``osprey init``'s materialization, ``osprey build`` and
``osprey validate`` call. Picks the base (a bundled preset or an on-disk file),
resolves ``extends``, then applies the command line's ``--set`` pairs and hands
the result to :func:`osprey.cli.build_profile_load._parse_profile`. Also owns the
``--set`` mini-parser, the top-level shorthand keys (model selection plus
``connector``) whose explicit use is recorded in the build manifest, and the
rule that lets a ``config.*`` key stated on the command line outrank the deeper
keys some other layer spells beneath it, instead of losing to them by key order.

Two things reach a profile, and they mean different things:

* **Inheritance** — ``extends:``, a persona delta over the profile beside it, a
  host-variant overlay over ``profile.yml``. A layer is a *difference*: it adds
  to what it inherits, string lists union, and taking something away is an
  explicit verb (``exclude:``, ``remove_deny``). That merge lives in
  :mod:`osprey.cli.build_profile_merge`.
* **An edit** — ``--set`` at ``osprey init`` and ``osprey set`` afterwards. An
  edit *states*: the value at the key it names is replaced, whatever was there.
  :func:`apply_cli_edits` makes that edit to the resolved document before it is
  written; :func:`write_back_cli_overrides` makes the same edit to the file once
  it exists. Neither is a layer, and nothing is layered at invocation time and
  thrown away afterwards: the profile is the source of truth, so an edit is an
  edit of that profile.
"""

from __future__ import annotations

import copy
import io
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import click
import yaml

from osprey.errors import BuildProfileError
from osprey.port_layout import PORT_BASE_CONFIG_KEY
from osprey.utils.logger import get_logger

from .build_profile_document import _read_profile_document
from .build_profile_load import (
    CONNECTOR_CONFIG_KEY,
    CONNECTOR_PROFILE_KEY,
    PORT_BASE_PROFILE_KEY,
    LoadedProfile,
    _apply_connector_shorthand,
    _apply_port_base_shorthand,
    _parse_profile,
)
from .build_profile_merge import _deep_merge, _resolve_extends, resolve_profile_document
from .build_profile_model import BuildProfile
from .build_profile_presets import _load_preset_raw

logger = get_logger("build")

#: Refusal shared by the two places CLI layers reach a materialized profile —
#: `osprey init` baking them in, and a build writing them back. One
#: constant because the rule is one rule: a materialized profile is standalone,
#: so nothing may give it an `extends:` parent. Two spellings of it would mean
#: the same override file is refused on the build that materializes and accepted
#: on the next one.
EXTENDS_OVERRIDE_REFUSAL = (
    "Cannot override 'extends' — a materialized profile is standalone and "
    "inherits nothing at build time."
)


class _MappingValue(dict):
    """A mapping given as a ``--set`` VALUE, as opposed to nesting spelled by key dots.

    ``--set bluesky.port=1`` and ``--set bluesky={port: 1}`` both parse to
    ``{"bluesky": {"port": 1}}``; the dict shape cannot tell them apart, and
    they mean different edits — the first names the leaf ``bluesky.port``, the
    second states the whole value of ``bluesky``. The VALUE mapping is wrapped
    in this type so :func:`_dotted_leaves` stops at it and writes it whole,
    while key nesting stays a plain dict and is descended into.
    """


def _parse_set_pairs(pairs: tuple[str, ...]) -> dict[str, Any]:
    """Parse ``--set KEY.PATH=VALUE`` pairs into a nested dict.

    The right-hand side is parsed with ``yaml.safe_load`` so callers get
    type coercion for free: ``true``/``false`` -> bool, ``[a,b]`` -> list,
    bare ints/floats -> numeric, anything else -> string. A mapping value is
    marked as one (:class:`_MappingValue`): it is the value of the key named,
    written whole, not a shorthand for one edit per leaf inside it.

    ``config.`` is the one prefix that stops the nesting. A profile's
    ``config:`` block is a flat bag of LITERAL DOTTED KEYS, each applied
    verbatim to the one rendered-config leaf it addresses, so
    ``--set config.a.b=1`` becomes ``{"config": {"a.b": 1}}`` — the same
    spelling :func:`write_back_cli_overrides` writes into ``profile.yml``, and
    the same one the presets ship. Nesting it instead would put a second,
    differently-shaped statement of the same path beside the preset's, where
    which of the two reaches ``config.yml`` depends on key order.
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
            if isinstance(value, dict):
                value = _MappingValue(value)
        except yaml.YAMLError as e:
            raise BuildProfileError(f"--set value for {key!r} is not valid YAML: {e}") from e
        target: dict[str, Any] = result
        parts = key.split(".")
        if parts[0] == _CONFIG_SET_PREFIX and len(parts) > 1:
            # Everything after `config.` is ONE key, dots and all.
            parts = [_CONFIG_SET_PREFIX, ".".join(parts[1:])]
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
    # Returned as the layer it is: the value loads above are scalar reads, and
    # a `--set` pair names a profile key directly, so there is nothing between
    # what was typed and what the merge sees.
    return result


# The model-selection shorthand keys a user can override via `--set`, whose
# explicit use is recorded in the project manifest (extract_build_args). A
# persona inherits them the same way it inherits everything else — `osprey set`
# writes the override INTO profile.yml, and every delta in `personas/` merges
# over that profile — so one repo-level override retints every persona the next
# build renders, with nothing replayed from a build invocation.
MODEL_SELECTION_OVERRIDE_KEYS = ("provider", "model", "channel_finder_mode")

# Every top-level shorthand whose explicit ``--set`` use is forwarded that way.
# `connector` joins the model-selection keys because it shapes the whole stack
# in the same sense: which control system a project talks to is a property of
# the deployment, not of one persona, so it belongs in profile.yml where every
# persona delta inherits it rather than in any one persona's own file.
SHORTHAND_OVERRIDE_KEYS = (
    *MODEL_SELECTION_OVERRIDE_KEYS,
    CONNECTOR_PROFILE_KEY,
    PORT_BASE_PROFILE_KEY,
)


def explicit_model_override_keys(set_pairs: tuple[str, ...]) -> list[str]:
    """Shorthand keys the user explicitly overrode via bare ``--set``.

    Only top-level shorthand keys count (``--set provider=x``,
    ``--set connector=epics``); a dotted path into ``config:`` addresses the
    rendered config directly and carries no whole-stack intent, so it is never
    forwarded to persona renders.

    Returns the matching keys in :data:`SHORTHAND_OVERRIDE_KEYS` order.
    """
    parsed = _parse_set_pairs(set_pairs)
    return [key for key in SHORTHAND_OVERRIDE_KEYS if key in parsed]


# The ``config:`` key prefix a ``--set`` pair uses to address the rendered
# config, and the segments of the literal key the ``connector`` shorthand
# resolves to.
_CONFIG_SET_PREFIX = "config"
_CONNECTOR_CONFIG_SEGMENTS = tuple(CONNECTOR_CONFIG_KEY.split("."))


def _stated_config_paths(layer: dict[str, Any]) -> list[tuple[str, ...]]:
    """The rendered-config paths one CLI layer states outright.

    A ``config:`` block's own keys ARE paths — the literal dotted key
    ``approval.tools.channel_read`` addresses one leaf, the plain key
    ``approval`` holding a mapping addresses that whole subtree — so a layer
    states exactly what its top-level ``config`` keys spell. That is the
    provenance :func:`_drop_shadowed_config_keys` needs, and it exists only for
    the layers the command line supplied: a key the base or an ``extends``
    parent contributes is inherited content, not a statement made now.
    """
    config = layer.get(_CONFIG_SET_PREFIX)
    if not isinstance(config, dict):
        return []
    return [tuple(key.split(".")) for key in config if isinstance(key, str)]


def _shadowed_config_keys(config: Mapping[str, Any], stated: list[tuple[str, ...]]) -> set[str]:
    """The ``config:`` keys strictly beneath a stated path (a stated path itself is kept)."""
    claimed = set(stated)
    shadowed: set[str] = set()
    for key in config:
        if not isinstance(key, str):
            continue
        path = tuple(key.split("."))
        if path in claimed:
            continue
        if any(len(above) < len(path) and path[: len(above)] == above for above in claimed):
            shadowed.add(key)
    return shadowed


def _drop_shadowed_config_keys(
    raw: dict[str, Any], stated: list[tuple[str, ...]]
) -> dict[str, Any]:
    """Drop inherited ``config:`` keys that a CLI-stated key sits above.

    ``config_update_fields`` applies dotted keys in iteration order and sets
    each addressed path verbatim, so a key and a deeper key beneath it are two
    different dict keys that both survive every merge — and then one of the two
    values is discarded, by an order nobody wrote down. ``--set
    config.approval.tools='{…}'`` beside a preset's
    ``approval.tools.channel_read`` is exactly that pair.

    An edit is the last word, so the key it names wins the whole subtree
    beneath it: every key BENEATH a stated path (segment-prefix match, so
    ``approval.tools`` claims ``approval.tools.channel_read`` and not
    ``approval.tools_extra``) is dropped, and the value the operator just gave
    is the one that renders. A stated path itself is never dropped.

    This is the first half of an edit and runs BEFORE the edit's own leaves are
    written (:func:`apply_cli_edits`, :func:`write_back_cli_overrides`): a
    mapping value is written as the dotted leaves beneath its key, and those
    leaves lie under the path the edit claims, so pruning afterwards would
    delete the edit itself. Run on the FULLY resolved raw, after ``extends``:
    a parent's literal dotted key is inherited content just as surely as the
    base preset's.

    Args:
        raw: Resolved raw profile dict.
        stated: Paths the edit states, from :func:`_stated_config_paths`.

    Returns:
        ``raw`` when nothing is shadowed, else a copy with a pruned ``config``.
    """
    config = raw.get(_CONFIG_SET_PREFIX)
    if not stated or not isinstance(config, dict):
        return raw
    shadowed = _shadowed_config_keys(config, stated)
    if not shadowed:
        return raw
    logger.debug(
        "Dropped %d config key(s) shadowed by a CLI override: %s",
        len(shadowed),
        ", ".join(sorted(shadowed)),
    )
    return {
        **raw,
        _CONFIG_SET_PREFIX: {key: value for key, value in config.items() if key not in shadowed},
    }


def _literal_control_system_type(layer: dict[str, Any]) -> Any:
    """The ``control_system.type`` an authored layer sets, in either spelling.

    Returns the value, or ``None`` when the layer sets it nowhere — which is
    also what a layer setting it to a literal ``null`` returns, a distinction
    with no meaning here (both leave the connector unstated).
    """
    config = layer.get("config")
    if not isinstance(config, dict):
        return None
    if CONNECTOR_CONFIG_KEY in config:
        return config[CONNECTOR_CONFIG_KEY]
    nested = config.get(_CONNECTOR_CONFIG_SEGMENTS[0])
    if isinstance(nested, dict):
        return nested.get(_CONNECTOR_CONFIG_SEGMENTS[1])
    return None


def _reject_connector_type_conflict(layers: list[dict[str, Any]]) -> None:
    """Reject CLI layers that name both ``connector`` and ``control_system.type``.

    The shorthand is the short spelling of that one config key, so a command
    line giving both states the connector twice, and nothing in the profile
    says which spelling wins. Scoped to what the command line states (the
    ``--set`` pairs): a preset or ``extends`` parent that already sets the
    literal key is exactly what the shorthand is for overriding, and must keep
    working.

    Raises:
        BuildProfileError: If any CLI layer names the shorthand while any names
            the literal key.
    """
    connector = next(
        (layer[CONNECTOR_PROFILE_KEY] for layer in layers if CONNECTOR_PROFILE_KEY in layer), None
    )
    if connector is None:
        return
    literal = next(
        (
            value
            for value in (_literal_control_system_type(layer) for layer in layers)
            if value is not None
        ),
        None,
    )
    if literal is None:
        return
    raise BuildProfileError(
        f"Conflicting connector overrides: {CONNECTOR_PROFILE_KEY}={connector!r} and "
        f"{_CONFIG_SET_PREFIX}.{CONNECTOR_CONFIG_KEY}={literal!r} were both given. "
        f"{CONNECTOR_PROFILE_KEY!r} is the short spelling of "
        f"config: {{{CONNECTOR_CONFIG_KEY}: ...}} — the two set the same key and "
        f"nothing states which wins. Keep one."
    )


def cli_edit_layer(set_pairs: tuple[str, ...]) -> dict[str, Any]:
    """The edit a command line's ``--set`` pairs make, as one profile fragment.

    Parsed (:func:`_parse_set_pairs`), checked for a connector named twice
    (:func:`_reject_connector_type_conflict`), and with the ``connector`` and
    ``port_base`` shorthands folded into the ``config:`` keys they stand for —
    so what an edit states is the literal key a reader of the profile would
    edit, never a shorthand that silently outranks the block printed beside it.

    Raises:
        BuildProfileError: On an unparseable pair, an invalid ``connector``
            value, or pairs naming both ``connector`` and
            ``config.control_system.type``.
    """
    layer = _parse_set_pairs(set_pairs)
    _reject_connector_type_conflict([layer])
    return _apply_port_base_shorthand(_apply_connector_shorthand(layer))


def apply_cli_edits(resolved: dict[str, Any], set_pairs: tuple[str, ...]) -> dict[str, Any]:
    """Apply ``--set`` pairs to a resolved raw profile, replacing at each key.

    A ``--set`` pair is an edit of the profile — the edit ``osprey set`` makes
    to the file once it exists — and it means the same thing here: the value at
    the key it names is replaced, whatever was there (a scalar, a mapping, a
    list). It is applied AFTER ``extends`` resolution, to the document as it
    would be written out, and never as one more layer under inheritance.
    Routed through :func:`~osprey.cli.build_profile_merge._deep_merge` instead,
    a list stated on the command line would union with the preset's, and a
    narrowing such as ``config.deployed_services=[]`` would be silently lost.
    Inheritance adds; an edit states.

    A ``config.*`` key the edit names wins the whole subtree beneath it: the
    inherited keys under that path go first (:func:`_drop_shadowed_config_keys`),
    then the edit's own leaves land, so ``config.approval.tools={…}`` leaves
    exactly the ``approval.tools.*`` entries it spelled and none of the preset's.
    The prune is done here, as part of the edit, because the edit is the only
    place the provenance exists — afterwards an edited key is indistinguishable
    from an inherited one.

    The shorthand folds run on the document BEFORE the edit lands, as well as
    on the edit itself, so a hand-written profile spelling ``connector:`` at
    the top level has it folded into ``config.control_system.type`` first and
    ``--set connector=…`` then replaces that key rather than being overwritten
    by the fold. Idempotent.

    Args:
        resolved: The profile as resolved so far, ``extends`` already applied.
        set_pairs: ``--set KEY=VALUE`` pairs.

    Raises:
        BuildProfileError: Whatever :func:`cli_edit_layer` raises.
    """
    folded = _apply_port_base_shorthand(_apply_connector_shorthand(resolved))
    if not set_pairs:
        return folded
    layer = cli_edit_layer(set_pairs)
    claimed = _drop_shadowed_config_keys(folded, _stated_config_paths(layer))
    return _replace_leaves(claimed, _flatten_override_layer(layer))


def _replace_leaves(
    document: dict[str, Any], updates: list[tuple[list[str], Any]]
) -> dict[str, Any]:
    """Set each ``key_path`` in a copy of ``document`` to its value.

    The in-memory twin of :func:`_write_profile_values`, walking the same leaf
    list :func:`_flatten_override_layer` produces, so an edit applied to a
    profile before it is written and one written into the file afterwards
    address the same keys and land the same values.
    """
    edited = copy.deepcopy(document)
    for key_path, value in updates:
        node: dict[str, Any] = edited
        for segment in key_path[:-1]:
            if not isinstance(node.get(segment), dict):
                node[segment] = {}
            node = node[segment]
        node[key_path[-1]] = value
    return edited


def _read_overlay(path: Path) -> dict[str, Any] | None:
    """Read one host-variant overlay: a YAML mapping, or ``None`` for an empty file.

    Raises:
        BuildProfileError: If the file is missing or not a mapping.
    """
    if not path.exists():
        raise BuildProfileError(f"Overlay not found: {path}")
    raw = _read_profile_document(path)
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise BuildProfileError(f"Overlay must be a YAML mapping: {path}")
    return raw


def resolve_build_profile(
    profile_path: Path | None,
    preset: str | None,
    overlays: tuple[Path, ...] = (),
    set_pairs: tuple[str, ...] = (),
) -> tuple[BuildProfile, Path]:
    """The two fields most callers need from :func:`resolve_build_document`.

    Answers "what does this profile say, and where does it anchor". A caller
    that must also honor what resolution *derived* — which convention artifacts
    the profile excludes — wants :func:`resolve_build_document` instead; the
    build does, because an excluded artifact it copies anyway would shadow the
    framework's own version of that file.

    Returns:
        ``(profile, profile_dir)``. ``profile_dir`` is the profile ROOT — where
        every profile-relative path anchors, and what
        :meth:`BuildProfile.validate` resolves overlay/services lookups
        against. For a persona delta that is the directory above ``personas/``,
        never the delta's own parent; for preset mode it is the bundled
        ``profiles/presets/`` package directory.

    Raises:
        BuildProfileError: Whatever :func:`resolve_build_document` raises.
    """
    document = resolve_build_document(profile_path, preset, overlays, set_pairs)
    return document.profile, document.profile_dir


def resolve_build_document(
    profile_path: Path | None,
    preset: str | None,
    overlays: tuple[Path, ...] = (),
    set_pairs: tuple[str, ...] = (),
) -> LoadedProfile:
    """Resolve a build profile from a preset or a file, plus overlays and edits.

    Mode is determined by which of ``profile_path`` and ``preset`` is given;
    they are mutually exclusive and exactly one is required.

    ``overlays`` are the host-variant profiles ``osprey build`` selects
    (:mod:`osprey.cli.variant_selection`): inheritance layers, merged over the
    file through :func:`_deep_merge` before ``extends:`` is resolved, so string
    lists union and ``exclude:`` subtracts, as in any other layer. ``set_pairs``
    are edits, applied by :func:`apply_cli_edits` AFTER resolution: each
    replaces the value at the key it names.

    The multi-source counterpart of
    :func:`~osprey.cli.build_profile_load.load_profile_document`, and it returns
    the same record for the same reason: resolution knows two things the parsed
    :class:`BuildProfile` does not — the profile ROOT, and the convention
    artifacts the profile excludes — and both are things the build must act on.
    Returning the record rather than a widening tuple is what keeps the next
    thing resolution learns from breaking every callsite.

    Returns:
        The parsed, validated profile with its root and exclusion record.

    Raises:
        BuildProfileError: For mutual-exclusion violations, missing files,
        invalid YAML, a ``data:`` tree in preset mode, or validation failures.
    """
    if profile_path is not None and preset is not None:
        raise BuildProfileError("Pass either a profile path or --preset, not both.")
    if profile_path is None and preset is None:
        raise BuildProfileError("Either a profile path or --preset is required.")

    # A preset is one file in a shared package directory: it is never a persona
    # delta and carries no convention material to exclude (compute_preset_hash
    # folds none either, for the same reason).
    is_persona_delta = False
    excluded_artifacts: frozenset[str] = frozenset()
    # Which bundled preset the resolved document came through — the one thing
    # that still says which packaged data bundle its tree is from, now that the
    # preset-side `app_template:` is consumed during resolution.
    inherited_preset: str | None = None

    if preset is not None:
        raw, base_anchor = _load_preset_raw(preset)
        profile_dir = base_anchor.parent
        raw = _resolve_extends(raw, base_anchor)
        # In preset mode the named preset IS the nearest one; what it extends is
        # followed from there by `preset_data_bundle`.
        inherited_preset = preset
        # The command line edits the RESOLVED document, so a list it states is
        # the list the profile then holds rather than one more layer for the
        # inheritance merge to union with the preset's, and a subtree it names
        # outranks the deeper keys the preset or a parent spells beneath.
        raw = apply_cli_edits(raw, set_pairs)

        # Checked after extends resolution so no injection path escapes: the
        # preset itself, a --set pair, or an extends parent. A preset
        # has no profile directory to anchor a data tree against (profile_dir is
        # the bundled package dir), so carrying one is always a mistake.
        if raw.get("data") is not None:
            raise BuildProfileError(
                f"Profile key 'data' is not supported with --preset (got {raw['data']!r}). "
                f"A preset carries no profile directory to resolve the data tree against. "
                f"Materialize the preset first — 'osprey init DIR --preset {preset}' — "
                f"then build from that directory."
            )
    else:
        assert profile_path is not None  # narrows for type-checkers
        if not profile_path.exists():
            raise BuildProfileError(f"Profile not found: {profile_path}")
        raw = _read_profile_document(profile_path)
        if not isinstance(raw, dict):
            raise BuildProfileError(f"Profile must be a YAML mapping, got {type(raw).__name__}")
        for overlay_path in overlays:
            overlay = _read_overlay(overlay_path)
            if overlay is not None:
                raw = _deep_merge(raw, overlay)
        # Resolution goes through the one call that decides what a profile file
        # *means* — the same one the loader and the content hash make. A file
        # under `personas/` is a delta merged over the `profile.yml` beside it
        # and anchors at that root; resolving `extends` here instead, against
        # the file's own parent, would build a hollow project from the delta
        # alone and read its data tree from `personas/`.
        document = resolve_profile_document(raw, profile_path.resolve())
        raw, profile_dir = document.raw, document.root_dir
        is_persona_delta = document.is_persona_delta
        excluded_artifacts = document.excluded_artifacts
        inherited_preset = document.inherited_preset
        # The command line's edits go onto the fully resolved document — the
        # file, its variant overlay, an extends parent, a persona base, all
        # merged — because only here does the whole picture an edit outranks
        # exist: every literal dotted key any layer contributes.
        raw = apply_cli_edits(raw, set_pairs)

    profile = _parse_profile(raw)
    profile.inherited_preset = inherited_preset
    profile.validate(profile_dir)
    return LoadedProfile(
        profile=profile,
        profile_dir=profile_dir,
        is_persona_delta=is_persona_delta,
        excluded_artifacts=excluded_artifacts,
    )


def preset_authored_config(preset: str) -> dict[str, Any]:
    """The ``config:`` block a bundled preset writes DOWN ITSELF.

    The counterpart of what :func:`resolve_build_document` returns for the same
    preset: this is the layer read straight off the preset file, before
    ``extends`` folds its parents in, so a key here is one the preset's own
    author typed rather than one it inherits.

    A build never wants this — it wants the resolved answer, which is the only
    thing that describes what will be deployed. A *guard* sometimes does: the
    difference between "this file says read-only" and "this file resolves to
    read-only" is the whole question a check about accidental inheritance asks,
    and it cannot be recovered from the merged document.

    Args:
        preset: A bundled preset name, in either spelling
            (``control-assistant`` / ``control_assistant``).

    Returns:
        The preset's own ``config:`` mapping, empty when it declares none.
        Dotted keys, nested blocks, or both — exactly as written.

    Raises:
        BuildProfileError: If the preset is unknown or is not a YAML mapping.
    """
    raw, _base_anchor = _load_preset_raw(preset)
    config = raw.get("config")
    return config if isinstance(config, dict) else {}


# ---------------------------------------------------------------------------
# The profile a build reads
# ---------------------------------------------------------------------------

#: The profile file at a deployment repo's root. Kept here for
#: :mod:`~osprey.cli.deploy_scaffold_templates`, which renders the name into the
#: CI pipeline; :data:`osprey.cli.repo_resolver.PROFILE_FILENAME` is the same
#: name in its role as the discovery marker.
PROFILE_FILENAME = "profile.yml"


def write_back_cli_overrides(
    profile_path: Path,
    set_pairs: tuple[str, ...] = (),
    tier: int | None = None,
) -> list[str]:
    """Write ``osprey set``'s pairs into the profile, before it is read back.

    The profile is the source of truth, so an explicit override is an edit *of
    that profile* — made here and then read back by the ordinary resolution path
    like any other profile content. Nothing is layered at invocation time and
    thrown away afterwards. Reporting the edit is the caller's job: the keys are
    returned rather than announced, so the command that asked for the edit
    describes it in its own words.

    The edit **replaces** the value at each dotted key path. A value written
    into a file has to be the value the file then holds, or the profile stops
    describing the deployment. A first materialization makes the same edit to
    the document before writing it (:func:`apply_cli_edits`), so
    ``osprey init --set`` and ``osprey set`` land identically.

    ``config:`` is written the way a profile spells it — one mapping key holding
    the whole dotted path (``control_system.type``) rather than a nested map, so
    a write-back addresses the same rendered-config leaf the profile's own
    entries do. A mapping VALUE is the other case: ``config.a={…}`` states the
    whole value of ``a``, so it is written whole and the entries beneath ``a``
    are removed first.

    Args:
        profile_path: The ``profile.yml`` (or persona delta) being edited.
        set_pairs: The ``osprey set`` pairs.
        tier: Written as the profile's ``tier:`` key.

    Returns:
        The dotted key paths written, in write order; empty when there was
        nothing to write.

    Raises:
        BuildProfileError: If a ``--set`` pair is malformed.
        click.UsageError: If a pair sets ``extends``, which a materialized
            profile cannot have — the same refusal materialization makes, so the
            same edit is answered the same way on every build.
    """
    # Only what the CLI supplied is written — the profile's own content is
    # never rewritten as a side effect.
    layer = cli_edit_layer(set_pairs)
    if tier is not None:
        layer["tier"] = tier
    if "extends" in layer:
        raise click.UsageError(EXTENDS_OVERRIDE_REFUSAL)
    if not layer:
        return []

    updates = _flatten_override_layer(layer)
    _write_profile_values(profile_path, updates, claimed=_stated_config_paths(layer))
    written = [".".join(key_path) for key_path, _ in updates]
    # Debug, not info: the two callers both report the write in their own words
    # — ``osprey set`` prints the keys it wrote, and a build prints them as part
    # of its render summary — from the list returned below. Announcing it here
    # too put the same sentence on the operator's screen twice.
    logger.debug(
        "Wrote %d override(s) into %s: %s",
        len(written),
        profile_path,
        ", ".join(written),
    )
    return written


def _flatten_override_layer(layer: dict[str, Any]) -> list[tuple[list[str], Any]]:
    """Flatten a CLI edit layer into ``(key_path, value)`` writes.

    Descends the nesting key dots spelled, so ``a.b=1`` touches only the leaf
    ``a.b``; scalars, lists and mapping VALUES (:class:`_MappingValue`) are
    written whole at the key named — which is exactly :func:`_dotted_leaves`,
    so the descent is done there rather than a second time here.

    ``config:`` is the one block that does not nest: its keys are dotted paths
    into the *rendered* config, held as single mapping keys. Its interior is
    therefore flattened into ONE such key (``config`` → ``a.b``) instead of into
    further profile levels. Only a top-level ``config`` means that — a ``config``
    key nested under something else addresses no rendered config — which is why
    the split is made here, over the layer's own keys, rather than inside the
    recursion.
    """
    flat: list[tuple[list[str], Any]] = []
    for key, value in layer.items():
        name = str(key)
        if name == "config" and isinstance(value, dict) and value:
            leaves = [
                (("config", ".".join(sub_path)), leaf) for sub_path, leaf in _dotted_leaves(value)
            ]
        else:
            leaves = _dotted_leaves({name: value})
        flat.extend((list(key_path), leaf) for key_path, leaf in leaves)
    return flat


def _dotted_leaves(
    mapping: dict[str, Any], prefix: tuple[str, ...] = ()
) -> list[tuple[tuple[str, ...], Any]]:
    """Every leaf of ``mapping`` as a ``(path_segments, value)`` pair.

    Key nesting (a plain dict) is descended; a mapping given as a value
    (:class:`_MappingValue`) is a leaf and comes back as a plain dict.
    """
    leaves: list[tuple[tuple[str, ...], Any]] = []
    for key, value in mapping.items():
        path = (*prefix, str(key))
        if isinstance(value, _MappingValue):
            leaves.append((path, dict(value)))
        elif isinstance(value, dict) and value:
            leaves.extend(_dotted_leaves(value, path))
        else:
            leaves.append((path, value))
    return leaves


#: Top-level shorthand keys and the rendered-config key each stands for.
_SHORTHAND_CONFIG_KEYS: tuple[tuple[str, str], ...] = (
    (CONNECTOR_PROFILE_KEY, CONNECTOR_CONFIG_KEY),
    (PORT_BASE_PROFILE_KEY, PORT_BASE_CONFIG_KEY),
)


def _write_profile_values(
    profile_path: Path,
    updates: list[tuple[list[str], Any]],
    claimed: list[tuple[str, ...]] = (),
) -> None:
    """Set each ``key_path`` in ``profile_path`` to its value, keeping comments.

    ``claimed`` are the rendered-config paths the edit names
    (:func:`_stated_config_paths`); every ``config:`` entry beneath one of them
    is removed before the new leaves are written, the same prune
    :func:`apply_cli_edits` makes in memory — so ``osprey set
    config.approval.tools={…}`` leaves the file holding the ``approval.tools.*``
    entries it spelled and none of the ones it replaced.

    Uses the shared round-trip YAML handle rather than a private one: the
    profile is a hand-edited, heavily commented document, and a second handle
    with a different line width would silently re-wrap the whole file on the
    first write-back.

    Rendered to text first and then written by :func:`_atomic_write_bytes`, so
    the document either lands whole or not at all: this file is the facility's
    source of truth, and a truncate-in-place write interrupted halfway would
    leave it neither the profile it was nor the one it was becoming.
    """
    from ruamel.yaml import CommentedMap

    from osprey.utils.config_writer import _yaml, load_config_document

    data = load_config_document(profile_path)
    config = data.get(_CONFIG_SET_PREFIX)
    if claimed and isinstance(config, dict):
        for key in _shadowed_config_keys(config, claimed):
            del config[key]
    for key_path, value in updates:
        node = data
        for segment in key_path[:-1]:
            if not isinstance(node.get(segment), dict):
                node[segment] = CommentedMap()
            node = node[segment]
        node[key_path[-1]] = value
    # An edit that lands on the config key a top-level shorthand stands for
    # retires the shorthand: left beside it, the parse-time fold would restore
    # the old value on the next read and the edit would silently not stick.
    written = {".".join(key_path) for key_path, _ in updates}
    for shorthand, config_key in _SHORTHAND_CONFIG_KEYS:
        if f"{_CONFIG_SET_PREFIX}.{config_key}" in written and shorthand in data:
            del data[shorthand]
    rendered = io.StringIO()
    _yaml.dump(data, rendered)
    _atomic_write_bytes(profile_path, rendered.getvalue().encode("utf-8"))


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Replace ``path`` with ``payload`` via a same-directory temp file.

    The same shape as the manifest and ``.env`` writers this feature added, for
    the same reason: a reader (or a crash) never sees a half-written file, and
    the previous contents survive any failure before the ``os.replace``. The
    existing file's mode is carried over so an atomic rewrite does not quietly
    re-permission the profile.
    """
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        if path.exists():
            os.chmod(tmp_name, path.stat().st_mode & 0o7777)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
