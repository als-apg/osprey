"""Multi-source profile resolution: preset / file + overlays + ``--set``.

The entry point ``osprey init``'s materialization and ``osprey validate`` call.
Picks the base layer
(bundled preset or on-disk file), deep-merges override files and ``--set``
values over it, then hands the assembled raw dict to ``extends`` resolution and
:func:`osprey.cli.build_profile_load._parse_profile`. Also owns the ``--set``
mini-parser, the top-level shorthand keys (model selection plus ``connector``)
whose explicit use is recorded in the build manifest, and the rule that lets a
``config.*`` key stated on the command line outrank the deeper keys some other
layer spells beneath it, instead of losing to them by key order.

The profile is the source of truth, so an explicit override *is* a profile edit:
:func:`write_back_cli_overrides` turns ``osprey set``'s pairs into a
comment-preserving edit of the repo's own ``profile.yml``, which the ordinary
resolution path then reads back like any other profile content. Nothing is
layered at invocation time and thrown away afterwards.
"""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import Any

import click
import yaml

from osprey.errors import BuildProfileError
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


def _parse_set_pairs(pairs: tuple[str, ...]) -> dict[str, Any]:
    """Parse ``--set KEY.PATH=VALUE`` pairs into a nested dict.

    The right-hand side is parsed with ``yaml.safe_load`` so callers get
    type coercion for free: ``true``/``false`` -> bool, ``[a,b]`` -> list,
    bare ints/floats -> numeric, anything else -> string.

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

    A command line is the last word, so the CLI's key wins the whole subtree it
    names: every inherited key BENEATH it (segment-prefix match, so
    ``approval.tools`` claims ``approval.tools.channel_read`` and not
    ``approval.tools_extra``) is dropped, and the value the operator just gave
    is the one that renders. A key the CLI states itself is never dropped —
    ``--set config.a=… --set config.a.b=…`` is one layer refining its own key,
    which the emitter's prefix collapse folds deeper-key-wins.

    Run on the FULLY resolved raw, after ``extends``: a parent's literal dotted
    key is inherited content just as surely as the base preset's, and the paths
    the CLI stated are carried across resolution to meet it here.

    Args:
        raw: Resolved raw profile dict.
        stated: Paths the CLI layers stated, from :func:`_stated_config_paths`.

    Returns:
        ``raw`` when nothing is shadowed, else a copy with a pruned ``config``.
    """
    config = raw.get(_CONFIG_SET_PREFIX)
    if not stated or not isinstance(config, dict):
        return raw
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
    says which spelling wins. Scoped to the layers the caller passed on the
    command line (``-O`` files and ``--set`` pairs): a preset or ``extends``
    parent that already sets the literal key is exactly what the shorthand is
    for overriding, and must keep working.

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


def merge_cli_overrides(
    base: dict[str, Any],
    overrides: tuple[Path, ...],
    set_pairs: tuple[str, ...],
    *,
    stated_config_paths: list[tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    """Layer ``-O`` override files and ``--set`` pairs over ``base``.

    The shared CLI layering step: override files deep-merge in declaration
    order, then ``--set`` values merge on top. Used by the project-render path
    (:func:`resolve_build_profile`) and by ``osprey init``, which bakes
    the merged result into the materialized ``profile.yml``.

    The ``connector`` shorthand is folded into ``config`` here rather than at
    parse time alone, so the profile ``osprey init`` bakes states the
    connector at the one literal key a reader would edit
    (``control_system.type``) instead of carrying a shorthand that silently
    outranks the ``config:`` block printed beside it. The fold happens after
    all layers merge, so the last layer to name the connector wins.

    Args:
        base: The base layer — bundled preset raw or profile-file raw.
        overrides: ``-O`` override files, deep-merged in declaration order.
        set_pairs: ``--set KEY=VALUE`` pairs, merged last.
        stated_config_paths: Optional collector, extended with every
            rendered-config path the CLI layers state
            (:func:`_stated_config_paths`). The layering step is where that
            provenance exists — afterwards a CLI key is indistinguishable from
            an inherited one — but what it shadows can only be judged against
            the fully-merged raw, so the caller carries the list across
            ``extends`` resolution and prunes there, once.

    Raises:
        BuildProfileError: On a missing or non-mapping override file, an
            unparseable ``--set`` pair, an invalid ``connector`` value, or CLI
            layers naming both ``connector`` and ``config.control_system.type``.
    """
    raw = base
    # The layers the caller passed on the command line, kept apart from ``base``:
    # the connector conflict is about what one command line states twice.
    cli_layers: list[dict[str, Any]] = []
    for override_path in overrides:
        if not override_path.exists():
            raise BuildProfileError(f"Override not found: {override_path}")
        override_raw = _read_profile_document(override_path)
        if override_raw is None:
            continue
        if not isinstance(override_raw, dict):
            raise BuildProfileError(f"Override must be a YAML mapping: {override_path}")
        cli_layers.append(override_raw)
        raw = _deep_merge(raw, override_raw)

    if set_pairs:
        set_raw = _parse_set_pairs(set_pairs)
        cli_layers.append(set_raw)
        raw = _deep_merge(raw, set_raw)

    _reject_connector_type_conflict(cli_layers)
    if stated_config_paths is not None:
        for layer in cli_layers:
            stated_config_paths.extend(_stated_config_paths(layer))
    return _apply_port_base_shorthand(_apply_connector_shorthand(raw))


def resolve_build_profile(
    profile_path: Path | None,
    preset: str | None,
    overrides: tuple[Path, ...] = (),
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
    document = resolve_build_document(profile_path, preset, overrides, set_pairs)
    return document.profile, document.profile_dir


def resolve_build_document(
    profile_path: Path | None,
    preset: str | None,
    overrides: tuple[Path, ...] = (),
    set_pairs: tuple[str, ...] = (),
) -> LoadedProfile:
    """Resolve a build profile from any combination of preset / file / overlays.

    Mode is determined by which of ``profile_path`` and ``preset`` is given;
    they are mutually exclusive and exactly one is required.

    Layers are applied in order: base -> override file(s) -> --set values.
    All layers are merged via :func:`_deep_merge` (string lists union-dedup,
    other lists concatenate) before ``extends:`` is resolved.

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
        stated: list[tuple[str, ...]] = []
        raw = merge_cli_overrides(raw, overrides, set_pairs, stated_config_paths=stated)
        raw = _resolve_extends(raw, base_anchor)
        # In preset mode the named preset IS the nearest one; what it extends is
        # followed from there by `preset_data_bundle`.
        inherited_preset = preset
        # Same pruning as the profile-file branch below, run after extends
        # resolution so a parent's literal dotted key is shadowed too.
        raw = _drop_shadowed_config_keys(raw, stated)

        # Checked after extends resolution so no injection path escapes: the
        # preset itself, a -O file, a --set pair, or an extends parent. A preset
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
        stated_file: list[tuple[str, ...]] = []
        raw = merge_cli_overrides(raw, overrides, set_pairs, stated_config_paths=stated_file)
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
        # One prune, after full resolution, because only here does the whole
        # picture exist: the paths the CLI stated (provenance, from before the
        # merge) and every literal dotted key any layer — the file, a -O
        # overlay, an extends parent, a persona base — contributes (after it).
        raw = _drop_shadowed_config_keys(raw, stated_file)

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
    overrides: tuple[Path, ...] = (),
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
    describing the deployment. (A first materialization still bakes its layers
    in through :func:`merge_cli_overrides`, so ``osprey init -O``/``--set`` on a
    *fresh* repo keeps layering semantics.)

    ``config:`` is written the way a profile spells it — one mapping key holding
    the whole dotted path (``control_system.type``) rather than a nested map, so
    a write-back addresses the same rendered-config leaf the profile's own
    entries do instead of wholesale-replacing a config subtree.

    Args:
        profile_path: The ``profile.yml`` (or persona delta) being edited.
        overrides: ``-O`` files, deep-merged in declaration order. Nothing in
            the shipped CLI passes them: ``osprey init -O`` layers at
            materialization instead (see :func:`merge_cli_overrides`), so this
            is API surface, not a live path.
        set_pairs: The ``osprey set`` pairs — the only argument its one
            production caller passes.
        tier: Written as the profile's ``tier:`` key. Caller-less for the same
            reason as *overrides*.

    Returns:
        The dotted key paths written, in write order; empty when there was
        nothing to write.

    Raises:
        BuildProfileError: If an override file is missing, unreadable, or not a
            YAML mapping, or a ``--set`` pair is malformed.
        click.UsageError: If a layer sets ``extends``, which a materialized
            profile cannot have — the same refusal materialization makes, so the
            same override file is answered the same way on every build.
    """
    # Layered against an empty base so only what the CLI supplied is written —
    # the profile's own content is never rewritten as a side effect.
    layer = merge_cli_overrides({}, overrides, set_pairs)
    if tier is not None:
        layer["tier"] = tier
    if "extends" in layer:
        raise click.UsageError(EXTENDS_OVERRIDE_REFUSAL)
    if not layer:
        return []

    updates = _flatten_override_layer(layer)
    _write_profile_values(profile_path, updates)
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
    """Flatten a CLI override layer into ``(key_path, value)`` leaf writes.

    Descends mappings so an override touches only the leaf it names; scalars,
    lists and empty mappings are leaves — which is exactly
    :func:`_dotted_leaves`, so the descent is done there rather than a second
    time here.

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
    """Every leaf of ``mapping`` as a ``(path_segments, value)`` pair."""
    leaves: list[tuple[tuple[str, ...], Any]] = []
    for key, value in mapping.items():
        path = (*prefix, str(key))
        if isinstance(value, dict) and value:
            leaves.extend(_dotted_leaves(value, path))
        else:
            leaves.append((path, value))
    return leaves


def _write_profile_values(profile_path: Path, updates: list[tuple[list[str], Any]]) -> None:
    """Set each ``key_path`` in ``profile_path`` to its value, keeping comments.

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
    for key_path, value in updates:
        node = data
        for segment in key_path[:-1]:
            if not isinstance(node.get(segment), dict):
                node[segment] = CommentedMap()
            node = node[segment]
        node[key_path[-1]] = value
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
