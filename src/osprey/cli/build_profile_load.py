"""YAML-to-``BuildProfile`` parsing and the single-file loader.

Turns a raw profile mapping into the typed dataclasses — validating the
per-server and per-section shapes that only the parser can see (MCP
command/url/port exclusivity, mapping-vs-scalar sections) and rejecting keys
the schema does not define, both at the top level and inside the closed
``environment:`` and ``bluesky:`` blocks. :func:`load_profile` is the
plain single-file entry point; the preset/override/``--set`` layering path in
:mod:`osprey.cli.build_profile_resolve` reuses :func:`_parse_profile` after it has
assembled its own raw dict.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from osprey.errors import BuildProfileError
from osprey.port_layout import (
    DEFAULT_PORT_BASE,
    PORT_BASE_CONFIG_KEY,
    default_port,
    resolve_port_base,
)
from osprey_connectors.types import SET_CONTROL_SYSTEM_TYPES

from .build_profile_archiver import parse_va_archiver_block
from .build_profile_deploy import parse_deploy_block
from .build_profile_document import _normalize_profile_aliases, _read_profile_document
from .build_profile_merge import resolve_profile_document
from .build_profile_model import BuildProfile
from .build_profile_schema import (
    BlueskyConfig,
    BlueskyWebConfig,
    DispatchConfig,
    EnvConfig,
    EnvironmentConfig,
    GChatBridgeProfileConfig,
    LifecycleConfig,
    LifecycleStep,
    McpServerDef,
    NextcloudBridgeProfileConfig,
    ProfileProvenance,
    ServiceDef,
    VAConfig,
)


@dataclass(frozen=True)
class LoadedProfile:
    """A validated profile plus what resolving it revealed.

    Attributes:
        profile: The parsed, validated profile.
        profile_dir: The profile ROOT — where every relative path in
            ``profile`` anchors. For a persona delta this is the directory
            above ``personas/``, so a persona reads the same data tree and
            convention directories as the profile it lives under.
        is_persona_delta: Whether this file was resolved as a persona delta.
        excluded_artifacts: Convention-dir artifacts to omit, as
            ``"<source>/<name>"``. The build skips these files and derives
            ownership from what remains.
    """

    profile: BuildProfile
    profile_dir: Path
    is_persona_delta: bool = False
    excluded_artifacts: frozenset[str] = field(default_factory=frozenset)


def load_profile(path: Path) -> BuildProfile:
    """Load a build profile from YAML.

    Args:
        path: Path to the profile YAML file.

    Returns:
        Parsed and validated BuildProfile.

    Raises:
        BuildProfileError: If the file is invalid or validation fails.
    """
    return load_profile_document(path).profile


def load_profile_document(path: Path) -> LoadedProfile:
    """Load a build profile, keeping what resolution derived alongside it.

    :func:`load_profile` is the answer to "what does this profile say"; this is
    the answer to "what should the build do with this file", which is more than
    the model carries. Two things resolution knows and :class:`BuildProfile`
    does not: the profile ROOT (for a persona delta, the directory above
    ``personas/`` — every relative path anchors there, not at the file's own
    parent), and the convention artifacts the profile excludes, which the build
    omits and ownership derivation scans post-exclusion.

    Args:
        path: Path to the profile YAML file.

    Returns:
        The parsed, validated profile with its root and exclusion record.

    Raises:
        BuildProfileError: If the file is invalid or validation fails.
    """
    if not path.exists():
        raise BuildProfileError(f"Profile not found: {path}")

    raw = _read_profile_document(path)

    if not isinstance(raw, dict):
        raise BuildProfileError(f"Profile must be a YAML mapping, got {type(raw).__name__}")

    document = resolve_profile_document(raw, path)

    profile = _parse_profile(document.raw)
    profile.validate(document.root_dir)
    return LoadedProfile(
        profile=profile,
        profile_dir=document.root_dir,
        is_persona_delta=document.is_persona_delta,
        excluded_artifacts=document.excluded_artifacts,
    )


# Minimum OSPREY release that understands the current profile schema — the one
# that ships ``app_template:`` and ``data:``. Emitted profiles stamp
# ``requires_osprey_version`` from this constant (as ``>=<value>``) rather than
# from the running ``__version__``: the stamp must name the floor a *reader*
# needs, and a dynamic stamp would let the emitting release satisfy its own
# gate while silently ignoring the keys it just wrote. Pinned by test — bump it
# only when a release adds profile keys older releases cannot honor.
_PROFILE_SCHEMA_MIN_OSPREY = "2026.9.0"


# Top-level keys recognized by BuildProfile. Anything else is almost certainly
# a typo of one of these (e.g. mcp_server vs mcp_servers).
_KNOWN_PROFILE_KEYS = frozenset(
    {
        "name",
        "extends",
        "exclude",
        "data_bundle",
        # YAML-surface spelling of data_bundle, never reached by the check
        # itself: _parse_profile normalizes it away before _reject_unknown_keys
        # runs. It is a member because this frozenset is also the "valid keys
        # are:" list in that check's error, where dropping the spelling users
        # are told to write would name it invalid.
        "app_template",
        "data",
        "deploy",
        "deploy_services",
        # Shorthand for config's `control_system.type`, consumed by
        # _apply_connector_shorthand before the profile is parsed. Listed here
        # because a materialized or hand-written profile may spell it, and
        # because this frozenset doubles as the "valid keys are:" list.
        "connector",
        # Shorthand for config's `deployment.port_base`, consumed by
        # _apply_port_base_shorthand — same contract as `connector` above.
        "port_base",
        "provider",
        "model",
        "channel_finder_mode",
        "tier",
        "config",
        "mcp_servers",
        "services",
        "lifecycle",
        "env",
        "environment",
        "dependencies",
        "requires_osprey_version",
        "osprey_install",
        "python_env",
        "hooks",
        "rules",
        "skills",
        "agents",
        "output_styles",
        "web_panels",
        "default_panel",
        "panel_presets",
        "claude_md_template",
        "artifact_server",
        "dispatch",
        "bluesky",
        "virtual_accelerator",
        "bluesky_web",
        "nextcloud_bridge",
        "gchat_bridge",
        "va_archiver",
        "provenance",
    }
)


# Keys recognized inside the ``environment:`` block. Rejected outright like the
# top-level schema (see _reject_unknown_keys): the block is small and closed,
# and a silently ignored ``package:`` would leave the built environment missing
# what the facility asked for.
_KNOWN_ENVIRONMENT_KEYS = frozenset({"python", "packages", "inherit_exclude"})


# Keys recognized inside the ``bluesky:`` block, derived from the dataclass the
# block is parsed into rather than listed by hand: every field below is read by
# _parse_profile, so the check can never drift from what the parser honors when
# a field is added or renamed. Rejected like the top-level schema — a dropped
# `tiled_enabld:` would ship a bridge without the catalog the facility asked
# for, and a knob a release removed (e.g. `demo_runner:`) has to announce its
# removal instead of being silently ignored.
_KNOWN_BLUESKY_KEYS = frozenset(f.name for f in fields(BlueskyConfig))


# Keys recognized inside the ``dispatch:`` block, derived from its dataclass for
# the same reason the bluesky set is. A dropped key here is expensive: a
# misspelled `netwrok: host` would leave the dispatcher and its workers on the
# default bridge network, and the deployment would come up looking healthy while
# unreachable from the address the facility asked for.
_KNOWN_DISPATCH_KEYS = frozenset(f.name for f in fields(DispatchConfig))


# Keys recognized inside the ``virtual_accelerator:`` block, derived from its
# dataclass like the two sets above. What a dropped key costs here is a
# deployment that quietly stays on simulated channels: a misspelled
# `live_standin` leaves the stand-in unbuilt, so the deployment never gains its
# `standin` target — `control_target_set standin` has no machine to point at,
# and a session meant for the soft IOC runs against the same lane the operator
# was already on.
_KNOWN_VA_KEYS = frozenset(f.name for f in fields(VAConfig))


def _profile_port_base(raw: dict[str, Any]) -> int:
    """Return the port base this profile's ``config:`` block resolves to.

    The one rule the port layout runs on is that a port is derived from the base
    the deployment actually resolved, never from the layout's own default — so a
    loader that places a service by slot has to read the profile's own
    ``deployment.port_base`` first. A ``config:`` block is a flat bag of dotted
    keys that may spell that path at any depth, which is what
    :func:`~osprey.cli.build_profile_emit.effective_config_subtree` folds into
    one subtree; re-wrapping the result under ``deployment`` keeps
    :func:`~osprey.port_layout.resolve_port_base` on its single input shape, so
    the range refusal fires here too.

    Args:
        raw: The fully-merged raw profile mapping — presets, ``extends``
            parents, ``-O`` layers and ``--set`` pairs already folded in.

    Returns:
        The base the profile configures, or
        :data:`~osprey.port_layout.DEFAULT_PORT_BASE` when it configures none.

    Raises:
        ValueError: If the profile names a base whose thousand-port block could
            not exist (below 1024, or running past port 65535).
    """
    # Imported inside the function on purpose: build_profile_emit imports this
    # module at import time, so a module-level import would close the cycle.
    from .build_profile_emit import effective_config_subtree

    config = raw.get("config")
    if not isinstance(config, Mapping):
        return DEFAULT_PORT_BASE
    # Only the keys that address `deployment` are folded. Handing the whole
    # block to the folder would make a prefix conflict anywhere in it — a
    # scalar `env:` beside an `env.required:`, say — a refusal raised here, at
    # profile parse, on behalf of a key this function never reads. Those
    # conflicts belong to the axis that owns the key and are reported there.
    deployment_keys = {
        key: value
        for key, value in config.items()
        if key == "deployment" or (isinstance(key, str) and key.startswith("deployment."))
    }
    return resolve_port_base(
        {"deployment": effective_config_subtree(deployment_keys, ("deployment",))}
    )


def _parse_live_standin(value: Any, base: int) -> int | None:
    """Normalise ``virtual_accelerator.live_standin`` to a port or ``None``.

    ``true`` is the spelling a profile should use: it asks for the stand-in
    without naming a number, and the number it gets is the layout's
    ``va_standin`` slot on *this deployment's* base, so two deployments on one
    host never collide over it. An explicit integer is still honoured — a
    facility may have to place the second soft-IOC somewhere specific — and the
    field stays an ``int | None`` either way, so nothing downstream has to know
    which spelling was used.

    ``false`` is refused rather than read as "off": a profile that inherits the
    key from a preset switches the stand-in off by excluding the key, and
    silently accepting ``false`` here would leave two spellings for absence.

    Args:
        value: The raw ``live_standin`` value, or ``None`` when unset.
        base: The base the profile resolved, from :func:`_profile_port_base`.

    Returns:
        The Channel Access port of the stand-in, or ``None`` when the profile
        does not deploy one.

    Raises:
        BuildProfileError: If the value is ``false`` or is neither ``true`` nor
            an integer.
    """
    if value is None:
        return None
    if value is True:
        return default_port("va_standin", base=base)
    if value is False:
        raise BuildProfileError(
            "virtual_accelerator.live_standin: false is not a way to switch the "
            "stand-in off. Write `true` to deploy it on the layout's stand-in port, "
            "or omit the key (exclude it, if a parent profile sets it) to deploy no "
            "stand-in at all."
        )
    if not isinstance(value, int):
        raise BuildProfileError(
            "virtual_accelerator.live_standin must be `true` — the layout's stand-in "
            f"port on this deployment's base — or a Channel Access port number (got {value!r})"
        )
    return value


def _parse_environment(raw: dict[str, Any]) -> EnvironmentConfig:
    """Parse the raw ``environment:`` block into an :class:`EnvironmentConfig`.

    Structural problems (wrong container types, unknown keys) raise here;
    semantic ones (interpreter must exist, inherit_exclude needs a venv base)
    are accumulated by :meth:`BuildProfile._validate_environment`.

    Args:
        raw: The full raw profile dict.

    Returns:
        The parsed config — all defaults when the block is absent or ``null``.

    Raises:
        BuildProfileError: If the block or one of its fields has the wrong
            shape, or names an unknown key.
    """
    block = raw.get("environment") or {}
    if not isinstance(block, dict):
        raise BuildProfileError(
            f"Profile 'environment' must be a mapping (got {type(block).__name__})"
        )

    _reject_unknown_block_keys(block, _KNOWN_ENVIRONMENT_KEYS, "environment")

    python = block.get("python")
    if python is not None and not isinstance(python, str):
        raise BuildProfileError(
            f"environment.python must be a string path (got {type(python).__name__})"
        )

    lists: dict[str, list[str]] = {}
    for key in ("packages", "inherit_exclude"):
        value = block.get(key, [])
        if not isinstance(value, list):
            raise BuildProfileError(
                f"environment.{key} must be a list of strings (got {type(value).__name__})"
            )
        lists[key] = list(value)

    return EnvironmentConfig(
        python=python,
        packages=lists["packages"],
        inherit_exclude=lists["inherit_exclude"],
    )


def _reject_unknown_block_keys(keys: Iterable[str], known_keys: frozenset[str], label: str) -> None:
    """Reject unrecognized keys in a closed profile block, naming every one at once.

    The shared body behind every closed-schema check, so an operator meets one
    wording whichever block they mistyped.

    Args:
        keys: The keys present in the block being checked.
        known_keys: The block's closed key set.
        label: What the block is called in the message (``profile`` for the
            top level, otherwise the block's own key, e.g. ``bluesky``).

    Raises:
        BuildProfileError: If any key is unrecognized. The message names every
            offender, its closest known spelling, and the full valid set.
    """
    unknown = sorted(set(keys) - known_keys)
    if not unknown:
        return

    known = sorted(known_keys)
    named = []
    for key in unknown:
        close = difflib.get_close_matches(key, known, n=1)
        named.append(f"{key!r} (did you mean {close[0]!r}?)" if close else repr(key))
    raise BuildProfileError(
        f"Unknown {label} key(s): {', '.join(named)}. "
        f"Remove or correct them — valid keys are: {', '.join(known)}."
    )


def _reject_unknown_keys(raw: dict[str, Any]) -> None:
    """Reject unknown top-level profile keys, naming every one at once.

    A key the schema does not define is a facility asking for something the
    build will never do; ignoring it with a warning buries that in the log and
    ships a deployment missing what the profile asked for.

    Args:
        raw: The resolved raw profile dict (``extends``/``exclude`` already
            consumed, though both stay allowlisted for pre-resolution callers).

    Raises:
        BuildProfileError: If any key is unrecognized.
    """
    _reject_unknown_block_keys(raw.keys(), _KNOWN_PROFILE_KEYS, "profile")


#: The ``claude_code.permissions`` keys whose value is a list of tool matchers.
#:
#: Every one of them is consumed by *iterating* it: ``settings.json.j2`` renders
#: each element as one JSON array entry, and the deny composition subtracts with
#: ``d not in remove_deny`` over the raw value
#: (:func:`osprey.cli.templates.claude_code._rendered_deny_list`). A value that
#: is not a list of strings therefore does not mean what its author meant — a
#: bare string iterates as its characters, and ``in`` against it is *substring*
#: containment. Refusing the shape here is what lets every later reader assume a
#: list and still agree with the render and with each other; see
#: :func:`osprey.cli.profile_conventions.permission_entries`, which reads a
#: non-list as no entries precisely because this refusal stands in front of it.
_PERMISSION_LIST_KEYS: tuple[str, ...] = ("allow", "ask", "deny", "remove_ask", "remove_deny")

#: The ``config:`` key the Claude Code block lives under, and the dotted path to
#: its permissions mapping. Both spellings of the block reach the same rendered
#: ``config.yml`` keys, so both are checked.
_CLAUDE_CODE_CONFIG_KEY = "claude_code"
_PERMISSIONS_CONFIG_PATH = f"{_CLAUDE_CODE_CONFIG_KEY}.permissions"


def _permission_lists(config: dict[str, Any]) -> list[tuple[str, Any]]:
    """Every permissions list a ``config:`` block spells, in each of its spellings.

    A ``config:`` block is a flat bag of dotted keys, but ``config_update_fields``
    accepts three ways of reaching the same rendered leaf: the fully dotted key
    (``claude_code.permissions.deny``), a dotted key carrying a mapping
    (``claude_code.permissions:`` with ``deny:`` under it), and the fully nested
    ``claude_code:`` mapping. A shape check that saw only the flattest one would
    wave the other two straight through to the render.

    Args:
        config: The profile's ``config:`` block.

    Returns:
        ``(label, value)`` pairs, where the label names the key AND the spelling
        it was written in, so a refusal points at the line the author wrote.
    """
    found: list[tuple[str, Any]] = []
    for key in _PERMISSION_LIST_KEYS:
        dotted = f"{_PERMISSIONS_CONFIG_PATH}.{key}"
        if dotted in config:
            found.append((dotted, config[dotted]))
    sources = (
        (
            config.get(_PERMISSIONS_CONFIG_PATH),
            f"nested under the {_PERMISSIONS_CONFIG_PATH!r} mapping",
        ),
        (
            block.get("permissions")
            if isinstance(block := config.get(_CLAUDE_CODE_CONFIG_KEY), dict)
            else None,
            f"nested under the {_CLAUDE_CODE_CONFIG_KEY + ':'!r} mapping",
        ),
    )
    for permissions, spelling in sources:
        if not isinstance(permissions, dict):
            continue
        for key in _PERMISSION_LIST_KEYS:
            if key in permissions:
                found.append((f"{_PERMISSIONS_CONFIG_PATH}.{key} ({spelling})", permissions[key]))
    return found


def _misshapen_permission_list(value: Any) -> str | None:
    """Why *value* cannot serve as a permissions list, or ``None`` when it can.

    Args:
        value: The value written under one of :data:`_PERMISSION_LIST_KEYS`.

    Returns:
        A phrase completing "…, but <reason>", or ``None`` if the value is a
        list of non-empty strings.
    """
    if isinstance(value, str):
        return (
            f"it is a bare string ({value!r}). The render iterates the value, so this "
            f"renders one entry per character and names no tool at all"
        )
    if not isinstance(value, list):
        return f"it is a {type(value).__name__} ({value!r})"
    bad = [entry for entry in value if not isinstance(entry, str) or not entry.strip()]
    if bad:
        shown = ", ".join(repr(entry) for entry in bad[:3])
        more = f" (and {len(bad) - 3} more)" if len(bad) > 3 else ""
        return f"it contains {shown}{more}, which is not a non-empty tool-matcher string"
    return None


def _reject_permission_list_shapes(config: Any) -> None:
    """Refuse a ``config:`` block whose permission lists are not lists of strings.

    The one place a loose spelling of ``claude_code.permissions.deny`` and
    friends is caught, so that everything downstream may assume the shape. Three
    readers compose these lists — the ``settings.json`` render, the container's
    setup-capability check
    (``build_cmd._profile_setup_patch_capable``), and the persona predicate
    :func:`osprey.cli.profile_conventions.is_setup_patch_capable` — and a
    non-list makes them disagree in *both* directions at once: a bare-string
    ``deny`` renders 34 single-character deny entries and denies the tool
    nowhere, while a bare-string ``remove_deny`` lifts by substring and can drop
    a deny nobody named. There is no reading that is right for all three, so the
    profile is refused instead of guessed at.

    Runs on the fully-resolved ``config:`` block, so no source escapes: the
    preset, an ``-O`` overlay, a ``--set`` pair, an ``extends`` parent or a
    persona base.

    Args:
        config: The profile's ``config:`` block (ignored when not a mapping —
            ``BuildProfile.validate`` rejects that by name).

    Raises:
        BuildProfileError: naming every offending key at once, in the spelling
            it was written in, with the shape that was expected.
    """
    if not isinstance(config, dict):
        return
    problems = [
        f"  {label} — {reason}"
        for label, value in _permission_lists(config)
        if (reason := _misshapen_permission_list(value))
    ]
    if not problems:
        return
    keys = ", ".join(repr(key) for key in _PERMISSION_LIST_KEYS)
    raise BuildProfileError(
        "Misshapen Claude Code permission list(s) in the profile's config: block:\n"
        + "\n".join(problems)
        + f"\n\nEach of {_PERMISSIONS_CONFIG_PATH}.<{keys}> must be a YAML list of "
        "non-empty tool-matcher strings. Write it as a list — '[]' or no key at all "
        "for none:\n"
        "  config:\n"
        f"    {_PERMISSIONS_CONFIG_PATH}.deny:\n"
        "      - mcp__osprey_workspace__setup_patch\n"
        "The build will not guess at a looser spelling: settings.json renders these "
        "lists by iterating them and subtracts remove_deny with 'not in', so a bare "
        "string renders one entry per character and lifts denies by substring."
    )


def _reject_mixed_claude_code_spellings(config: Any) -> None:
    """Refuse a ``config:`` block that addresses one ``claude_code`` path twice over.

    Two keys where one path-PREFIXES the other are different dict keys, so both
    survive every merge — and then one of them is silently discarded.
    ``config_update_fields`` applies keys in iteration order and sets each
    addressed path verbatim (``node[leaf] = value``), so the shallower key
    applied second REPLACES the whole subtree the deeper one just wrote;
    :func:`~osprey.cli.build_profile_emit._collapse_config_prefixes` folds the
    same pair the other way, deeper-key-wins. Which of the author's two values
    reaches ``config.yml`` depends on where they left the lines.

    Prefixing is by SEGMENT and the rule covers every split point, not just the
    bare ``claude_code:`` mapping it was originally written for. The middle
    spelling is the one that got away: ``claude_code.permissions:`` holding a
    ``deny`` list beside a dotted ``claude_code.permissions.deny`` is exactly
    the same hazard — measured, it renders whichever of the two comes last —
    and the guard that only looked for the bare key let it through.

    That is the same hazard :func:`~osprey.cli.build_profile_resolve._reject_set_config_collisions`
    refuses for a ``--set config.*`` path, raised to the whole ``config:`` block:
    that guard only sees paths ``--set`` addressed, so a nested mapping arriving
    from a preset, an ``-O`` overlay or an ``extends`` parent went unrefused.

    It is a privilege question and not only a tidiness one. The container's
    setup-capability check reads BOTH spellings and unions them, which is exact
    only while at most one of them is present: a profile carrying
    ``claude_code.permissions.remove_deny: [setup_patch]`` beside
    ``claude_code: {permissions: {deny: [setup_patch]}}`` renders a
    ``config.yml`` with the deny and no lift — settings.json denies the tool —
    while the union reads the lift and chowns ``build/config.yml`` to the agent.
    Refusing the shape is what makes that union honest.

    Args:
        config: The profile's ``config:`` block.

    Raises:
        BuildProfileError: naming the nested key and every dotted key beside it.
    """
    if not isinstance(config, dict):
        return
    # Every key that addresses somewhere inside `claude_code`, as segments. The
    # bare key is one of them: a nested `claude_code:` mapping is just the
    # shallowest way to address the subtree, not a different kind of thing.
    addressed = {
        key: tuple(key.split("."))
        for key in config
        if isinstance(key, str)
        and (key == _CLAUDE_CODE_CONFIG_KEY or key.startswith(f"{_CLAUDE_CODE_CONFIG_KEY}."))
    }
    for shallow, shallow_segments in sorted(addressed.items()):
        deeper = sorted(
            key
            for key, segments in addressed.items()
            if len(segments) > len(shallow_segments)
            and segments[: len(shallow_segments)] == shallow_segments
        )
        if not deeper:
            continue
        raise BuildProfileError(
            f"The profile's config: block addresses {shallow!r} twice over: the key "
            f"{shallow!r} itself, and the deeper key(s) "
            f"{', '.join(repr(key) for key in deeper)} inside it. Both survive the merge "
            "as separate keys, and which one reaches config.yml depends on key order — "
            f"the value at {shallow!r} is written verbatim and replaces that whole "
            "subtree — so one of the two would be discarded silently. That includes the "
            "permissions lists the build's setup-capability check reads, which is why "
            "this is refused rather than resolved by a rule. Keep one spelling: fold the "
            "shallower key's leaves into the deeper spelling, e.g.\n"
            "  config:\n"
            f"    {_PERMISSIONS_CONFIG_PATH}.deny:\n"
            "      - mcp__osprey_workspace__setup_patch"
        )


# The top-level shorthand for the control-system connector, and the literal
# dotted `config:` key it resolves to. `connector: epics` is the short spelling
# of `config: {control_system.type: epics}` — one place in the schema sets the
# connector, so the two can never disagree in the rendered project.
CONNECTOR_PROFILE_KEY = "connector"
CONNECTOR_CONFIG_KEY = "control_system.type"

#: Top-level shorthand for ``config: {deployment.port_base: N}`` — the
#: convenient spelling that moves a whole deployment off the default port
#: block (``osprey init --set port_base=42000``), so a dev or CI stack never
#: collides with a real deployment running on the defaults.
PORT_BASE_PROFILE_KEY = "port_base"


def _apply_connector_shorthand(raw: dict[str, Any]) -> dict[str, Any]:
    """Fold a top-level ``connector:`` shorthand into the ``config:`` block.

    Applied to the merged CLI layers
    (:func:`~osprey.cli.build_profile_resolve.merge_cli_overrides`) and again
    here at parse time, so no entry path — preset, ``-O`` file, ``--set`` pair,
    ``extends`` parent, or a hand-written profile loaded directly — can carry
    the shorthand and have it silently ignored. Idempotent: a mapping without
    the key is returned unchanged.

    The value is validated against
    :data:`~osprey_connectors.types.SET_CONTROL_SYSTEM_TYPES` — the built-in
    connector types the CLI offers, plus ``live_standin``. That list rather
    than :data:`~osprey_connectors.types.CLI_CONTROL_SYSTEM_TYPES` because the
    two questions differ: a deployment that already runs a stand-in may be
    pointed at it (``osprey set connector=live_standin``), while ``osprey
    init`` never materializes a project onto one, having no stand-in to point
    at. The nearest-type suggestion draws from the same list, so a typo for the
    stand-in is corrected rather than told the type does not exist.

    A custom connector is addressed by its dotted module path, which the
    shorthand does not cover — write ``config: {control_system.type: ...}``
    for those.

    Args:
        raw: Raw profile mapping, mutated in place like the alias
            normalization it sits beside.

    Returns:
        The same mapping, with the shorthand consumed.

    Raises:
        BuildProfileError: If the value is not one of the built-in connector
            types, or ``config:`` is not a mapping to fold it into.
    """
    if CONNECTOR_PROFILE_KEY not in raw:
        return raw

    value = raw.pop(CONNECTOR_PROFILE_KEY)
    known = sorted(SET_CONTROL_SYSTEM_TYPES)
    if not isinstance(value, str) or not value.strip():
        raise BuildProfileError(
            f"Profile key 'connector' must name a connector type (got {value!r}) — "
            f"one of: {', '.join(known)}."
        )
    value = value.strip()
    if value not in SET_CONTROL_SYSTEM_TYPES:
        # Case-insensitive first: difflib scores 'EPICS' against 'epics' at
        # zero, so the likeliest mistake would otherwise get no suggestion.
        close = [name for name in known if name.lower() == value.lower()] or (
            difflib.get_close_matches(value, known, n=1)
        )
        suggestion = f" (did you mean {close[0]!r}?)" if close else ""
        raise BuildProfileError(
            f"Unknown connector {value!r}{suggestion}. Valid connectors are: "
            f"{', '.join(known)}. A custom connector is set by its dotted module "
            f"path instead: config: {{{CONNECTOR_CONFIG_KEY}: mypackage.MyConnector}}."
        )

    config = raw.setdefault("config", {})
    if not isinstance(config, dict):
        raise BuildProfileError(
            f"Profile 'config' must be a mapping to carry the 'connector' shorthand "
            f"(got {type(config).__name__})"
        )
    config[CONNECTOR_CONFIG_KEY] = value
    return raw


def _apply_port_base_shorthand(raw: dict[str, Any]) -> dict[str, Any]:
    """Fold a top-level ``port_base:`` shorthand into the ``config:`` block.

    Applied beside :func:`_apply_connector_shorthand` on both entry paths
    (CLI-layer merge and parse), for the same reason: no path a profile can
    arrive by may carry the shorthand and have it silently ignored. Idempotent:
    a mapping without the key is returned unchanged.

    The value is range-checked through
    :func:`~osprey.port_layout.resolve_port_base` — the resolver every runtime
    consumer uses — so a base whose thousand-port block cannot exist fails the
    parse with the layout's own refusal rather than surfacing at deploy.

    Args:
        raw: Raw profile mapping, mutated in place like the connector
            shorthand it sits beside.

    Returns:
        The same mapping, with the shorthand consumed.

    Raises:
        BuildProfileError: If the value is not an in-range integer, or
            ``config:`` is not a mapping to fold it into.
    """
    if PORT_BASE_PROFILE_KEY not in raw:
        return raw

    value = raw.pop(PORT_BASE_PROFILE_KEY)
    if isinstance(value, bool) or not isinstance(value, int):
        raise BuildProfileError(
            f"Profile key 'port_base' must be an integer port base (got {value!r}) "
            "— e.g. port_base: 42000."
        )
    try:
        resolve_port_base({"deployment": {"port_base": value}})
    except ValueError as exc:
        raise BuildProfileError(f"Profile key 'port_base': {exc}") from exc

    config = raw.setdefault("config", {})
    if not isinstance(config, dict):
        raise BuildProfileError(
            f"Profile 'config' must be a mapping to carry the 'port_base' shorthand "
            f"(got {type(config).__name__})"
        )
    config[PORT_BASE_CONFIG_KEY] = value
    return raw


def _parse_profile(raw: dict[str, Any]) -> BuildProfile:
    """Parse raw YAML dict into a BuildProfile.

    Callers that read documents have already normalized their YAML-surface
    spellings; normalizing again here covers the hand-assembled dicts that
    reach the parser directly, where an ``app_template`` key would otherwise
    be allowlisted, ignored, and silently replaced by the loader default.
    """
    _normalize_profile_aliases(raw, "profile")
    _reject_unknown_keys(raw)
    _apply_connector_shorthand(raw)
    _apply_port_base_shorthand(raw)
    mcp_servers: dict[str, McpServerDef] = {}
    for name, sdef in raw.get("mcp_servers", {}).items():
        if not isinstance(sdef, dict):
            raise BuildProfileError(f"MCP server '{name}' must be a mapping")
        perms = sdef.get("permissions", {})
        url = sdef.get("url")
        command = sdef.get("command", "")
        port = sdef.get("port")
        if port is not None and (
            not isinstance(port, int) or isinstance(port, bool) or not (1 <= port <= 65535)
        ):
            raise BuildProfileError(
                f"MCP server '{name}' port must be an integer in 1..65535 (got {port!r})"
            )
        if url and command:
            raise BuildProfileError(
                f"MCP server '{name}' has both 'command' and 'url' — use one or the other"
            )
        transport = sdef.get("transport", "http")
        if transport not in ("http", "sse"):
            raise BuildProfileError(
                f"MCP server '{name}' transport must be 'http' or 'sse' (got {transport!r})"
            )
        if "transport" in sdef and command:
            raise BuildProfileError(
                f"MCP server '{name}' declares 'transport' with 'command' — stdio "
                "servers have no transport choice"
            )
        if transport == "sse" and not url:
            # The port-derived URL below is a streamable-HTTP /mcp endpoint;
            # an SSE server must spell out where its event stream lives.
            raise BuildProfileError(
                f"MCP server '{name}' transport 'sse' requires an explicit 'url'"
            )
        if port is not None and command:
            raise BuildProfileError(
                f"MCP server '{name}' has both 'command' and 'port' — stdio servers cannot declare a port"
            )
        # Derive url from port when only port is set (HTTP host-published service).
        # Web terminals run host-networked, so localhost is the right host for .mcp.json.
        if port is not None and not url:
            url = f"http://localhost:{port}/mcp"
        if not url and not command:
            raise BuildProfileError(f"MCP server '{name}' must have either 'command' or 'url'")
        mcp_servers[name] = McpServerDef(
            command=command,
            args=sdef.get("args", []),
            env=sdef.get("env", {}),
            permissions={
                "allow": perms.get("allow", []),
                "ask": perms.get("ask", []),
            },
            url=url,
            transport=transport,
            port=port,
        )

    services: dict[str, ServiceDef] = {}
    for name, sdef in raw.get("services", {}).items():
        if not isinstance(sdef, dict):
            raise BuildProfileError(f"Service '{name}' must be a mapping")
        services[name] = ServiceDef(
            template=sdef.get("template", ""),
            config=sdef.get("config", {}),
        )

    lifecycle_raw = raw.get("lifecycle", {})
    lifecycle = LifecycleConfig(
        pre_build=[LifecycleStep(**s) for s in lifecycle_raw.get("pre_build", [])],
        post_build=[LifecycleStep(**s) for s in lifecycle_raw.get("post_build", [])],
        validate=[LifecycleStep(**s) for s in lifecycle_raw.get("validate", [])],
    )

    env_raw = raw.get("env", {})
    env = EnvConfig(
        required=env_raw.get("required", []),
        pinned=env_raw.get("pinned", []),
        defaults=env_raw.get("defaults", {}),
        file=env_raw.get("file"),
    )

    environment = _parse_environment(raw)

    dependencies = raw.get("dependencies", [])

    # Resolved ONCE, here, and handed to every block below. The layout's rule is
    # that a port comes from the base the deployment actually resolved, so an
    # unspelled port key cannot fall back to the dataclass default: those are
    # computed at the layout's own base and would bake 10010 into a deployment
    # that asked for 20000, leaving the rendered config disagreeing with the
    # compose templates that derive from `osprey_ports`.
    port_base = _profile_port_base(raw)

    dispatch_raw = raw.get("dispatch")
    dispatch = None
    if dispatch_raw is not None:
        if not isinstance(dispatch_raw, dict):
            raise BuildProfileError("Profile 'dispatch' must be a mapping")
        # Checked on the merged block, like bluesky's: parents, -O layers and
        # --set pairs are all folded in by the time the parser runs.
        _reject_unknown_block_keys(dispatch_raw, _KNOWN_DISPATCH_KEYS, "dispatch")
        dispatch = DispatchConfig(
            triggers=dispatch_raw.get("triggers", ""),
            worker_count=dispatch_raw.get("worker_count", 1),
            workspace_mode=dispatch_raw.get("workspace_mode", "isolated"),
            max_concurrent_runs=dispatch_raw.get("max_concurrent_runs", 2),
            max_queue_depth=dispatch_raw.get("max_queue_depth", 50),
            dispatcher_port=dispatch_raw.get(
                "dispatcher_port", default_port("dispatcher", base=port_base)
            ),
            worker_port_base=dispatch_raw.get(
                "worker_port_base", default_port("worker", 1, base=port_base)
            ),
            worker_port_stride=dispatch_raw.get(
                "worker_port_stride", DispatchConfig.worker_port_stride
            ),
            timeout_sec=dispatch_raw.get("timeout_sec", 300),
            inactivity_sec=dispatch_raw.get("inactivity_sec", 120),
            facility_name=dispatch_raw.get("facility_name", ""),
            pv_strip_prefix=dispatch_raw.get("pv_strip_prefix", ""),
            network=dispatch_raw.get("network", "bridge"),
        )

    bluesky_raw = raw.get("bluesky")
    bluesky = None
    if bluesky_raw is not None:
        if not isinstance(bluesky_raw, dict):
            raise BuildProfileError("Profile 'bluesky' must be a mapping")
        # Checked on the merged block: extends parents, -O layers and --set
        # pairs have all been folded in by the time the parser runs, so a key
        # a parent declares is not an unknown key seen from its child.
        _reject_unknown_block_keys(bluesky_raw, _KNOWN_BLUESKY_KEYS, "bluesky")
        excluded_plans = bluesky_raw.get("excluded_plans", [])
        if not isinstance(excluded_plans, list) or not all(
            isinstance(p, str) for p in excluded_plans
        ):
            raise BuildProfileError(
                "bluesky.excluded_plans must be a list of plan-name strings "
                f"(got {excluded_plans!r})"
            )
        devices_file = bluesky_raw.get("devices_file", BlueskyConfig.devices_file)
        if not isinstance(devices_file, str) or not devices_file:
            raise BuildProfileError(
                f"bluesky.devices_file must be a non-empty path string (got {devices_file!r})"
            )
        device_page_size = bluesky_raw.get("device_page_size", BlueskyConfig.device_page_size)
        if (
            not isinstance(device_page_size, int)
            or isinstance(device_page_size, bool)
            or device_page_size < 1
        ):
            raise BuildProfileError(
                f"bluesky.device_page_size must be an integer >= 1 (got {device_page_size!r})"
            )
        bluesky = BlueskyConfig(
            port=bluesky_raw.get("port", default_port("bluesky", base=port_base)),
            tiled_enabled=bluesky_raw.get("tiled_enabled", False),
            tiled_port=bluesky_raw.get("tiled_port", default_port("tiled", base=port_base)),
            second_lane=bool(bluesky_raw.get("second_lane", False)),
            plan_dir=bluesky_raw.get("plan_dir"),
            excluded_plans=excluded_plans,
            devices_file=devices_file,
            device_page_size=device_page_size,
        )

    va_raw = raw.get("virtual_accelerator")
    virtual_accelerator = None
    if va_raw is not None:
        if not isinstance(va_raw, dict):
            raise BuildProfileError("Profile 'virtual_accelerator' must be a mapping")
        _reject_unknown_block_keys(va_raw, _KNOWN_VA_KEYS, "virtual_accelerator")
        live_standin = _parse_live_standin(va_raw.get("live_standin"), port_base)
        virtual_accelerator = VAConfig(
            port=va_raw.get("port", VAConfig.port),
            live_standin=live_standin,
        )

    bluesky_web_raw = raw.get("bluesky_web")
    bluesky_web = None
    if bluesky_web_raw is not None:
        if not isinstance(bluesky_web_raw, dict):
            raise BuildProfileError("Profile 'bluesky_web' must be a mapping")
        bluesky_web = BlueskyWebConfig(
            port=bluesky_web_raw.get("port", default_port("bluesky_web", base=port_base)),
        )

    nextcloud_bridge_raw = raw.get("nextcloud_bridge")
    nextcloud_bridge = None
    if nextcloud_bridge_raw is not None:
        if not isinstance(nextcloud_bridge_raw, dict):
            raise BuildProfileError("Profile 'nextcloud_bridge' must be a mapping")
        nextcloud_bridge = NextcloudBridgeProfileConfig(
            trigger=nextcloud_bridge_raw.get("trigger", "nextcloud-question"),
        )

    gchat_bridge_raw = raw.get("gchat_bridge")
    gchat_bridge = None
    if gchat_bridge_raw is not None:
        if not isinstance(gchat_bridge_raw, dict):
            raise BuildProfileError("Profile 'gchat_bridge' must be a mapping")
        gchat_bridge = GChatBridgeProfileConfig(
            trigger=gchat_bridge_raw.get("trigger", "gchat-question"),
        )

    provenance_raw = raw.get("provenance")
    provenance = None
    if provenance_raw is not None:
        if not isinstance(provenance_raw, dict):
            raise BuildProfileError("Profile 'provenance' must be a mapping")
        missing = [key for key in ("preset", "preset_hash") if not provenance_raw.get(key)]
        if missing:
            raise BuildProfileError(
                f"Profile 'provenance' is missing {', '.join(missing)}. It is written by "
                f"`osprey init` and records what the profile was materialized from; "
                f"drop the block rather than half-filling it."
            )
        provenance = ProfileProvenance(
            preset=str(provenance_raw["preset"]),
            preset_hash=str(provenance_raw["preset_hash"]),
        )

    # A `config:` key present but empty — every entry commented out, say —
    # parses to None, and an empty block means exactly "no config entries".
    # Narrowed to None on purpose: `or {}` would swallow an empty LIST too, and
    # a list is a real mistake that `BuildProfile.validate()` rejects by name.
    config_raw = raw.get("config")
    config = {} if config_raw is None else config_raw
    # Checked on the fully-resolved block, like the bluesky/dispatch shapes
    # above: every layer has been folded in by the time the parser runs, so a
    # spelling a preset or an `extends` parent contributes is visible here.
    # Ambiguity first, then shape — a block spelling `claude_code` both ways has
    # no single permissions list to check the shape of.
    _reject_mixed_claude_code_spellings(config)
    _reject_permission_list_shapes(config)

    return BuildProfile(
        name=raw.get("name", ""),
        data_bundle=raw.get("data_bundle", "control_assistant"),
        data=raw.get("data"),
        deploy=parse_deploy_block(raw),
        deploy_services=raw.get("deploy_services", True),
        provider=raw.get("provider"),
        model=raw.get("model"),
        channel_finder_mode=raw.get("channel_finder_mode"),
        tier=(int(raw["tier"]) if raw.get("tier") is not None else None),
        config=config,
        mcp_servers=mcp_servers,
        services=services,
        lifecycle=lifecycle,
        env=env,
        environment=environment,
        dependencies=dependencies,
        requires_osprey_version=raw.get("requires_osprey_version"),
        osprey_install=raw.get("osprey_install", "local"),
        python_env=raw.get("python_env", "project"),
        hooks=raw.get("hooks", []),
        rules=raw.get("rules", []),
        skills=raw.get("skills", []),
        agents=raw.get("agents", []),
        output_styles=raw.get("output_styles", []),
        web_panels=raw.get("web_panels", []),
        default_panel=raw.get("default_panel"),
        panel_presets=raw.get("panel_presets", {}),
        claude_md_template=raw.get("claude_md_template"),
        artifact_server=raw.get("artifact_server", {}),
        dispatch=dispatch,
        bluesky=bluesky,
        virtual_accelerator=virtual_accelerator,
        bluesky_web=bluesky_web,
        nextcloud_bridge=nextcloud_bridge,
        gchat_bridge=gchat_bridge,
        va_archiver=parse_va_archiver_block(raw, base=port_base),
        provenance=provenance,
    )
