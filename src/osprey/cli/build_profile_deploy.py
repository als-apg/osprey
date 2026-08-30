"""The ``deploy:`` block — where a profile's project is built, pushed, and run.

The profile directory is the single source of facility truth, so the coordinates
a deployment needs — which CI platform builds it, which registry holds the
images, which host runs them — live in the profile alongside everything else,
not in a second file with its own schema.

The block is deliberately narrow. It carries only what a pipeline consumes and
nothing a profile already answers: no LLM provider, no timezone, no control
system, no user roster. Those keys are *rejected by name* rather than ignored,
because a facility that writes them here has put a fact somewhere the build
will never read it. Deploy-scoped environment variables are named here but
declared through the profile's own ``env.required`` / ``env.defaults`` channel,
which is the one place a built project's ``.env`` template comes from.

Shapes, parsing and the block's rules are all here rather than split across the
loader modules: the block is closed and small, and the CI-scaffolding verbs that
consume it want exactly one import.
"""

from __future__ import annotations

import difflib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from osprey.errors import BuildProfileError
from osprey_connectors.types import (
    LIMITS_CHECKING_LEAF,
    LIMITS_LEAVES,
    SET_CONTROL_SYSTEM_TYPES,
)

from .build_profile_schema import _ENV_VAR_RE

#: CI platforms with a shipped pipeline template. ``deploy.ci`` selects one of
#: these by name; the scaffolding verbs key their template lookup off the same
#: value, so a platform absent here has nothing to render and is refused at
#: validation rather than at scaffold time.
SUPPORTED_CI_PLATFORMS: tuple[str, ...] = ("gitlab",)

#: How the deploy host obtains its images. ``registry`` pulls what CI built;
#: ``local`` builds them on the host, which is what a facility with no CI or no
#: reachable registry does.
IMAGE_SOURCES: tuple[str, ...] = ("registry", "local")

#: Keys the profile already owns, mapped to where the fact actually goes. The
#: legacy facility-config file held these beside the deploy coordinates, so a
#: facility porting one will paste them in; naming the real home turns a
#: silently-ignored key into a one-line fix.
_PROFILE_OWNED_KEYS: dict[str, str] = {
    "llm": "the profile's own `provider:` and `model:` keys",
    "facility": "the profile's `name:` key, plus `config:` entries for the rest",
    "timezone": "the profile's `config:` key `system.timezone`",
    "control_system": "the profile's `config:` key `control_system.type`",
    "users": "the profile's `config:` key `modules.web_terminals.users`",
    "personas": "the profile's `config:` key `modules.web_terminals.personas`",
    "default_persona": "the profile's `config:` key `modules.web_terminals.default_persona`",
}

#: Environment declarations have one channel for the whole profile, so the
#: deploy block names variables but never declares them.
_ENV_CHANNEL_KEYS: tuple[str, ...] = ("env", "environment")

#: The rendered config section a per-type limits posture lives in, and the
#: dotted spelling of the same path for a spelling-agnostic lookup. The block's
#: own key and the leaves that make it complete come from
#: :mod:`osprey_connectors.types`, which is where the resolvers read them: a
#: build that refused a different pair than the runtime answers with would be
#: refusing profiles the deployment can honour, or passing ones it cannot.
_CONTROL_SYSTEM_KEY = "control_system"
_CONNECTOR_SEGMENTS: tuple[str, ...] = (_CONTROL_SYSTEM_KEY, "connector")
_CONNECTOR_PREFIX = ".".join(_CONNECTOR_SEGMENTS)

#: The rendered-config subtree the multi-user web stack reads at deploy time.
#: ``deploy.image_source`` is propagated into it at build time, which is what
#: makes the deploy block the fact's only home in the profile.
WEB_TERMINALS_CONFIG_PATH = "modules.web_terminals"

#: The leaf inside it that ``deploy.image_source`` owns.
IMAGE_SOURCE_CONFIG_KEY = f"{WEB_TERMINALS_CONFIG_PATH}.image_source"

_KNOWN_DEPLOY_KEYS = frozenset({"ci", "registry", "host", "image_source", "external_projects"})
_KNOWN_REGISTRY_KEYS = frozenset({"url", "token_env_var"})
_KNOWN_HOST_KEYS = frozenset({"name", "fqdn", "user", "project_path"})
_KNOWN_EXTERNAL_PROJECT_KEYS = frozenset({"name", "url", "image", "token_env_var"})


@dataclass
class DeployRegistry:
    """Where CI pushes images and the deploy host pulls them from."""

    url: str
    """Registry URL including port and project path, without a scheme —
    ``git.example.org:5050/physics/production/facility-profiles``."""
    token_env_var: str | None = None
    """Name of the variable holding the registry credential. The value lives in
    the deployment's ``.env``; declare the variable under ``env.required``."""


@dataclass
class DeployHost:
    """The server the deployment runs on."""

    name: str
    """SSH-resolvable hostname — ``ssh <name>`` must work for the operator."""
    project_path: str
    """Absolute path to the facility repo's checkout on that server."""
    user: str
    """SSH user that owns the checkout and runs the containers."""
    fqdn: str | None = None
    """Fully-qualified name developers reach the host by, when the short name
    only resolves on the facility network."""


@dataclass
class ExternalProject:
    """Another project's image this deployment also pulls."""

    name: str
    url: str
    image: str
    token_env_var: str | None = None


@dataclass
class DeployConfig:
    """The parsed ``deploy:`` block."""

    ci: str
    host: DeployHost
    registry: DeployRegistry | None = None
    image_source: str = "registry"
    external_projects: list[ExternalProject] = field(default_factory=list)


def parse_deploy_block(raw: dict[str, Any]) -> DeployConfig | None:
    """Parse and validate a profile's ``deploy:`` block.

    Every problem in the block is reported at once rather than one per run: a
    facility filling this in for the first time is transcribing a handful of
    coordinates, and finding them wrong one round-trip at a time is the slowest
    possible way to learn the shape.

    Args:
        raw: The resolved raw profile dict.

    Returns:
        The parsed config, or ``None`` when the profile declares no ``deploy:``
        block — deployment coordinates are opt-in, and a profile that only ever
        builds locally needs none.

    Raises:
        BuildProfileError: If the block is not a mapping, names a key the
            profile owns elsewhere, or is missing/misshapen anywhere.
    """
    block = raw.get("deploy")
    if block is None:
        return None
    if not isinstance(block, dict):
        raise BuildProfileError(f"Profile 'deploy' must be a mapping (got {type(block).__name__})")

    problems: list[str] = []
    _reject_relocated_keys(block, problems)
    _reject_unknown_deploy_keys(block, problems)

    ci = _parse_ci(block, problems)
    image_source = _parse_image_source(block, problems)
    registry = _parse_registry(block, image_source, problems)
    host = _parse_host(block, problems)
    external_projects = _parse_external_projects(block, problems)
    _reject_duplicate_image_source(raw.get("config"), image_source, problems)

    if problems:
        raise BuildProfileError(
            "Profile 'deploy' block is invalid:\n  - " + "\n  - ".join(problems)
        )

    assert ci is not None and host is not None  # narrows: problems was empty
    return DeployConfig(
        ci=ci,
        host=host,
        registry=registry,
        image_source=image_source,
        external_projects=external_projects,
    )


# ---------------------------------------------------------------------------
# image_source: one home in the profile, one value in the rendered config
# ---------------------------------------------------------------------------


def config_image_source_spelling(config: Any) -> str | None:
    """How a profile's ``config:`` block spells ``image_source``, if it does.

    Two spellings reach the same rendered leaf, and both are in use: the dotted
    key ``modules.web_terminals.image_source``, and an ``image_source`` inside a
    ``modules.web_terminals`` subtree mapping. Callers need to know *which* one
    a profile wrote in order to name it back.

    Returns:
        The offending ``config:`` key, or ``None`` when the block spells it
        nowhere. A ``modules:`` mapping nested under ``config:`` is checked too
        — it is a discouraged spelling but a legal one, and a duplicate hiding
        there would defeat the whole point of looking.
    """
    if not isinstance(config, dict):
        return None
    if IMAGE_SOURCE_CONFIG_KEY in config:
        return IMAGE_SOURCE_CONFIG_KEY
    subtree = config.get(WEB_TERMINALS_CONFIG_PATH)
    if isinstance(subtree, dict) and "image_source" in subtree:
        return f"{WEB_TERMINALS_CONFIG_PATH}: image_source"
    modules = config.get("modules")
    if isinstance(modules, dict):
        nested = modules.get("web_terminals")
        if isinstance(nested, dict) and "image_source" in nested:
            return "modules: web_terminals: image_source"
    return None


def declares_web_terminals(config: Any) -> bool:
    """Whether a profile's ``config:`` block configures the web-terminal stack.

    The propagation target has to already exist: writing ``image_source`` into
    a config with no ``modules.web_terminals`` would invent a module the
    facility never asked for, and the deploy runtime reads the leaf only from
    inside that module.
    """
    if not isinstance(config, dict):
        return False
    for key, value in config.items():
        if not isinstance(key, str):
            continue
        if key == WEB_TERMINALS_CONFIG_PATH or key.startswith(f"{WEB_TERMINALS_CONFIG_PATH}."):
            return True
        if key == "modules" and isinstance(value, dict) and "web_terminals" in value:
            return True
    return False


def deploy_config_overrides(deploy: DeployConfig | None, config: Any) -> dict[str, Any]:
    """The ``config:`` entries the ``deploy:`` block contributes to the render.

    Today that is one entry: ``image_source``. The deploy block is where a
    facility says how its host gets images, and the multi-user web stack reads
    that answer from ``modules.web_terminals.image_source`` — so the build
    writes it there rather than making the facility say the same thing twice.

    Emitted only when the profile actually configures the web-terminal stack;
    for everything else the deploy block's answer has no rendered home yet and
    inventing one would add a module the facility never declared.

    Returns:
        Overrides to apply *after* the profile's own ``config:`` entries, so the
        leaf lands inside whatever ``modules.web_terminals`` subtree those
        wrote. Empty when there is nothing to contribute.
    """
    if deploy is None or not declares_web_terminals(config):
        return {}
    return {IMAGE_SOURCE_CONFIG_KEY: deploy.image_source}


def deploy_aware_config_errors(
    deploy: DeployConfig | None, config: Any, *, profile_root: Path | None = None
) -> list[str]:
    """Lint a profile's web stack against the config the BUILD will render.

    The lint reads a ``config:`` block, but not every fact it checks is spelled
    there: ``image_source`` lives in the ``deploy:`` block and reaches the
    rendered config through :func:`deploy_config_overrides`. Linting the raw
    block therefore judges a profile on a view no deployment ever runs — a
    facility that correctly states ``image_source: local`` once, in the deploy
    block, would be told its (defaulted) ``registry`` mode is missing a
    ``registry.url`` it does not need.

    So the lint runs on the merged view, and this function is where the merge
    and the lint are paired. Both command surfaces that gate on the lint
    (``osprey profile validate`` and ``osprey build``'s pre-check) call THIS,
    not the two halves: a profile that validates must build, and one function is
    the only way to keep that true.

    Nothing changes for a profile without deployment coordinates.
    :func:`deploy_config_overrides` contributes nothing when there is no
    ``deploy:`` block, and nothing when the profile does not configure the
    web-terminal stack, so in both cases the merged view IS the raw block.

    This stays out of ``BuildProfile.validate()``, which also runs during
    profile *resolution* — ``lint_profile_config``'s own docstring has the
    reason the engine belongs to the commands.

    Args:
        deploy: The profile's parsed ``deploy:`` block, or ``None``.
        config: The profile's ``config:`` block.
        profile_root: The directory the profile being linted lives in. Persona
            deltas (``build_profile: personas/<name>.yml``) are named relative
            to it, so a caller that omits it gets a lint that reads no delta
            unless it happens to be running from the repo root — which is how
            ``osprey build`` from a subdirectory, and ``--repo`` from anywhere,
            used to pass a roster the profile gate exists to refuse. Every
            command surface passes it.

    Returns:
        The lint messages that must fail the command, empty when it passes.
    """
    # Imported in-function: this module is on `BuildProfile`'s import chain, and
    # the lint engine pulls in the whole web-terminals package behind it.
    from osprey.deployment.web_terminals.lint import profile_config_errors

    return profile_config_errors(
        {**config, **deploy_config_overrides(deploy, config)}, profile_root=profile_root
    )


def deploy_aware_config_warnings(
    deploy: DeployConfig | None, config: Any, *, profile_root: Path | None = None
) -> list[str]:
    """The advisory half of :func:`deploy_aware_config_errors`, on the same merged view.

    Paired with the errors here for the same reason the merge is: a profile that
    validates must build, and a profile that draws an advisory from one command
    must draw the same one from the other. Both command surfaces print these
    above their success line — an exposure nobody prints is an exposure nobody
    has, and the whole-deployment one (no login wall — ``auth.method: token`` or
    ``none`` — in front of a privileged terminal) is deliberately not an error,
    so printing is the only way it reaches an operator at all.
    """
    from osprey.deployment.web_terminals.lint import profile_config_warnings

    return profile_config_warnings(
        {**config, **deploy_config_overrides(deploy, config)}, profile_root=profile_root
    )


def limits_block_errors(config: Mapping[str, Any]) -> list[str]:
    """Refuse a profile whose per-type ``limits_checking`` block will not render.

    ``control_system.connector.<type>.limits_checking`` overrides the
    deployment-wide ``enabled`` / ``allow_unlisted_channels`` pair as a WHOLE
    block — there is no leaf inheritance — so a block stating one leaf has no
    posture to answer with, and every reader falls back to the failsafe
    validator. That is a deployment quietly stricter (or, for the operator who
    meant to relax a simulator, quietly unchanged) than the profile says.

    The other half of the check is about the render rather than the block. A
    ``config:`` entry reaches ``config.yml`` through
    :func:`osprey.utils.config_writer.config_update_fields`, which splits only
    the TOP-LEVEL key of each entry on dots and assigns the value verbatim at
    the leaf it lands on. Every dot below that key therefore stays inside a
    literal key name. So the flat leaf a preset writes
    (``control_system.connector.virtual_accelerator.limits_checking.enabled``)
    renders where it reads, while the same block written for a dotted custom
    type flattened into one key renders four levels too deep, and
    ``control_system.connector.<type>: {"limits_checking.enabled": true}``
    renders a key literally named ``limits_checking.enabled``. Both look right
    in the profile and are invisible afterwards; a custom type spelled as its
    own map key under a ``control_system.connector`` prefix is the spelling
    that works.

    Mixed spellings are otherwise legal and are not judged here: a profile is
    free to write a leaf flat, as a dotted prefix over a mapping, or fully
    nested (:func:`osprey.cli.build_profile_reach._spelled_values` reads all of
    them), and two dotted keys at different depths below ``control_system``
    merge at render. The one exception is a bare top-level ``control_system:``
    mapping beside flat ``control_system.*`` keys, which does not merge and
    leaves no way to attribute a limits leaf to one spelling or the other —
    :func:`_mixed_depth_control_system_errors` has the mechanism.

    Registry-free by construction: a per-type block is checked for every
    built-in connector type and for every type the profile itself names,
    whether or not the deployment selects it. A stray block is still a stated
    posture, and a deployment pointed at that type later would inherit it.

    Args:
        config: The profile's ``config:`` block, as merged — presets, ``-O``
            overlays, ``--set`` pairs and ``extends`` parents folded together.
            A non-mapping has nothing to refuse.

    Returns:
        One message per problem, empty when the profile's limits blocks are
        sound. Each message names the ``config:`` line(s) to fix:

        * a per-type ``limits_checking`` path that does not render as
          ``control_system.connector.<type>.limits_checking.<leaf>`` — the
          message names the entry as the profile wrote it, and the path it
          actually renders to;
        * a ``config:`` block addressing ``control_system`` at two depths — the
          message names the shallower key and every deeper key inside it;
        * a per-type block stating one of its two leaves — the message names
          the connector type, the leaf the profile did state, and the missing
          leaf.
    """
    if not isinstance(config, dict):
        return []

    # Imported in-function: this module sits on `BuildProfile`'s import chain,
    # and the reach registry behind `_spelled_values` pulls the whole
    # service-resolution package in with it.
    from .build_profile_reach import _spelled_values

    errors: list[str] = []
    named_types: set[str] = set()

    for written, rendered, _value in _rendered_leaf_paths(config):
        if tuple(rendered[: len(_CONNECTOR_SEGMENTS)]) != _CONNECTOR_SEGMENTS:
            continue
        below = rendered[len(_CONNECTOR_SEGMENTS) :]
        if not any(LIMITS_CHECKING_LEAF in segment.split(".") for segment in below):
            continue
        # The one shape that renders where it reads: <type>, the block, a leaf.
        if len(below) == 3 and below[1] == LIMITS_CHECKING_LEAF:
            named_types.add(below[0])
            continue
        errors.append(
            f"The profile's config: block writes `{written}`, which renders as "
            f"`{'.'.join(rendered)}` — not a per-type limits block. A per-type posture is "
            f"exactly `control_system.connector.<type>.limits_checking.<leaf>`, and the build "
            f"splits only the top-level key of each config: entry on dots, so every dot below "
            f"it becomes part of a literal key name. Write a built-in type's leaves flat, and "
            f"a dotted custom type as its own map key:\n"
            f"  config:\n"
            f"    control_system.connector:\n"
            f"      mypkg.TangoConnector:\n"
            f"        limits_checking:\n"
            f"          enabled: true\n"
            f"          allow_unlisted_channels: false"
        )

    errors.extend(_mixed_depth_control_system_errors(config))

    candidates = set(SET_CONTROL_SYSTEM_TYPES) | named_types
    for _spelling, value in _spelled_values(config, _CONNECTOR_PREFIX):
        if isinstance(value, dict):
            candidates.update(key for key in value if isinstance(key, str))

    for connector_type in sorted(candidates):
        spelled = {
            leaf: _spelled_values(
                config, f"{_CONNECTOR_PREFIX}.{connector_type}.{LIMITS_CHECKING_LEAF}.{leaf}"
            )
            for leaf in LIMITS_LEAVES
        }
        stated = [leaf for leaf in LIMITS_LEAVES if spelled[leaf]]
        if not stated or len(stated) == len(LIMITS_LEAVES):
            continue
        missing = [leaf for leaf in LIMITS_LEAVES if not spelled[leaf]]
        stated_spelling, _stated_value = spelled[stated[0]][0]
        errors.append(
            f"The per-type limits block for connector type {connector_type!r} is incomplete: "
            f"the profile's config: block writes `{stated_spelling}` but never "
            f"{', '.join(repr(leaf) for leaf in missing)}. A per-type block overrides the "
            f"deployment-wide `control_system.limits_checking` pair as a whole — no leaf is "
            f"inherited — so a block missing one has no posture to answer with, and every "
            f"reader falls back to refusing unlisted channels. State both "
            f"{', '.join(repr(leaf) for leaf in LIMITS_LEAVES)}, or remove the block "
            f"and let the deployment-wide pair answer for this type."
        )

    return errors


def _rendered_leaf_paths(config: Mapping[str, Any]) -> list[tuple[str, list[str], Any]]:
    """Every leaf a ``config:`` block writes, with where the render will put it.

    Args:
        config: The profile's ``config:`` block.

    Returns:
        One entry per non-mapping value reachable in the block:
        ``(spelling, rendered path, value)``. The spelling is the chain of map
        keys as written, joined ``key: key`` the way
        :func:`osprey.cli.build_profile_reach._spelled_values` names a line.
        The rendered path is the top-level key split on dots followed by every
        deeper map key *unsplit* — which is what
        :func:`osprey.utils.config_writer.config_update_fields` produces.
    """
    found: list[tuple[str, list[str], Any]] = []
    for key, value in config.items():
        if isinstance(key, str):
            _collect_leaves([key], key.split("."), value, found)
    return found


def _collect_leaves(
    written: list[str],
    rendered: list[str],
    value: Any,
    found: list[tuple[str, list[str], Any]],
) -> None:
    """Descend *value*, appending one entry per leaf to *found*."""
    if isinstance(value, dict):
        for key, sub_value in value.items():
            if isinstance(key, str):
                _collect_leaves([*written, key], [*rendered, key], sub_value, found)
        return
    found.append((": ".join(written), rendered, value))


def _mixed_depth_control_system_errors(config: Mapping[str, Any]) -> list[str]:
    """Refuse a bare top-level ``control_system:`` key beside flat ``control_system.*`` ones.

    Exactly that pair, and no deeper one. A deeper pair —
    ``control_system.connector:`` holding one type's block beside
    ``control_system.connector.epics.writes_enabled`` — renders correctly
    today: :func:`~osprey.utils.config_writer._set_dotted_anchored` runs with
    ``create_only=True``, so it walks INTO whatever an existing intermediate
    key holds rather than replacing it, and the two entries merge. Refusing
    those would refuse profiles the build already honours.

    The bare key is the one that does not merge. It is a top-level ``config:``
    entry with no dots to walk, so the render assigns its value verbatim at
    ``control_system`` and replaces the whole rendered section — the flat
    entries beside it land in a subtree that is then thrown away, or throw the
    bare one away, depending on key order. The reach helpers read both
    spellings and so cannot say which of the two a limits leaf belongs to,
    which is exactly the attribution :func:`limits_block_errors` needs before
    it can call a per-type block complete. ``claude_code`` is guarded against
    the same shape by
    :func:`osprey.cli.build_profile_load._reject_mixed_claude_code_spellings`.

    The bare key's value is not inspected: a nested mapping is what a profile
    writing this actually has, and a scalar there is no better.
    """
    if _CONTROL_SYSTEM_KEY not in config:
        return []
    flat = sorted(
        key for key in config if isinstance(key, str) and key.startswith(f"{_CONTROL_SYSTEM_KEY}.")
    )
    if not flat:
        return []
    return [
        f"The profile's config: block addresses {_CONTROL_SYSTEM_KEY!r} at two depths: the "
        f"bare key {_CONTROL_SYSTEM_KEY!r} itself, and the flat key(s) "
        f"{', '.join(repr(key) for key in flat)} beside it. Both survive the merge as "
        f"separate keys, and the bare key's value is written verbatim over the whole "
        f"rendered control_system section, so which of the two reaches config.yml depends "
        f"on key order — a limits posture, a write posture or a gateway among them. Keep "
        f"one spelling: fold the bare mapping's leaves into flat keys, or write every "
        f"control_system fact inside the nested mapping."
    ]


def _reject_duplicate_image_source(config: Any, image_source: str, problems: list[str]) -> None:
    """Refuse a profile that states ``image_source`` in both of its homes.

    The build propagates ``deploy.image_source`` into the rendered config, so a
    ``config:`` entry saying the same thing is not an override — it is a second
    home for one fact, and the two can disagree. Disagreeing is the dangerous
    case: it decides whether the deploy host builds its own images or pulls
    them, and a facility that meant ``local`` while the deploy block defaulted
    to ``registry`` would find out by watching a pull fail on the host.

    Rejected rather than overridden-with-a-warning, for the same reason the
    profile-owned keys above are: a warning leaves both spellings in the file,
    and the next reader still cannot tell which one the deployment runs on. The
    message carries the current value so moving it cannot silently change the
    mode.
    """
    spelling = config_image_source_spelling(config)
    if spelling is None:
        return
    problems.append(
        f"'image_source' is set both here and in the profile's `config:` block "
        f"({spelling}) — one fact, two homes, free to disagree. The deploy block "
        f"is the home: the build writes its value into "
        f"`{IMAGE_SOURCE_CONFIG_KEY}` for you. Delete the `config:` entry, and "
        f"make sure this block says what you meant (`image_source: "
        f"{image_source}` right now)."
    )


def _reject_relocated_keys(block: dict[str, Any], problems: list[str]) -> None:
    """Name the real home of every key that belongs to the profile proper."""
    for key, home in _PROFILE_OWNED_KEYS.items():
        if key in block:
            problems.append(
                f"'{key}' is not a deploy key — that fact belongs in {home}. "
                f"Remove it from the deploy block."
            )

    for key in _ENV_CHANNEL_KEYS:
        if key in block:
            problems.append(
                f"'{key}' is not a deploy key — a profile declares every variable "
                f"through `env.required` / `env.defaults`, including deploy-scoped "
                f"ones. The deploy block names variables (`registry.token_env_var`) "
                f"but never declares them."
            )

    if "gitlab" in block:
        problems.append(
            "'gitlab' is not a deploy key — name the platform instead: "
            '`ci: "gitlab"`. Its coordinates come from the pipeline itself, so '
            "there is nothing else to carry over."
        )


def _reject_unknown_deploy_keys(block: dict[str, Any], problems: list[str]) -> None:
    """Reject anything the block does not define, naming the closest spelling."""
    already_named = set(_PROFILE_OWNED_KEYS) | set(_ENV_CHANNEL_KEYS) | {"gitlab"}
    unknown = sorted(set(block) - _KNOWN_DEPLOY_KEYS - already_named)
    if not unknown:
        return
    known = sorted(_KNOWN_DEPLOY_KEYS)
    for key in unknown:
        close = difflib.get_close_matches(str(key), known, n=1)
        suggestion = f" (did you mean '{close[0]}'?)" if close else ""
        problems.append(
            f"unknown key '{key}'{suggestion} — valid deploy keys are: {', '.join(known)}."
        )


def _parse_ci(block: dict[str, Any], problems: list[str]) -> str | None:
    """Read ``ci``, the platform name that selects the pipeline template."""
    supported = ", ".join(repr(name) for name in SUPPORTED_CI_PLATFORMS)
    ci = block.get("ci")
    if ci is None:
        problems.append(
            f"'ci' is required — it names the CI platform whose pipeline the "
            f"scaffolding renders. Supported: {supported}."
        )
        return None
    if isinstance(ci, dict):
        problems.append(
            f"'ci' names the platform as a string ({supported}), not a mapping. The "
            f"pipeline's own coordinates — project id, path, branch — come from the "
            f"CI environment at run time, so the profile carries only the name."
        )
        return None
    if not isinstance(ci, str):
        problems.append(f"'ci' must be a string (got {type(ci).__name__}).")
        return None
    if ci not in SUPPORTED_CI_PLATFORMS:
        problems.append(
            f"'ci' is {ci!r}, which has no pipeline template. Supported platforms: {supported}."
        )
        return None
    return ci


def _parse_image_source(block: dict[str, Any], problems: list[str]) -> str:
    """Read ``image_source``, defaulting to pulling what CI built."""
    value = block.get("image_source")
    if value is None:
        return "registry"
    if value not in IMAGE_SOURCES:
        problems.append(
            f"'image_source' is {value!r} — must be one of "
            f"{', '.join(repr(name) for name in IMAGE_SOURCES)}."
        )
        return "registry"
    return str(value)


def _parse_registry(
    block: dict[str, Any], image_source: str, problems: list[str]
) -> DeployRegistry | None:
    """Read the ``registry:`` sub-block, required only when images are pulled."""
    raw = block.get("registry")
    if raw is None:
        if image_source == "registry":
            problems.append(
                "'registry' is required when image_source is 'registry' — that is "
                "where the deploy host pulls images from. Set `image_source: local` "
                "for a host that builds its own."
            )
        return None
    if not isinstance(raw, dict):
        problems.append(f"'registry' must be a mapping (got {type(raw).__name__}).")
        return None

    _reject_unknown_subkeys(raw, _KNOWN_REGISTRY_KEYS, "registry", problems)

    url = raw.get("url")
    if not isinstance(url, str) or not url.strip():
        problems.append("'registry.url' is required and must be a non-empty string.")
        url = ""
    else:
        _reject_scheme(url, "registry.url", problems)

    token_env_var = _parse_env_var_name(
        raw.get("token_env_var"), "registry.token_env_var", problems
    )
    return DeployRegistry(url=url, token_env_var=token_env_var)


def _parse_host(block: dict[str, Any], problems: list[str]) -> DeployHost | None:
    """Read the ``host:`` sub-block — where the deployment actually runs."""
    raw = block.get("host")
    if raw is None:
        problems.append(
            "'host' is required — a deploy block describes a deployment, and a "
            "deployment runs somewhere. It needs `name`, `user` and `project_path`."
        )
        return None
    if not isinstance(raw, dict):
        problems.append(f"'host' must be a mapping (got {type(raw).__name__}).")
        return None

    _reject_unknown_subkeys(raw, _KNOWN_HOST_KEYS, "host", problems)

    values: dict[str, str] = {}
    for key in ("name", "user", "project_path"):
        value = raw.get(key)
        if not isinstance(value, str) or not value.strip():
            problems.append(f"'host.{key}' is required and must be a non-empty string.")
            values[key] = ""
        else:
            values[key] = value

    project_path = values["project_path"]
    if project_path and not project_path.startswith("/"):
        problems.append(
            f"'host.project_path' must be absolute (got {project_path!r}) — it is "
            f"resolved on the deploy host, where the operator's working directory "
            f"is nobody's business but their own."
        )

    fqdn = raw.get("fqdn")
    if fqdn is not None and (not isinstance(fqdn, str) or not fqdn.strip()):
        problems.append("'host.fqdn' must be a non-empty string when present.")
        fqdn = None

    if not values["name"]:
        return None
    return DeployHost(
        name=values["name"],
        project_path=project_path,
        user=values["user"],
        fqdn=fqdn,
    )


def _parse_external_projects(block: dict[str, Any], problems: list[str]) -> list[ExternalProject]:
    """Read ``external_projects`` — other projects' images this deploy pulls."""
    raw = block.get("external_projects")
    if raw is None:
        return []
    if not isinstance(raw, list):
        problems.append(f"'external_projects' must be a list (got {type(raw).__name__}).")
        return []

    parsed: list[ExternalProject] = []
    seen: set[str] = set()
    for index, entry in enumerate(raw):
        label = f"external_projects[{index}]"
        if not isinstance(entry, dict):
            problems.append(f"'{label}' must be a mapping (got {type(entry).__name__}).")
            continue
        _reject_unknown_subkeys(entry, _KNOWN_EXTERNAL_PROJECT_KEYS, label, problems)

        values: dict[str, str] = {}
        for key in ("name", "url", "image"):
            value = entry.get(key)
            if not isinstance(value, str) or not value.strip():
                problems.append(f"'{label}.{key}' is required and must be a non-empty string.")
                values[key] = ""
            else:
                values[key] = value
        if values["url"]:
            _reject_scheme(values["url"], f"{label}.url", problems)

        name = values["name"]
        if name and name in seen:
            problems.append(
                f"'{label}.name' repeats {name!r} — each external project is pulled "
                f"under its own name, so two entries sharing one is ambiguous."
            )
        seen.add(name)

        token_env_var = _parse_env_var_name(
            entry.get("token_env_var"), f"{label}.token_env_var", problems
        )
        if name:
            parsed.append(
                ExternalProject(
                    name=name,
                    url=values["url"],
                    image=values["image"],
                    token_env_var=token_env_var,
                )
            )
    return parsed


def _reject_unknown_subkeys(
    block: dict[str, Any], known: frozenset[str], label: str, problems: list[str]
) -> None:
    """Reject unrecognized keys inside one of the block's sub-mappings."""
    unknown = sorted(set(block) - known)
    if unknown:
        problems.append(
            f"'{label}' has unknown key(s): {', '.join(repr(key) for key in unknown)} "
            f"— valid keys are: {', '.join(sorted(known))}."
        )


def _reject_scheme(url: str, label: str, problems: list[str]) -> None:
    """Refuse a registry coordinate written as a URL with a scheme.

    Registry references are host-and-path, never scheme-prefixed: an image ref
    built from ``https://host/path`` is unpullable, and the failure surfaces
    only on the deploy host at pull time.
    """
    if "://" in url:
        problems.append(
            f"'{label}' must not include a scheme (got {url!r}) — a registry "
            f"reference is host[:port]/path, as it appears in an image name."
        )


def _parse_env_var_name(value: Any, label: str, problems: list[str]) -> str | None:
    """Validate a variable *name* the deploy block points at."""
    if value is None:
        return None
    if not isinstance(value, str) or not _ENV_VAR_RE.match(value):
        problems.append(
            f"'{label}' must be an environment variable NAME "
            f"(uppercase letters, digits, underscores), got {value!r}. The value "
            f"belongs in the deployment's .env; declare the name under `env.required`."
        )
        return None
    return value
