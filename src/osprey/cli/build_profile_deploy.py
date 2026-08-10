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
from dataclasses import dataclass, field
from typing import Any

from osprey.errors import BuildProfileError

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


def deploy_aware_config_errors(deploy: DeployConfig | None, config: Any) -> list[str]:
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

    Returns:
        The lint messages that must fail the command, empty when it passes.
    """
    # Imported in-function: this module is on `BuildProfile`'s import chain, and
    # the lint engine pulls in the whole web-terminals package behind it.
    from osprey.deployment.web_terminals.lint import profile_config_errors

    return profile_config_errors({**config, **deploy_config_overrides(deploy, config)})


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
