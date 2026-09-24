"""The ``osprey set`` verb — the one sanctioned CLI write into ``profile.yml``.

Every other lifecycle verb reads the deployment repo's source zone; this one
writes it. The write lands in ``profile.yml`` and nowhere else: ``build/`` is
100% derived, so a CLI edit of the rendered ``build/config.yml`` would be
undone by the next build without ever having reached the file that describes
the deployment. That failure mode — a setting that works until someone
rebuilds — is what having exactly one writer prevents.

The machinery is the build's own write-back
(:func:`~osprey.cli.build_profile_resolve.write_back_cli_overrides`), promoted
rather than reimplemented: comment-preserving round-trip YAML, one atomic
replace of the whole document, ``--set`` key spellings unchanged, and the same
refusal of an ``extends:`` write that materialization makes. A second writer
with its own spelling rules would mean ``osprey set connector=epics`` and
``osprey init --set connector=epics`` landing differently in the same file.

One shorthand is accepted as a key spelling: ``connector=``, folded into
``config.control_system.type`` by the shared layering step. It writes the
literal dotted key a reader would otherwise type by hand, so nothing lands in
the profile that only this command can understand.

``epics_gateway=<facility>`` was a second one. It expanded a table of named
facilities' gateway hostnames that core no longer carries — a gateway address
is site infrastructure, not something a framework can know — so it is refused
here with the dotted keys that do the job.
"""

from __future__ import annotations

from pathlib import Path

import click

from .output import note, report, warn
from .profile_expand import RETIRED_TEMPLATE_KEY
from .repo_resolver import PROFILE_FILENAME, find_repo_root, repo_option
from .styles import Styles

#: Retired shorthand. It stood for a facility out of a table of named
#: institutions' gateway hostnames that core no longer ships; refused rather
#: than passed through, because as a profile key it is one the next
#: ``osprey build`` rejects.
RETIRED_GATEWAY_KEY = "epics_gateway"

#: Where the EPICS connector's gateway table lives in the rendered config, as
#: the ``config.``-prefixed key path ``--set`` addresses it by.
GATEWAY_KEY_PREFIX = "config.control_system.connector.epics.gateways"


def _refuse_retired_shorthands(pairs: tuple[str, ...]) -> None:
    """Refuse the retired ``epics_gateway=`` spelling before anything is written.

    Runs before the shared parser sees anything, so the refusal costs no
    partial write: the whole command line is rejected the way a malformed pair
    is.
    """
    for pair in pairs:
        key, separator, _ = pair.partition("=")
        if separator and key.strip() == RETIRED_GATEWAY_KEY:
            raise click.UsageError(
                f"`{RETIRED_GATEWAY_KEY}=` named a facility out of a gateway table "
                "OSPREY no longer ships. Write the gateway your control network "
                "actually has:\n"
                f"  osprey set {GATEWAY_KEY_PREFIX}.read_only.address=gw.example.org "
                f"{GATEWAY_KEY_PREFIX}.read_only.port=5064"
            )


def _unrecognized_top_level_keys(pairs: tuple[str, ...]) -> list[str]:
    """Top-level keys in *pairs* that the profile schema does not recognize.

    The write itself is happy to put any key into the file — it is a YAML edit,
    not a schema-aware one — but the profile schema is CLOSED, so an unknown
    top-level key makes the very next ``osprey build`` refuse the whole profile.
    Without this the operator learns that at the build, from a message about
    ``profile.yml`` rather than about the command that just edited it.

    A warning rather than a refusal: the value did reach the file, saying so is
    the honest report, and an operator writing a key ahead of the release that
    reads it should not be blocked by this command. The build still refuses.

    ``config.``-prefixed keys are excluded — they address the rendered config,
    whose key space is open, and the profile's ``config:`` block holds them as
    the dotted paths they are.
    """
    from .build_profile import _KNOWN_PROFILE_KEYS

    # Deduplicated through a dict, not a `set()`: the command in this module is
    # NAMED `set`, so at module scope that name is a click Command and calling
    # it re-enters the CLI instead of building a set.
    unknown: dict[str, None] = {}
    for pair in pairs:
        key, separator, _ = pair.partition("=")
        if not separator:
            continue
        head = key.strip().split(".", 1)[0]
        if head and head != "config" and head not in _KNOWN_PROFILE_KEYS:
            unknown[head] = None
    return sorted(unknown)


#: Claude Code's own alias names. They name no model, so ``model=`` refuses them.
_CLAUDE_CODE_ALIAS_WORDS = ("haiku", "sonnet", "opus")


#: The key a pin on one agent's model is written under, as ``--set`` spells it.
AGENT_MODELS_KEY = "config.claude_code.agent_models"


def _read_profile(repo_root: Path) -> dict:
    """The parsed ``profile.yml``, or ``{}`` when it cannot be read as a mapping."""
    import yaml

    try:
        profile = yaml.safe_load((repo_root / PROFILE_FILENAME).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return profile if isinstance(profile, dict) else {}


def _served_models(repo_root: Path, provider: str) -> list[str]:
    """The model ids the catalog entry for *provider* lists; empty when it lists none."""
    from osprey.errors import BuildProfileError
    from osprey.profiles.providers import load_provider_catalog

    try:
        entry = load_provider_catalog(repo_root).entries.get(provider) or {}
    except BuildProfileError:
        entry = {}
    return [str(m) for m in entry.get("models") or []]


def _refuse_alias_word(key: str, model: str, provider: str, served: list[str]) -> None:
    """Refuse *model* when it is one of Claude Code's alias names rather than a model id."""
    if model in _CLAUDE_CODE_ALIAS_WORDS:
        raise click.UsageError(
            f"`{key}={model}` is not a model id. Provider '{provider}' serves: "
            f"{', '.join(served) or 'no listed models'}."
        )


def _configured_agents(profile: dict) -> list[str]:
    """The agent names the profile's ``config:`` block defines under ``claude_code.agents``."""
    config = profile.get("config")
    if not isinstance(config, dict):
        return []
    names: dict[str, None] = {}
    prefix = "claude_code.agents."
    for key, value in config.items():
        key = str(key)
        if key.startswith(prefix):
            names[key[len(prefix) :].split(".", 1)[0]] = None
        elif key == "claude_code.agents" and isinstance(value, dict):
            names.update(dict.fromkeys(str(name) for name in value))
        elif key == "claude_code" and isinstance(value, dict):
            agents = value.get("agents")
            if isinstance(agents, dict):
                names.update(dict.fromkeys(str(name) for name in agents))
    return sorted(names)


def _model_check(repo_root: Path, pairs: tuple[str, ...]) -> list[tuple[str, str, list[str]]]:
    """Each model id *pairs* writes, the provider it runs on, and what that provider serves.

    Covers ``model=`` and every ``config.claude_code.agent_models`` pin, written
    one agent at a time or as a whole mapping. The provider is a ``provider=``
    pair on the same command line, else the profile's own ``provider:``. Before
    anything is written, a bare alias word is refused with the ids the provider
    serves, and so is every pin ``osprey build`` refuses.

    Returns:
        One ``(model, provider, served)`` per id written, ``model=`` first and
        then the pins sorted by agent; empty when *pairs* sets no model and no
        pin, or no provider can be named.
    """
    import yaml

    from osprey.registry.mcp import FRAMEWORK_AGENTS

    from .validate_claude_artifacts import agent_model_pin_errors

    values: dict[str, str] = {}
    pins: dict[str, str] = {}
    for pair in pairs:
        key, separator, raw = pair.partition("=")
        if not separator:
            continue
        key = key.strip()
        value = yaml.safe_load(raw) if raw.strip() else None
        if key in ("model", "provider"):
            values[key] = "" if value is None else str(value)
        elif key == AGENT_MODELS_KEY and isinstance(value, dict):
            pins.update({str(a): str(m) for a, m in value.items() if m is not None})
        elif key.startswith(AGENT_MODELS_KEY + ".") and value is not None:
            pins[key[len(AGENT_MODELS_KEY) + 1 :]] = str(value)
    model = values.get("model")
    if not model and not pins:
        return []
    profile = _read_profile(repo_root)
    provider = values.get("provider") or profile.get("provider")
    if not provider:
        return []
    provider = str(provider)
    served = _served_models(repo_root, provider)

    if model:
        _refuse_alias_word("model", model, provider, served)
    for agent in sorted(pins):
        _refuse_alias_word(f"{AGENT_MODELS_KEY}.{agent}", pins[agent], provider, served)

    errors = agent_model_pin_errors(
        repo_root / "agents", pins, {*FRAMEWORK_AGENTS, *_configured_agents(profile)}
    )
    if errors:
        raise click.UsageError("\n".join(errors))

    checked = [(model, provider, served)] if model else []
    checked.extend((pins[agent], provider, served) for agent in sorted(pins))
    return checked


@click.command(name="set")
@click.argument("pairs", nargs=-1, metavar="KEY=VALUE...")
@repo_option
def set(pairs: tuple[str, ...], repo: Path | None) -> None:
    """Write settings into the deployment profile.

    Each KEY=VALUE is written into this repo's profile.yml in place, comments
    intact. That file is the source of truth, so this is the only command that
    edits configuration for you — the rendered build/config.yml is generated
    from it and is never hand-edited or CLI-edited. Run `osprey build` to carry
    a setting through to build/, then `osprey up` to deploy it.

    KEY is a top-level profile key (provider, model, tier, channel_finder_mode,
    connector) or a dotted path. `model` is a model id the provider serves, and so is
    each `config.claude_code.agent_models.<agent>` pin. Keys under `config.` address the rendered
    config: `config.control_system.type=epics` writes that literal dotted entry
    into the profile's config: block, replacing the value already there. A
    mapping value states the whole block at that key: `config.approval.tools={…}`
    replaces every `approval.tools.*` entry, `config.approval.tools.execute=skip`
    changes that one leaf.

    VALUE is read as YAML — true/false become booleans, bare numbers become
    numbers, everything else is text.

    One shorthand stands in for a longer key path: `connector=` writes
    config.control_system.type.

    Examples:

    \b
      $ osprey set model=claude-sonnet-5
      $ osprey set config.claude_code.agent_models.logbook-deep-research=claude-opus-5-5
      $ osprey set connector=epics
      $ osprey set tier=1 channel_finder_mode=in_context
      $ osprey set config.facility.name='Storage Ring'
      $ osprey set --repo ~/my-assistant config.control_system.writes_enabled=true
      $ osprey set config.control_system.connector.virtual_accelerator.writes_enabled=true
    """
    from osprey.deployment.staleness import check_drift
    from osprey.errors import BuildProfileError

    from .build_profile import write_back_cli_overrides

    if not pairs:
        raise click.UsageError(
            "Nothing to set. Name at least one KEY=VALUE, e.g. `osprey set model=claude-sonnet-5`.\n\n"
            "To see what the profile currently says, run `osprey config`."
        )

    # Resolved before anything is parsed: standing outside a deployment repo is
    # the operator's first problem, and a message about a malformed pair would
    # send them looking in the wrong place.
    repo_root = find_repo_root(repo)
    profile_path = repo_root / PROFILE_FILENAME

    _refuse_retired_shorthands(pairs)
    model_check = _model_check(repo_root, pairs)

    try:
        # Every pair is merged into one layer before any of it is written, so a
        # command line with one bad pair in it changes nothing at all.
        written = write_back_cli_overrides(profile_path, set_pairs=pairs)
    except BuildProfileError as e:
        raise click.UsageError(str(e)) from e

    report(f"✓ Wrote {len(written)} setting(s) into {profile_path}", style=Styles.SUCCESS)
    for key in written:
        note(key)
    for model, provider, served in model_check:
        if served and model not in served:
            note(
                f"{model} is not in provider '{provider}''s served list; the build trusts the gateway."
            )

    unrecognized = _unrecognized_top_level_keys(pairs)
    # The retired app-template key is unknown for a REASON the generic advice
    # gets wrong: prefixing it with `config.` writes a key nothing reads, and
    # what the operator actually wants is the verb that fills the profile in.
    # Its own message says so, and it is the same one the loader refuses with.
    if RETIRED_TEMPLATE_KEY in unrecognized:
        from .build_profile_load import _RETIRED_APP_TEMPLATE_REFUSAL

        unrecognized.remove(RETIRED_TEMPLATE_KEY)
        warn(f"Not a profile key: {RETIRED_TEMPLATE_KEY}", _RETIRED_APP_TEMPLATE_REFUSAL)
    if unrecognized:
        warn(
            f"Not a profile key: {', '.join(unrecognized)}",
            "It was written, but the profile schema is closed, so `osprey build` will "
            "refuse this profile until it is corrected or removed. To address the "
            "rendered config instead, prefix the key with `config.`.",
        )

    # The build this edit just invalidated, named while the operator is still
    # here. `check_drift` never raises, so a repo with no build/ — the common
    # case right after `init` — reports that rather than failing the write.
    report(check_drift(repo_root).status_line)
