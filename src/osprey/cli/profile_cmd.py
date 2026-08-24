"""Profile command group — inspect build profiles, and materialize one.

A build profile is the editable source a deployment owns: the ``profile.yml`` at
its repo root plus the data tree and convention directories beside it. This
group holds the read-only verbs that act on that source, kept separate from
``osprey build``, which consumes a profile and renders ``build/`` from it.

The write half lives elsewhere by design: ``osprey init`` creates the deployment
repo (through :func:`_materialize_profile_directory` here, which is the one
function in this module that writes anything) and ``osprey set`` edits the
profile.

``osprey validate`` is the top-level spelling of ``profile validate``, and the
check both run is :func:`osprey.cli.validate_cmd.check_profile_file` — one
implementation, because the only thing that differs between the spellings is
whether the target is required. ``presets`` is the other end of the same rule:
:func:`echo_preset_names` here is also what ``osprey init --list-presets``
prints.

Usage:
    osprey profile presets
    osprey profile validate ~/deployments/als-assistant
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import click

from osprey.errors import BuildProfileError
from osprey.utils.logger import get_logger

from .output import report, section
from .profile_conventions import (
    BUILD_OUTPUT_DIR,
    CONTEXT_BASELINE_FILENAME,
    PER_USER_CONTEXT_DIRNAME,
    context_baseline_slot,
)

if TYPE_CHECKING:
    # Annotation only — the profile model is imported lazily inside the command
    # bodies to keep `osprey --help` off the build-profile import chain (the
    # lazy-import budget test in tests/cli/test_main.py pins this).
    from .build_profile_model import BuildProfile
    from .templates.manager import TemplateManager

logger = get_logger("profile")


@click.group()
def profile() -> None:
    """Author, validate, and inspect build profiles."""


def echo_preset_names() -> None:
    """Print every bundled preset name, one per line.

    Two surfaces ask this question — ``osprey profile presets`` below and
    ``osprey init --list-presets``, an eager flag that answers before anything
    else parses — and an operator comparing the two lists is entitled to the
    same answer from both. One writer, so there is nothing to drift.
    """
    from .build_profile import list_presets

    for name in list_presets():
        report(name)


@profile.command()
def presets() -> None:
    """List bundled preset names, one per line.

    Every name printed here is usable as 'osprey init --preset NAME'.
    """
    echo_preset_names()


@profile.command()
def artifacts() -> None:
    """List every artifact the six profile lists can name.

    One section per list key (hooks, rules, skills, agents, output_styles,
    web_panels), each entry with a one-line description. The same menu
    appears as commented entries in an emitted profile.yml.
    """
    from .build_profile_emit import artifact_menu_catalog

    for kind, entries in artifact_menu_catalog().items():
        section(kind, list(entries))


@profile.command()
@click.argument("target", type=click.Path(exists=True, path_type=Path))
def validate(target: Path) -> None:
    """Check a profile without building anything.

    TARGET is a profile directory (its profile.yml is used) or a path to a
    profile file. Resolves 'extends:' chains and runs the full consistency
    check — convention directories, the 'data:' tree, service templates,
    lifecycle steps, env vars — reporting every problem found, not just the
    first.

    Exits 0 when the profile is valid, 2 with the accumulated errors when it is
    not.

    Examples:

    \b
      $ osprey profile validate ~/deployments/als-assistant
      $ osprey profile validate ~/deployments/als-assistant/personas/readonly.yml
    """
    # The check itself belongs to the verb, not to either spelling of it: this
    # command differs from `osprey validate` only in requiring its TARGET, so
    # that requirement is the whole of what lives here.
    from .validate_cmd import check_profile_file, profile_file_at

    check_profile_file(profile_file_at(target))


# Directory name the materialized data tree gets, and the value written to the
# profile's ``data:`` key. One constant so the copy target and the emitted key
# cannot drift apart.
_PROFILE_DATA_DIRNAME = "data"

# Sibling persona deltas live here, one file per web-terminal persona. A file
# in this directory is merged over the `profile.yml` beside it (FR-10), so the
# whole stack shares ONE facility data tree and ONE set of convention dirs: edit
# `<profile>/` once and every persona render sees it.
_PERSONA_PROFILE_DIRNAME = "personas"

# File name the resolved trigger config gets at the profile root, and the value
# written to the emitted `dispatch.triggers` key (FR-3). One constant so the
# copy target and the emitted key cannot drift apart.
_PROFILE_TRIGGERS_FILENAME = "triggers.yml"

# Convention directory holding one subdirectory of per-user web-terminal context
# per roster user. Named from the convention table rather than spelled again, so
# the slots seeded here are the ones the build copies from.
_CONTEXT_CONVENTION_DIRNAME = PER_USER_CONTEXT_DIRNAME

# The deployment's secret channel (FR-1). `.env.example` is the documented
# variable list, rendered from the SAME template every other render uses, so the
# two can never document different variables. `.env` beside it holds the values
# and IS the deployment's secret store — the file compose is handed as
# `--env-file` and every `${VAR}` expansion reads. Nothing derives a second copy
# of it, and a build only ever appends, which is what makes a secret survive a
# rebuild.
_PROFILE_ENV_FILENAME = ".env"
_PROFILE_ENV_EXAMPLE_FILENAME = ".env.example"
_ENV_EXAMPLE_TEMPLATE = "project/env.example.j2"

# Section header the shell-harvested keys are written under, distinct from the
# banner `osprey up` appends its minted service tokens beneath: the two have
# different origins, and a reader should be able to tell which values came from
# their own shell. Named after the verb that actually ran, because the banner is
# what tells a reader months later where a value in their `.env` came from — and
# an emitted artifact naming a command that does not exist is worse than no
# attribution at all (SC-8).
REPO_SEEDED_ENV_BANNER = "# ── Seeded by `osprey init` from your shell ──"

# Section header for values the profile itself declares under `env.defaults`.
# A third origin, and a third banner for the same reason the two above are
# distinct: these came from the preset's author, not from this operator's shell
# and not from a deploy mint, and the reader deciding whether a value is safe
# to change needs to know that.
PROFILE_DEFAULTS_ENV_BANNER = "# ── Declared by this profile (env.defaults) — edit freely ──"

#: Source-zone entries a repo-root materialization owns, and therefore the exact
#: set a re-materialization is allowed to replace. Everything else in a
#: deployment repo — ``.git``, ``.env``, ``var/``, ``build/``, ``ci-extra.yml``,
#: the CI files — belongs to the operator or to another command, so re-running
#: ``osprey init`` over an existing repo can never cost a secret, an agent's
#: memory, or a hand-written CI job.
MATERIALIZED_SOURCE_ENTRIES: tuple[str, ...] = (
    "profile.yml",
    _PROFILE_DATA_DIRNAME,
    _PERSONA_PROFILE_DIRNAME,
    _PROFILE_TRIGGERS_FILENAME,
    _CONTEXT_CONVENTION_DIRNAME,
    _PROFILE_ENV_EXAMPLE_FILENAME,
)

# Build exhaust that a source checkout may hold inside a bundle's data tree but
# a wheel install never does — hatch excludes it from the package (see the
# `exclude` in pyproject.toml). Copying it would make an emission from a
# checkout differ from an emission from a wheel, so the copy drops it. Paths are
# relative to the data root, matched segment-wise so an unrelated `results/`
# elsewhere in the tree is untouched.
_EXCLUDED_DATA_SUBTREES: tuple[tuple[str, ...], ...] = (("benchmarks", "results"),)


def _directory_derived_name(dirname: str) -> str:
    """The facility's display name, read off the directory the operator chose.

    ``my-facility`` becomes ``My Facility``. A guess, and a replaceable one —
    ``--set name=...`` overrides it, and the emitted ``profile.yml`` carries the
    result as an ordinary editable key.
    """
    return dirname.replace("-", " ").replace("_", " ").title()


def _data_copy_ignore(source_root: Path) -> Callable[[str, list[str]], set[str]]:
    """``shutil.copytree`` ignore callable dropping :data:`_EXCLUDED_DATA_SUBTREES`.

    Matching is by position in the tree, not by name: only the directory at
    exactly ``benchmarks/results`` under ``source_root`` is dropped, so a
    ``results/`` a bundle legitimately ships anywhere else comes across.
    """
    root = source_root.resolve()

    def ignore(directory: str, names: list[str]) -> set[str]:
        try:
            here = Path(directory).resolve().relative_to(root).parts
        except ValueError:
            # Outside the tree (a symlinked subdirectory): nothing to drop.
            return set()
        return {
            subtree[-1]
            for subtree in _EXCLUDED_DATA_SUBTREES
            if subtree[:-1] == here and subtree[-1] in names
        }

    return ignore


def _persona_catalog_layer(persona_names: Iterable[str], *, repo_name: str) -> dict[str, Any]:
    """A raw profile fragment repointing each persona's ``build_profile`` at its
    emitted sibling profile.

    Written at the DEEPEST spelling on purpose. ``_collapse_config_prefixes``
    resolves a prefix pair deeper-key-wins, so these keys survive whatever
    shallower spelling the preset used for the module subtree
    (``modules.web_terminals:``, or even a nested ``modules:``) and land inside
    it — while a shallower fragment of ours would instead be overwritten
    wholesale by the preset's own subtree.

    Args:
        persona_names: Personas the profile's catalog declares.
        repo_name: Directory name of the deployment repo. A persona render is
            build output like every other render, so its ``project_path`` lands
            under the repo's ``build/`` zone and its ``project`` is keyed off
            the deployment's own name rather than the preset's. This is why a
            shipped preset cannot spell either value correctly on its own —
            neither is knowable until a repo has a name.
    """
    layer: dict[str, str] = {}
    for name in persona_names:
        prefix = f"modules.web_terminals.personas.{name}"
        layer[f"{prefix}.build_profile"] = f"{_PERSONA_PROFILE_DIRNAME}/{name}.yml"
        # `project` must equal `project_path`'s basename. Both are derived from
        # the repo name here, and `osprey build` derives the render's own name
        # the same way, which is how the render lands exactly where the catalog
        # mounts it.
        layer[f"{prefix}.project"] = f"{repo_name}-{name}"
        layer[f"{prefix}.project_path"] = f"{BUILD_OUTPUT_DIR}/{repo_name}-{name}"
    return {"config": layer}


def _triggers_source(resolved: BuildProfile, preset_dir: Path) -> Path | None:
    """The trigger-config file ``resolved``'s ``dispatch:`` block names.

    ``None`` for a profile that declares no dispatch block — there is nothing to
    materialize and nothing to repoint.

    Resolution mirrors the build's exactly
    (:func:`~osprey.cli.build_injectors._inject_dispatch`): profile-relative
    first, then the bundled triggers directory. ``resolve_build_profile`` has
    already rejected a value that resolves to neither, so a miss here is a
    packaging problem rather than something the caller could have got wrong.

    Raises:
        BuildProfileError: If neither candidate exists.
    """
    if resolved.dispatch is None:
        return None

    from .build_profile_presets import _triggers_dir

    for candidate in (
        preset_dir / resolved.dispatch.triggers,
        _triggers_dir() / resolved.dispatch.triggers,
    ):
        if candidate.is_file():
            return candidate
    raise BuildProfileError(
        f"dispatch.triggers not found: {resolved.dispatch.triggers!r} — looked in "
        f"{preset_dir} and the bundled triggers directory."
    )


def _roster_user_names(config: Mapping[str, Any]) -> list[str]:
    """Web-terminal user names a profile's roster declares, in roster order.

    Empty for a profile with no web-terminal module and for one whose module is
    switched off — a persona delta, say, which attaches to a hosting project's
    web tier and stands up no roster of its own.

    Only the *route* to the subtree is this function's own: a profile spells its
    settings as a flat ``config:`` block, so the dotted keys have to be folded
    (:func:`~.build_profile_emit.effective_web_terminals`) before there is a
    subtree to read. Deriving names from it is
    :func:`~osprey.deployment.web_terminals.personas.roster_user_names`, the
    same call the build makes against the built project's ``config.yml`` — which
    is what makes the slots seeded here exactly the ones the build looks for.
    """
    from osprey.deployment.web_terminals.personas import roster_user_names

    from .build_profile_emit import effective_web_terminals

    return roster_user_names(effective_web_terminals(config))


def _stands_up_a_web_tier(config: Mapping[str, Any]) -> bool:
    """Whether the profile switches the web-terminal module on.

    An empty roster covers two cases the seeding has to tell apart: a profile
    with no web-terminal module at all, which has no context convention to
    seed, and one whose module is on but names no users yet, which does — its
    shared baseline is roster-independent, so it belongs in the profile whether
    or not anyone is on the roster. The subtree is read through the same fold
    :func:`_roster_user_names` uses, on the same "not switched on means no web
    tier" rule as
    :func:`~osprey.deployment.web_terminals.personas.roster_user_names`.
    """
    from .build_profile_emit import effective_web_terminals

    return bool(effective_web_terminals(config).get("enabled"))


# The two config paths a profile names a provider at. `claude_code.provider`
# picks the one the agent runs on; every entry under `api.providers` is a
# provider the profile configures (a proxy's base_url, its model tier map), and
# a configured provider is a referenced one. Kept as segment tuples because a
# `config:` block addresses paths, not strings.
_AGENT_PROVIDER_PATH = ("claude_code", "provider")
_API_PROVIDERS_PATH = ("api", "providers")


def _config_node(path: tuple[str, ...], value: Any, wanted: tuple[str, ...]) -> Any:
    """What a single ``config:`` key sets at ``wanted``, or ``None``.

    A ``config:`` block addresses paths in whatever spelling its author chose:
    the dotted key itself (``claude_code.provider``), or an ancestor key
    (``claude_code``) carrying the path nested inside its value. Both are read
    here, so a profile cannot hide a provider selection behind a spelling.

    A key DEEPER than ``wanted`` addresses something inside the value rather
    than the value itself and returns ``None``; :func:`_config_entry_names`
    handles the one case where that is meaningful.
    """
    if path == wanted:
        return value
    if wanted[: len(path)] != path:
        return None
    probe: Any = value
    for part in wanted[len(path) :]:
        probe = probe.get(part) if isinstance(probe, Mapping) else None
    return probe


def _config_entry_names(path: tuple[str, ...], value: Any, wanted: tuple[str, ...]) -> set[str]:
    """The names a single ``config:`` key puts in the mapping at ``wanted``.

    ``api.providers`` is a mapping keyed by provider name, and a profile may
    populate it either wholesale (a mapping value) or one leaf at a time
    (``api.providers.my-proxy.base_url``), so both spellings have to yield the
    same names.
    """
    depth = len(wanted)
    if path[:depth] == wanted and len(path) > depth:
        return {path[depth]}
    node = _config_node(path, value, wanted)
    if isinstance(node, Mapping):
        return {name for name in node if isinstance(name, str)}
    return set()


def _providers_named_by(provider: Any, config: Any) -> set[str]:
    """Provider names one profile layer selects or configures.

    Args:
        provider: The layer's top-level ``provider:`` key.
        config: The layer's ``config:`` block.

    Returns:
        Every provider name the layer references, by any spelling. A union
        rather than a resolution: this decides which secrets a profile may
        need, so a name that only one spelling reaches still counts.
    """
    names: set[str] = set()
    if isinstance(provider, str) and provider.strip():
        names.add(provider.strip())
    if not isinstance(config, Mapping):
        return names
    for key, value in config.items():
        if not isinstance(key, str):
            continue
        path = tuple(key.split("."))
        selected = _config_node(path, value, _AGENT_PROVIDER_PATH)
        if isinstance(selected, str) and selected.strip():
            names.add(selected.strip())
        names |= _config_entry_names(path, value, _API_PROVIDERS_PATH)
    return names


def _parsed_persona_deltas(persona_texts: Mapping[str, str]) -> dict[str, Mapping[str, Any]]:
    """Parse every emitted persona delta, naming the file a bad one came from.

    A delta gets no resolution round-trip of its own — it is meaningless without
    the host beside it — so this parse is the ONLY check that the line-level
    ``extends:`` surgery (:func:`~.build_profile_emit.emit_persona_delta_yaml`)
    left valid YAML behind. It therefore happens once, here, and its result
    serves every later reader: a second parse elsewhere would either raise a
    bare ``YAMLError`` first or silently disagree with this one.

    A delta that parses to something other than a mapping (an empty file, a
    stray scalar) is carried as an empty mapping: readers ask it for keys, and
    the emitted text is written out either way.

    Raises:
        BuildProfileError: If any delta is not valid YAML.
    """
    import yaml

    parsed: dict[str, Mapping[str, Any]] = {}
    for persona_name, persona_text in persona_texts.items():
        try:
            delta = yaml.safe_load(persona_text)
        except yaml.YAMLError as e:
            raise BuildProfileError(
                f"Emitted persona delta {_PERSONA_PROFILE_DIRNAME}/{persona_name}.yml "
                f"is not valid YAML: {e}"
            ) from e
        parsed[persona_name] = delta if isinstance(delta, Mapping) else {}
    return parsed


def _referenced_providers(
    resolved: BuildProfile, persona_deltas: Mapping[str, Mapping[str, Any]]
) -> set[str]:
    """Every provider the materialized profile — host and personas — references.

    The persona deltas are read too because they share this profile's ``.env``:
    a delta sits in ``personas/`` and anchors its secrets at the profile root
    (:func:`~.profile_root.resolve_profile_root`), so a persona that switches
    provider needs its key in the same file. A delta that overrides neither key
    inherits the host's selection, which is already counted.

    Args:
        resolved: The resolved host profile.
        persona_deltas: The parsed deltas (:func:`_parsed_persona_deltas`).
            Parsed rather than raw text so the one parse that validates them
            is the one read here.
    """
    names = _providers_named_by(resolved.provider, resolved.config)
    for delta in persona_deltas.values():
        names |= _providers_named_by(delta.get("provider"), delta.get("config"))
    return names


class _ShellProviderKeys(NamedTuple):
    """The exported provider keys, split by whether this profile references them."""

    seeded: dict[str, str]
    """``{VAR: value}`` to write into the profile ``.env``, in registry order."""

    skipped: tuple[str, ...]
    """Variables the shell exports for providers the profile never names."""


def _exported_provider_keys(providers: Collection[str]) -> _ShellProviderKeys:
    """Split the shell's provider API keys against the providers ``providers`` names.

    ``os.environ`` is the ONLY source (FR-1). A ``.env`` that happens to sit in
    whatever directory ``osprey init`` was run from is ambient state the profile
    cannot reproduce, so nothing is harvested from it.

    This is the one place the shell may seed a key, and it seeds the repo's own
    ``.env`` — the file an operator can read, edit, and account for, and the one
    store compose and every ``${VAR}`` expansion read. There is no second copy
    to derive: a build only ever appends to that file
    (:func:`~osprey.utils.dotenv.append_profile_env`), so a key seeded here
    survives every rebuild and a key that reaches a running deployment was
    written to this file first.

    Only the keys of providers the RESOLVED PROFILE references are seeded: a
    whole-keyring import copies secrets the profile has no use for into a file
    it then owns forever, which is more surface than the profile needs. The rest
    are reported by name rather than dropped silently (:func:`_skipped_keys_note`)
    — they were seen, and the operator decides whether the omission is right.

    The variable list comes from
    :func:`~.templates.scaffolding.provider_api_key_entries`, the same registry
    derivation the ``.env.example`` beside it is rendered from, so the file that
    holds the values and the file that documents them cannot name different
    variables.

    Args:
        providers: Provider names the profile references
            (:func:`_referenced_providers`).

    Returns:
        The exported keys split into ``seeded`` and ``skipped``. Both are empty
        when the caller exported none.
    """
    import os

    from .templates.scaffolding import provider_api_key_entries

    seeded: dict[str, str] = {}
    skipped: list[str] = []
    for entry in provider_api_key_entries():
        value = os.environ.get(entry["var"])
        if not value:
            continue
        if entry["provider"] in providers:
            seeded[entry["var"]] = value
        else:
            skipped.append(entry["var"])
    return _ShellProviderKeys(seeded, tuple(skipped))


def _skipped_keys_note(skipped: Collection[str]) -> str:
    """One line naming the exported keys this profile did not take.

    One wording, used by both the materializer's log and ``osprey init``'s
    summary: a skipped secret is a thing the operator has to be able to account
    for, and two spellings of the same fact read as two different facts.
    """
    return (
        f"Left out {', '.join(skipped)}. Your shell exports them, but this "
        f"assistant uses a different provider."
    )


def _write_secret_channel(
    target: Path,
    manager: TemplateManager,
    resolved: BuildProfile,
    profile_name: str,
    exported: Mapping[str, str],
    providers: Collection[str] = (),
) -> list[str]:
    """Write the profile's secret channel: ``.env.example``, ``.env``, ``.gitignore``.

    The profile owns its secrets (FR-1): the build derives a project's ``.env``
    from the one written here, so a value set once survives every rebuild.

    ``.env.example`` is always written, and comes from the project template
    (:data:`_ENV_EXAMPLE_TEMPLATE`) rather than from prose of its own — one
    template documents the variable set wherever it is rendered. ``.env`` is
    written ONLY when there is something to seed it with — shell-exported
    provider keys (``exported``) and/or the profile's own ``env.defaults``
    values: an empty secrets file reads as a configured one, and ``cp
    .env.example .env`` is the honest starting point when there is nothing to
    seed.

    ``env.defaults`` values are seeded as real starting values, under their own
    banner (:data:`PROFILE_DEFAULTS_ENV_BANNER`), because a preset declares one
    for exactly the deployments that should come up working without a hand
    edit — a demo login password, say. The append-only writer keeps every
    later authority intact: a value the operator has already set (or that
    ``osprey up`` minted) always wins over the declared default.

    Args:
        target: The profile directory, already created.
        manager: Template manager whose Jinja environment renders the example.
        resolved: The resolved profile, for the ``env:`` block the example
            documents.
        profile_name: Display name, for the example's title line.
        exported: Provider keys to seed (:attr:`_ShellProviderKeys.seeded`).
            Passed in rather than read here because the README rendered earlier
            says whether a ``.env`` was seeded, and the two must agree.

    Returns:
        Profile-relative names of the files written, for the caller's summary.
    """
    from osprey.utils.dotenv import append_profile_env

    from .templates.scaffolding import provider_api_key_entries, service_token_var_entries

    manager.render_template(
        _ENV_EXAMPLE_TEMPLATE,
        {
            "project_name": profile_name,
            "provider_api_keys": provider_api_key_entries(),
            # Which of those this profile actually uses, so the example puts
            # the one or two keys someone has to fill in above the rest rather
            # than in a list of eight they have to search. Empty from the
            # programmatic path, where the template lists them all as before.
            "active_provider_vars": [
                entry["var"]
                for entry in provider_api_key_entries()
                if entry["provider"] in providers
            ],
            "service_token_vars": service_token_var_entries(),
            # The profile's `env:` block, for the example's documentation —
            # the same two keys `osprey build` feeds this template. The
            # `env.defaults` VALUES are additionally seeded into `.env` below.
            "env_required": list(resolved.env.required or []),
            "env_defaults": dict(resolved.env.defaults or {}),
            # No project exists yet; the key is a commented hint either way.
            "project_root": "/path/to/your/project",
        },
        target / _PROFILE_ENV_EXAMPLE_FILENAME,
    )
    written = [_PROFILE_ENV_EXAMPLE_FILENAME]

    if exported:
        # Written through the same append-only, 0600, atomic path `osprey up`
        # uses for its write-back, so there is one writer discipline for the
        # profile `.env` rather than a second one that only new profiles get.
        append_profile_env(target / _PROFILE_ENV_FILENAME, exported, REPO_SEEDED_ENV_BANNER)
        written.append(_PROFILE_ENV_FILENAME)

    defaults = dict(resolved.env.defaults or {})
    if defaults:
        # Same writer, own banner: a declared default an operator has already
        # overridden (or that a mint got to first) is never rewritten.
        append_profile_env(target / _PROFILE_ENV_FILENAME, defaults, PROFILE_DEFAULTS_ENV_BANNER)
        if _PROFILE_ENV_FILENAME not in written:
            written.append(_PROFILE_ENV_FILENAME)

    return written


def _triggers_layer() -> dict[str, Any]:
    """A raw profile fragment repointing ``dispatch.triggers`` at the profile's own file.

    The materialized ``triggers.yml`` is what the build must read (FR-3), so the
    emitted key names it rather than the bundled trigger set it was copied from.
    Merged through the same channel as every other emitted value, so nothing
    here is a second path into the resolved content.
    """
    return {"dispatch": {"triggers": _PROFILE_TRIGGERS_FILENAME}}


def _off_chain_problem(persona_name: str, persona_preset: str, host_preset: str) -> str | None:
    """Why ``persona_preset`` cannot be emitted as a delta over ``host_preset``.

    ``None`` when it can: its ``extends`` names the host preset directly, so the
    preset's own layer IS the delta and nothing is lost by dropping the key.
    Two distinct failures get two distinct messages, because the fix differs —
    a preset that never reaches the host is pointed somewhere else entirely,
    while one that reaches it through an intermediate would emit a delta with
    that intermediate's layer silently missing.
    """
    from .build_profile_presets import (
        _load_preset_raw,
        _normalize_preset_name,
        _preset_extends_chain_reaches,
    )

    raw, _path = _load_preset_raw(persona_preset)
    parent = raw.get("extends")
    if isinstance(parent, str) and parent and _normalize_preset_name(parent) == host_preset:
        return None
    if _preset_extends_chain_reaches(persona_preset, host_preset):
        return (
            f"{persona_name!r}: build_profile {persona_preset!r} reaches {host_preset!r} only "
            f"through {_normalize_preset_name(str(parent))!r}. A persona file holds its own "
            f"layer and nothing else, so emitting one here would drop that preset's settings "
            f"— point the catalog entry at a preset that extends {host_preset!r} directly, or "
            f"drop this entry from the catalog (a `-O` override removing it), materialize, "
            f"then hand-write {_PERSONA_PROFILE_DIRNAME}/{persona_name}.yml as a delta over "
            f"profile.yml and add the entry back to the emitted profile"
        )
    return (
        f"{persona_name!r}: build_profile {persona_preset!r} does not extend {host_preset!r}, "
        f"so it is not a delta over this profile — a persona file is merged over the profile "
        f"beside it, and this preset carries its own base. Point the catalog entry at a preset "
        f"that extends {host_preset!r}, or materialize that preset as a profile of its own"
    )


def _privileged_persona_problems(
    config: Mapping[str, Any],
    privileges: Mapping[str, tuple[str, ...]],
    absolute_privileges: Mapping[str, tuple[str, ...]],
) -> list[str]:
    """Why this profile's roster must not deploy as written, from the tiers it names.

    The build-time half of the deployment-editing guard, run where the persona
    presets have just been resolved and their privileges are therefore known
    exactly (see
    :func:`~osprey.deployment.web_terminals.personas.persona_privileges`). Two
    rules, both owned by that module so the lint belt behind this reports the
    same thing in the same words:

    * ``default_persona`` must not be a privileged tier — it is what every entry
      that names no tier inherits;
    * no entry served without a login may resolve to one.

    **The two rules read two different maps**, and handing one map to both is
    what left this surface exempt on a floorless host preset: every persona
    there sits at the baseline, so the relative map is empty for all of them and
    the login rule had nothing to report about a terminal holding everything.
    See
    :func:`~osprey.deployment.web_terminals.personas.privileges_beyond_baseline`
    for why the ``default_persona`` rule keeps the relative reading.

    Args:
        config: The host profile's ``config:`` block, from which the roster and
            the ``auth`` stanza are folded out — and the baseline both the
            relative map and the remedy are measured against.
        privileges: Persona name → what it holds BEYOND that baseline, for the
            personas whose presets resolved. A persona absent from this mapping
            contributes nothing — its entry already failed to resolve and is
            reported as such.
        absolute_privileges: Persona name → what it holds outright, same
            personas. What the login rule is judged on.

    Returns:
        One message per problem, ``default_persona`` first, then roster order.
    """
    from osprey.deployment.web_terminals.personas import (
        persona_privileges,
        privileged_default_persona_problem,
        resolve_personas,
        unauthenticated_privileged_terminal_problems,
    )

    from .build_profile_emit import effective_web_terminals

    web_terminals = effective_web_terminals(config)
    problems: list[str] = []
    default_persona = web_terminals.get("default_persona")
    default_problem = privileged_default_persona_problem(
        default_persona,
        privileges.get(default_persona, ()) if isinstance(default_persona, str) else (),
    )
    if default_problem is not None:
        problems.append(default_problem)

    # Lenient resolution: an entry naming a persona that is not in the catalog
    # is a different mistake, reported by the deploy lint that owns it, and
    # raising here would replace that report with a worse one.
    resolved = resolve_personas(web_terminals, {}, "", strict=False)
    problems.extend(
        unauthenticated_privileged_terminal_problems(
            resolved,
            absolute_privileges,
            # The host profile's own posture, which decides which remedy is
            # honest: a profile that floors nothing has no unprivileged tier to
            # point the entry at, so it is told how to create one instead.
            baseline_privileges=persona_privileges(config),
        )
    )
    return problems


def _persona_profile_texts(
    resolved: BuildProfile,
    profile_name: str,
    persona_path_prefix: str,
    host_preset: str,
) -> dict[str, str]:
    """Emit one delta text per persona the profile deploys.

    Empty unless the profile stands up a persona stack of its own (see
    :func:`~osprey.cli.build_profile_emit.emits_persona_profiles`) — a persona
    preset inherits the catalog but disables the module, and emitting from one
    of those would produce personas-of-a-persona.

    ``persona_path_prefix`` is what the emitted headers prefix ``personas/``
    with when they name a persona file: the profile's directory name and a
    slash for a nested profile directory, and empty for a repo root, where the
    reader already stands where those paths are relative to.

    Each entry is emitted as a pure DELTA — the persona preset's own layer, no
    ``extends:`` (:func:`~.build_profile_emit.emit_persona_delta_yaml`) — over
    the host profile it sits beside. The host therefore stays the single source
    of truth: edits there, and the caller's baked ``-O``/``--set`` layers with
    them, reach every persona through the implicit merge instead of being copied
    around. A catalog entry whose preset is not a delta over ``host_preset``
    (the bundled shape: ``control-assistant-readonly`` over
    ``control-assistant``) is rejected rather than approximated — see
    :func:`_off_chain_problem`.

    One thing beyond emission is decided here, because this is the only moment
    the resolved persona tiers are all in hand: whether the roster hands
    deployment editing to a terminal that must not have it (see
    :func:`_privileged_persona_problems`). Refusing at materialization is what
    keeps the bad roster from reaching a repo at all.

    Raises:
        click.UsageError: With every unusable catalog entry named at once — or,
            once they all resolve, with every privileged-terminal problem named
            at once.
    """
    from osprey.deployment.web_terminals.personas import (
        persona_privileges,
        privileges_beyond_baseline,
    )

    from .build_profile import resolve_build_profile
    from .build_profile_emit import emit_persona_delta_yaml, persona_catalog
    from .build_profile_presets import _normalize_preset_name

    catalog = persona_catalog(resolved.config)
    texts: dict[str, str] = {}
    problems: list[str] = []
    # What each persona can edit, recorded as its preset resolves — this is the
    # one moment the whole composed tier is in hand, and the emitted delta
    # deliberately drops the `extends:` that produced it. Recorded BOTH ways,
    # because the two rules in `_privileged_persona_problems` read two different
    # answers: relative to the host for the inherited `default_persona`,
    # absolute for the `login: false` key an author typed on purpose. See
    # `privileges_beyond_baseline`.
    host_privileges = persona_privileges(resolved.config)
    privileges: dict[str, tuple[str, ...]] = {}
    absolute: dict[str, tuple[str, ...]] = {}
    for persona_name in sorted(catalog):
        preset_ref = catalog[persona_name].get("build_profile")
        # `..` needs naming explicitly: `Path("..").name` is `".."`, not the
        # empty string, so the plain-name check below passes it through and the
        # emission would write a `personas/..yml` nobody meant.
        if (
            not persona_name
            or persona_name in (".", "..")
            or Path(persona_name).name != persona_name
        ):
            problems.append(
                f"{persona_name!r}: the persona name becomes a file name under "
                f"{_PERSONA_PROFILE_DIRNAME}/, so it must be a plain name — no "
                "path separators, and not empty"
            )
            continue
        if not isinstance(preset_ref, str) or not preset_ref:
            problems.append(
                f"{persona_name!r}: no build_profile, so there is no preset to "
                "materialize its profile from"
            )
            continue
        try:
            persona_resolved, _preset_dir = resolve_build_profile(None, preset_ref)
        except BuildProfileError as e:
            problems.append(f"{persona_name!r}: build_profile {preset_ref!r} does not resolve: {e}")
            continue
        if persona_resolved.data_bundle != resolved.data_bundle:
            # The sibling profiles share ONE data tree, materialized from the
            # host's app template. A persona rendering a different template
            # would read a tree that was never built for it — caught here
            # rather than surfacing as missing files at deploy time.
            problems.append(
                f"{persona_name!r}: build_profile {preset_ref!r} renders app template "
                f"{persona_resolved.data_bundle!r}, but this profile materializes "
                f"{resolved.data_bundle!r} — one shared data tree cannot serve both"
            )
            continue
        persona_preset = _normalize_preset_name(preset_ref)
        off_chain = _off_chain_problem(persona_name, persona_preset, host_preset)
        if off_chain is not None:
            problems.append(off_chain)
            continue
        held = persona_privileges(persona_resolved.config)
        if held:
            absolute[persona_name] = held
        lifted = privileges_beyond_baseline(held, host_privileges)
        if lifted:
            privileges[persona_name] = lifted
        texts[persona_name] = emit_persona_delta_yaml(
            preset_name=persona_preset,
            profile_name=f"{profile_name} ({persona_name})",
            profile_filename=(
                f"{persona_path_prefix}{_PERSONA_PROFILE_DIRNAME}/{persona_name}.yml"
            ),
        )
    if problems:
        raise click.UsageError(
            "Cannot materialize the persona profiles this profile's web-terminal "
            "catalog calls for:\n  - " + "\n  - ".join(problems)
        )
    # Reported only once every entry resolved: a capability verdict drawn from a
    # catalog half of which did not resolve would name the wrong tiers, and the
    # failure above is the one the operator has to fix first anyway.
    privilege_problems = _privileged_persona_problems(resolved.config, privileges, absolute)
    if privilege_problems:
        raise click.UsageError(
            "This profile's web-terminal roster would hand deployment editing to a "
            "terminal that must not have it:\n  - " + "\n  - ".join(privilege_problems)
        )
    return texts


def _cleanup(target: Path) -> str:
    """Remove what a failed materialization wrote, and say what is left.

    Only the entries a materialization owns (:data:`MATERIALIZED_SOURCE_ENTRIES`)
    are removed: the target is a deployment repo root, which routinely holds an
    operator's own files — a ``.git``, an ``.env``, a clone's README — and a
    failed run must never cost one of those.

    Returns:
        A sentence for the refusal it is appended to, naming anything that could
        not be removed so the operator knows a retry will refuse too.
    """
    import shutil

    for name in MATERIALIZED_SOURCE_ENTRIES:
        entry = target / name
        if entry.is_dir():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            entry.unlink(missing_ok=True)
    remaining = [name for name in MATERIALIZED_SOURCE_ENTRIES if (target / name).exists()]
    if remaining:
        return (
            f"Partly-written files remain in {target}: {', '.join(remaining)} — "
            f"remove them before retrying."
        )
    return "Nothing was materialized."


def _context_baseline_source(manager: TemplateManager, data_bundle: str) -> Path:
    """The ``base.md`` text a new profile's context baseline starts from.

    Bundle first: an app template that ships its own
    ``web-terminal-context/base.md`` (the control assistant does — its text
    describes that family's personas) seeds the profile with it. Any other
    bundle falls back to the framework's generic baseline, the same file
    ``osprey build`` installs into every render — so the materialized slot
    starts byte-identical to what the build would have used anyway, and every
    edit to it from then on is visible in the operator's own repo.
    """
    bundle = context_baseline_slot(
        manager.template_root / "apps" / data_bundle / PER_USER_CONTEXT_DIRNAME
    )
    if bundle.is_file():
        return bundle
    return context_baseline_slot(manager.template_root / "claude_code" / PER_USER_CONTEXT_DIRNAME)


def _packaged_data_source(manager: TemplateManager, data_bundle: str) -> Path:
    """The packaged ``data/`` tree this command copies verbatim.

    Checked up front, before anything is written, so a packaging regression
    surfaces as an actionable error here rather than as a missing-file error
    mid-copy. It cannot be caused by anything the caller passed.

    Args:
        manager: The :class:`~.templates.manager.TemplateManager` locating the
            installed template root.
        data_bundle: App template whose ``data/`` tree gets materialized.

    Raises:
        BuildProfileError: If the tree is absent from the installation.
    """
    data_source = Path(manager.template_root) / "apps" / data_bundle / "data"
    if not data_source.is_dir():
        raise BuildProfileError(
            f"App template {data_bundle!r} ships no data tree at {data_source}. "
            f"This is a packaging bug — reinstall osprey-framework."
        )

    return data_source


class _MaterializedProfile(NamedTuple):
    """What :func:`_materialize_profile_directory` produced, for the caller's summary."""

    target: Path
    """The resolved profile directory."""

    skipped_shell_keys: tuple[str, ...]
    """Exported provider keys left out because the profile references none of
    their providers (:func:`_exported_provider_keys`). Reported rather than
    returned as a courtesy: the seeded keys can be read back from the ``.env``,
    the ones that were deliberately not written cannot."""

    profile_name: str
    """The display name written into the profile — the caller's, the directory's,
    or whatever ``--set name=`` said. Returned because the caller cannot
    reconstruct which of those won."""

    deploy_declared: bool
    """Whether the emitted profile carries an ACTIVE ``deploy:`` block, read back
    from the file on disk rather than from the inputs. Presets ship the block
    commented out, so this is normally false on a fresh materialization; it
    decides whether there is anything to render a CI pipeline from."""

    resolved: BuildProfile
    """The materialized profile, resolved back from the written ``profile.yml``
    — the same read-back that validates the round-trip, so what the caller
    summarizes (the composition card) is a fact about what is on disk rather
    than about what the inputs asked for."""

    persona_deltas: dict[str, Mapping[str, Any]]
    """The emitted persona deltas, parsed, keyed by persona name — empty for a
    profile that emits none. The per-persona facts a summary needs (each tier's
    panels and its write posture) live in these, not in the host profile."""


def _materialize_profile_directory(
    target_dir: Path,
    preset_name: str,
    overrides: tuple[Path, ...] = (),
    set_pairs: tuple[str, ...] = (),
    *,
    profile_name: str | None = None,
) -> _MaterializedProfile:
    """Materialize an editable, standalone profile directory from ``preset_name``.

    Writes ``profile.yml`` — the preset's fully resolved content as an
    explicit, self-contained profile (comments preserved, no ``extends:``) —
    the bundle's ``data/`` tree copied verbatim, the profile's ``.env`` channel
    (:func:`_write_secret_channel`), and a tutorial ``README.md`` explaining the
    convention directories. ``-O`` files and ``--set`` pairs are merged with the
    same layering as the render path, so a validated build one-liner carries
    into the profile without hand-editing.

    Fail-before-mutating: the preset, its layers, and the rendered profile text
    are all produced before the first ``mkdir``, and anything that fails after
    it removes the target rather than leaving a half-materialized directory.
    With ``force``, an existing target is deleted at that same point — never
    earlier — and only when it is a materialized profile (has ``profile.yml``)
    or an empty directory, so a failed run or a mistyped target never costs an
    unrelated directory.

    Args:
        target_dir: The profile directory to create.
        preset_name: Bundled preset to materialize, in either spelling.
        overrides: ``-O`` files, layered in order.
        set_pairs: ``--set`` pairs, layered last.
        profile_name: Display name for the emitted profile. Defaults to one
            derived from the repo directory's own name. ``--set name=`` wins
            over both.

    Returns:
        What was written, for the caller's summary (:class:`_MaterializedProfile`).

    Raises:
        click.UsageError: For user errors — existing target, an ``extends``
            override, or layers that produce an invalid profile.
        BuildProfileError: For packaging problems (missing seed or data tree).
    """
    import shutil

    from .build_profile import (
        EXTENDS_OVERRIDE_REFUSAL,
        _normalize_preset_name,
        merge_cli_overrides,
        resolve_build_profile,
    )
    from .build_profile_emit import emit_standalone_profile_yaml
    from .templates.manager import TemplateManager

    # Resolving through the public path validates the preset AND its -O/--set
    # layers up front, and names the bundle whose data tree gets copied. It also
    # rejects a user-supplied `data:` in preset mode, which is right: this
    # command materializes the tree, so pointing it elsewhere is a mistake.
    # Everything it rejects is a user error, so it surfaces as one.
    try:
        resolved, preset_dir = resolve_build_profile(None, preset_name, overrides, set_pairs)
    except BuildProfileError as e:
        raise click.UsageError(f"Cannot materialize {preset_name!r}: {e}") from e

    baked = merge_cli_overrides({}, overrides, set_pairs)
    if "extends" in baked:
        # The shared refusal: the same override file must be answered the same
        # way here and on a later build's write-back into this profile.
        raise click.UsageError(EXTENDS_OVERRIDE_REFUSAL)
    name_override = baked.get("name")

    target = target_dir.resolve()

    normalized_preset = _normalize_preset_name(preset_name)
    # The caller's name, or one read off the directory the operator chose
    # (e.g. "my-profile" → "My Profile"). `--set name=` outranks both: it is an
    # explicit statement of the same fact.
    profile_name_default = profile_name or _directory_derived_name(target.name)
    if name_override is not None:
        profile_name_default = str(name_override)

    manager = TemplateManager()
    data_source = _packaged_data_source(manager, resolved.data_bundle)

    # How the emitted persona comments spell their own paths: repo-relative,
    # because the repo root is where a reader stands.
    persona_dirname = ""

    # A profile that deploys per-persona web terminals owns those personas too:
    # their profiles are materialized beside this one and the catalog is
    # repointed at them, so the whole stack reads this profile's data tree
    # rather than the bundled preset's (D7a). Emitted before the first mkdir,
    # like everything else here, so a bad catalog entry fails before mutating.
    # The trigger config a dispatch profile runs on is facility state, so the
    # profile owns a copy of it (FR-3) and the emitted key names that copy.
    # Located before the first mkdir like everything else here.
    triggers_src = _triggers_source(resolved, preset_dir)

    persona_texts = _persona_profile_texts(
        resolved, profile_name_default, persona_dirname, normalized_preset
    )
    # Parsed here, before the first mkdir and before `--force` replaces
    # anything: the parse is what validates the emitted deltas, so a bad one
    # must cost nothing. Its result is what every later reader sees.
    persona_deltas = _parsed_persona_deltas(persona_texts)

    extra_layers: tuple[dict[str, Any], ...] = (
        *((_persona_catalog_layer(persona_texts, repo_name=target.name),) if persona_texts else ()),
        *((_triggers_layer(),) if triggers_src is not None else ()),
    )

    # Read once, before anything is written: the README rendered below tells the
    # reader whether a `.env` was seeded for them, and the seeding itself happens
    # further down. Two reads of the environment could disagree.
    referenced_providers = _referenced_providers(resolved, persona_deltas)
    shell_keys = _exported_provider_keys(referenced_providers)
    exported_keys = shell_keys.seeded

    # Derived once for the same reason: the README lists the per-user context
    # slots and the loop below creates them, and a roster those two disagreed on
    # would document a directory that was never made.
    roster = _roster_user_names(resolved.config)

    # The materialized tree is what the build must read, so `data:` is emitted
    # as an active key — injected through the same --set layering a user would
    # use, rather than through a second path into the resolved content.
    profile_text = emit_standalone_profile_yaml(
        preset_name=normalized_preset,
        overrides=overrides,
        set_pairs=(*set_pairs, f"data={_PROFILE_DATA_DIRNAME}"),
        profile_name=profile_name_default,
        extra_layers=extra_layers,
        include_flow_diagram=True,
    )

    # The repo root is allowed to exist — it usually does (an empty clone, the
    # operator's own mkdir). Whether writing into THIS one is acceptable was
    # settled by the caller, which also owns what to clear if this fails.
    target.mkdir(parents=True, exist_ok=True)

    try:
        # Verbatim copy (D1/FR2): staging subdirectories and any stray `.j2`
        # come across byte-identical — a profile data tree is content, never
        # templates, so nothing here is rendered. The one exclusion is build
        # exhaust the wheel does not ship either (_EXCLUDED_DATA_SUBTREES).
        shutil.copytree(
            data_source,
            target / _PROFILE_DATA_DIRNAME,
            ignore=_data_copy_ignore(data_source),
        )
        (target / "profile.yml").write_text(profile_text, encoding="utf-8")

        if triggers_src is not None:
            shutil.copy2(triggers_src, target / _PROFILE_TRIGGERS_FILENAME)
            logger.debug("  Trigger config: %s", _PROFILE_TRIGGERS_FILENAME)

        # The profile owns its secrets from the first minute (FR-1) — the
        # documented variable list, whatever the shell already exports, and the
        # .gitignore that keeps the values out of version control.
        secret_files = _write_secret_channel(
            target,
            manager,
            resolved,
            profile_name_default,
            exported_keys,
            referenced_providers,
        )
        logger.debug("  Secrets: %s", ", ".join(secret_files))
        if shell_keys.skipped:
            # Debug only. `osprey init`'s summary prints the same sentence from
            # the same helper, and this is the only caller, so logging it here
            # at info put the fact on screen twice in one run.
            logger.debug("  %s", _skipped_keys_note(shell_keys.skipped))

        # One empty slot per roster user, so the per-user context a facility
        # writes has an obvious home from the first minute (FR-5). Only the
        # directories are seeded: what goes in them is the facility's, and the
        # build derives the copy from the roster it resolves at the time, so
        # nothing about the roster is frozen here.
        for user in roster:
            user_dir = target / _CONTEXT_CONVENTION_DIRNAME / user
            user_dir.mkdir(parents=True, exist_ok=True)
            (user_dir / ".gitkeep").touch()
        if _stands_up_a_web_tier(resolved.config):
            # The shared baseline, beside the slots — the one seeded entry that
            # carries content, because it IS content the deployment ships: the
            # build copies this file over the framework's fallback, so the text
            # every seeded user starts from lives in the operator's repo where
            # they can read and edit it. Seeded on the module rather than the
            # roster, because the copy side is roster-independent too
            # (plan_convention_copies): a profile that names no users yet still
            # starts from text it can see.
            context_dir = target / _CONTEXT_CONVENTION_DIRNAME
            context_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(
                _context_baseline_source(manager, resolved.data_bundle),
                context_baseline_slot(context_dir),
            )
            logger.debug(
                "  Per-user context: %s (+ shared %s)",
                ", ".join(f"{_CONTEXT_CONVENTION_DIRNAME}/{user}/" for user in roster)
                or "no roster yet",
                CONTEXT_BASELINE_FILENAME,
            )

        if persona_texts:
            # Validity was settled by `_parsed_persona_deltas` above, before the
            # first mkdir — nothing reaching here is unparsed, so these writes
            # need no guard of their own.
            persona_dir = target / _PERSONA_PROFILE_DIRNAME
            persona_dir.mkdir()
            for persona_name, persona_text in persona_texts.items():
                (persona_dir / f"{persona_name}.yml").write_text(persona_text, encoding="utf-8")
            logger.debug(
                "  Persona deltas: %s",
                ", ".join(f"{_PERSONA_PROFILE_DIRNAME}/{name}.yml" for name in persona_texts),
            )

        # The round-trip runs last because it validates `data:` against the tree
        # that must already be on disk. Only the host profile is resolved: a
        # persona file is a delta, meaningless on its own, and resolving one is
        # resolving the host with that delta merged in — which the host's own
        # round-trip already covers.
        #
        # Its result is what the caller is told about the profile: read back
        # from the written file, so "declares a deploy block" is a fact about
        # what is on disk rather than about what the inputs asked for.
        written, _written_dir = resolve_build_profile((target / "profile.yml").resolve(), None)
    except BuildProfileError as e:
        # Emission round-trips for every bundled preset (guarded by tests), so
        # with layers present they are the thing to look at; without them this
        # is a framework bug and blaming the user's flags would misdirect.
        blame = (
            "Overrides produce an invalid profile"
            if (overrides or set_pairs)
            else "The materialized profile does not validate"
        )
        raise click.UsageError(f"{blame}: {e}\n{_cleanup(target)}") from e
    except Exception:
        # Any other failure (a copy error, a full disk) must not leave a
        # half-materialized directory that looks buildable.
        _cleanup(target)
        raise

    logger.debug("Wrote profile directory: %s", target)
    return _MaterializedProfile(
        target,
        shell_keys.skipped,
        profile_name_default,
        written.deploy is not None,
        written,
        persona_deltas,
    )
