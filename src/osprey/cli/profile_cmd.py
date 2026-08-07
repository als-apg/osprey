"""Profile command group — author and inspect build profiles.

A build profile is the editable source a facility owns: a ``profile.yml`` plus
the data tree and convention directories beside it. This group holds the verbs that act on
that source, keeping them separate from ``osprey build``, which consumes a
profile and derives a project from it.

Usage:
    osprey profile presets
    osprey profile validate my-profile/
    osprey profile new my-profile --preset control-assistant
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import click

from osprey.errors import BuildProfileError
from osprey.utils.logger import get_logger

from .profile_conventions import PER_USER_CONTEXT_DIRNAME

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


def _resolve_profile_file(target: Path) -> Path:
    """Return the profile file named by *target* (a profile file or its directory).

    A profile directory is the unit users work with, so both spellings are
    accepted: the directory itself (``profile.yml`` inside it, the name
    ``profile new`` writes) or an explicit path to any profile file.

    Raises:
        click.UsageError: When *target* is a directory without a profile.yml.
    """
    if target.is_dir():
        candidate = target / "profile.yml"
        if not candidate.is_file():
            raise click.UsageError(
                f"No profile.yml in {target}. Pass the profile file directly, or "
                f"materialize a profile directory with "
                f"`osprey profile new {target} --preset <NAME>`."
            )
        return candidate.resolve()
    return target.resolve()


@profile.command()
def presets() -> None:
    """List bundled preset names, one per line.

    Every name printed here is usable as ``--preset NAME`` for
    ``osprey profile new`` and ``osprey build``.
    """
    from .build_profile import list_presets

    for name in list_presets():
        click.echo(name)


@profile.command()
@click.argument("target", type=click.Path(exists=True, path_type=Path))
def validate(target: Path) -> None:
    """Check a profile without building anything.

    TARGET is a profile directory (its ``profile.yml`` is used) or a path to a
    profile file. Resolves ``extends:`` chains and runs the full consistency
    check — convention directories, the ``data:`` tree, service templates,
    lifecycle steps, env vars — reporting every problem found, not just the
    first.

    Exits 0 when the profile is valid, 2 with the accumulated errors when it is
    not.

    Examples:

    \b
      $ osprey profile validate my-profile/
      $ osprey profile validate my-profile/personas/reader.yml
    """
    from .build_profile import resolve_build_profile

    profile_file = _resolve_profile_file(target)
    try:
        build_profile, profile_dir = resolve_build_profile(profile_file, None)
        # Named explicitly rather than left to resolution's internals: this
        # command exists to run exactly this check, so it must not become a
        # no-op if resolution ever stops validating on its own.
        build_profile.validate(profile_dir)
    except BuildProfileError as e:
        raise click.UsageError(str(e)) from e

    click.echo(f"✓ Profile is valid: {profile_file}")
    click.echo(f"  Name: {build_profile.name}")
    click.echo("\nNext steps:")
    click.echo(f"  1. Build a project from it: osprey build <PROJECT_NAME> {profile_file}")
    click.echo("  2. Re-run this command after editing the profile")


@profile.command()
@click.argument("target_dir", type=click.Path(path_type=Path))
@click.option(
    "--preset",
    required=True,
    metavar="NAME",
    help="Bundled preset to materialize (see `osprey profile presets`).",
)
@click.option(
    "--override",
    "-O",
    "overrides",
    multiple=True,
    type=click.Path(exists=False, dir_okay=False, path_type=Path),
    help="Layer a YAML file on top of the preset before writing (repeatable).",
)
@click.option(
    "--set",
    "set_pairs",
    multiple=True,
    metavar="KEY.PATH=VALUE",
    help="Inline scalar/list override baked into the emitted profile (repeatable). "
    "RHS parsed as YAML.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Replace an existing profile directory, deleting its current contents "
    "(including any edits). Refuses a target that is not a materialized "
    "profile (no profile.yml) and not empty.",
)
def new(
    target_dir: Path,
    preset: str,
    overrides: tuple[Path, ...],
    set_pairs: tuple[str, ...],
    force: bool,
) -> None:
    """Materialize an editable profile directory from a bundled preset.

    TARGET_DIR is created and populated with a standalone ``profile.yml`` (the
    preset's full configuration written out explicitly, no ``extends:``), the
    preset's data tree, and the profile's own ``.env`` channel — the editable
    source a facility then owns and builds from.

    Examples:

    \b
      $ osprey profile new my-profile --preset control-assistant
      $ osprey profile new my-profile --preset hello-world --set model=sonnet
    """
    try:
        target, skipped_keys = _materialize_profile_directory(
            target_dir, preset, overrides, set_pairs, force=force
        )
    except BuildProfileError as e:
        # Reaching here means a packaging problem, not a user mistake — the
        # helper raises UsageError for everything the caller could have got
        # wrong. Abort (exit 1) keeps that distinct from usage errors (exit 2).
        logger.error("✗ %s", e)
        raise click.Abort() from e

    click.echo(f"✓ Materialized profile at: {target}")
    # Read back from disk rather than threading a second return value out of the
    # helper: what the user is told they own is exactly what was written.
    persona_files = sorted((target / _PERSONA_PROFILE_DIRNAME).glob("*.yml"))
    if persona_files:
        click.echo(
            f"\nWeb-terminal personas — one delta each, merged over {target.name}/profile.yml:"
        )
        for persona_file in persona_files:
            click.echo(f"  {target.name}/{_PERSONA_PROFILE_DIRNAME}/{persona_file.name}")

    # Secrets get their own block: this directory is now where they live, and a
    # reader has to be able to tell at a glance whether a value was seeded for
    # them or is still theirs to supply.
    click.echo(f"\nSecrets — kept out of git by {target.name}/.gitignore:")
    click.echo(f"  {target.name}/{_PROFILE_ENV_EXAMPLE_FILENAME} — every variable this agent reads")
    env_path = target / _PROFILE_ENV_FILENAME
    if env_path.is_file():
        from osprey.utils.dotenv import parse_dotenv_file

        seeded = ", ".join(sorted(parse_dotenv_file(env_path)))
        click.echo(f"  {target.name}/{_PROFILE_ENV_FILENAME} — seeded from your shell: {seeded}")
    else:
        # Two different absences, and the remedy differs: nothing exported at
        # all, or keys exported for providers this profile does not use.
        reason = (
            "your shell exports no key for the providers it references"
            if skipped_keys
            else "your shell exports no provider key"
        )
        click.echo(
            f"  {target.name}/{_PROFILE_ENV_FILENAME} — not written: {reason}. "
            f"Copy the example and fill it in."
        )
    if skipped_keys:
        # Named rather than dropped in silence: the operator exported these, and
        # has to be able to tell "seen and not needed" from "lost".
        click.echo(f"  {_skipped_keys_note(skipped_keys)}")

    click.echo("\nNext steps:")
    click.echo(f"  1. Read {target.name}/README.md — it explains what you now own")
    click.echo(f"  2. Edit {target.name}/profile.yml and the files under {target.name}/data/")
    click.echo(
        f"  3. Build a project from it: osprey build <PROJECT_NAME> {target.name}/profile.yml"
    )


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

# The profile's secret channel (FR-1). `.env.example` is the documented variable
# list, rendered from the SAME template the build renders into a project, so the
# two can never document different variables. `.env` beside it holds the values,
# and is what the build derives the project's own `.env` from — which is what
# makes a secret survive a rebuild.
_PROFILE_ENV_FILENAME = ".env"
_PROFILE_ENV_EXAMPLE_FILENAME = ".env.example"
_ENV_EXAMPLE_TEMPLATE = "project/env.example.j2"

# Section header the shell-harvested keys are written under, distinct from the
# banner `osprey deploy up` appends its minted service tokens beneath: the two
# have different origins, and a reader should be able to tell which values came
# from their own shell.
_SEEDED_ENV_BANNER = "# ── Seeded by `osprey profile new` from your shell ──"

# The profile's own `.gitignore`. A profile is meant to live in version control
# — it is the facility's source of truth — so it ships with the one rule that
# keeps its secrets out. `.env*` deliberately covers every variant the directory
# accumulates, including the `.env.lock` the write-back path creates beside the
# `.env` it appends to; `.env.example` carries no values and is the exception.
_PROFILE_GITIGNORE = """\
# This profile is your facility's source of truth — keep it in version control.
# Its secrets are the one thing that must stay out.

# Every .env variant holds values (and .env.lock is the write-back lock file).
# .env.example is the documented variable list and carries none, so it is the
# single exception.
.env*
!.env.example

# OS / editor noise
.DS_Store
*.swp
*.swo
"""

# Build exhaust that a source checkout may hold inside a bundle's data tree but
# a wheel install never does — hatch excludes it from the package (see the
# `exclude` in pyproject.toml). Copying it would make an emission from a
# checkout differ from an emission from a wheel, so the copy drops it. Paths are
# relative to the data root, matched segment-wise so an unrelated `results/`
# elsewhere in the tree is untouched.
_EXCLUDED_DATA_SUBTREES: tuple[tuple[str, ...], ...] = (("benchmarks", "results"),)

_ALREADY_EXISTS = (
    "Target directory already exists: {target}. Remove it, choose a different "
    "path, or re-run with --force to replace it."
)

_NOT_REPLACEABLE = (
    "Refusing --force: {target} is not a materialized profile directory (no "
    "profile.yml) and not empty. Remove it yourself if you really mean to "
    "replace it."
)

# Build-time staging directories a bundle may ship inside its data tree, paired
# with what the README should say about each. Data-driven rather than
# per-preset prose: only the ones a bundle actually ships are described, so a
# hello-world profile is not told about tiers it does not have. Paths are
# relative to the data root.
# Notes are pre-wrapped with a two-space continuation indent so the rendered
# markdown reads as well in a plain editor as it does formatted.
_STAGING_TREE_NOTES: tuple[tuple[str, str], ...] = (
    (
        "channel_databases/tiers",
        "per-tier source databases. The build materializes the one your\n"
        "  `tier:` and `channel_finder_mode:` settings select into the flat\n"
        "  location the deployment reads, and prunes the rest.",
    ),
    (
        "benchmarks/cross_paradigm",
        "benchmark question sets for comparing channel-finder paradigms.\n"
        "  Build-time input, never deployed.",
    ),
    (
        "raw",
        "staging inputs the channel databases are generated from, kept so\n"
        "  they can be regenerated.",
    ),
)


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


def _persona_catalog_layer(persona_names: Iterable[str]) -> dict[str, Any]:
    """A raw profile fragment repointing each persona's ``build_profile`` at its
    emitted sibling profile.

    Written at the DEEPEST spelling on purpose. ``_collapse_config_prefixes``
    resolves a prefix pair deeper-key-wins, so these keys survive whatever
    shallower spelling the preset used for the module subtree
    (``modules.web_terminals:``, or even a nested ``modules:``) and land inside
    it — while a shallower fragment of ours would instead be overwritten
    wholesale by the preset's own subtree.
    """
    return {
        "config": {
            f"modules.web_terminals.personas.{name}.build_profile": (
                f"{_PERSONA_PROFILE_DIRNAME}/{name}.yml"
            )
            for name in persona_names
        }
    }


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

    Read through the same two helpers the deployment uses
    (:func:`~.build_profile_emit.effective_web_terminals` for the subtree,
    :func:`~osprey.deployment.web_terminals.personas.normalize_users` for the
    entries), so the directories seeded here are exactly the ones the build
    later looks for.
    """
    from osprey.deployment.web_terminals.personas import normalize_users

    from .build_profile_emit import effective_web_terminals

    web_terminals = effective_web_terminals(config)
    if not web_terminals.get("enabled"):
        return []
    return [entry["name"] for entry in normalize_users(web_terminals.get("users"))]


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
    whatever directory ``osprey profile new`` was run from is ambient state the
    profile cannot reproduce, so nothing is harvested from it.

    This is the one place the shell may seed a key, and it seeds the *profile* —
    the file an operator can read, edit, and account for. ``osprey build`` reads
    nothing from the environment at all: it derives the project ``.env`` from
    the profile written here, so a key that reaches a built project was always
    recorded in the profile first.

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

    One wording, used by both the materializer's log and ``profile new``'s
    summary: a skipped secret is a thing the operator has to be able to account
    for, and two spellings of the same fact read as two different facts.
    """
    subject = "this variable" if len(skipped) == 1 else "these variables"
    return (
        f"Not seeded: {', '.join(skipped)} — exported by your shell, but this "
        f"profile references no provider that reads {subject}."
    )


def _write_secret_channel(
    target: Path,
    manager: TemplateManager,
    resolved: BuildProfile,
    profile_name: str,
    exported: Mapping[str, str],
) -> list[str]:
    """Write the profile's secret channel: ``.env.example``, ``.env``, ``.gitignore``.

    The profile owns its secrets (FR-1): the build derives a project's ``.env``
    from the one written here, so a value set once survives every rebuild.

    ``.env.example`` is always written, and comes from the project template
    (:data:`_ENV_EXAMPLE_TEMPLATE`) rather than from prose of its own — one
    template documents the variable set wherever it is rendered. ``.env`` is
    written ONLY when ``exported`` is non-empty: an empty secrets file reads as
    a configured one, and ``cp .env.example .env`` is the honest starting point
    when there is nothing to seed.

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
            "service_token_vars": service_token_var_entries(),
            # The profile's `env:` block is documentation, not values — the
            # same two keys `osprey build` feeds this template.
            "env_required": list(resolved.env.required or []),
            "env_defaults": dict(resolved.env.defaults or {}),
            # No project exists yet; the key is a commented hint either way.
            "project_root": "/path/to/your/project",
        },
        target / _PROFILE_ENV_EXAMPLE_FILENAME,
    )
    (target / ".gitignore").write_text(_PROFILE_GITIGNORE, encoding="utf-8")
    written = [_PROFILE_ENV_EXAMPLE_FILENAME, ".gitignore"]

    if exported:
        # Written through the same append-only, 0600, atomic path `deploy up`
        # uses for its write-back, so there is one writer discipline for the
        # profile `.env` rather than a second one that only new profiles get.
        append_profile_env(target / _PROFILE_ENV_FILENAME, exported, _SEEDED_ENV_BANNER)
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


def _persona_profile_texts(
    resolved: BuildProfile,
    profile_name: str,
    profile_dirname: str,
    host_preset: str,
) -> dict[str, str]:
    """Emit one delta text per persona the profile deploys.

    Empty unless the profile stands up a persona stack of its own (see
    :func:`~osprey.cli.build_profile_emit.emits_persona_profiles`) — a persona
    preset inherits the catalog but disables the module, and emitting from one
    of those would produce personas-of-a-persona.

    Each entry is emitted as a pure DELTA — the persona preset's own layer, no
    ``extends:`` (:func:`~.build_profile_emit.emit_persona_delta_yaml`) — over
    the host profile it sits beside. The host therefore stays the single source
    of truth: edits there, and the caller's baked ``-O``/``--set`` layers with
    them, reach every persona through the implicit merge instead of being copied
    around. A catalog entry whose preset is not a delta over ``host_preset``
    (the bundled shape: ``control-assistant-readonly`` over
    ``control-assistant``) is rejected rather than approximated — see
    :func:`_off_chain_problem`.

    Raises:
        click.UsageError: With every unusable catalog entry named at once.
    """
    from .build_profile import resolve_build_profile
    from .build_profile_emit import emit_persona_delta_yaml, persona_catalog
    from .build_profile_presets import _normalize_preset_name

    catalog = persona_catalog(resolved.config)
    texts: dict[str, str] = {}
    problems: list[str] = []
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
        texts[persona_name] = emit_persona_delta_yaml(
            preset_name=persona_preset,
            profile_name=f"{profile_name} ({persona_name})",
            profile_filename=f"{profile_dirname}/{_PERSONA_PROFILE_DIRNAME}/{persona_name}.yml",
        )
    if problems:
        raise click.UsageError(
            "Cannot materialize the persona profiles this profile's web-terminal "
            "catalog calls for:\n  - " + "\n  - ".join(problems)
        )
    return texts


def _cleanup(target: Path) -> str:
    """Remove a partially materialized ``target``; report what is actually left.

    ``rmtree`` runs with ``ignore_errors`` so a cleanup failure never masks the
    original error — which means the directory may survive, and the message
    must not claim otherwise.
    """
    import shutil

    shutil.rmtree(target, ignore_errors=True)
    if target.exists():
        return f"A partial directory remains at {target} — remove it before retrying."
    return "Nothing was materialized."


def _packaged_sources(manager: TemplateManager, data_bundle: str) -> tuple[Path, Path]:
    """The two packaged trees this command copies: the seed, and ``data_bundle``'s data.

    Both are checked up front, before anything is written, so a packaging
    regression surfaces as an actionable error here rather than as a Jinja
    ``TemplateNotFound`` deep in the loader or a missing-file error mid-copy.
    Neither can be caused by anything the caller passed.

    Args:
        manager: The :class:`~.templates.manager.TemplateManager` locating the
            installed template root.
        data_bundle: App template whose ``data/`` tree gets materialized.

    Returns:
        ``(seed_root, data_source)``.

    Raises:
        BuildProfileError: If either tree is absent from the installation.
    """
    seed_root = manager.template_root / "profile_seed"
    if not seed_root.is_dir():
        raise BuildProfileError(
            f"Profile seed templates missing at {seed_root}. "
            f"This is a packaging bug — reinstall osprey-framework."
        )

    data_source = manager.template_root / "apps" / data_bundle / "data"
    if not data_source.is_dir():
        raise BuildProfileError(
            f"App template {data_bundle!r} ships no data tree at {data_source}. "
            f"This is a packaging bug — reinstall osprey-framework."
        )

    return seed_root, data_source


class _MaterializedProfile(NamedTuple):
    """What :func:`_materialize_profile_directory` produced, for the caller's summary."""

    target: Path
    """The resolved profile directory."""

    skipped_shell_keys: tuple[str, ...]
    """Exported provider keys left out because the profile references none of
    their providers (:func:`_exported_provider_keys`). Reported rather than
    returned as a courtesy: the seeded keys can be read back from the ``.env``,
    the ones that were deliberately not written cannot."""


def _materialize_profile_directory(
    target_dir: Path,
    preset_name: str,
    overrides: tuple[Path, ...] = (),
    set_pairs: tuple[str, ...] = (),
    *,
    force: bool = False,
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

    Returns:
        The resolved target directory and the exported provider keys that were
        deliberately not seeded (:class:`_MaterializedProfile`).

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
    from .templates.scaffolding import _copy_data_tree

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
    replacing = False
    if target.exists():
        if not force:
            raise click.UsageError(_ALREADY_EXISTS.format(target=target))
        # --force only replaces what this command itself produces: a
        # materialized profile (profile.yml present) or an empty directory.
        # Anything else could be an arbitrary directory named by mistake —
        # refuse rather than delete it.
        if not (
            target.is_dir() and ((target / "profile.yml").is_file() or not any(target.iterdir()))
        ):
            raise click.UsageError(_NOT_REPLACEABLE.format(target=target))
        replacing = True

    normalized_preset = _normalize_preset_name(preset_name)
    # `target.name` is the user-chosen directory name (e.g. "my-profile") —
    # the emitted profile's display name unless the user passed --set name=...
    profile_name_default = target.name.replace("-", " ").replace("_", " ").title()
    if name_override is not None:
        profile_name_default = str(name_override)

    manager = TemplateManager()
    seed_root, data_source = _packaged_sources(manager, resolved.data_bundle)

    profile_filename = f"{target.name}/profile.yml"

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
        resolved, profile_name_default, target.name, normalized_preset
    )
    # Parsed here, before the first mkdir and before `--force` replaces
    # anything: the parse is what validates the emitted deltas, so a bad one
    # must cost nothing. Its result is what every later reader sees.
    persona_deltas = _parsed_persona_deltas(persona_texts)

    extra_layers: tuple[dict[str, Any], ...] = (
        *((_persona_catalog_layer(persona_texts),) if persona_texts else ()),
        *((_triggers_layer(),) if triggers_src is not None else ()),
    )

    # Read once, before anything is written: the README rendered below tells the
    # reader whether a `.env` was seeded for them, and the seeding itself happens
    # further down. Two reads of the environment could disagree.
    shell_keys = _exported_provider_keys(_referenced_providers(resolved, persona_deltas))
    exported_keys = shell_keys.seeded

    # The materialized tree is what the build must read, so `data:` is emitted
    # as an active key — injected through the same --set layering a user would
    # use, rather than through a second path into the resolved content.
    profile_text = emit_standalone_profile_yaml(
        preset_name=normalized_preset,
        overrides=overrides,
        set_pairs=(*set_pairs, f"data={_PROFILE_DATA_DIRNAME}"),
        profile_name=profile_name_default,
        profile_filename=profile_filename,
        extra_layers=extra_layers,
        include_flow_diagram=True,
    )

    # The replacement happens here, after every input has resolved and the new
    # profile text is fully rendered — a failure above (bad preset, invalid
    # override) leaves the existing directory exactly as it was.
    if replacing:
        shutil.rmtree(target)

    try:
        target.mkdir(parents=True)
    except FileExistsError as e:
        # Lost the race against another process between the check above and
        # here — same user-facing outcome, so say the same thing.
        raise click.UsageError(_ALREADY_EXISTS.format(target=target)) from e

    try:
        ctx = {
            "preset_name": normalized_preset,
            "profile_name": profile_name_default,
            "profile_dirname": target.name,
            "profile_filename": profile_filename,
            # Only the staging directories this bundle actually ships, so the
            # README never explains a directory the reader does not have.
            "staging_notes": [
                {"path": rel, "note": note}
                for rel, note in _STAGING_TREE_NOTES
                if (data_source / rel).is_dir()
            ],
            # Empty for a profile that deploys no persona stack, so the README
            # never explains a directory the reader does not have.
            "persona_profiles": list(persona_texts),
            # Same: only a profile with a dispatch block owns a triggers.yml.
            "has_triggers": triggers_src is not None,
            # Same again: the per-user context slots exist only for a profile
            # that stands up a web-terminal roster.
            "context_users": _roster_user_names(resolved.config),
            # Whether the reader has a `.env` already, so the README either
            # explains what was seeded or tells them how to create one.
            "has_seeded_env": bool(exported_keys),
        }
        _copy_data_tree(seed_root, target, manager.template_root, manager.jinja_env, ctx)
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
            logger.info("  Trigger config: %s", _PROFILE_TRIGGERS_FILENAME)

        # The profile owns its secrets from the first minute (FR-1) — the
        # documented variable list, whatever the shell already exports, and the
        # .gitignore that keeps the values out of version control.
        secret_files = _write_secret_channel(
            target, manager, resolved, profile_name_default, exported_keys
        )
        logger.info("  Secrets: %s", ", ".join(secret_files))
        if shell_keys.skipped:
            # Also logged here, not only in `profile new`'s summary: a build
            # that materializes its own profile takes this path too, and a
            # skipped secret must be visible wherever the skipping happened.
            logger.info("  %s", _skipped_keys_note(shell_keys.skipped))

        # One empty slot per roster user, so the per-user context a facility
        # writes has an obvious home from the first minute (FR-5). Only the
        # directories are seeded: what goes in them is the facility's, and the
        # build derives the copy from the roster it resolves at the time, so
        # nothing about the roster is frozen here.
        roster = _roster_user_names(resolved.config)
        for user in roster:
            user_dir = target / _CONTEXT_CONVENTION_DIRNAME / user
            user_dir.mkdir(parents=True, exist_ok=True)
            (user_dir / ".gitkeep").touch()
        if roster:
            logger.info(
                "  Per-user context: %s",
                ", ".join(f"{_CONTEXT_CONVENTION_DIRNAME}/{user}/" for user in roster),
            )

        if persona_texts:
            # Validity was settled by `_parsed_persona_deltas` above, before the
            # first mkdir — nothing reaching here is unparsed, so these writes
            # need no guard of their own.
            persona_dir = target / _PERSONA_PROFILE_DIRNAME
            persona_dir.mkdir()
            for persona_name, persona_text in persona_texts.items():
                (persona_dir / f"{persona_name}.yml").write_text(persona_text, encoding="utf-8")
            logger.info(
                "  Persona deltas: %s",
                ", ".join(f"{_PERSONA_PROFILE_DIRNAME}/{name}.yml" for name in persona_texts),
            )

        # The round-trip runs last because it validates `data:` against the tree
        # that must already be on disk. Only the host profile is resolved: a
        # persona file is a delta, meaningless on its own, and resolving one is
        # resolving the host with that delta merged in — which the host's own
        # round-trip already covers.
        resolve_build_profile((target / "profile.yml").resolve(), None)
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

    logger.info("✓ Materialized profile at: %s", target)
    return _MaterializedProfile(target, shell_keys.skipped)
