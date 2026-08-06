"""Profile command group — author and inspect build profiles.

A build profile is the editable source a facility owns: a ``profile.yml`` plus
the overlay and data trees beside it. This group holds the verbs that act on
that source, keeping them separate from ``osprey build``, which consumes a
profile and derives a project from it.

Usage:
    osprey profile presets
    osprey profile validate my-profile/
    osprey profile new my-profile --preset control-assistant
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from osprey.errors import BuildProfileError
from osprey.utils.logger import get_logger

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
    check — overlay sources, the ``data:`` tree, service templates, lifecycle
    steps, env vars — reporting every problem found, not just the first.

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
    preset's data tree, and an overlay seed — the editable source a facility
    then owns and builds from.

    Examples:

    \b
      $ osprey profile new my-profile --preset control-assistant
      $ osprey profile new my-profile --preset hello-world --set model=sonnet
    """
    try:
        target = _materialize_profile_directory(
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
            f"\nWeb-terminal personas — one profile each, all reading "
            f"{target.name}/{_PROFILE_DATA_DIRNAME}/:"
        )
        for persona_file in persona_files:
            click.echo(f"  {target.name}/{_PERSONA_PROFILE_DIRNAME}/{persona_file.name}")
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

# Sibling persona profiles live here, one file per web-terminal persona, and
# each reads the profile's own data tree one level up (D7a). The whole stack
# therefore shares ONE facility data tree: edit `<profile>/data/` once and every
# persona render sees it.
_PERSONA_PROFILE_DIRNAME = "personas"
_PERSONA_DATA_REF = f"../{_PROFILE_DATA_DIRNAME}"

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


def _replayed_model_selection(baked: dict[str, Any]) -> tuple[str, ...]:
    """``--set`` pairs re-applying the caller's baked model selection to a persona.

    Only the flattened fallback emission needs this: a delta persona inherits
    the host profile wholesale, baked model selection included, through its
    ``extends: ../profile.yml``.

    Mirrors :func:`osprey.deployment.web_terminals.persona_images._parent_set_override_args`
    exactly — the same key set (:data:`MODEL_SELECTION_OVERRIDE_KEYS`) and the
    same ``str | int`` value filter — because the two are the SAME guarantee at
    two different moments: one parent-level model choice retints the whole
    stack. That function covers a choice made at ``osprey build`` time; this one
    covers a choice baked in at ``osprey profile new`` time, which reaches no
    manifest and would otherwise never reach a persona at all.

    Only top-level shorthand keys are replayed, matching
    :func:`~osprey.cli.build_profile_resolve.explicit_model_override_keys`: a
    dotted path into ``config:`` addresses the rendered config directly and
    carries no whole-stack intent.
    """
    from .build_profile_resolve import MODEL_SELECTION_OVERRIDE_KEYS

    return tuple(
        f"{key}={baked[key]}"
        for key in MODEL_SELECTION_OVERRIDE_KEYS
        if isinstance(baked.get(key), str | int) and baked.get(key)
    )


def _persona_profile_texts(
    resolved: BuildProfile,
    profile_name: str,
    profile_dirname: str,
    baked: dict[str, Any],
    host_preset: str,
) -> dict[str, str]:
    """Emit one sibling profile text per persona the profile deploys.

    Empty unless the profile stands up a persona stack of its own (see
    :func:`~osprey.cli.build_profile_emit.emits_persona_profiles`) — a persona
    preset inherits the catalog but disables the module, and emitting from one
    of those would produce personas-of-a-persona.

    A persona whose preset ``extends``-chains through ``host_preset`` (the
    bundled shape: ``control-assistant-readonly`` over ``control-assistant``)
    is emitted as a DELTA — ``extends: ../profile.yml`` plus the preset's own
    few overrides (:func:`~.build_profile_emit.emit_persona_delta_yaml`) — so
    the host profile stays the single source of truth: edits there, and the
    caller's baked ``-O``/``--set`` layers with them, reach every persona
    through resolution instead of being copied around. A catalog entry whose
    preset sits outside that chain cannot be rebased without changing what it
    resolves to; it falls back to the flattened standalone emission, with the
    model-selection subset of the caller's overrides replayed into it
    (:func:`_replayed_model_selection`) so a whole-stack provider choice still
    reaches it.

    Raises:
        click.UsageError: With every unusable catalog entry named at once.
    """
    from .build_profile import resolve_build_profile
    from .build_profile_emit import (
        emit_persona_delta_yaml,
        emit_standalone_profile_yaml,
        persona_catalog,
    )
    from .build_profile_presets import _normalize_preset_name, _preset_extends_chain_reaches

    catalog = persona_catalog(resolved.config)
    model_selection = _replayed_model_selection(baked)
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
        persona_filename = f"{profile_dirname}/{_PERSONA_PROFILE_DIRNAME}/{persona_name}.yml"
        if _preset_extends_chain_reaches(persona_preset, host_preset):
            texts[persona_name] = emit_persona_delta_yaml(
                preset_name=persona_preset,
                profile_name=f"{profile_name} ({persona_name})",
                profile_filename=persona_filename,
            )
        else:
            texts[persona_name] = emit_standalone_profile_yaml(
                preset_name=persona_preset,
                overrides=(),
                set_pairs=(f"data={_PERSONA_DATA_REF}", *model_selection),
                profile_name=f"{profile_name} ({persona_name})",
                profile_filename=persona_filename,
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


def _materialize_profile_directory(
    target_dir: Path,
    preset_name: str,
    overrides: tuple[Path, ...] = (),
    set_pairs: tuple[str, ...] = (),
    *,
    force: bool = False,
) -> Path:
    """Materialize an editable, standalone profile directory from ``preset_name``.

    Writes ``profile.yml`` — the preset's fully resolved content as an
    explicit, self-contained profile (comments preserved, no ``extends:``) —
    the bundle's ``data/`` tree copied verbatim, the ``overlays/`` seed, and a
    tutorial ``README.md``. ``-O`` files and ``--set`` pairs are merged with the
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
        The resolved target directory.

    Raises:
        click.UsageError: For user errors — existing target, an ``extends``
            override, or layers that produce an invalid profile.
        BuildProfileError: For packaging problems (missing seed or data tree).
    """
    import shutil

    from .build_profile import (
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
        resolved, _preset_dir = resolve_build_profile(None, preset_name, overrides, set_pairs)
    except BuildProfileError as e:
        raise click.UsageError(f"Cannot materialize {preset_name!r}: {e}") from e

    baked = merge_cli_overrides({}, overrides, set_pairs)
    if "extends" in baked:
        raise click.UsageError(
            "`osprey profile new` cannot override 'extends' — the materialized "
            "profile is standalone and inherits nothing at build time."
        )
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
    persona_texts = _persona_profile_texts(
        resolved, profile_name_default, target.name, baked, normalized_preset
    )

    # The materialized tree is what the build must read, so `data:` is emitted
    # as an active key — injected through the same --set layering a user would
    # use, rather than through a second path into the resolved content.
    profile_text = emit_standalone_profile_yaml(
        preset_name=normalized_preset,
        overrides=overrides,
        set_pairs=(*set_pairs, f"data={_PROFILE_DATA_DIRNAME}"),
        profile_name=profile_name_default,
        profile_filename=profile_filename,
        extra_layers=(_persona_catalog_layer(persona_texts),) if persona_texts else (),
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

        persona_dir = target / _PERSONA_PROFILE_DIRNAME
        if persona_texts:
            persona_dir.mkdir()
            for persona_name, persona_text in persona_texts.items():
                (persona_dir / f"{persona_name}.yml").write_text(persona_text, encoding="utf-8")
            logger.info(
                "  Persona profiles: %s",
                ", ".join(f"{_PERSONA_PROFILE_DIRNAME}/{name}.yml" for name in persona_texts),
            )

        # The round-trips run last because they validate `data:` against the
        # trees that must already be on disk — `data: ../data` for a persona.
        resolve_build_profile((target / "profile.yml").resolve(), None)
        for persona_name in persona_texts:
            resolve_build_profile((persona_dir / f"{persona_name}.yml").resolve(), None)
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
    return target
