"""Build command — assemble a facility-specific assistant from a build profile.

Reads a YAML build profile that specifies a base template, config overrides,
and MCP server definitions, alongside the convention directories the profile
carries its own artifacts in. Produces a standalone, self-contained project
directory (wipe-and-rebuild safe).

Every build reads a profile. ``--preset`` is not a second way to build: it
materializes ``<output_dir>/<PROJECT_NAME>-profile/`` on first use and builds
from that, so the thing a facility then edits and rebuilds from exists from the
first command it runs.

Usage:
    osprey build my-assistant profile.yml
    osprey build my-assistant --preset hello-world
    osprey build my-assistant --preset education -O override.yml --set model=claude-sonnet-4-6
    osprey build --list-presets

The build pipeline's helper concerns live in sibling modules that this command
orchestrates: venv + ``.env`` templating in :mod:`osprey.cli.build_environment`,
lifecycle-phase execution in :mod:`osprey.cli.build_lifecycle`, service
injectors in :mod:`osprey.cli.build_injectors`, and project-directory
persistence (config overrides, convention artifacts, MCP servers, git init) in
:mod:`osprey.cli.build_persistence`. They are re-exported below so
``from osprey.cli.build_cmd import <helper>`` keeps working.
"""

from __future__ import annotations

import shlex
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import click

from osprey.errors import BuildProfileError
from osprey.utils.logger import get_logger

from .build_environment import (
    _copy_env_file,
    _create_project_venv,
    _resolve_osprey_spec,
    report_provider_credentials,
)
from .build_injectors import (
    _copy_service_templates,
    _inject_bluesky,
    _inject_bluesky_panels,
    _inject_dispatch,
    _inject_gchat_bridge,
    _inject_nextcloud_bridge,
    _inject_profile_services,
    _inject_va,
    _inject_va_archiver,
    _locate_pkg_services,
)
from .build_lifecycle import (
    _SHELL_METACHARACTERS,
    _format_junit_summary,
    _run_lifecycle_phase,
)
from .build_persistence import (
    _apply_config_overrides,
    _apply_conventions,
    _clear_rendered_project_dir,
    _git_init_and_commit,
    _persist_artifact_server,
    _persist_mcp_servers,
    _profile_known_root_entries,
    _register_convention_artifacts,
    _resolve_context_roster,
)
from .templates.manager import TemplateManager

if TYPE_CHECKING:
    from .build_profile_load import LoadedProfile

logger = get_logger("build")

__all__ = [
    "_SHELL_METACHARACTERS",
    "_apply_config_overrides",
    "_apply_conventions",
    "_clear_rendered_project_dir",
    "_copy_env_file",
    "_copy_service_templates",
    "_create_project_venv",
    "_format_junit_summary",
    "_git_init_and_commit",
    "_inject_bluesky",
    "_inject_bluesky_panels",
    "_inject_dispatch",
    "_inject_gchat_bridge",
    "_inject_nextcloud_bridge",
    "_inject_profile_services",
    "_inject_va",
    "_inject_va_archiver",
    "_locate_pkg_services",
    "_persist_artifact_server",
    "_persist_mcp_servers",
    "_profile_known_root_entries",
    "_register_convention_artifacts",
    "_resolve_context_roster",
    "_resolve_osprey_spec",
    "_run_lifecycle_phase",
    "build",
    "report_provider_credentials",
]


def _list_presets_callback(ctx: click.Context, param: click.Parameter, value: bool) -> None:
    """Eager --list-presets: print bundled presets and exit before any args parse."""
    if not value or ctx.resilient_parsing:
        return
    from .build_profile import list_presets

    for name in list_presets():
        click.echo(name)
    ctx.exit(0)


class _SettledProfile(NamedTuple):
    """The profile a build reads, resolved, and where the build renders."""

    profile_path: Path
    """The profile file the build was settled on — for a ``--preset`` build the
    ``profile.yml`` of the directory it materialized or reused."""

    document: LoadedProfile
    """The resolved document: the parsed profile, its ROOT, and the convention
    artifacts it excludes."""

    output_path: Path
    """Where the project directory is created."""


def _settle_profile(
    profile_arg: Path | None,
    preset: str | None,
    project_name: str,
    output_dir: str | None,
    output_path: Path,
    overrides: tuple[Path, ...],
    set_pairs: tuple[str, ...],
    tier_value: int | None,
) -> _SettledProfile:
    """Settle the profile this build reads, then resolve it.

    Every build reads a profile: it is the facility's source of truth, so there
    is no path that renders a project straight out of a bundled preset. A
    ``--preset`` build materializes ``<output_dir>/<PROJECT_NAME>-profile/`` the
    first time and reuses it verbatim afterwards; a positional build already has
    one. Either way ``--set`` / ``-O`` / ``--tier`` end up IN the profile — baked
    in at materialization, written into it on a reuse — so what is resolved here
    is the profile's own content and nothing else.

    The write-back guard is deliberately NOT constructed here. It has to exist
    before the first of these calls can edit anything and it has to outlive a
    failure, so it belongs to the caller's ``try`` rather than to a function
    that may not return.

    Args:
        profile_arg: The positional profile file, already resolved from either
            spelling, or ``None`` for a ``--preset`` build.
        preset: ``--preset``, or ``None``.
        project_name: Names the materialized profile directory.
        output_dir: ``--output-dir`` as given, or ``None``.
        output_path: The provisional render destination, as the caller derived
            it before there was a profile to ask.
        overrides: ``-O`` files.
        set_pairs: ``--set`` pairs.
        tier_value: ``--tier`` as an int, or ``None``.

    Returns:
        The settled profile path, the resolved document, and the FINAL render
        destination.

    Raises:
        click.UsageError: For anything the operator could have got wrong; the
            refusal is logged here too, so they need not know which layer made
            it.
        BuildProfileError: For a genuine profile problem, which the build
            reports as itself.
    """
    from .build_profile import (
        materialize_or_reuse_profile,
        resolve_build_document,
        resolve_build_output_dir,
        write_back_cli_overrides,
    )

    try:
        if preset is not None:
            profile_arg = materialize_or_reuse_profile(
                preset, output_path, project_name, overrides, set_pairs, tier_value
            )
        else:
            assert profile_arg is not None  # guarded: no preset means a profile path
            logger.info("  Profile: %s", profile_arg)
            write_back_cli_overrides(profile_arg, overrides, set_pairs, tier_value)

        # The document, not just the profile: resolution also derives which
        # convention artifacts this profile excludes, and step 11 has to
        # honor that record or an excluded artifact is copied anyway — and
        # then registered as user-owned, shadowing the framework's own
        # version of the file the persona asked to drop.
        document = resolve_build_document(profile_arg, None)
        # Now that the profile ROOT is known, the destination is too. A
        # --preset build keeps the provisional answer: its profile is a
        # `<name>-profile/` beside the project, not one nested in a
        # facility repo, so there is no repo `build/` for it to belong to.
        if preset is None:
            output_path = resolve_build_output_dir(output_dir, document.profile_dir)
        return _SettledProfile(profile_arg, document, output_path)
    except click.UsageError as e:
        # Materialization reports user errors the way `osprey profile new`
        # does, as a UsageError. The build reports its own through the log,
        # and an operator should not have to know which layer refused —
        # so it is logged here too, then re-raised for its exit code.
        logger.error("✗ %s", e)
        raise
    except BuildProfileError as e:
        # Mutual-exclusion / missing-input / unknown-preset errors are
        # user errors, not bugs — promote to UsageError so the outer
        # except chain produces exit code 2.
        msg = str(e)
        lower = msg.lower()
        if "either" in lower or "not both" in lower or lower.startswith("unknown preset"):
            raise click.UsageError(msg) from e
        raise


@click.command()
@click.argument("project_name", required=False)
@click.argument(
    "profile",
    required=False,
    default=None,
    type=click.Path(exists=False),
)
@click.option(
    "--preset",
    default=None,
    metavar="NAME",
    help="Materialize <PROJECT_NAME>-profile/ from a bundled preset and build "
    "from it (see --list-presets). Reused as-is if it already exists.",
)
@click.option(
    "--override",
    "-O",
    "overrides",
    multiple=True,
    type=click.Path(exists=False, dir_okay=False, path_type=Path),
    help="Layer a YAML file on top of the profile (repeatable). Written into "
    "the profile when it already exists.",
)
@click.option(
    "--set",
    "set_pairs",
    multiple=True,
    metavar="KEY.PATH=VALUE",
    help="Inline scalar/list override (repeatable). RHS parsed as YAML. "
    "Written into the profile when it already exists, replacing the value "
    "at the dotted key path.",
)
@click.option(
    "--list-presets",
    is_flag=True,
    is_eager=True,
    expose_value=False,
    callback=_list_presets_callback,
    help="List bundled preset names and exit.",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default=None,
    help="Render the project under this directory. Default: the facility "
    "repo's build/ when the profile is nested in one, otherwise the current "
    "directory.",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help=(
        "Re-render an existing project directory in place "
        "(.env, _agent_data/, and .git are preserved). Never touches the "
        "profile — replace one with `osprey profile new --force`."
    ),
)
@click.option("--stream", "-s", is_flag=True, help="Stream lifecycle step output in real-time")
@click.option(
    "--skip-lifecycle", is_flag=True, help="Skip pre_build, post_build, and validate phases"
)
@click.option(
    "--skip-deps", is_flag=True, help="Skip venv creation and dependency installation (CI mode)"
)
@click.option(
    "--runtime-root",
    type=click.Path(),
    default=None,
    help="Override project_root in rendered config (for container builds where the "
    "build path differs from the runtime path, e.g. --runtime-root /app/als-assistant)",
)
@click.option(
    "--tier",
    type=click.Choice(["1", "3"]),
    default=None,
    help="Channel-database tier (1|3). Selects which "
    "data/channel_databases/tiers/tier{N}/ DB the rendered config points at. "
    "Advanced: override the paradigm-derived default "
    "(in_context → tier 1, hierarchical/middle_layer → tier 3). "
    "Tier 1 is in_context-only.",
)
def build(
    project_name: str | None,
    profile: str | None,
    preset: str | None,
    overrides: tuple[Path, ...],
    set_pairs: tuple[str, ...],
    output_dir: str | None,
    force: bool,
    stream: bool,
    skip_lifecycle: bool,
    skip_deps: bool,
    runtime_root: str | None,
    tier: str | None,
) -> None:
    """Build a facility-specific assistant from a build profile.

    Assembles a standalone project by rendering a base template, applying
    config overrides, applying the profile's convention directories, and
    injecting MCP servers.

    The profile is the source of truth. `--preset NAME` materializes
    `<PROJECT_NAME>-profile/` beside the project the first time and builds from
    it; every later build reuses that directory as it stands, so an edit made
    there is what the next build renders. `--set`, `-O` and `--tier` on such a
    build are written into the profile before it is read.

    A profile nested in a facility repo (`<repo>/profile/`) renders into
    `<repo>/build/<PROJECT_NAME>/`, whichever directory the command is run
    from. Anything else renders under the current directory. `--output-dir`
    overrides both.

    PROJECT_NAME: Name of the project directory to create

    PROFILE: A profile directory (its `profile.yml` is used) or a path to a
    profile file — the two spellings `osprey profile validate` takes. Mutually
    exclusive with --preset.

    Examples:

    \b
      # Materialize a profile from a bundled preset and build from it
      $ osprey build my-assistant --preset hello-world

      # Build from a facility repo's profile directory
      $ osprey build my-facility profile/

      # Build from a profile file
      $ osprey build als-test ~/profiles/als-dev.yml

      # Bake overrides into the profile this build materializes
      $ osprey build als-test --preset control-assistant -O als-overrides.yml \\
            --set model=claude-sonnet-4-6

      # List available presets
      $ osprey build --list-presets
    """
    from .build_profile import (
        PROFILE_FILENAME,
        ProfileWriteBackGuard,
        preset_profile_dir,
    )
    from .build_profile_archiver import va_archiver_config_overrides
    from .build_profile_deploy import deploy_config_overrides
    from .profile_cmd import _resolve_profile_file
    from .project_utils import _clear_claude_code_project_state

    if not project_name:
        raise click.UsageError("PROJECT_NAME is required. Run 'osprey build --help' for usage.")

    logger.info("Building project: %s", project_name)

    # The CLI overrides are written into the profile before the build reads it
    # back, so the edit is only as good as the build that made it: a build that
    # does not complete leaves the source of truth exactly as it found it. Named
    # here, inert, so every exit path below has a guard to ask.
    profile_guard = ProfileWriteBackGuard(None)
    build_completed = False

    try:
        # 1. Settle the profile this build reads — `_settle_profile` does that.
        #    What stays here is only what has to happen AROUND the write-back
        #    guard: the guard must exist before the first call that can edit the
        #    profile, and it must outlive one that fails.
        # Provisional destination: it is what a --preset build materializes its
        # profile under, and that has to be settled before there is a profile to
        # ask. A positional build re-derives it inside `_settle_profile`, once
        # resolution has reported the profile ROOT the answer depends on.
        output_path = Path(output_dir).resolve() if output_dir is not None else Path.cwd().resolve()
        # A profile directory is the unit a facility works with, so the build
        # takes one wherever it takes a profile file — the same two spellings
        # `osprey profile validate` accepts, resolved by the same function. In a
        # facility repo `osprey build <name> profile/` is the natural thing to
        # type, and one verb refusing what its sibling accepts is a trap.
        profile_arg = _resolve_profile_file(Path(profile)) if profile else None
        if profile_arg is not None and preset is not None:
            raise click.UsageError("Pass either a profile path or --preset, not both.")
        if profile_arg is None and preset is None:
            raise click.UsageError("Either a profile path or --preset is required.")

        # click constrains --tier to {1, 3}; the profile model carries the int.
        tier_value = int(tier) if tier is not None else None

        # Snapshot the profile about to be edited — for a --preset build the one
        # the materialization would reuse. On a first build there is nothing to
        # snapshot, so the guard is told which directory this invocation is
        # about to materialize instead: that one it removes rather than
        # restores, since a build that failed has no earlier state to put back.
        preset_dir = preset_profile_dir(output_path, project_name) if preset is not None else None
        profile_guard = ProfileWriteBackGuard(
            (preset_dir / PROFILE_FILENAME) if preset_dir is not None else profile_arg,
            tuple(overrides),
            tuple(set_pairs),
            tier_value,
            materializes_into=preset_dir,
        )

        settled = _settle_profile(
            profile_arg,
            preset,
            project_name,
            output_dir,
            output_path,
            tuple(overrides),
            tuple(set_pairs),
            tier_value,
        )
        profile_arg = settled.profile_path
        resolved = settled.document
        build_profile = resolved.profile
        profile_dir = resolved.profile_dir
        output_path = settled.output_path

        # Re-run validation after the write-back so the tier rule (tier 1
        # requires channel_finder_mode: in_context) fails here with a
        # rule-naming error rather than downstream as a scaffolding
        # FileNotFoundError.
        if tier_value is not None:
            build_profile.validate(profile_dir)

        # The multi-user web stack the profile declares — the roster, the
        # persona catalog, and the port families — checked before the build
        # renders a project from them, and checked against the config this build
        # is about to render (the `config:` block plus what the `deploy:` block
        # contributes to it). Run here rather than inside
        # `BuildProfile.validate()`, which also runs during profile resolution
        # (see `lint_profile_config`).
        from .build_profile_deploy import deploy_aware_config_errors

        web_errors = deploy_aware_config_errors(build_profile.deploy, build_profile.config)
        if web_errors:
            raise click.UsageError(
                "Build profile validation failed:\n  - " + "\n  - ".join(web_errors)
            )

        # Provider is required — no implicit fallback. Each provider has
        # different auth gating (CBORG: LBLnet; als-apg: ALS_APG_API_KEY;
        # anthropic: ANTHROPIC_API_KEY), so silently defaulting masks
        # misconfiguration as a credential failure at runtime.
        if not build_profile.provider:
            raise click.UsageError(
                "Profile does not specify a provider. Add `provider: "
                "<als-apg|cborg|anthropic|amsc-i2|argo>` to your profile or "
                "pass `--set provider=<...>` on the build command."
            )

        logger.info("  Profile: %s", build_profile.name)
        logger.info("  Data bundle: %s", build_profile.data_bundle)
        logger.info("  Tier: %d", build_profile.resolved_tier())

        # 1b. Collect and validate profile artifact selections
        artifacts: dict[str, list[str]] = {}
        for artifact_type in ("hooks", "rules", "skills", "agents", "output_styles"):
            names = getattr(build_profile, artifact_type, [])
            if names:
                artifacts[artifact_type] = list(names)

        if artifacts:
            from osprey.cli.templates.artifact_library import validate_artifacts

            validate_artifacts(artifacts)
            total = sum(len(v) for v in artifacts.values())
            logger.info(
                "  ✓ Validated %d artifact(s): %s",
                total,
                ", ".join(f"{len(v)} {k}" for k, v in artifacts.items()),
            )

        # web_panels is validated at manifest load time (warn-only) — not file-backed,
        # so it bypasses validate_artifacts. Flow it into the template context via the
        # same dict the manager consumes.
        if build_profile.web_panels:
            artifacts["web_panels"] = list(build_profile.web_panels)

        # 1d. Check OSPREY version requirement
        if build_profile.requires_osprey_version:
            from packaging.specifiers import SpecifierSet
            from packaging.version import Version

            from osprey.version import get_release_version

            from .build_profile_load import _PROFILE_SCHEMA_MIN_OSPREY

            # Compare the release lineage, not the running version: a development
            # build carries a post/local segment that no release specifier is
            # written against. But the lineage alone is not the whole capability:
            # between releases a checkout ships the *next* release's profile
            # schema (it stamps `_PROFILE_SCHEMA_MIN_OSPREY` into every profile
            # it writes) while its tag still names the previous release, so
            # judging it by tag alone would make it refuse profiles it just
            # wrote. The schema floor this code carries is therefore also a
            # floor on what it satisfies. `prereleases=True` keeps a pre-release
            # lineage from being excluded by PEP 440's default filtering.
            release_version = get_release_version()
            spec = SpecifierSet(build_profile.requires_osprey_version, prereleases=True)
            current = max(Version(release_version), Version(_PROFILE_SCHEMA_MIN_OSPREY))
            if current not in spec:
                logger.error(
                    "  ✗ OSPREY %s does not satisfy requires_osprey_version: %s",
                    current,
                    build_profile.requires_osprey_version,
                )
                logger.info("     Upgrade OSPREY or run: osprey --version")
                raise click.Abort()
            logger.info(
                "  ✓ OSPREY %s satisfies %s",
                current,
                build_profile.requires_osprey_version,
            )

        # 2. Resolve output path. Logged because the default now depends on
        #    where the profile lives rather than on where the operator is
        #    standing, and a destination the operator did not name is one they
        #    should still be told.
        project_path = output_path / project_name
        logger.info("  Output: %s", project_path)

        # 3. Handle --force / directory existence. The profile beside the
        #    project is never touched: it is the facility's own source of
        #    truth, and only `osprey profile new --force` replaces one.
        if project_path.exists():
            if force:
                logger.warning("  Clearing rendered files in existing directory: %s", project_path)
                preserved = _clear_rendered_project_dir(project_path)
                if preserved:
                    logger.info("  ✓ Preserved user state: %s", ", ".join(preserved))
                logger.info("  ✓ Cleared rendered files")
            else:
                logger.error(
                    "  ✗ Directory '%s' already exists. Use --force to overwrite, or choose a different name.",
                    project_path,
                )
                raise click.Abort()

        # 4. Run pre_build lifecycle commands
        if build_profile.lifecycle.pre_build and not skip_lifecycle:
            _run_lifecycle_phase(
                "pre_build",
                build_profile.lifecycle.pre_build,
                profile_dir,
                project_path,
                stream=stream,
            )

        # 5. Clear Claude Code project state
        _clear_claude_code_project_state(project_path)

        # 6. Build context from profile fields
        context: dict[str, Any] = {}
        # Gates the rendered config.yml's `services:`/`deployed_services:`
        # sections. An attached project (deploy_services: false) renders an
        # empty `services: {}` and `deployed_services: []` so `osprey deploy`
        # finds an explicit empty list and scaffolds nothing.
        context["deploy_services"] = build_profile.deploy_services
        if build_profile.provider:
            context["default_provider"] = build_profile.provider
        if build_profile.model:
            context["default_model"] = build_profile.model
        if build_profile.channel_finder_mode is not None:
            context["channel_finder_mode"] = build_profile.channel_finder_mode
        if build_profile.default_panel:
            context["default_panel"] = build_profile.default_panel
        if build_profile.panel_presets:
            context["panel_presets"] = build_profile.panel_presets
        if build_profile.claude_md_template:
            context["claude_md_template"] = build_profile.claude_md_template

        # 6a. The project `.env` this build starts from, read before anything
        # writes to it. Only the runtime-written keys survive a rebuild, and
        # step 13b is where that is decided — but by then the render has already
        # overwritten the file, so the "before" has to be captured here.
        project_env_path = project_path / ".env"
        existing_env_text = (
            project_env_path.read_text(encoding="utf-8") if project_env_path.is_file() else ""
        )

        # 6b. Create project directory early (venv creation needs it)
        project_path.mkdir(parents=True, exist_ok=True)

        # 6c. Create project venv with OSPREY + profile deps
        # Moved before template rendering so templates get the real project Python path.
        # The merged requirement list it installed — and recorded in the
        # project's pyproject.toml — is carried forward to step 6f, which
        # renders it into the generated Dockerfile.
        if not skip_deps:
            project_deps = _create_project_venv(project_path, build_profile)
        else:
            # No venv was created, so nothing was frozen and nothing merged.
            # The profile's own `dependencies` are all that is known to declare,
            # and `environment.packages` describes an environment this build
            # never assembled.
            project_deps = list(build_profile.dependencies or [])

        # 6d. Resolve the OSPREY-runtime interpreter for the template context.
        # This key feeds .mcp.json server commands, framework hook commands, and
        # registry `{current_python_env}` substitution — all processes that
        # import `osprey`, so every branch below must yield an interpreter that
        # has it. The build profile chooses here; config.yml never does, at build
        # time or later (`osprey claude regen` derives `sys.executable` for the
        # same reason). Agent-authored code is the other half of the split and is
        # resolved separately at run time by `resolve_agent_interpreter`.
        python_env = build_profile.python_env or "project"
        if skip_deps:
            # No venv created — pin to the python running osprey-build, which is
            # guaranteed to have osprey importable (else this command couldn't
            # run). Bare "python" gambles on PATH and breaks for subprocess
            # contexts that don't inherit the venv's PATH (Claude Code SDK,
            # containerized launchers).
            resolved_python_env = sys.executable
        elif python_env == "project":
            resolved_python_env = str(project_path / ".venv" / "bin" / "python")
        elif python_env == "build":
            resolved_python_env = sys.executable
        else:
            resolved_python_env = python_env
        context["current_python_env"] = resolved_python_env

        # 6d2. Record the environment declaration for the rendered config.yml.
        # This is provenance, not configuration: it is the profile's `environment:`
        # block verbatim, so a config.yml states which environment the project was
        # built from and the build can be reproduced from it. Deliberately no
        # interpreter path — the runtime resolves the interpreter conventionally
        # (project `.venv`, else the interpreter running OSPREY), so a config.yml
        # never carries an absolute host path that a move or a container would
        # invalidate.
        environment = build_profile.environment
        context["environment_python"] = environment.python
        context["environment_packages"] = list(environment.packages or [])
        context["environment_inherit_exclude"] = list(environment.inherit_exclude or [])

        # 6e. Override project_root for container builds
        if runtime_root:
            context["project_root"] = str(runtime_root)

        # 6f. Project dependencies for the generated Dockerfile's install line.
        # Not re-derived from the profile: this is the very list step 6c
        # installed into the project venv and recorded in its pyproject.toml, so
        # the image and the host it was built from carry the same packages. A
        # second derivation here is how the two drifted before — the profile's
        # `dependencies` alone omit both the frozen base and
        # `environment.packages`, leaving the image short of what the host has.
        context["dependencies"] = project_deps
        context["pip_dependency_args"] = " ".join(shlex.quote(d) for d in project_deps)

        manager = TemplateManager()

        # 6g. Prepare the virtual-accelerator channel manifest from the data
        # tree this build sources. It has to happen before step 7 on both
        # counts: step 7 prunes the tiers/ subtree the paradigm DBs live in,
        # and it renders the .env whose VA_CHANNELS_FILE/VA_LATTICE keys this
        # decides. Preparing (not writing) here is what makes the step atomic
        # — every way generation can fail has happened by the time the two
        # context keys are set. Most trees can't back a manifest and get None,
        # leaving the container's package fallback exactly as it was.
        # Imported here, not at module scope: this pulls in the channel-finder
        # database parsers, which no other build path needs.
        from osprey.services.virtual_accelerator.manifest.build import (
            MANIFEST_FILENAME as VA_MANIFEST_FILENAME,
        )
        from osprey.services.virtual_accelerator.manifest.build import (
            prepare_project_manifest,
            write_project_manifest,
        )

        va_data_root = build_profile.resolved_data_root(profile_dir) or (
            manager.template_root / "apps" / build_profile.data_bundle / "data"
        )
        prepared_va_manifest = prepare_project_manifest(va_data_root, build_profile.resolved_tier())
        if prepared_va_manifest is not None:
            context["va_channels_file"] = VA_MANIFEST_FILENAME
            context["va_lattice"] = "builtin"

        # 7. Create project from template (also materializes tier-specific
        # channel DBs from the tiers/ subtree, before the Claude Code
        # hierarchy probe reads the flat data/channel_databases/<name>.json
        # path). A profile carrying `data:` replaces the bundle's data tree
        # wholesale with its own; materialization then runs against that tree
        # unchanged, and the convention artifacts applied in step 11 still land
        # last.
        # The project `.env` is derived, not rendered: the profile's `.env` is
        # where a facility secret lives, so it wins over the render, and the
        # only thing kept from the project's previous `.env` is what a runtime
        # writer put there (see osprey.utils.dotenv.derive_project_env).
        profile_env = profile_dir / ".env"
        project_path = manager.create_project(
            project_name=project_name,
            output_dir=output_path,
            data_bundle=build_profile.data_bundle,
            context=context,
            force=True,  # Directory already exists from step 6b (venv created there)
            artifacts=artifacts or None,
            tier=build_profile.resolved_tier(),
            data_root=build_profile.resolved_data_root(profile_dir),
            profile_env_text=(
                profile_env.read_text(encoding="utf-8") if profile_env.is_file() else ""
            ),
            existing_env_text=existing_env_text,
        )
        logger.info("  ✓ Base template rendered")

        # 7b. The profile's `.env.example` is the project's: one documented
        # variable list for the whole stack, written where the operator edits
        # values. Rendering a second one into the project from the same
        # template would only invite the two to drift.
        profile_env_example = profile_dir / ".env.example"
        if profile_env_example.is_file():
            shutil.copy2(profile_env_example, project_path / ".env.example")

        # 8. Apply config overrides, plus what the deploy and va_archiver blocks
        #    contribute. Appended last and written in one pass, so the derived
        #    leaf lands inside whatever `modules.web_terminals` subtree the
        #    profile's own entries wrote rather than being replaced by it.
        #    Derived keys the profile also spells are rejected at validation, so
        #    winning here can never silently overwrite a facility's own value.
        derived_by_block = {
            "deploy": deploy_config_overrides(build_profile.deploy, build_profile.config),
            "va_archiver": va_archiver_config_overrides(build_profile.va_archiver),
        }
        derived = {
            key: value for block in derived_by_block.values() for key, value in block.items()
        }
        config_overrides = {**build_profile.config, **derived}
        if config_overrides:
            _apply_config_overrides(project_path, config_overrides)
            logger.info("  ✓ Applied %d config override(s)", len(config_overrides))
            for block, entries in derived_by_block.items():
                for key, value in entries.items():
                    logger.info("      %s: %s (from the profile's %s block)", key, value, block)

        # 9-10d. Service scaffolding + injection. Skipped wholesale for an
        # attached project (deploy_services: false): its service sections were
        # parsed and validated, but it deploys nothing of its own and instead
        # connects to a services stack another OSPREY project deployed on the
        # same host. The rendered config.yml already carries an empty
        # `deployed_services: []` (config.yml.j2 gates on `deploy_services`), so
        # nothing below needs to run.
        if build_profile.deploy_services:
            # 9. Copy service templates for `osprey deploy up`
            svc_count = _copy_service_templates(project_path)
            if svc_count:
                logger.info("  ✓ Copied %d service template(s) for deploy", svc_count)

            # 10. Inject profile-defined services (facility containers)
            if build_profile.services:
                psvc_count = _inject_profile_services(
                    profile_dir, project_path, build_profile.services
                )
                logger.info("  ✓ Injected %d profile service(s) for deploy", psvc_count)

            # 10b. Inject event-dispatch services + triggers
            if build_profile.dispatch is not None:
                _inject_dispatch(build_profile.dispatch, profile_dir, project_path)

            # 10b2. Inject the Nextcloud Talk bridge. Must follow step 10b: the
            # bridge's compose template gates its `depends_on` and its
            # in-network DISPATCHER_URL/WORKER_URL on `event_dispatcher` /
            # `dispatch_worker` being in `deployed_services`, which is exactly
            # what _inject_dispatch writes there. `validate()` already rejected a
            # bridge declared without a `dispatch:` block, so by here a declared
            # bridge means step 10b ran.
            if build_profile.nextcloud_bridge is not None:
                _inject_nextcloud_bridge(build_profile.nextcloud_bridge, project_path)

            # 10b3. Inject the Google Chat bridge. Must follow step 10b for the
            # same reason as 10b2: its compose template gates `depends_on` and
            # the in-network DISPATCHER_URL/WORKER_URL on `event_dispatcher` /
            # `dispatch_worker` being in `deployed_services`, which is what
            # _inject_dispatch writes there. `validate()` already rejected a
            # bridge declared without a `dispatch:` block, so by here a declared
            # bridge means step 10b ran. The two chat bridges are independent —
            # a project may deploy either, both, or neither.
            if build_profile.gchat_bridge is not None:
                _inject_gchat_bridge(build_profile.gchat_bridge, project_path)

            # 10c. Inject the Bluesky scan-bridge service
            if build_profile.bluesky is not None:
                _inject_bluesky(build_profile.bluesky, project_path)

            # 10c2. Inject the bluesky-panels sidecar + its three web panels (depends
            # on bluesky — the sidecar read-proxies the bridge — so this must run
            # after step 10c).
            if build_profile.bluesky_panels is not None:
                _inject_bluesky_panels(build_profile.bluesky_panels, project_path)

            # 10d. Inject the Virtual Accelerator soft-IOC service
            if build_profile.virtual_accelerator is not None:
                _inject_va(build_profile.virtual_accelerator, project_path)

            # 10e. Inject the archiver store + recorder. Must follow step 10d:
            # the recorder's compose template gates its image source, its
            # startup ordering and its Channel Access addressing on
            # `virtual_accelerator` being in `deployed_services`, which is
            # exactly what _inject_va writes there. The connection block and
            # the archive's knobs are not written here — they reached the
            # rendered config through step 8, which an attached project also
            # runs (see va_archiver_config_overrides).
            if build_profile.va_archiver is not None:
                _inject_va_archiver(build_profile.va_archiver, project_path)
        else:
            logger.info(
                "deploy_services: false — attached project; no services scaffolded "
                "(connects to a shared OSPREY services stack)"
            )

        # 11. Apply the profile's convention directories. The roster comes from
        # the project's own config.yml, so it carries every layer that shaped it
        # (step 8's overrides included); a persona build disables
        # modules.web_terminals and so resolves an empty roster, skipping
        # per-user context entirely.
        # Unconditional: every build resolves an operator-owned profile root,
        # so there is no longer a build whose "profile directory" is the
        # installed presets package.
        # Non-default data: directories, profile filenames, and other
        # profile-relative paths the profile declares are known root entries,
        # not typos of a convention directory.
        applied = _apply_conventions(
            profile_dir,
            project_path,
            _resolve_context_roster(project_path),
            extra_known=_profile_known_root_entries(build_profile, profile_arg),
            excluded=resolved.excluded_artifacts,
        )
        if applied.copied:
            logger.info(
                "  ✓ Applied %d profile artifact(s): %s",
                applied.copied,
                ", ".join(f"{count} {name}" for name, count in sorted(applied.by_category.items())),
            )

            # 11b. Register convention artifacts in config.yml
            reg_count = _register_convention_artifacts(project_path, applied)
            if reg_count:
                logger.info("  ✓ Registered %d profile artifact(s) in config.yml", reg_count)

        # 11c. Write the prepared VA manifest and its drive limits into
        # data/simulation/, the directory the container already bind-mounts —
        # so neither file needs a compose change. The decision is atomic (step
        # 6g settled it before anything was written); these writes carry it
        # out. After the convention artifacts land, so the limits the
        # accelerator enforces are the ones the project ships.
        # The two files come from different points on purpose: the limits from
        # the built project (post-conventions), the manifest from the source tree,
        # which is the only place the paradigm DBs still exist by now.
        if prepared_va_manifest is not None:
            write_project_manifest(prepared_va_manifest, project_path / "data")
            logger.info(
                "  ✓ Generated virtual-accelerator channel manifest (%d channels)",
                prepared_va_manifest.manifest["_metadata"]["total_channels"],
            )

        # 12. Persist profile MCP servers to config.yml
        if build_profile.mcp_servers:
            _persist_mcp_servers(project_path, build_profile.mcp_servers)
            logger.info(
                "  ✓ Persisted %d MCP server(s) to config.yml", len(build_profile.mcp_servers)
            )

        # 12b. Merge artifact_server overrides (gallery settings + custom
        # artifact categories) into config.yml
        if build_profile.artifact_server:
            _persist_artifact_server(project_path, build_profile.artifact_server)
            logger.info(
                "  ✓ Merged artifact_server overrides into config.yml (%d category/ies)",
                len(build_profile.artifact_server.get("categories", {})),
            )

        # 13. Copy profile .env file (if provided)
        if build_profile.env.file:
            _copy_env_file(profile_dir, project_path, build_profile.env.file)

        # 15. Report provider credentials. Runs after every .env write above so
        # the summary reflects what the project actually ships with.
        report_provider_credentials(project_path, build_profile.provider, profile_dir=profile_dir)

        # 16. Generate manifest
        manifest_context = {
            "default_provider": build_profile.provider,
            "default_model": build_profile.model,
        }
        if build_profile.channel_finder_mode is not None:
            manifest_context["channel_finder_mode"] = build_profile.channel_finder_mode
        if build_profile.claude_md_template:
            manifest_context["claude_md_template"] = build_profile.claude_md_template
        # The profile this project was built from, recorded on every build —
        # a preset build's materialized one included. It is what the deploy
        # follows to write minted secrets back to their source, so a project
        # that omitted it would degrade to keeping them only in its own `.env`.
        # Absolute: `deploy` runs from the project directory, where a relative
        # CLI string re-resolves to something else (usually nothing).
        manifest_context["profile_path_abs"] = str(profile_arg)
        # Carry the invocation source forward so build_reproducible_command
        # renders the matching --preset or positional form (C12).
        if preset:
            from .build_profile import _normalize_preset_name

            # The preset the PROFILE records, not the one the command line
            # named. A reused profile is built verbatim, so its own provenance
            # is what the project actually came from — and what the manifest's
            # reproducible command has to name to reproduce it. Falls back to
            # the CLI spelling for a profile that records no provenance.
            #
            # Read off the profile this build already resolved rather than
            # re-reading the file: a --preset build has read it once for the
            # provenance check `materialize_or_reuse_profile` makes, and a
            # second read could only disagree with the profile actually built.
            provenance = build_profile.provenance
            stamped = _normalize_preset_name(provenance.preset) if provenance is not None else None
            manifest_preset = stamped or _normalize_preset_name(preset)
            manifest_profile_path = None
        else:
            manifest_preset = None
            manifest_profile_path = profile  # the original CLI string

        manager.generate_manifest(
            project_dir=project_path,
            project_name=project_name,
            data_bundle=build_profile.data_bundle,
            context=manifest_context,
            artifacts=artifacts or None,
            preset_name=manifest_preset,
            profile_path=manifest_profile_path,
        )

        # 16b. Re-render Claude Code files with complete config
        # Profile MCP servers are now in config.yml (step 12), so regen
        # picks them up alongside framework servers.
        manager.regenerate_claude_code(
            project_path,
            project_root_override=runtime_root,
        )
        logger.info("  ✓ Re-rendered Claude Code artifacts")

        # 16c. Validate agent tools are backed by permissions.allow.
        # Catches wildcards in agent frontmatter and bug-class where a
        # facility author adds a tool to an agent's tools: allowlist but
        # forgets to add it to the MCP server's permissions.allow.
        from .validate_claude_artifacts import validate_agent_tools_against_permissions

        validation_errors = validate_agent_tools_against_permissions(project_path)
        if validation_errors:
            raise BuildProfileError(
                "Agent tool/permission drift detected:\n  " + "\n  ".join(validation_errors)
            )

        # 16d. Validate the model selection against the rendered provider map.
        # The web terminal runs the same resolver strict at startup, so a value
        # that only warns here would deploy per-user terminals that crash-loop
        # behind the reverse proxy. Failing the build stops the
        # `build && deploy` chain at the checkpoint the operator watches.
        from osprey.build.claude_code_resolver import load_provider_spec

        # Telemetry credentials are exempt from this strictness: a profile may
        # legitimately leave them as ${VAR} for the *deployment* to supply, and
        # the runtime re-resolves them at agent-spawn (degrading telemetry, not
        # the agent, if they are still unset). Aborting here would force every
        # such build to hand the builder production observability secrets.
        try:
            load_provider_spec(project_path, defer_unresolved_telemetry_creds=True)
        except ValueError as e:
            raise BuildProfileError(str(e)) from e

        # 17. Git init + commit
        _git_init_and_commit(project_path)

        # 18. Run post_build lifecycle commands
        if build_profile.lifecycle.post_build and not skip_lifecycle:
            _run_lifecycle_phase(
                "post_build",
                build_profile.lifecycle.post_build,
                project_path,
                project_path,
                stream=stream,
            )

        # 19. Run validate lifecycle commands
        if build_profile.lifecycle.validate and not skip_lifecycle:
            _run_lifecycle_phase(
                "validate",
                build_profile.lifecycle.validate,
                project_path,
                project_path,
                abort_on_failure=False,
                stream=stream,
            )

        logger.info("✓ Project built successfully at: %s", project_path)

        # Sim-backed presets ship scenario bundles whose logbook entries are
        # seeded into ARIEL on demand (build must never require a running
        # Postgres). Point the user at the one command that makes them live.
        if (project_path / "data" / "simulation" / "scenarios").is_dir():
            logger.info(
                "  → Seed the demo logbook with: cd %s && osprey sim apply nominal",
                project_path,
            )

        # Reached only by a build that produced a project: from here the edit
        # the CLI overrides made to the profile is what the project was built
        # from, and stays.
        build_completed = True

    except click.Abort:
        raise
    except click.UsageError:
        raise
    except BuildProfileError as e:
        logger.error("✗ Build error: %s", e)
        raise click.Abort() from e
    except ValueError as e:
        logger.error("✗ Error: %s", e)
        raise click.Abort() from e
    except Exception as e:
        logger.error("✗ Unexpected error: %s", e)
        import traceback

        logger.debug(traceback.format_exc())
        raise click.Abort() from e
    finally:
        # After the handlers above, so the reason the build failed is logged
        # first and the rollback reads as a consequence of it. In a `finally`
        # rather than in each handler so an interrupt rolls back too.
        if not build_completed:
            profile_guard.rollback()
