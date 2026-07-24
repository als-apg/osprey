"""Build command — assemble a facility-specific assistant from a build profile.

Reads a YAML build profile (or a bundled ``--preset``) that specifies a base
template, config overrides, file overlays, and MCP server definitions.
Produces a standalone, self-contained project directory (wipe-and-rebuild
safe).

Usage:
    osprey build my-assistant profile.yml
    osprey build my-assistant --preset hello-world
    osprey build my-assistant --preset education -O override.yml --set model=claude-sonnet-4-6
    osprey build --list-presets

The build pipeline's helper concerns live in sibling modules that this command
orchestrates: venv + ``.env`` templating in :mod:`osprey.cli.build_environment`,
lifecycle-phase execution in :mod:`osprey.cli.build_lifecycle`, service
injectors in :mod:`osprey.cli.build_injectors`, and project-directory
persistence (config overrides, overlays, MCP servers, git init) in
:mod:`osprey.cli.build_persistence`. They are re-exported below so
``from osprey.cli.build_cmd import <helper>`` keeps working.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

import click

from osprey.errors import BuildProfileError
from osprey.utils.logger import get_logger

from .build_environment import (
    _copy_env_file,
    _create_project_venv,
    _generate_env_template,
    _resolve_osprey_spec,
)
from .build_injectors import (
    _copy_service_templates,
    _inject_bluesky,
    _inject_bluesky_panels,
    _inject_dispatch,
    _inject_profile_services,
    _inject_va,
    _locate_pkg_services,
)
from .build_lifecycle import (
    _SHELL_METACHARACTERS,
    _format_junit_summary,
    _run_lifecycle_phase,
)
from .build_persistence import (
    _apply_config_overrides,
    _clear_rendered_project_dir,
    _copy_overlay_files,
    _git_init_and_commit,
    _persist_categories,
    _persist_mcp_servers,
    _register_overlay_artifacts,
)
from .templates.manager import TemplateManager

logger = get_logger("build")

__all__ = [
    "_SHELL_METACHARACTERS",
    "_apply_config_overrides",
    "_clear_rendered_project_dir",
    "_copy_env_file",
    "_copy_overlay_files",
    "_copy_service_templates",
    "_create_project_venv",
    "_format_junit_summary",
    "_generate_env_template",
    "_git_init_and_commit",
    "_inject_bluesky",
    "_inject_bluesky_panels",
    "_inject_dispatch",
    "_inject_profile_services",
    "_inject_va",
    "_locate_pkg_services",
    "_persist_categories",
    "_persist_mcp_servers",
    "_register_overlay_artifacts",
    "_resolve_osprey_spec",
    "_run_lifecycle_phase",
    "build",
]


def _list_presets_callback(ctx: click.Context, param: click.Parameter, value: bool) -> None:
    """Eager --list-presets: print bundled presets and exit before any args parse."""
    if not value or ctx.resilient_parsing:
        return
    from .build_profile import list_presets

    for name in list_presets():
        click.echo(name)
    ctx.exit(0)


@click.command()
@click.argument("project_name", required=False)
@click.argument(
    "profile",
    required=False,
    default=None,
    type=click.Path(exists=False, dir_okay=False),
)
@click.option(
    "--preset",
    default=None,
    metavar="NAME",
    help="Use a bundled preset profile (see --list-presets).",
)
@click.option(
    "--override",
    "-O",
    "overrides",
    multiple=True,
    type=click.Path(exists=False, dir_okay=False, path_type=Path),
    help="Layer a YAML file on top of the base profile/preset (repeatable).",
)
@click.option(
    "--set",
    "set_pairs",
    multiple=True,
    metavar="KEY.PATH=VALUE",
    help="Inline scalar/list override (repeatable). RHS parsed as YAML.",
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
    default=".",
    help="Output directory for project (default: current directory)",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help=(
        "Re-render an existing project directory in place "
        "(.env, _agent_data/, and .git are preserved)"
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
@click.option(
    "--emit-profile",
    "emit_profile",
    type=click.Path(path_type=Path),
    default=None,
    metavar="DIR",
    help="Scaffold an editable profile directory at DIR that extends --preset, "
    "then exit without rendering a project. Build the project from it with "
    "`osprey build <PROJECT_NAME> DIR/profile.yml`.",
)
def build(
    project_name: str | None,
    profile: str | None,
    preset: str | None,
    overrides: tuple[Path, ...],
    set_pairs: tuple[str, ...],
    output_dir: str,
    force: bool,
    stream: bool,
    skip_lifecycle: bool,
    skip_deps: bool,
    runtime_root: str | None,
    tier: str | None,
    emit_profile: Path | None,
) -> None:
    """Build a facility-specific assistant from a profile or bundled preset.

    Assembles a standalone project by rendering a base template, applying
    config overrides, copying overlay files, and injecting MCP servers.

    PROJECT_NAME: Name of the project directory to create

    PROFILE: Optional path to a YAML build profile (mutually exclusive with --preset)

    Examples:

    \b
      # Build from a bundled preset
      $ osprey build my-assistant --preset hello-world

      # Build from a profile file
      $ osprey build als-test ~/profiles/als-dev.yml

      # Layer overrides on top of a preset
      $ osprey build als-test --preset control-assistant -O als-overrides.yml \\
            --set model=claude-sonnet-4-6

      # List available presets
      $ osprey build --list-presets
    """
    from .build_profile import explicit_model_override_keys, resolve_build_profile
    from .project_utils import _clear_claude_code_project_state

    # --emit-profile is a project-less scaffold mode. Validate its constraints
    # and dispatch before the normal "PROJECT_NAME is required" check fires.
    if emit_profile is not None:
        if not preset:
            raise click.UsageError("--emit-profile requires --preset.")
        # Reject every flag that only makes sense for rendering a project.
        _incompatible: list[str] = []
        if project_name:
            _incompatible.append("PROJECT_NAME")
        if profile:
            _incompatible.append("PROFILE")
        if overrides:
            _incompatible.append("--override")
        if set_pairs:
            _incompatible.append("--set")
        if output_dir != ".":
            _incompatible.append("--output-dir")
        if force:
            _incompatible.append("--force")
        if stream:
            _incompatible.append("--stream")
        if skip_lifecycle:
            _incompatible.append("--skip-lifecycle")
        if skip_deps:
            _incompatible.append("--skip-deps")
        if runtime_root:
            _incompatible.append("--runtime-root")
        if tier is not None:
            _incompatible.append("--tier")
        if _incompatible:
            raise click.UsageError(
                "--emit-profile cannot be combined with project-rendering flags: "
                + ", ".join(_incompatible)
            )
        try:
            _emit_profile_directory(emit_profile, preset)
        except BuildProfileError as e:
            # Unknown-preset is a user error; promote to UsageError so the
            # exit code is 2 (same convention as the project-render path).
            if str(e).lower().startswith("unknown preset"):
                raise click.UsageError(str(e)) from e
            raise
        return

    if not project_name:
        raise click.UsageError("PROJECT_NAME is required. Run 'osprey build --help' for usage.")

    logger.info("Building project: %s", project_name)

    try:
        # 1. Resolve profile from any combination of preset / file / overlays.
        #    resolve_build_profile() enforces mutual exclusion (preset XOR profile)
        #    and merges layers in order: base -> override file(s) -> --set values.
        profile_arg = Path(profile).resolve() if profile else None
        try:
            build_profile, profile_dir = resolve_build_profile(
                profile_arg, preset, tuple(overrides), tuple(set_pairs)
            )
        except BuildProfileError as e:
            # Mutual-exclusion / missing-input / unknown-preset errors are
            # user errors, not bugs — promote to UsageError so the outer
            # except chain produces exit code 2.
            msg = str(e)
            lower = msg.lower()
            if "either" in lower or "not both" in lower or lower.startswith("unknown preset"):
                raise click.UsageError(msg) from e
            raise

        # CLI --tier overrides any value coming from the profile/preset/overrides.
        # Equivalent to --set tier=N but more discoverable in --help. click
        # constrains the choice to {1, 3}; convert the string form to the int
        # the profile model carries. Re-run validation so the tier rule (tier 1
        # requires channel_finder_mode: in_context) fails here with a
        # rule-naming error rather than downstream as a scaffolding
        # FileNotFoundError.
        if tier is not None:
            build_profile.tier = int(tier)
            build_profile.validate(profile_dir)

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

            from osprey import __version__

            spec = SpecifierSet(build_profile.requires_osprey_version)
            current = Version(__version__)
            if current not in spec:
                logger.error(
                    "  ✗ OSPREY %s does not satisfy requires_osprey_version: %s",
                    __version__,
                    build_profile.requires_osprey_version,
                )
                logger.info("     Upgrade OSPREY or run: osprey --version")
                raise click.Abort()
            logger.info(
                "  ✓ OSPREY %s satisfies %s",
                __version__,
                build_profile.requires_osprey_version,
            )

        # 2. Resolve output path
        output_path = Path(output_dir).resolve()
        project_path = output_path / project_name

        # 3. Handle --force / directory existence
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

        # 6b. Create project directory early (venv creation needs it)
        project_path.mkdir(parents=True, exist_ok=True)

        # 6c. Create project venv with OSPREY + profile deps
        # Moved before template rendering so templates get the real project Python path.
        if not skip_deps:
            _create_project_venv(project_path, build_profile)

        # 6d. Resolve python_env for template context
        python_env = build_profile.python_env or "project"
        if skip_deps:
            # No venv created — pin to the python running osprey-build, which is
            # guaranteed to have osprey importable (else this command couldn't
            # run). Bare "python" gambles on PATH and breaks for subprocess
            # contexts that don't inherit the venv's PATH (Claude Code SDK,
            # containerized launchers).
            import sys

            resolved_python_env = sys.executable
        elif python_env == "project":
            resolved_python_env = str(project_path / ".venv" / "bin" / "python")
        elif python_env == "build":
            import sys

            resolved_python_env = sys.executable
        else:
            resolved_python_env = python_env
        context["current_python_env"] = resolved_python_env

        # 6e. Override project_root for container builds
        if runtime_root:
            context["project_root"] = str(runtime_root)

        # 6f. Profile pip dependencies for the generated Dockerfile's install line
        deps = list(build_profile.dependencies or [])
        context["dependencies"] = deps
        context["pip_dependency_args"] = " ".join(shlex.quote(d) for d in deps)

        # 7. Create project from template (also materializes tier-specific
        # channel DBs from the preset's tiers/ subtree, before the Claude Code
        # hierarchy probe reads the flat data/channel_databases/<name>.json
        # path).
        manager = TemplateManager()
        project_path = manager.create_project(
            project_name=project_name,
            output_dir=output_path,
            data_bundle=build_profile.data_bundle,
            context=context,
            force=True,  # Directory already exists from step 6b (venv created there)
            artifacts=artifacts or None,
            tier=build_profile.resolved_tier(),
        )
        logger.info("  ✓ Base template rendered")

        # 8. Apply config overrides
        if build_profile.config:
            _apply_config_overrides(project_path, build_profile.config)
            logger.info("  ✓ Applied %d config override(s)", len(build_profile.config))

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
        else:
            logger.info(
                "deploy_services: false — attached project; no services scaffolded "
                "(connects to a shared OSPREY services stack)"
            )

        # 11. Copy overlay files
        if build_profile.overlay:
            _copy_overlay_files(profile_dir, project_path, build_profile.overlay)
            logger.info("  ✓ Copied %d overlay(s)", len(build_profile.overlay))

            # 11b. Register overlay artifacts in config.yml
            reg_count = _register_overlay_artifacts(project_path, build_profile.overlay)
            if reg_count:
                logger.info("  ✓ Registered %d overlay artifact(s) in config.yml", reg_count)

        # 12. Persist profile MCP servers to config.yml
        if build_profile.mcp_servers:
            _persist_mcp_servers(project_path, build_profile.mcp_servers)
            logger.info(
                "  ✓ Persisted %d MCP server(s) to config.yml", len(build_profile.mcp_servers)
            )

        # 12b. Persist custom artifact categories to config.yml
        if build_profile.categories:
            _persist_categories(project_path, build_profile.categories)
            logger.info(
                "  ✓ Persisted %d custom category/ies to config.yml",
                len(build_profile.categories),
            )

        # 13. Copy profile .env file (if provided)
        if build_profile.env.file:
            _copy_env_file(profile_dir, project_path, build_profile.env.file)

        # 14. Generate .env.template
        if build_profile.env.required or build_profile.env.defaults:
            _generate_env_template(project_path, build_profile.env)

        # 16. Generate manifest
        manifest_context = {
            "default_provider": build_profile.provider,
            "default_model": build_profile.model,
        }
        if build_profile.channel_finder_mode is not None:
            manifest_context["channel_finder_mode"] = build_profile.channel_finder_mode
        if build_profile.claude_md_template:
            manifest_context["claude_md_template"] = build_profile.claude_md_template
        # Mark which model-selection keys came from an explicit `--set` so the
        # manifest can distinguish user intent from resolved preset defaults
        # (persona auto-render forwards only the former).
        explicit_set_keys = explicit_model_override_keys(tuple(set_pairs))
        if explicit_set_keys:
            manifest_context["explicit_set_keys"] = explicit_set_keys
        # Carry the invocation source forward so build_reproducible_command
        # renders the matching --preset or positional form (C12).
        if preset:
            from .build_profile import _normalize_preset_name

            manifest_preset = _normalize_preset_name(preset)
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


def _emit_profile_directory(target_dir: Path, preset_name: str) -> None:
    """Scaffold an editable profile directory that extends ``preset_name``.

    Writes ``profile.yml`` (with ``extends: <preset>`` + commented override
    sections), an explanatory ``README.md``, and the ``overlays/{rules,skills,
    agents}/`` tree (with ``.gitkeep`` sentinels). The user then drops overlay
    artifacts in, edits ``profile.yml``, and builds the project with
    ``osprey build <PROJECT_NAME> <target_dir>/profile.yml``.
    """
    from .build_profile import _load_preset_raw, _normalize_preset_name
    from .templates.scaffolding import _copy_data_tree

    # Resolve and validate the preset name up-front so the error is clean
    # (raises BuildProfileError → caught by the outer except chain).
    preset_raw, _preset_path = _load_preset_raw(preset_name)

    target = target_dir.resolve()
    if target.exists():
        raise click.UsageError(
            f"Target directory already exists: {target}. Remove it or choose a different path."
        )

    normalized_preset = _normalize_preset_name(preset_name)
    # `target.name` is the user-chosen directory name (e.g. "my-profile").
    # Derive a human display name only when the preset itself has no `name:`.
    profile_name_default = target.name.replace("-", " ").replace("_", " ").title()
    preset_display_name = preset_raw.get("name") or profile_name_default

    manager = TemplateManager()
    seed_root = manager.template_root / "profile_seed"
    if not seed_root.is_dir():
        # Defensive: catch packaging regressions early with an actionable error
        # rather than letting Jinja raise TemplateNotFound deep in the loader.
        raise BuildProfileError(
            f"Profile seed templates missing at {seed_root}. "
            f"This is a packaging bug — reinstall osprey-framework."
        )

    target.mkdir(parents=True)
    ctx = {
        "preset_name": normalized_preset,
        "preset_display_name": preset_display_name,
        "profile_name": profile_name_default,
        "profile_dirname": target.name,
        "profile_filename": f"{target.name}/profile.yml",
    }
    _copy_data_tree(seed_root, target, manager.template_root, manager.jinja_env, ctx)

    logger.info("✓ Scaffolded profile at: %s", target)
    logger.info(
        "  Next: edit %s/profile.yml, then run `osprey build <PROJECT_NAME> %s/profile.yml`",
        target,
        target,
    )
