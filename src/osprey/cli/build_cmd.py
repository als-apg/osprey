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
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import threading
import time
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlparse

import click

from osprey.errors import BuildProfileError
from osprey.utils.dotenv import parse_dotenv_file as _load_dotenv
from osprey.utils.logger import get_logger

from .templates.manager import TemplateManager

if TYPE_CHECKING:
    from osprey.cli.build_profile import (
        BlueskyConfig,
        BlueskyPanelsConfig,
        DispatchConfig,
        VAConfig,
    )

logger = get_logger("build")


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
@click.option("--force", "-f", is_flag=True, help="Force overwrite if project directory exists")
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
    type=click.IntRange(1, 3),
    default=None,
    help="Channel-database tier (1|2|3). Selects which "
    "data/channel_databases/tiers/tier{N}/ DB the rendered config points at. "
    "Advanced: override the paradigm-derived default "
    "(in_context → tier 1, hierarchical/middle_layer → tier 3).",
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
    tier: int | None,
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
    from .build_profile import resolve_build_profile
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
        # Equivalent to --set tier=N but more discoverable in --help.
        if tier is not None:
            build_profile.tier = tier

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
                logger.warning("  Removing existing directory: %s", project_path)
                shutil.rmtree(project_path)
                logger.info("  ✓ Removed existing directory")
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

        # 9. Copy service templates for `osprey deploy up`
        svc_count = _copy_service_templates(project_path)
        if svc_count:
            logger.info("  ✓ Copied %d service template(s) for deploy", svc_count)

        # 10. Inject profile-defined services (facility containers)
        if build_profile.services:
            psvc_count = _inject_profile_services(profile_dir, project_path, build_profile.services)
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


_SHELL_METACHARACTERS = ("|", "&&", "||", "$(", "`")


def _format_junit_summary(xml_path: Path) -> None:
    """Parse JUnit XML and print a Rich summary table of test results."""
    import xml.etree.ElementTree as ET

    from rich.console import Console
    from rich.table import Table

    if not xml_path.exists():
        return

    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return

    root = tree.getroot()

    table = Table(title="Integration Test Results", show_header=True, padding=(0, 1))
    table.add_column("Test", style="bold", no_wrap=True)
    table.add_column("Status", justify="center", width=6)
    table.add_column("Time", justify="right", width=8)

    for testsuite in root.iter("testsuite"):
        for testcase in testsuite.iter("testcase"):
            name = testcase.get("name", "unknown")
            time_s = testcase.get("time", "0")

            failure = testcase.find("failure")
            error = testcase.find("error")
            skipped = testcase.find("skipped")

            if failure is not None or error is not None:
                status = "[red]✗[/red]"
            elif skipped is not None:
                status = "[dim]skip[/dim]"
            else:
                status = "[green]✓[/green]"

            table.add_row(name, status, f"{float(time_s):.2f}s")

    if table.row_count > 0:
        render_console = Console(force_terminal=True, width=120)
        render_console.print(table)


def _run_lifecycle_phase(
    phase_name: str,
    steps: list[Any],
    default_cwd: Path,
    project_path: Path,
    *,
    abort_on_failure: bool = True,
    stream: bool = False,
) -> None:
    """Run lifecycle commands for a build phase.

    Args:
        phase_name: Phase name for display (pre_build, post_build, validate).
        steps: List of LifecycleStep objects.
        default_cwd: Default working directory for steps without explicit cwd.
        project_path: Project root path for {project_root} substitution.
        abort_on_failure: If True, raise BuildProfileError on failure.
            If False, warn and continue (used for validate phase).
        stream: If True, stream stdout/stderr in real-time instead of capturing.
    """
    # Auto-inject .env vars into subprocess environment
    env_file = project_path / ".env"
    if env_file.is_file():
        dotenv_vars = _load_dotenv(env_file)
        sub_env = {**os.environ, **dotenv_vars}
        logger.info("Loaded %d vars from %s into lifecycle environment", len(dotenv_vars), env_file)
    else:
        sub_env = os.environ.copy()

    # Prepend project venv to PATH so `python` resolves to the project's
    # Python (with profile deps) rather than OSPREY's Python.
    venv_bin = project_path / ".venv" / "bin"
    if venv_bin.is_dir():
        sub_env["PATH"] = f"{venv_bin}{os.pathsep}{sub_env.get('PATH', '')}"
        logger.info("Prepended project venv to lifecycle PATH: %s", venv_bin)

    # Prepend _mcp_servers to PYTHONPATH so lifecycle commands can
    # ``import integration_tests`` (and other MCP server packages)
    # without manual PYTHONPATH wrappers in profile YAML.
    mcp_servers_dir = project_path / "_mcp_servers"
    if mcp_servers_dir.is_dir():
        existing = sub_env.get("PYTHONPATH", "")
        sub_env["PYTHONPATH"] = (
            f"{mcp_servers_dir}{os.pathsep}{existing}" if existing else str(mcp_servers_dir)
        )
        logger.info("Prepended _mcp_servers to lifecycle PYTHONPATH: %s", mcp_servers_dir)

    logger.info("  Running %s commands...", phase_name)
    for step in steps:
        cmd_str = step.run.replace("{project_root}", str(project_path))

        # Resolve cwd
        if step.cwd:
            cwd_str = step.cwd.replace("{project_root}", str(project_path))
            cwd = (default_cwd / cwd_str).resolve()
        else:
            cwd = default_cwd

        # Detect shell metacharacters
        use_shell = any(meta in cmd_str for meta in _SHELL_METACHARACTERS)

        t0 = time.monotonic()
        try:
            cmd = cmd_str if use_shell else shlex.split(cmd_str)

            if stream or step.stream:
                # Stream mode: show output in real-time, prefix with step name.
                # Uses a threaded reader so proc.wait(timeout=...) can enforce
                # the timeout even when the subprocess stalls mid-output.
                logger.info("  > %s", step.name)
                proc = subprocess.Popen(
                    cmd,
                    shell=use_shell,
                    cwd=cwd,
                    env=sub_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                assert proc.stdout is not None  # noqa: S101

                def _drain_stdout(stdout=proc.stdout) -> None:
                    for line in stdout:
                        print(f"    {line}", end="", flush=True)

                reader = threading.Thread(target=_drain_stdout, daemon=True)
                reader.start()
                # Wait for stdout to drain (tests finished) with the full
                # timeout, then give the process a short grace period to
                # exit.  Some test frameworks (pyepics CA context) keep
                # background threads alive that prevent clean exit.
                reader.join(timeout=step.timeout)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                elapsed = time.monotonic() - t0
                if proc.returncode != 0:
                    msg = f"Lifecycle {phase_name} step '{step.name}' failed (exit {proc.returncode}, {elapsed:.1f}s)"
                    if abort_on_failure:
                        logger.error("  ✗ %s", msg)
                        _format_junit_summary(project_path / "check_results.xml")
                        raise BuildProfileError(msg)
                    else:
                        logger.warning("  ! %s", msg)
                else:
                    logger.info("  ✓ %s (%.1fs)", step.name, elapsed)
                # Show JUnit summary if test results were produced
                _format_junit_summary(project_path / "check_results.xml")
            else:
                # Quiet mode: capture output, show one-line summary
                result = subprocess.run(
                    cmd,
                    shell=use_shell,
                    cwd=cwd,
                    env=sub_env,
                    capture_output=True,
                    text=True,
                    timeout=step.timeout,
                )
                elapsed = time.monotonic() - t0

                if result.returncode != 0:
                    output = (result.stdout + result.stderr).strip()
                    msg = f"Lifecycle {phase_name} step '{step.name}' failed (exit {result.returncode}, {elapsed:.1f}s)"
                    if output:
                        msg += f":\n{output}"
                    if abort_on_failure:
                        logger.error("  ✗ %s", msg)
                        _format_junit_summary(project_path / "check_results.xml")
                        raise BuildProfileError(msg)
                    else:
                        logger.warning("  ! %s", msg)
                else:
                    success_msg = f"{step.name} ({elapsed:.1f}s)"
                    output = (result.stdout + result.stderr).strip()
                    if output:
                        summary = output.rstrip().rsplit("\n", 1)[-1].strip()
                        if summary:
                            success_msg += f" — {summary}"
                    logger.info("  ✓ %s", success_msg)
                # Show JUnit summary if test results were produced
                _format_junit_summary(project_path / "check_results.xml")

        except subprocess.TimeoutExpired as e:
            elapsed = time.monotonic() - t0
            msg = f"Lifecycle {phase_name} step '{step.name}' timed out ({elapsed:.0f}s)"
            # Show partial output captured before timeout (quiet mode only;
            # stream mode already printed output in real-time).
            _out = (
                e.stdout.decode(errors="replace")
                if isinstance(e.stdout, bytes)
                else (e.stdout or "")
            )
            _err = (
                e.stderr.decode(errors="replace")
                if isinstance(e.stderr, bytes)
                else (e.stderr or "")
            )
            partial = _out + _err
            if partial.strip():
                tail = "\n".join(partial.strip().splitlines()[-20:])
                msg += f"\n  Last output:\n{tail}"
            if abort_on_failure:
                logger.error("  ✗ %s", msg)
                _format_junit_summary(project_path / "check_results.xml")
                raise BuildProfileError(msg) from None
            else:
                logger.warning("  ! %s", msg)
            _format_junit_summary(project_path / "check_results.xml")
        except OSError as exc:
            msg = f"Lifecycle {phase_name} step '{step.name}' failed to start: {exc}"
            if abort_on_failure:
                logger.error("  ✗ %s", msg)
                raise BuildProfileError(msg) from exc
            else:
                logger.warning("  ! %s", msg)


def _copy_env_file(profile_dir: Path, project_path: Path, env_file: str) -> None:
    """Copy a profile-provided .env file to the built project."""
    src = (profile_dir / env_file).resolve()
    dst = project_path / ".env"
    shutil.copy2(src, dst)
    logger.info("  ✓ Copied %s → .env", env_file)


def _generate_env_template(project_path: Path, env_config: Any) -> None:
    """Generate a .env.template file from the profile's env configuration."""
    lines: list[str] = []
    if env_config.required:
        lines.append("# Required")
        for var in env_config.required:
            lines.append(f"{var}=")
    if env_config.defaults:
        if lines:
            lines.append("")
        lines.append("# Defaults")
        for var, value in env_config.defaults.items():
            lines.append(f"{var}={value}")
    lines.append("")  # Trailing newline

    env_path = project_path / ".env.template"
    env_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("  ✓ Generated .env.template")
    if not (project_path / ".env").exists():
        logger.info("  Hint: Copy .env.template to .env and fill in required values")


def _resolve_osprey_spec(osprey_install: str) -> tuple[str, str]:
    """Resolve the osprey install spec for the project venv.

    Returns ``(spec, label)`` where ``spec`` is the pip/uv install argument
    and ``label`` is a human-readable identifier used in logs and the
    generated requirements.txt comment.

    The ``osprey_install`` value drives the resolution:
      - ``"local"`` (default): consult ``importlib.metadata``. Editable
        installs (``pip install -e .``, ``uv sync``) install from the source
        tree; non-editable installs (``uv tool install``, wheels from PyPI)
        pin to the running version (``osprey-framework==<version>``).
      - ``"pip"``: install ``osprey-framework`` from PyPI, unpinned.
      - anything else: treated as a PEP 508 spec, passed through verbatim.
    """
    if osprey_install == "local":
        try:
            dist = distribution("osprey-framework")
        except PackageNotFoundError:
            dist = None

        direct_url_text = dist.read_text("direct_url.json") if dist else None
        info = json.loads(direct_url_text) if direct_url_text else {}
        if info.get("dir_info", {}).get("editable"):
            src_path = unquote(urlparse(info["url"]).path)
            return src_path, f"editable: {src_path}"

        if dist is not None:
            spec = f"osprey-framework=={dist.version}"
            return spec, spec

        # Metadata unavailable (rare: e.g. running osprey directly from a
        # source tree without installing it). Fall back to the source root
        # one final time so dev workflows that bypass install still work.
        osprey_root = Path(__file__).resolve().parents[3]
        if (osprey_root / "pyproject.toml").exists():
            return str(osprey_root), f"local: {osprey_root}"
        raise BuildProfileError(
            "Cannot resolve osprey install location: package metadata is "
            f"missing and no source tree is present at {osprey_root}. "
            "Install osprey-framework with `uv tool install osprey-framework` "
            "or set `osprey_install` explicitly in your profile."
        )

    if osprey_install == "pip":
        return "osprey-framework", "osprey-framework"

    return osprey_install, osprey_install


def _create_project_venv(project_path: Path, profile: Any) -> None:
    """Create the project venv and install osprey + profile deps.

    This is the single place where the project's Python environment is set up.
    One venv, one install command, one resolver pass. The resolver sees all
    dependencies together (osprey + profile deps) and either succeeds or fails.

    See :func:`_resolve_osprey_spec` for how ``profile.osprey_install`` is
    interpreted.
    """
    import sys

    venv_path = project_path / ".venv"
    uv_path = os.environ.get("UV") or shutil.which("uv")

    # --- Create venv ---
    logger.info("  Creating project virtual environment...")
    if uv_path:
        result = subprocess.run(
            [uv_path, "venv", str(venv_path), "--python", sys.executable, "--quiet"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    else:
        result = subprocess.run(
            [sys.executable, "-m", "venv", str(venv_path)],
            capture_output=True,
            text=True,
            timeout=60,
        )
    if result.returncode != 0:
        output = (result.stdout + result.stderr).strip()
        raise BuildProfileError(f"Failed to create project venv: {output}")

    # --- Resolve osprey install spec ---
    osprey_install = profile.osprey_install or "local"
    osprey_spec, osprey_label = _resolve_osprey_spec(osprey_install)

    # --- Install osprey + profile deps ---
    all_deps = [osprey_spec] + list(profile.dependencies or [])
    venv_python = venv_path / "bin" / "python"
    dep_count = len(profile.dependencies or [])

    if uv_path:
        cmd = [uv_path, "pip", "install", "--quiet", "-p", str(venv_python), *all_deps]
    else:
        cmd = [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "--quiet",
            "--disable-pip-version-check",
            *all_deps,
        ]

    from rich.live import Live
    from rich.spinner import Spinner

    spinner = Spinner("dots", text=f"  Installing osprey ({osprey_label}) + {dep_count} deps...")
    with Live(spinner, transient=True):
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode == 0:
        logger.info("  ✓ Installed osprey + %d profile deps into project venv", dep_count)
    elif "litellm" in (result.stdout + result.stderr).lower():
        # ---------------------------------------------------------------
        # TEMPORARY WORKAROUND — litellm supply chain attack (2026-03-24)
        #
        # litellm versions 1.82.7-1.82.8 were compromised with credential-
        # stealing malware (TeamPCP attack chain). PyPI has quarantined the
        # entire package, so uv refuses to resolve it.
        #
        # Workaround: install osprey --no-deps + profile deps into the
        # project venv, then add a .pth file pointing to OSPREY's own
        # site-packages so the project inherits litellm and other
        # transitive deps from the known-good build environment.
        #
        # REVERT THIS when litellm is restored on PyPI:
        #   1. Remove this entire elif block
        #   2. The normal install path above will work again
        # ---------------------------------------------------------------
        logger.warning(
            "  litellm unavailable on PyPI (quarantined) — inheriting from build environment"
        )
        # Install osprey (no transitive deps) + profile deps
        if uv_path:
            cmd_nodeps = [
                uv_path,
                "pip",
                "install",
                "--quiet",
                "-p",
                str(venv_python),
                "--no-deps",
                osprey_spec,
            ]
            cmd_profile = (
                [
                    uv_path,
                    "pip",
                    "install",
                    "--quiet",
                    "-p",
                    str(venv_python),
                    *list(profile.dependencies or []),
                ]
                if profile.dependencies
                else None
            )
        else:
            cmd_nodeps = [
                str(venv_python),
                "-m",
                "pip",
                "install",
                "--quiet",
                "--disable-pip-version-check",
                "--no-deps",
                osprey_spec,
            ]
            cmd_profile = (
                [
                    str(venv_python),
                    "-m",
                    "pip",
                    "install",
                    "--quiet",
                    "--disable-pip-version-check",
                    *list(profile.dependencies or []),
                ]
                if profile.dependencies
                else None
            )

        spinner = Spinner("dots", text="  Installing osprey (--no-deps)...")
        with Live(spinner, transient=True):
            r = subprocess.run(cmd_nodeps, capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            raise BuildProfileError(
                f"Failed to install osprey --no-deps:\n{(r.stdout + r.stderr).strip()}"
            )

        if cmd_profile:
            spinner = Spinner("dots", text=f"  Installing {dep_count} profile deps...")
            with Live(spinner, transient=True):
                r = subprocess.run(cmd_profile, capture_output=True, text=True, timeout=120)
            if r.returncode != 0:
                raise BuildProfileError(
                    f"Failed to install profile deps:\n{(r.stdout + r.stderr).strip()}"
                )

        # Add .pth file so project venv can import osprey's transitive deps
        # (litellm, pandas, etc.) from the build environment's site-packages
        build_site_packages = Path(sys.prefix) / "lib"
        # Find the actual site-packages dir (python version varies)
        sp_dirs = list(build_site_packages.glob("python*/site-packages"))
        if sp_dirs:
            pth_path = venv_path / "lib"
            proj_sp = list(pth_path.glob("python*/site-packages"))
            if proj_sp:
                pth_file = proj_sp[0] / "_osprey_build_env.pth"
                pth_file.write_text(f"{sp_dirs[0]}\n")
                logger.info("  ✓ Linked build environment site-packages via .pth")

        logger.info("  ✓ Installed osprey (--no-deps) + %d profile deps", dep_count)
    else:
        output = (result.stdout + result.stderr).strip()
        raise BuildProfileError(
            f"Failed to install project dependencies (exit {result.returncode}):\n{output}"
        )

    # --- Record deps in requirements.txt for documentation ---
    req_path = project_path / "requirements.txt"
    lines = ["\n", f"# osprey ({osprey_label})\n", f"{osprey_spec}\n"]
    if profile.dependencies:
        lines.append("\n# Profile dependencies\n")
        for dep in profile.dependencies:
            lines.append(f"{dep}\n")
    with open(req_path, "a", encoding="utf-8") as f:
        f.writelines(lines)


def _apply_config_overrides(project_path: Path, config_dict: dict[str, Any]) -> None:
    """Apply dot-notation config overrides to the project's config.yml."""
    from osprey.utils.config_writer import config_update_fields

    config_path = project_path / "config.yml"
    if not config_path.exists():
        logger.warning("config.yml not found at %s — skipping config overrides", config_path)
        return
    config_update_fields(config_path, config_dict)


def _locate_pkg_services() -> Path:
    """Locate the OSPREY package's bundled ``templates/services`` directory.

    Prefers the installed ``osprey.templates`` package's location; falls back to
    a path relative to this module for source/editable checkouts where the
    package metadata is unavailable. Callers check ``.is_dir()`` on the result,
    since the directory may be absent in a stripped-down install.
    """
    try:
        import osprey.templates

        return Path(osprey.templates.__file__).parent / "services"
    except (ImportError, AttributeError):
        return Path(__file__).parent.parent / "templates" / "services"


def _copy_service_templates(project_path: Path) -> int:
    """Copy service compose templates from the OSPREY package into the project.

    Copies each service's compose template directory from the package to the
    project's ``services/`` tree for the UNION of ``deployed_services`` and
    every service merely DECLARED under ``services:`` that ships a package
    template.  This makes the project self-contained so that ``osprey deploy
    up`` works directly from the project directory, and — crucially — bundles
    opt-in add-ons (declared but not deployed) so they can be switched on later
    via a ``deployed_services`` edit + ``osprey deploy up`` without rebuilding.
    A bundled-but-not-deployed template is inert until deployed.

    Returns:
        Number of service template directories copied.
    """
    from ruamel.yaml import YAML

    config_path = project_path / "config.yml"
    if not config_path.exists():
        return 0

    yaml = YAML()
    with open(config_path) as fh:
        config = yaml.load(fh)

    # Locate the package's service templates directory
    pkg_services = _locate_pkg_services()

    if not pkg_services.is_dir():
        logger.warning("Service templates directory not found — skipping")
        return 0

    dest_services_root = project_path / "services"
    dest_services_root.mkdir(exist_ok=True)

    # Always copy the root compose template so `osprey deploy up` works even
    # for presets with no deployed_services (the renderer references it
    # unconditionally; without it deploy fails with TemplateNotFound).
    root_template = pkg_services / "docker-compose.yml.j2"
    if root_template.exists():
        shutil.copy2(root_template, dest_services_root / "docker-compose.yml.j2")

    services_config = config.get("services", {})

    # Bundle the UNION of deployed services and every service merely DECLARED
    # under `services:`.  deployed_services come first (preserving prior
    # behavior exactly), then any declared key not already present.  Bundling a
    # declared-but-not-deployed template keeps it inert until deployed, so
    # opt-in add-ons (e.g. the openobserve telemetry backend) can be turned on
    # later via a `deployed_services` edit + `osprey deploy up`, no rebuild.
    deployed = [str(s) for s in config.get("deployed_services", [])]
    names = list(deployed)
    for declared in services_config:
        name = str(declared)
        if name not in names:
            names.append(name)

    if not names:
        return 0

    deployed_set = set(deployed)

    count = 0
    for name in names:
        # Resolve package source directory
        parts = name.split(".")
        if parts[0] == "osprey" and len(parts) == 2:
            src_dir = pkg_services / parts[1]
        elif len(parts) == 1:
            src_dir = pkg_services / name
        else:
            logger.warning("Skipping service %r — unsupported naming for template copy", name)
            continue

        if not src_dir.is_dir():
            # A declared-but-not-deployed service may legitimately ship no
            # package template (e.g. facility-injected elsewhere) — skip it
            # silently.  Only warn when a *deployed* service is missing its
            # template, which would break `osprey deploy up`.
            if name in deployed_set:
                logger.warning("No package template for service %r at %s", name, src_dir)
            continue

        # Determine destination from the service config's path field
        svc_config = services_config.get(parts[-1], {})
        dest_rel = svc_config.get("path", f"./services/{parts[-1]}")
        dest_dir = project_path / dest_rel.lstrip("./")

        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.copytree(src_dir, dest_dir)
        count += 1

    return count


def _inject_profile_services(
    profile_dir: Path, project_path: Path, services: dict[str, Any]
) -> int:
    """Copy facility-defined service templates and register them in config.yml.

    For each service declared in the profile's ``services:`` section:
    1. Copies the template directory to ``{project}/services/{name}/``
    2. Writes ``services.{name}`` config entries to config.yml
    3. Appends the service to ``deployed_services``

    This lets facilities define their own containers (Typesense, Redis, etc.)
    alongside OSPREY's built-in services (PostgreSQL).

    Returns:
        Number of profile services injected.
    """
    from ruamel.yaml import YAML

    if not services:
        return 0

    config_path = project_path / "config.yml"
    if not config_path.exists():
        return 0

    yaml = YAML()
    yaml.preserve_quotes = True
    with open(config_path) as fh:
        config = yaml.load(fh)

    dest_services_root = project_path / "services"
    dest_services_root.mkdir(exist_ok=True)

    count = 0
    for name, svc_def in services.items():
        # Copy template directory
        src_dir = profile_dir / svc_def.template
        dest_dir = dest_services_root / name
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.copytree(src_dir, dest_dir)

        # Register service config in config.yml
        if "services" not in config:
            config["services"] = {}
        svc_config = {"path": f"./services/{name}"}
        svc_config.update(svc_def.config)
        config["services"][name] = svc_config

        # Add to deployed_services
        deployed = config.get("deployed_services", [])
        if name not in [str(s) for s in deployed]:
            deployed.append(name)
            config["deployed_services"] = deployed

        count += 1

    with open(config_path, "w") as fh:
        yaml.dump(config, fh)

    return count


def _inject_dispatch(dispatch: DispatchConfig, profile_dir: Path, project_path: Path) -> None:
    """Wire the event-dispatch feature into a built project.

    1. Resolve and copy the triggers file to ``<project>/triggers.yml``.
    2. Copy the bundled event_dispatcher + dispatch_worker compose templates
       into ``<project>/services/``.
    3. Write ``services.{event_dispatcher,dispatch_worker}`` config + register
       both in ``deployed_services``.
    4. Print a post-build hint (dashboard URL + sample curl + image prerequisite).

    Args:
        dispatch: Validated dispatch configuration from the build profile.
        profile_dir: Directory containing the build profile (triggers source).
        project_path: Root of the built project.

    Raises:
        BuildProfileError: If the configured triggers file cannot be resolved.
    """
    from ruamel.yaml import YAML

    from osprey.cli.build_profile import _triggers_dir

    # 1. Resolve + copy triggers file (profile-relative path or bundled triggers name).
    if (profile_dir / dispatch.triggers).is_file():
        triggers_src = profile_dir / dispatch.triggers
    elif (_triggers_dir() / dispatch.triggers).is_file():
        triggers_src = _triggers_dir() / dispatch.triggers
    else:
        raise BuildProfileError(f"dispatch.triggers not found: {dispatch.triggers!r}")
    triggers_dest = project_path / "triggers.yml"
    shutil.copy2(triggers_src, triggers_dest)

    # 1a. Make the preset the single source of truth for pool limits. The bundled
    # triggers file hardcodes its own dispatcher.max_concurrent_runs/max_queue_depth
    # (the dispatcher reads them from triggers.yml at runtime), so a profile that
    # overrides dispatch.max_concurrent_runs/max_queue_depth would otherwise be
    # silently ignored. Patch the copied file's dispatcher block to match the
    # validated DispatchConfig.
    _trigger_yaml = YAML()
    _trigger_yaml.preserve_quotes = True
    with open(triggers_dest) as fh:
        triggers_doc = _trigger_yaml.load(fh)
    if triggers_doc is not None:
        dispatcher_block = triggers_doc.setdefault("dispatcher", {})
        dispatcher_block["max_concurrent_runs"] = dispatch.max_concurrent_runs
        dispatcher_block["max_queue_depth"] = dispatch.max_queue_depth
        with open(triggers_dest, "w") as fh:
            _trigger_yaml.dump(triggers_doc, fh)

    # 2. Copy bundled compose templates (located the same way as service templates).
    pkg_services = _locate_pkg_services()

    dest_services_root = project_path / "services"
    dest_services_root.mkdir(exist_ok=True)

    for name in ("event_dispatcher", "dispatch_worker"):
        src_dir = pkg_services / name
        if not src_dir.is_dir():
            logger.warning("No package template for dispatch service %r at %s", name, src_dir)
            continue
        dest_dir = dest_services_root / name
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.copytree(src_dir, dest_dir)

    # 3. Write config.yml entries + register in deployed_services.
    config_path = project_path / "config.yml"
    if not config_path.exists():
        logger.warning("config.yml not found — skipping dispatch config registration")
        return

    yaml = YAML()
    yaml.preserve_quotes = True
    with open(config_path) as fh:
        config = yaml.load(fh)

    # No ``image`` key on either service, so each falls to its compose default:
    # the event-dispatcher builds the shared osprey-dispatch:local image (its own
    # compose ``build:`` block), and the dispatch worker runs <project>:local —
    # the project image ``osprey deploy up`` builds from the project Dockerfile
    # (the worker has no build block of its own, to avoid racing the dispatcher).
    # Override with OSPREY_DISPATCH_IMAGE/OSPREY_WORKER_IMAGE, or set
    # ``services.<name>.image`` here, to use a prebuilt/published image.
    config.setdefault("services", {})
    config["services"]["event_dispatcher"] = {
        "path": "./services/event_dispatcher",
        "port": dispatch.dispatcher_port,
        "facility_name": dispatch.facility_name,
        "pv_strip_prefix": dispatch.pv_strip_prefix,
        # Copy the project's triggers.yml into the service build context so the
        # compose ``./triggers.yml`` bind-mount resolves to a file (otherwise the
        # container runtime auto-creates an empty directory at the mount source).
        "additional_dirs": [{"src": "triggers.yml", "dst": "triggers.yml"}],
    }
    config["services"]["dispatch_worker"] = {
        "path": "./services/dispatch_worker",
        "worker_count": dispatch.worker_count,
        "worker_port_base": dispatch.worker_port_base,
        "workspace_mode": dispatch.workspace_mode,
        "timeout_sec": dispatch.timeout_sec,
        "inactivity_sec": dispatch.inactivity_sec,
    }
    deployed = config.get("deployed_services", []) or []
    for name in ("event_dispatcher", "dispatch_worker"):
        if name not in [str(s) for s in deployed]:
            deployed.append(name)
    config["deployed_services"] = deployed

    # Derive web.panels.events.url from dispatcher_port so the port is a single
    # source of truth.  Write only if the profile has not already set an explicit
    # ``web.panels.events.url`` via a config override (merged earlier in the
    # build); explicit overrides take precedence.
    #
    # Emit a bare-host ``url`` plus a ``/dashboard`` ``path`` (rather than baking
    # ``/dashboard`` into ``url``) to match the custom-panel proxy convention:
    # the web terminal composes ``url.rstrip('/') + '/' + path``, so a path baked
    # into ``url`` double-prefixes sub-routes. ``setdefault`` on ``path`` honors a
    # facility that pinned its own ``web.panels.events.path``.
    existing_events_url = config.get("web", {}).get("panels", {}).get("events", {}).get("url", "")
    if not existing_events_url:
        config.setdefault("web", {}).setdefault("panels", {}).setdefault("events", {})
        events_panel = config["web"]["panels"]["events"]
        events_panel["url"] = f"http://localhost:{dispatch.dispatcher_port}"
        events_panel.setdefault("path", "/dashboard")

    with open(config_path, "w") as fh:
        yaml.dump(config, fh)

    # 4. Post-build hint.
    logger.info(
        "  ✓ Injected event dispatch (%d worker(s), port %d)",
        dispatch.worker_count,
        dispatch.dispatcher_port,
    )
    logger.info("    Dashboard:  http://localhost:%d/dashboard", dispatch.dispatcher_port)
    logger.info(
        "    Token:      `osprey deploy up` writes EVENT_DISPATCHER_TOKEN to .env; "
        "load it with: export $(grep -E '^EVENT_DISPATCHER_TOKEN=' .env | xargs)"
    )
    logger.info(
        "    Try it:     curl -X POST http://localhost:%d/webhook/hello-dispatch "
        '-H "Authorization: Bearer $EVENT_DISPATCHER_TOKEN" '
        "-H 'Content-Type: application/json' -d '{}'",
        dispatch.dispatcher_port,
    )
    logger.info(
        "    Images:     `osprey deploy up` builds the dispatch image and the "
        "worker's project image locally (first run is slow). Use `--dev` to bake "
        "in your local osprey checkout; set OSPREY_DISPATCH_IMAGE/OSPREY_WORKER_IMAGE "
        "to use a published image."
    )


def _inject_bluesky(bluesky: BlueskyConfig, project_path: Path) -> None:
    """Wire the Bluesky scan-bridge feature into a built project.

    1. Copy the bundled ``templates/services/bluesky/`` compose template into
       ``<project>/services/bluesky/``.
    2. Write ``services.bluesky`` config + register it in ``deployed_services``
       (so ``find_service_config`` resolves it, mirroring ``_inject_dispatch``).
    3. Print a post-build hint (launch-token env var + image prerequisite).

    Simpler than ``_inject_dispatch``: no triggers file to resolve and no
    multi-instance worker loop — a project deploys exactly one bluesky-bridge
    process. The ``scan`` MCP server itself is a separate, always-available
    framework server (see ``osprey.mcp_server.bluesky``); this step only wires
    the *deploy-time* container that server talks to over HTTP.

    Args:
        bluesky: Validated bluesky configuration from the build profile.
        project_path: Root of the built project.
    """
    from ruamel.yaml import YAML

    # 1. Copy the bundled compose template (located the same way as service templates).
    pkg_services = _locate_pkg_services()

    src_dir = pkg_services / "bluesky"
    if not src_dir.is_dir():
        logger.warning("No package template for bluesky service at %s", src_dir)
        return

    dest_services_root = project_path / "services"
    dest_services_root.mkdir(exist_ok=True)
    dest_dir = dest_services_root / "bluesky"
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    shutil.copytree(src_dir, dest_dir)

    # 2. Write config.yml entries + register in deployed_services.
    config_path = project_path / "config.yml"
    if not config_path.exists():
        logger.warning("config.yml not found — skipping bluesky config registration")
        return

    yaml = YAML()
    yaml.preserve_quotes = True
    with open(config_path) as fh:
        config = yaml.load(fh)

    # No ``image`` key: the service builds the local bluesky-bridge image on
    # first ``osprey deploy up``. Override with OSPREY_BLUESKY_BRIDGE_IMAGE, or
    # set ``services.bluesky.image`` here, to use a prebuilt/published image.
    config.setdefault("services", {})
    config["services"]["bluesky"] = {
        "path": "./services/bluesky",
        "port": bluesky.port,
        "tiled_enabled": bluesky.tiled_enabled,
        "tiled_port": bluesky.tiled_port,
        "demo_runner": bluesky.demo_runner,
    }
    if bluesky.plan_dir:
        # Only written when configured — its absence is what keeps a
        # bridge-only deploy (no facility plan directory) rendering exactly
        # as before: the compose template's {% if %} guard reads this same
        # key, so an unset plan_dir means no mount and no BLUESKY_PLAN_DIRS
        # env var at all (Task 1.4).
        config["services"]["bluesky"]["plan_dir"] = bluesky.plan_dir
    deployed = config.get("deployed_services", []) or []
    if "bluesky" not in [str(s) for s in deployed]:
        deployed.append("bluesky")
    config["deployed_services"] = deployed

    with open(config_path, "w") as fh:
        yaml.dump(config, fh)

    # 3. Post-build hint.
    logger.info("  ✓ Injected Bluesky scan bridge (port %d)", bluesky.port)
    logger.info(
        "    Token:      `osprey deploy up` writes BLUESKY_LAUNCH_TOKEN to .env; "
        "the `scan` MCP server's launch_run tool reads it automatically."
    )
    logger.info(
        "    Images:     `osprey deploy up` builds the bluesky-bridge image locally "
        "(first run is slow). Use `--dev` to bake in your local osprey checkout; "
        "set OSPREY_BLUESKY_BRIDGE_IMAGE to use a published image."
    )
    if bluesky.tiled_enabled:
        logger.info("    Tiled:      enabled on port %d", bluesky.tiled_port)
    if bluesky.plan_dir:
        logger.info(
            "    Plan dir:   %s mounted read-only into the bridge; its plans "
            "load as the 'facility' trust tier (BLUESKY_PLAN_DIRS)",
            bluesky.plan_dir,
        )
    if bluesky.demo_runner:
        logger.warning(
            "    Demo mode:  BLUESKY_DEMO_RUNNER is set — the bridge runs a real "
            "bluesky RunEngine against MOCK devices only. Never enable this for a "
            "facility wiring real EPICS hardware."
        )


def _inject_va(va: VAConfig, project_path: Path) -> None:
    """Wire the Virtual Accelerator soft-IOC into a built project.

    1. Copy the bundled ``templates/services/virtual_accelerator/`` compose
       template into ``<project>/services/virtual_accelerator/``.
    2. Write ``services.virtual_accelerator`` config + register it in
       ``deployed_services`` (so ``find_service_config`` resolves it,
       mirroring ``_inject_bluesky``).
    3. Print a post-build hint (data/simulation prerequisite + image note).

    Thin mirror of :func:`_inject_bluesky`: one soft-IOC container, one
    config block — no source-tree staging, no registry logic.

    Args:
        va: Validated Virtual Accelerator configuration from the build profile.
        project_path: Root of the built project.
    """
    from ruamel.yaml import YAML

    # 1. Copy the bundled compose template (located the same way as service templates).
    pkg_services = _locate_pkg_services()

    src_dir = pkg_services / "virtual_accelerator"
    if not src_dir.is_dir():
        logger.warning("No package template for virtual_accelerator service at %s", src_dir)
        return

    dest_services_root = project_path / "services"
    dest_services_root.mkdir(exist_ok=True)
    dest_dir = dest_services_root / "virtual_accelerator"
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    shutil.copytree(src_dir, dest_dir)

    # 2. Write config.yml entries + register in deployed_services.
    config_path = project_path / "config.yml"
    if not config_path.exists():
        logger.warning("config.yml not found — skipping virtual_accelerator config registration")
        return

    yaml = YAML()
    yaml.preserve_quotes = True
    with open(config_path) as fh:
        config = yaml.load(fh)

    # No ``image`` key: the service builds the local VA image on first
    # ``osprey deploy up``. Override with OSPREY_VA_IMAGE, or set
    # ``services.virtual_accelerator.image`` here, to use a prebuilt/published image.
    config.setdefault("services", {})
    config["services"]["virtual_accelerator"] = {
        "path": "./services/virtual_accelerator",
        "port": va.port,
    }
    deployed = config.get("deployed_services", []) or []
    if "virtual_accelerator" not in [str(s) for s in deployed]:
        deployed.append("virtual_accelerator")
    config["deployed_services"] = deployed

    with open(config_path, "w") as fh:
        yaml.dump(config, fh)

    # 3. Post-build hint.
    logger.info("  ✓ Injected Virtual Accelerator soft-IOC (CA port %d)", va.port)
    logger.info(
        "    Data:       requires <project>/data/simulation/machine.json "
        "(the simulation preset provisions this; without it the IOC SystemExits)."
    )
    logger.info(
        "    Images:     `osprey deploy up` builds the virtual-accelerator image "
        "locally for your native architecture (first run is slow — the native deps "
        "PyAT/softioc are compiled from source, so no prebuilt aarch64 wheels are "
        "needed). Use `--dev` to bake in your local osprey checkout; "
        "set OSPREY_VA_IMAGE to use a published image."
    )


def _inject_bluesky_panels(bluesky_panels: BlueskyPanelsConfig, project_path: Path) -> None:
    """Wire the bluesky-panels sidecar + its three web panels into a built project.

    1. Copy the bundled ``templates/services/bluesky_panels/`` compose template
       into ``<project>/services/bluesky_panels/``.
    2. Write ``services.bluesky_panels`` config + register it in
       ``deployed_services`` (so ``find_service_config`` resolves it,
       mirroring ``_inject_bluesky``).
    3. Register the three ``web.panels.<id>`` entries (``plan``,
       ``results``, ``health``) pointing at the sidecar's root URL,
       mirroring ``_inject_dispatch``'s ``events`` panel registration: each
       panel points the proxy at the sidecar ROOT and uses ``path`` to select
       the panel's static mount, so the panel HTML loads there while its
       prefix-relative API fetches reach the sidecar root.
    4. Print a post-build hint (image prerequisite).

    Thin mirror of :func:`_inject_va`/:func:`_inject_bluesky` for the compose
    + config wiring, plus :func:`_inject_dispatch`'s ``web.panels`` setdefault
    idiom for the panel registration.

    Args:
        bluesky_panels: Validated bluesky-panels configuration from the build profile.
        project_path: Root of the built project.
    """
    from ruamel.yaml import YAML

    # 1. Copy the bundled compose template (located the same way as service templates).
    pkg_services = _locate_pkg_services()

    src_dir = pkg_services / "bluesky_panels"
    if not src_dir.is_dir():
        logger.warning("No package template for bluesky_panels service at %s", src_dir)
        return

    dest_services_root = project_path / "services"
    dest_services_root.mkdir(exist_ok=True)
    dest_dir = dest_services_root / "bluesky_panels"
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    shutil.copytree(src_dir, dest_dir)

    # 2. Write config.yml entries + register in deployed_services.
    config_path = project_path / "config.yml"
    if not config_path.exists():
        logger.warning("config.yml not found — skipping bluesky_panels config registration")
        return

    yaml = YAML()
    yaml.preserve_quotes = True
    with open(config_path) as fh:
        config = yaml.load(fh)

    # No ``image`` key: the service builds the local bluesky-panels image on
    # first ``osprey deploy up``. Override with OSPREY_BLUESKY_PANELS_IMAGE, or
    # set ``services.bluesky_panels.image`` here, to use a prebuilt/published image.
    config.setdefault("services", {})
    config["services"]["bluesky_panels"] = {
        "path": "./services/bluesky_panels",
        "port": bluesky_panels.port,
    }
    deployed = config.get("deployed_services", []) or []
    if "bluesky_panels" not in [str(s) for s in deployed]:
        deployed.append("bluesky_panels")
    config["deployed_services"] = deployed

    # 3. Register the three web.panels.<id> entries. Derive each url from
    # bluesky_panels.port so the port is a single source of truth (mirroring the
    # events-panel comment in _inject_dispatch), but write only when the
    # profile has not already set an explicit `web.panels.<id>.url` via a
    # config override (merged earlier in the build); explicit overrides take
    # precedence. Emit a bare sidecar-root `url` plus a per-panel `path`
    # (rather than baking the panel path into `url`) to match the
    # custom-panel proxy convention: the web terminal composes
    # `url.rstrip('/') + '/' + path`, so a path baked into `url` would
    # double-prefix sub-routes. `setdefault` on `path`/`label`
    # (`health_endpoint` for health) honors a facility override.
    default_url = f"${{BLUESKY_PANELS_URL:-http://localhost:{bluesky_panels.port}}}"
    panel_specs = (
        ("plan", "/plan/", "PLAN", None),
        ("results", "/results/", "RESULTS", None),
        ("health", "/health-panel/", "HEALTH", "/health"),
    )
    for panel_id, panel_path, label, health_endpoint in panel_specs:
        panel_cfg = config.setdefault("web", {}).setdefault("panels", {}).setdefault(panel_id, {})
        if not panel_cfg.get("url"):
            panel_cfg["url"] = default_url
        panel_cfg.setdefault("path", panel_path)
        panel_cfg.setdefault("label", label)
        if health_endpoint:
            panel_cfg.setdefault("health_endpoint", health_endpoint)

    with open(config_path, "w") as fh:
        yaml.dump(config, fh)

    # 4. Post-build hint.
    logger.info("  ✓ Injected bluesky-panels sidecar (port %d)", bluesky_panels.port)
    logger.info(
        "    Panels:     PLAN, RESULTS, HEALTH — reached through the "
        "web-terminal proxy at /panel/{plan,results,health}."
    )
    logger.info(
        "    Images:     `osprey deploy up` builds the bluesky-panels image locally "
        "(first run is slow). Use `--dev` to bake in your local osprey checkout; "
        "set OSPREY_BLUESKY_PANELS_IMAGE to use a published image."
    )


def _copy_overlay_files(
    profile_dir: Path, project_path: Path, overlay_dict: dict[str, str]
) -> None:
    """Copy overlay files/directories from profile dir into the project.

    Args:
        profile_dir: Directory containing the profile and overlay sources.
        project_path: Root of the built project.
        overlay_dict: Mapping of source (relative to profile_dir) → destination
            (relative to project_path).
    """
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

    with Progress(
        TextColumn("  Copying overlays"),
        BarColumn(),
        MofNCompleteColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("overlays", total=len(overlay_dict))
        for src_rel, dst_rel in overlay_dict.items():
            src = (profile_dir / src_rel).resolve()
            dst = (project_path / dst_rel).resolve()

            # Path traversal guard
            if not dst.is_relative_to(project_path.resolve()):
                raise ValueError(f"Overlay destination escapes project root: {dst_rel}")

            dst.parent.mkdir(parents=True, exist_ok=True)

            if src.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

            logger.debug("Overlay: %s → %s", src_rel, dst_rel)
            progress.advance(task)


def _register_overlay_artifacts(project_path: Path, overlay_dict: dict[str, str]) -> int:
    """Register overlay files landing in .claude/ as user_owned in config.yml.

    The Scaffold Gallery flags .claude/ files that aren't in the BuildArtifactCatalog
    or config.yml's scaffold.user_owned as "untracked."  Profile overlay files
    (agents, skills, rules) aren't framework artifacts, so they must be
    registered as user_owned to avoid the untracked warning.
    """
    from osprey.services.build_artifacts.ownership import update_config_add_user_owned

    config_path = project_path / "config.yml"
    if not config_path.exists():
        return 0

    # Subdirectories the Scaffold Gallery scans for untracked files
    # (mirrors ScaffoldGalleryService._scan_dirs)
    scan_prefixes = tuple(
        f".claude/{d}/" for d in ("agents", "commands", "output-styles", "rules", "skills")
    )

    registered = 0
    for _src_rel, dst_rel in overlay_dict.items():
        dst_path = project_path / dst_rel

        if dst_path.is_dir():
            # Directory overlay — find all .md files within
            md_files = [
                str(f.relative_to(project_path)) for f in dst_path.rglob("*.md") if f.is_file()
            ]
        elif dst_path.is_file() and dst_rel.endswith(".md"):
            md_files = [dst_rel]
        else:
            continue

        for rel_path in md_files:
            if not any(rel_path.startswith(p) for p in scan_prefixes):
                continue
            # Derive canonical name: .claude/rules/foo.md → rules/foo
            canonical = rel_path[len(".claude/") : -len(".md")]
            if update_config_add_user_owned(project_path, canonical):
                registered += 1

    return registered


def _persist_mcp_servers(project_path: Path, mcp_servers: dict[str, Any]) -> None:
    """Persist profile MCP server definitions into config.yml's claude_code.servers.

    Servers are written in the format that ``_custom_server_from_spec()`` parses,
    so ``regenerate_claude_code()`` can reconstruct them into the rendered
    ``.mcp.json`` and ``settings.json``.  Placeholders like ``{project_root}``
    are preserved as-is — resolution happens during regen.
    """
    from osprey.utils.config_writer import _load, _save

    from .build_profile import McpServerDef

    config_path = project_path / "config.yml"
    data = _load(config_path)

    # Ensure claude_code.servers section exists
    if "claude_code" not in data:
        from ruamel.yaml import CommentedMap

        data["claude_code"] = CommentedMap()
    cc = data["claude_code"]
    if "servers" not in cc:
        from ruamel.yaml import CommentedMap

        cc["servers"] = CommentedMap()
    servers_section = cc["servers"]

    for name, server in mcp_servers.items():
        if not isinstance(server, McpServerDef):
            continue

        spec: dict[str, Any] = {}
        if server.url:
            spec["transport"] = "http"
            spec["url"] = server.url
        else:
            spec["transport"] = "stdio"
            if server.command:
                spec["command"] = server.command
            if server.args:
                spec["args"] = list(server.args)
            if server.env:
                spec["env"] = dict(server.env)
        if server.port is not None and server.url:
            # Emit a derived network block so non-Claude consumers
            # (compose-port checkers, integration-tests probes) can read
            # host/docker URLs without re-deriving them.
            # NOTE: docker_url uses the MCP server's YAML key (`name`) as the
            # container hostname. This assumes the operator names the
            # docker-compose service identically to the mcp_servers entry
            # (e.g. mcp_servers.matlab → service: matlab). If they diverge,
            # docker_url will point at a non-existent host.
            spec["network"] = {
                "port": int(server.port),
                "host_url": f"http://localhost:{server.port}/mcp",
                "docker_url": f"http://{name}:{server.port}/mcp",
            }
        if server.permissions:
            spec["permissions"] = dict(server.permissions)

        servers_section[name] = spec

    _save(config_path, data)


def _persist_categories(project_path: Path, categories: dict[str, dict[str, str]]) -> None:
    """Persist custom artifact categories into config.yml's ``categories`` section."""
    from osprey.utils.config_writer import _load, _save

    config_path = project_path / "config.yml"
    data = _load(config_path)

    if "categories" not in data:
        from ruamel.yaml import CommentedMap

        data["categories"] = CommentedMap()
    cat_section = data["categories"]

    for key, spec in categories.items():
        from ruamel.yaml import CommentedMap

        entry = CommentedMap()
        entry["label"] = spec["label"]
        entry["color"] = spec["color"]
        cat_section[key] = entry

    _save(config_path, data)


def _git_init_and_commit(project_path: Path) -> None:
    """Initialize a git repo and create an initial commit."""
    import os
    import subprocess

    # Check if project is inside an existing git repo
    inside_existing_repo = False
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=project_path,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            parent_root = Path(result.stdout.strip()).resolve()
            if parent_root != project_path.resolve():
                inside_existing_repo = True
    except FileNotFoundError:
        pass

    try:
        subprocess.run(["git", "init"], cwd=project_path, check=True, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=project_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial project from osprey build"],
            cwd=project_path,
            check=True,
            capture_output=True,
            env={
                **os.environ,
                "GIT_AUTHOR_NAME": "osprey",
                "GIT_AUTHOR_EMAIL": "osprey@build",
                "GIT_COMMITTER_NAME": "osprey",
                "GIT_COMMITTER_EMAIL": "osprey@build",
            },
        )
        logger.info("  ✓ Initialized git repository")
        if inside_existing_repo:
            logger.warning(
                "  Note: created a nested git repo inside %s.\n"
                "     This is required for Claude Code project isolation (it uses\n"
                "     the git root to discover .claude/ settings). The parent repo\n"
                "     will treat this directory as opaque.",
                parent_root,
            )
    except FileNotFoundError:
        logger.warning(
            "  git not found — project created but not initialized as a git repo.\n"
            "     Claude Code requires git. Run 'git init && git add . && git commit'"
            " manually."
        )
    except subprocess.CalledProcessError:
        logger.warning(
            "  git init succeeded but initial commit failed.\n"
            "     Run 'git add . && git commit' manually."
        )


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
