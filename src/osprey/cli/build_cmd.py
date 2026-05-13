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

import os
import shlex
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import click

from osprey.errors import BuildProfileError
from osprey.utils.logger import get_logger

from .templates.manager import TemplateManager

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
    "Defaults to the profile's tier (which itself defaults to 1).",
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
                "<als-apg|cborg|anthropic|amsc|argo>` to your profile or "
                "pass `--set provider=<...>` on the build command."
            )

        logger.info("  Profile: %s", build_profile.name)
        logger.info("  Data bundle: %s", build_profile.data_bundle)
        logger.info("  Tier: %d", build_profile.tier)

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
            tier=build_profile.tier,
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


def _load_dotenv(path: Path) -> dict[str, str]:
    """Parse a .env file into a dict of environment variables.

    Handles KEY=VALUE lines, #comments, blank lines, and quoted values
    (single or double quotes stripped from value boundaries).
    """
    env: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Skip lines without =
        if "=" not in line:
            continue
        # Skip `export` prefix (common in .env files)
        if line.startswith("export "):
            line = line[7:]
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        # Strip matching surrounding quotes
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        env[key] = value
    return env


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


def _create_project_venv(project_path: Path, profile: Any) -> None:
    """Create the project venv and install osprey + profile deps.

    This is the single place where the project's Python environment is set up.
    One venv, one install command, one resolver pass. The resolver sees all
    dependencies together (osprey + profile deps) and either succeeds or fails.

    The ``osprey_install`` profile field controls where osprey comes from:
      - ``"local"`` (default): install from the source tree running this build
      - ``"pip"``: install ``osprey-framework`` from PyPI
      - anything else: treated as a PEP 508 spec (e.g. ``"osprey-framework==2026.5.0"``)
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
    if osprey_install == "local":
        # build_cmd.py is at src/osprey/cli/build_cmd.py → repo root is parents[3]
        osprey_root = Path(__file__).resolve().parents[3]
        if not (osprey_root / "pyproject.toml").exists():
            raise BuildProfileError(
                f"osprey_install: local — but no pyproject.toml at {osprey_root}"
            )
        osprey_spec = str(osprey_root)
    elif osprey_install == "pip":
        osprey_spec = "osprey-framework"
    else:
        osprey_spec = osprey_install

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

    spinner = Spinner("dots", text=f"  Installing osprey ({osprey_install}) + {dep_count} deps...")
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
    lines = ["\n", f"# osprey ({osprey_install})\n", f"{osprey_spec}\n"]
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


def _copy_service_templates(project_path: Path) -> int:
    """Copy service compose templates from the OSPREY package into the project.

    Reads ``deployed_services`` from the generated config.yml and copies each
    service's compose template directory from the package to the project's
    ``services/`` tree.  This makes the project self-contained so that
    ``osprey deploy up`` works directly from the project directory.

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
    try:
        import osprey.templates

        pkg_services = Path(osprey.templates.__file__).parent / "services"
    except (ImportError, AttributeError):
        pkg_services = Path(__file__).parent.parent / "templates" / "services"

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

    deployed = config.get("deployed_services", [])
    if not deployed:
        return 0

    services_config = config.get("services", {})

    count = 0
    for service_name in deployed:
        name = str(service_name)

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

    The Prompts Gallery flags .claude/ files that aren't in the PromptCatalog
    or config.yml's prompts.user_owned as "untracked."  Profile overlay files
    (agents, skills, rules) aren't framework artifacts, so they must be
    registered as user_owned to avoid the untracked warning.
    """
    from osprey.services.prompts.ownership import update_config_add_user_owned

    config_path = project_path / "config.yml"
    if not config_path.exists():
        return 0

    # Subdirectories the Prompts Gallery scans for untracked files
    # (mirrors PromptGalleryService._scan_dirs)
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
            spec["url"] = server.url
        else:
            if server.command:
                spec["command"] = server.command
            if server.args:
                spec["args"] = list(server.args)
            if server.env:
                spec["env"] = dict(server.env)
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
