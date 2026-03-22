"""Build command — assemble a facility-specific assistant from a build profile.

Reads a YAML build profile that specifies a base template, config overrides,
file overlays, and MCP server definitions. Produces a standalone, self-contained
project directory (wipe-and-rebuild safe).

Usage:
    osprey build my-assistant profile.yml
    osprey build my-assistant profile.yml --force
"""

from __future__ import annotations

import json
import logging
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import click

from osprey.errors import BuildProfileError

from .styles import Messages, Styles, console
from .templates.manager import TemplateManager

logger = logging.getLogger(__name__)


@click.command()
@click.argument("project_name")
@click.argument("profile", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default=".",
    help="Output directory for project (default: current directory)",
)
@click.option("--force", "-f", is_flag=True, help="Force overwrite if project directory exists")
def build(
    project_name: str,
    profile: str,
    output_dir: str,
    force: bool,
) -> None:
    """Build a facility-specific assistant from a profile.

    Assembles a standalone project by rendering a base template, applying
    config overrides, copying overlay files, and injecting MCP servers.

    PROJECT_NAME: Name of the project directory to create

    PROFILE: Path to a YAML build profile

    Examples:

    \b
      # Build from profile
      $ osprey build als-test ~/profiles/als-dev.yml

      # Force overwrite
      $ osprey build als-test ~/profiles/als-dev.yml --force
    """
    from .build_profile import load_profile
    from .init_cmd import _clear_claude_code_project_state

    console.print(f"🔨 Building project: [header]{project_name}[/header]")

    try:
        # 1. Load and validate profile
        profile_path = Path(profile).resolve()
        profile_dir = profile_path.parent
        build_profile = load_profile(profile_path)

        console.print(f"  📋 Profile: [accent]{build_profile.name}[/accent]")
        console.print(f"  📦 Template: [accent]{build_profile.base_template}[/accent]")

        # 1b. Check OSPREY version requirement
        if build_profile.requires_osprey_version:
            from packaging.specifiers import SpecifierSet
            from packaging.version import Version

            from osprey import __version__

            spec = SpecifierSet(build_profile.requires_osprey_version)
            current = Version(__version__)
            if current not in spec:
                console.print(
                    f"  ❌ OSPREY {__version__} does not satisfy "
                    f"requires_osprey_version: {build_profile.requires_osprey_version}",
                    style=Styles.ERROR,
                )
                console.print(
                    f"     Upgrade OSPREY or run: osprey --version",
                    style=Styles.DIM,
                )
                raise click.Abort()
            console.print(
                f"  ✓ OSPREY {__version__} satisfies "
                f"{build_profile.requires_osprey_version}",
                style=Styles.SUCCESS,
            )

        # 2. Resolve output path
        output_path = Path(output_dir).resolve()
        project_path = output_path / project_name

        # 3. Handle --force / directory existence
        if project_path.exists():
            if force:
                msg = Messages.warning(f"Removing existing directory: {project_path}")
                console.print(f"  ⚠️  {msg}")
                shutil.rmtree(project_path)
                console.print(f"  {Messages.success('Removed existing directory')}")
            else:
                console.print(
                    f"❌ Directory '{project_path}' already exists.\n"
                    f"   Use --force to overwrite, or choose a different name.",
                    style=Styles.ERROR,
                )
                raise click.Abort()

        # 4. Run pre_build lifecycle commands
        if build_profile.lifecycle.pre_build:
            _run_lifecycle_phase(
                "pre_build", build_profile.lifecycle.pre_build, profile_dir, project_path
            )

        # 5. Clear Claude Code project state
        _clear_claude_code_project_state(project_path)

        # 6. Build context from profile fields
        context: dict[str, Any] = {}
        if build_profile.provider:
            context["default_provider"] = build_profile.provider
        if build_profile.model:
            context["default_model"] = build_profile.model
        if build_profile.channel_finder_mode:
            context["channel_finder_mode"] = build_profile.channel_finder_mode

        # 7. Create project from template
        manager = TemplateManager()
        project_path = manager.create_project(
            project_name=project_name,
            output_dir=output_path,
            template_name=build_profile.base_template,
            registry_style="extend",
            context=context,
        )
        console.print("  ✓ Base template rendered", style=Styles.SUCCESS)

        # 8. Apply config overrides
        if build_profile.config:
            _apply_config_overrides(project_path, build_profile.config)
            console.print(
                f"  ✓ Applied {len(build_profile.config)} config override(s)",
                style=Styles.SUCCESS,
            )

        # 9. Copy service templates for `osprey deploy up`
        svc_count = _copy_service_templates(project_path)
        if svc_count:
            console.print(
                f"  ✓ Copied {svc_count} service template(s) for deploy",
                style=Styles.SUCCESS,
            )

        # 10. Inject profile-defined services (facility containers)
        if build_profile.services:
            psvc_count = _inject_profile_services(
                profile_dir, project_path, build_profile.services
            )
            console.print(
                f"  ✓ Injected {psvc_count} profile service(s) for deploy",
                style=Styles.SUCCESS,
            )

        # 11. Copy overlay files
        if build_profile.overlay:
            _copy_overlay_files(profile_dir, project_path, build_profile.overlay)
            console.print(
                f"  ✓ Copied {len(build_profile.overlay)} overlay(s)", style=Styles.SUCCESS
            )

        # 12. Inject MCP servers
        if build_profile.mcp_servers:
            _inject_mcp_servers(project_path, build_profile.mcp_servers)
            console.print(
                f"  ✓ Injected {len(build_profile.mcp_servers)} MCP server(s)",
                style=Styles.SUCCESS,
            )

        # 13. Copy profile .env file (if provided)
        if build_profile.env.file:
            _copy_env_file(profile_dir, project_path, build_profile.env.file)

        # 14. Generate .env.template
        if build_profile.env.required or build_profile.env.defaults:
            _generate_env_template(project_path, build_profile.env)

        # 15. Append to requirements.txt and install into build environment
        if build_profile.dependencies:
            _append_requirements(project_path, build_profile.dependencies)
            _install_dependencies(build_profile.dependencies)

        # 16. Generate manifest
        manifest_context = {
            "default_provider": build_profile.provider or "anthropic",
            "default_model": build_profile.model or "haiku",
        }
        if build_profile.channel_finder_mode:
            manifest_context["channel_finder_mode"] = build_profile.channel_finder_mode
        manager.generate_manifest(
            project_dir=project_path,
            project_name=project_name,
            template_name=build_profile.base_template,
            registry_style="extend",
            context=manifest_context,
        )

        # 17. Git init + commit
        _git_init_and_commit(project_path)

        # 18. Run post_build lifecycle commands
        if build_profile.lifecycle.post_build:
            _run_lifecycle_phase(
                "post_build", build_profile.lifecycle.post_build, project_path, project_path
            )

        # 19. Run validate lifecycle commands
        if build_profile.lifecycle.validate:
            _run_lifecycle_phase(
                "validate",
                build_profile.lifecycle.validate,
                project_path,
                project_path,
                abort_on_failure=False,
            )

        console.print(f"\n✅ Project built successfully at: [bold]{project_path}[/bold]")

    except click.Abort:
        raise
    except BuildProfileError as e:
        console.print(f"❌ Build error: {e}", style=Styles.ERROR)
        raise click.Abort() from e
    except ValueError as e:
        console.print(f"❌ Error: {e}", style=Styles.ERROR)
        raise click.Abort() from e
    except Exception as e:
        console.print(f"❌ Unexpected error: {e}", style=Styles.ERROR)
        import traceback

        console.print(traceback.format_exc(), style=Styles.DIM)
        raise click.Abort() from e


_SHELL_METACHARACTERS = ("|", "&&", "||", "$(", "`")


def _run_lifecycle_phase(
    phase_name: str,
    steps: list[Any],
    default_cwd: Path,
    project_path: Path,
    *,
    abort_on_failure: bool = True,
) -> None:
    """Run lifecycle commands for a build phase.

    Args:
        phase_name: Phase name for display (pre_build, post_build, validate).
        steps: List of LifecycleStep objects.
        default_cwd: Default working directory for steps without explicit cwd.
        project_path: Project root path for {project_root} substitution.
        abort_on_failure: If True, raise BuildProfileError on failure.
            If False, warn and continue (used for validate phase).
    """
    console.print(f"  ⚙️  Running {phase_name} commands...", style=Styles.DIM)
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
            if use_shell:
                result = subprocess.run(
                    cmd_str,
                    shell=True,
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    timeout=step.timeout,
                )
            else:
                result = subprocess.run(
                    shlex.split(cmd_str),
                    cwd=cwd,
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
                    console.print(f"  ❌ {msg}", style=Styles.ERROR)
                    raise BuildProfileError(msg)
                else:
                    console.print(f"  ⚠️  {msg}", style=Styles.WARNING)
            else:
                # Build success message with timing
                success_msg = f"  ✓ {step.name} ({elapsed:.1f}s)"
                # Extract a one-line summary from the last non-empty line of
                # combined output (scripts can print a summary as their last line)
                output = (result.stdout + result.stderr).strip()
                if output:
                    summary = output.rstrip().rsplit("\n", 1)[-1].strip()
                    if summary:
                        success_msg += f" — {summary}"
                console.print(success_msg, style=Styles.SUCCESS)

        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - t0
            msg = f"Lifecycle {phase_name} step '{step.name}' timed out ({elapsed:.0f}s)"
            if abort_on_failure:
                console.print(f"  ❌ {msg}", style=Styles.ERROR)
                raise BuildProfileError(msg)
            else:
                console.print(f"  ⚠️  {msg}", style=Styles.WARNING)
        except OSError as exc:
            msg = f"Lifecycle {phase_name} step '{step.name}' failed to start: {exc}"
            if abort_on_failure:
                console.print(f"  ❌ {msg}", style=Styles.ERROR)
                raise BuildProfileError(msg) from exc
            else:
                console.print(f"  ⚠️  {msg}", style=Styles.WARNING)


def _copy_env_file(profile_dir: Path, project_path: Path, env_file: str) -> None:
    """Copy a profile-provided .env file to the built project."""
    src = (profile_dir / env_file).resolve()
    dst = project_path / ".env"
    shutil.copy2(src, dst)
    console.print(f"  ✓ Copied {env_file} → .env", style=Styles.SUCCESS)


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
    console.print("  ✓ Generated .env.template", style=Styles.SUCCESS)
    if not (project_path / ".env").exists():
        console.print(
            "  💡 Copy .env.template to .env and fill in required values",
            style=Styles.DIM,
        )


def _append_requirements(project_path: Path, dependencies: list[str]) -> None:
    """Append profile dependencies to requirements.txt."""
    req_path = project_path / "requirements.txt"
    lines = ["\n", "# Profile dependencies\n"]
    for dep in dependencies:
        lines.append(f"{dep}\n")

    with open(req_path, "a", encoding="utf-8") as f:
        f.writelines(lines)

    console.print(
        f"  ✓ Added {len(dependencies)} profile dependency/ies to requirements.txt",
        style=Styles.SUCCESS,
    )


def _install_dependencies(dependencies: list[str]) -> None:
    """Install profile dependencies into the build environment.

    Tries ``uv pip install`` first (uv-managed venvs don't include pip),
    then falls back to ``python -m pip install``.  This is best-effort:
    if installation fails the build continues and the lifecycle step itself
    will report the real error.

    Args:
        dependencies: PEP 508 dependency specifiers (e.g. ``["typesense>=0.21"]``).
    """
    import os
    import shutil
    import sys

    console.print("  📦 Installing profile dependencies...", style=Styles.DIM)

    # Prefer uv (uv-managed venvs don't ship pip).
    # Inside `uv run`, $UV holds the path to the uv binary.
    uv_path = os.environ.get("UV") or shutil.which("uv")
    if uv_path:
        cmd = [uv_path, "pip", "install", "--quiet", *dependencies]
    else:
        cmd = [
            sys.executable, "-m", "pip", "install",
            "--quiet", "--disable-pip-version-check", *dependencies,
        ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            console.print(
                f"  ✓ Installed {len(dependencies)} build dependency/ies",
                style=Styles.SUCCESS,
            )
        else:
            output = (result.stdout + result.stderr).strip()
            console.print(
                f"  ⚠️  Failed to install profile dependencies (exit {result.returncode})",
                style=Styles.WARNING,
            )
            if output:
                last_lines = "\n".join(output.split("\n")[-5:])
                console.print(f"     {last_lines}", style=Styles.DIM)
            console.print(
                "     Lifecycle steps may fail if they import these packages.",
                style=Styles.DIM,
            )
    except subprocess.TimeoutExpired:
        console.print(
            "  ⚠️  Dependency installation timed out (120s)",
            style=Styles.WARNING,
        )
    except FileNotFoundError:
        console.print(
            "  ⚠️  Package installer not available — skipping dependency installation",
            style=Styles.WARNING,
        )


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

    deployed = config.get("deployed_services", [])
    if not deployed:
        return 0

    # Locate the package's service templates directory
    try:
        import osprey.templates

        pkg_services = Path(osprey.templates.__file__).parent / "services"
    except (ImportError, AttributeError):
        pkg_services = Path(__file__).parent.parent / "templates" / "services"

    if not pkg_services.is_dir():
        logger.warning("Service templates directory not found — skipping")
        return 0

    services_config = config.get("services", {})
    dest_services_root = project_path / "services"
    dest_services_root.mkdir(exist_ok=True)

    # Copy the root services compose template (shared network definition)
    root_template = pkg_services / "docker-compose.yml.j2"
    if root_template.exists():
        shutil.copy2(root_template, dest_services_root / "docker-compose.yml.j2")

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

        logger.info("Overlay: %s → %s", src_rel, dst_rel)


def _inject_mcp_servers(project_path: Path, mcp_servers: dict[str, Any]) -> None:
    """Inject MCP server definitions into .mcp.json and .claude/settings.json.

    For each server:
      - Adds command/args/env to .mcp.json mcpServers
      - Adds tool permissions to .claude/settings.json permissions.allow/ask
      - Resolves {project_root} placeholders in args and env values
    """
    from .build_profile import McpServerDef

    # --- .mcp.json ---
    mcp_json_path = project_path / ".mcp.json"
    if mcp_json_path.exists():
        mcp_data = json.loads(mcp_json_path.read_text(encoding="utf-8"))
    else:
        mcp_data = {}

    mcp_servers_section = mcp_data.setdefault("mcpServers", {})

    for name, server in mcp_servers.items():
        if not isinstance(server, McpServerDef):
            continue
        if name in mcp_servers_section:
            console.print(
                f"  ⚠️  MCP server '{name}' already exists in .mcp.json — skipping",
                style=Styles.WARNING,
            )
            continue

        entry: dict[str, Any] = {
            "command": server.command,
            "args": [_resolve_placeholders(a, project_path) for a in server.args],
        }
        if server.env:
            entry["env"] = {
                k: _resolve_placeholders(v, project_path) for k, v in server.env.items()
            }
        mcp_servers_section[name] = entry

    mcp_json_path.write_text(json.dumps(mcp_data, indent=2) + "\n", encoding="utf-8")

    # --- .claude/settings.json ---
    settings_path = project_path / ".claude" / "settings.json"
    if settings_path.exists():
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
    else:
        settings = {}

    permissions = settings.setdefault("permissions", {})
    allow_list: list[str] = permissions.setdefault("allow", [])
    ask_list: list[str] = permissions.setdefault("ask", [])

    for name, server in mcp_servers.items():
        if not isinstance(server, McpServerDef):
            continue
        for tool in server.permissions.get("allow", []):
            entry = f"mcp__{name}__{tool}"
            if entry not in allow_list:
                allow_list.append(entry)
        for tool in server.permissions.get("ask", []):
            entry = f"mcp__{name}__{tool}"
            if entry not in ask_list:
                ask_list.append(entry)

    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")


def _resolve_placeholders(value: str, project_path: Path) -> str:
    """Replace {project_root} with the actual project path."""
    return value.replace("{project_root}", str(project_path))


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
        console.print("  ✓ Initialized git repository", style=Styles.SUCCESS)
        if inside_existing_repo:
            console.print(
                f"  ⚠️  Note: created a nested git repo inside {parent_root}.\n"
                "     This is required for Claude Code project isolation (it uses\n"
                "     the git root to discover .claude/ settings). The parent repo\n"
                "     will treat this directory as opaque.",
                style=Styles.WARNING,
            )
    except FileNotFoundError:
        console.print(
            "  ⚠️  git not found — project created but not initialized as a git repo.\n"
            "     Claude Code requires git. Run 'git init && git add . && git commit'"
            " manually.",
            style=Styles.WARNING,
        )
    except subprocess.CalledProcessError:
        console.print(
            "  ⚠️  git init succeeded but initial commit failed.\n"
            "     Run 'git add . && git commit' manually.",
            style=Styles.WARNING,
        )
