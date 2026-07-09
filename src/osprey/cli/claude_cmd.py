"""Claude Code integration commands.

This module provides the 'osprey claude' command group for managing
Claude Code configuration and artifact regeneration.

Commands:
    - claude regen: Regenerate Claude Code artifacts from config.yml
    - claude status: Show Claude Code configuration status
    - claude chat: Launch Claude Code with regenerated artifacts
"""

import os
from pathlib import Path

import click
import yaml

from osprey.cli.styles import console
from osprey.utils.claude_launcher import build_claude_launch_argv


def get_claude_skills_dir() -> Path:
    """Get the Claude Code skills directory."""
    return Path.cwd() / ".claude" / "skills"


def get_installed_skills() -> list[str]:
    """Get list of installed Claude Code skills."""
    skills_dir = get_claude_skills_dir()
    if not skills_dir.exists():
        return []
    return sorted([d.name for d in skills_dir.iterdir() if d.is_dir()])


@click.group(name="claude", invoke_without_command=True)
@click.pass_context
def claude(ctx):
    """Manage Claude Code integration.

    Regenerate artifacts, check status, and launch Claude Code.

    Examples:

    \b
      # Regenerate Claude Code artifacts from config.yml
      osprey claude regen

      # Launch Claude Code with fresh artifacts
      osprey claude chat

      # Check configuration status
      osprey claude status
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@claude.command(name="regen")
@click.option(
    "--project",
    "-p",
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    default=None,
    help="Project directory (default: current directory)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would change without writing files",
)
@click.option(
    "--runtime-root",
    type=click.Path(),
    default=None,
    help="Rewrite project_root in config.yml and re-render artifacts for a "
    "relocated project (e.g. inside a container: --runtime-root /app/my-project)",
)
def regen(project, dry_run, runtime_root):
    """Regenerate Claude Code artifacts from config.yml.

    Re-reads config.yml and re-renders all Claude Code integration files
    (.mcp.json, .claude/settings.json, CLAUDE.md, agents). Existing files
    are backed up to _agent_data/backup/ before overwriting.

    Prompt overrides (in ``overrides/``) are used instead of framework templates.

    With ``--runtime-root``, recorded host paths are rewritten for the new
    location: ``project_root`` is set to the given path, and a recorded
    ``execution.python_env_path`` that does not exist on this filesystem is
    replaced with the current interpreter. Use this after copying a built
    project into a container image.

    Examples:

    \b
      # Regenerate in current directory
      osprey claude regen

      # Preview changes without writing
      osprey claude regen --dry-run

      # Regenerate for a specific project
      osprey claude regen --project /path/to/project

      # Relocate a project copied into a container
      osprey claude regen --project /app/myproj --runtime-root /app/myproj
    """
    from osprey.cli.templates.manager import TemplateManager

    project_dir = Path(project) if project else Path.cwd()

    try:
        if runtime_root and not dry_run:
            _rewrite_runtime_paths(project_dir, runtime_root)
        manager = TemplateManager()
        result = manager.regenerate_claude_code(
            project_dir, dry_run=dry_run, project_root_override=runtime_root
        )
    except FileNotFoundError as e:
        console.print(f"[error]Error:[/error] {e}", style="red")
        raise SystemExit(1) from e

    if dry_run:
        console.print("\n[bold]Dry run — no files modified[/bold]\n")
        if result["changed"]:
            console.print("[dim]Would change:[/dim]")
            for f in result["changed"]:
                console.print(f"  [warning]~[/warning] {f}")
        if result["unchanged"]:
            console.print("[dim]Unchanged:[/dim]")
            for f in result["unchanged"]:
                console.print(f"  [dim]  {f}[/dim]")
        if not result["changed"]:
            console.print("[success]All artifacts are up to date.[/success]")
    else:
        console.print("\n[bold]Claude Code artifacts regenerated[/bold]\n")
        if result["changed"]:
            console.print("[dim]Changed:[/dim]")
            for f in result["changed"]:
                console.print(f"  [success]✓[/success] {f}")
        if result["unchanged"]:
            console.print("[dim]Unchanged:[/dim]")
            for f in result["unchanged"]:
                console.print(f"  [dim]  {f}[/dim]")
        if not result["changed"]:
            console.print("[success]All artifacts were already up to date.[/success]")
        else:
            console.print(f"\n[dim]Backup saved to: {result['backup_dir']}[/dim]")

        if result["changed"]:
            console.print(
                "\n[dim]Tip: commit the regenerated files so Claude Code"
                " picks up the changes:[/dim]"
            )
            console.print(
                "  [dim]git add .claude/ CLAUDE.md .mcp.json && git commit -m"
                ' "regen: update Claude Code artifacts"[/dim]'
            )

    # Display active/disabled summary
    _print_regen_summary(result)
    console.print()


def _rewrite_runtime_paths(project_dir: Path, runtime_root: str) -> None:
    """Rewrite recorded host paths in config.yml for a relocated project.

    Sets ``project_root`` to *runtime_root*. When the recorded
    ``execution.python_env_path`` no longer exists on this filesystem
    (typical after copying a host-built project into a container), it is
    replaced with the current interpreter; a valid path is left untouched.
    Comments in config.yml are preserved.
    """
    import sys

    from osprey.utils.config_writer import config_update_fields

    config_file = project_dir / "config.yml"
    if not config_file.exists():
        raise FileNotFoundError(f"No config.yml found in {project_dir}")

    updates: dict = {"project_root": str(runtime_root)}

    config = yaml.safe_load(config_file.read_text()) or {}
    env_path = (config.get("execution") or {}).get("python_env_path")
    if env_path and not Path(env_path).exists():
        console.print(f"[warning]⚠ Recorded python_env_path not found here: {env_path}[/warning]")
        console.print(f"  [dim]Replacing with current interpreter: {sys.executable}[/dim]")
        updates["execution.python_env_path"] = sys.executable

    config_update_fields(config_file, updates)
    console.print(f"[dim]Rewrote project_root → {runtime_root} in config.yml[/dim]")


def _print_regen_summary(result: dict):
    """Print active/disabled server and agent summary."""
    active_servers = result.get("active_servers", [])
    disabled_servers = result.get("disabled_servers", [])
    extra_servers = result.get("extra_servers", [])
    active_agents = result.get("active_agents", [])
    disabled_agents = result.get("disabled_agents", [])

    if not active_servers and not active_agents:
        return

    console.print("\n[bold]Active MCP Servers:[/bold]")
    for s in active_servers:
        label = s
        if s in extra_servers:
            label += " [dim](custom)[/dim]"
        console.print(f"  [success]*[/success] {label}")
    if disabled_servers:
        console.print("[dim]Disabled servers:[/dim]")
        for s in disabled_servers:
            console.print(f"  [dim]- {s}[/dim]")

    console.print("\n[bold]Active Agents:[/bold]")
    for a in active_agents:
        console.print(f"  [success]*[/success] {a}")
    if disabled_agents:
        console.print("[dim]Disabled agents:[/dim]")
        for a in disabled_agents:
            console.print(f"  [dim]- {a}[/dim]")


def _launch_companion_servers(project_dir: Path) -> list[tuple[str, str]]:
    """Launch companion web servers enabled in config.

    Sets ``OSPREY_CONFIG``, resets the config cache, then iterates all
    registered servers.  Each server's ``auto_launch_checker`` decides
    whether it actually starts.

    Returns:
        List of ``(display_name, url)`` for servers that ended up running.
    """
    import logging

    from osprey.infrastructure.server_launcher import _launchers, ensure_web_server
    from osprey.registry.web import FRAMEWORK_WEB_SERVERS
    from osprey.utils.workspace import reset_config_cache

    config_file = project_dir / "config.yml"
    if config_file.exists():
        os.environ["OSPREY_CONFIG"] = str(config_file)
    reset_config_cache()

    # Silence ALL logging so daemon-thread output cannot interfere with the TUI.
    # In CLI mode, server logs are not useful — debug via `osprey web` instead.
    logging.getLogger().setLevel(logging.CRITICAL)

    started: list[tuple[str, str]] = []
    for key, defn in FRAMEWORK_WEB_SERVERS.items():
        try:
            ensure_web_server(key)
            launcher = _launchers[key]
            if launcher._launched:
                host, port = launcher._config_reader()
                started.append((defn.name, f"http://{host}:{port}"))
        except Exception:
            pass
    return started


@claude.command(name="status")
@click.option(
    "--project",
    "-p",
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    default=None,
    help="Project directory (default: current directory)",
)
def status(project):
    """Show Claude Code configuration status.

    Displays provider configuration, model tier mappings, per-agent model
    assignments, and artifact sync status (drift detection via dry-run).

    Examples:

    \b
      # Show status for current directory
      osprey claude status

      # Show status for a specific project
      osprey claude status --project /path/to/project
    """
    from osprey.cli.claude_code_resolver import (
        AGENT_DEFAULT_TIERS,
        load_provider_spec,
    )
    from osprey.cli.templates.manager import TemplateManager

    project_dir = Path(project) if project else Path.cwd()

    config_file = project_dir / "config.yml"
    if not config_file.exists():
        console.print("[error]Error:[/error] No config.yml found.", style="red")
        console.print(f"  Looked in: {project_dir}")
        raise SystemExit(1)

    config = yaml.safe_load(config_file.read_text()) or {}
    claude_code_config = config.get("claude_code", {})

    console.print("\n[bold]Claude Code Status[/bold]\n")

    # ── Provider ──────────────────────────────────────────────
    provider_name = claude_code_config.get("provider")
    if not provider_name:
        console.print("[dim]Provider:[/dim]  not configured")
        console.print(
            "  [dim]Set claude_code.provider in config.yml to enable "
            "automatic env/model resolution.[/dim]"
        )
    else:
        try:
            spec = load_provider_spec(project_dir)
        except ValueError as exc:
            console.print(f"[error]Provider error:[/error] {exc}")
            raise SystemExit(1) from exc

        console.print(f"[dim]Provider:[/dim]  {spec.provider}")

        # Env block
        console.print("\n[bold]Environment Variables[/bold]  (settings.json env block)")
        for key, value in spec.env_block.items():
            console.print(f"  {key} = [dim]{value}[/dim]")

        # Shell exports
        if spec.shell_exports:
            console.print("\n[bold]Required Shell Exports[/bold]  (add to ~/.zshrc)")
            for line in spec.shell_exports:
                console.print(f"  [dim]{line}[/dim]")

        # Model tiers
        console.print("\n[bold]Model Tiers[/bold]")
        model_overrides = claude_code_config.get("models", {}) or {}
        for tier in ("haiku", "sonnet", "opus"):
            model_id = spec.tier_to_model.get(tier, "?")
            suffix = " [dim](override)[/dim]" if tier in model_overrides else ""
            console.print(f"  {tier:8s} → {model_id}{suffix}")

        # Agent models
        console.print("\n[bold]Agent Models[/bold]")
        agent_overrides = claude_code_config.get("agent_models", {}) or {}
        for agent_name, default_tier in sorted(AGENT_DEFAULT_TIERS.items()):
            model_id = spec.agent_model(agent_name)
            if agent_name in agent_overrides:
                note = f" [dim](override: {agent_overrides[agent_name]})[/dim]"
            else:
                note = f" [dim]({default_tier})[/dim]"
            console.print(f"  {agent_name:28s} → {model_id}{note}")

        # ── Environment conflict check ──
        conflicts = spec.detect_env_conflicts(dict(os.environ))
        if conflicts:
            console.print("\n[warning]⚠ Shell environment conflicts:[/warning]")
            for var, (shell_val, settings_val) in sorted(conflicts.items()):
                console.print(f"  {var}:")
                console.print(f"    shell:    {shell_val}")
                console.print(f"    settings: {settings_val}")
            console.print("\n[dim]Use 'osprey claude chat' to auto-resolve.[/dim]")

        secret_available = bool(os.environ.get(spec.auth_secret_env))
        icon = "[success]✓[/success]" if secret_available else "[error]✗[/error]"
        console.print(
            f"\n  Auth: {icon} ${spec.auth_secret_env} "
            f"{'available' if secret_available else 'NOT FOUND'}"
        )

    # ── Artifact drift ────────────────────────────────────────
    console.print("\n[bold]Artifact Status[/bold]")
    try:
        manager = TemplateManager()
        result = manager.regenerate_claude_code(project_dir, dry_run=True)
    except FileNotFoundError:
        console.print("  [dim]Could not check artifact status[/dim]")
        result = None

    if result:
        if result["changed"]:
            console.print("  [warning]Out of sync — run `osprey claude regen`:[/warning]")
            for f in result["changed"]:
                console.print(f"    [warning]~[/warning] {f}")
        else:
            console.print("  [success]All artifacts up to date[/success]")
        if result["unchanged"]:
            console.print(f"  [dim]{len(result['unchanged'])} files in sync[/dim]")

        # Reuse the server/agent summary
        _print_regen_summary(result)

    console.print()


@claude.command(name="chat")
@click.option(
    "--project",
    "-p",
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    default=None,
    help="Project directory (default: current directory)",
)
@click.option("--resume", default=None, help="Resume a previous Claude Code session")
@click.option("--print", "print_mode", is_flag=True, help="Use print mode (non-interactive)")
@click.option(
    "--effort",
    type=click.Choice(["low", "medium", "high", "max"]),
    default=None,
    help="Claude Code effort level",
)
@click.option(
    "--no-pin",
    is_flag=True,
    help="Ignore claude_code.cli_version and launch the globally-installed `claude` binary",
)
def chat_claude(project, resume, print_mode, effort, no_pin):
    """Launch Claude Code with regenerated artifacts.

    Regenerates Claude Code integration files from config.yml,
    then launches the Claude Code CLI in the project directory.

    Examples:

    \b
      # Launch Claude Code
      osprey claude chat

      # Resume a previous session
      osprey claude chat --resume SESSION_ID

      # Non-interactive mode
      osprey claude chat --print
    """
    from osprey.cli.templates.manager import TemplateManager

    project_dir = Path(project) if project else Path.cwd()

    # Regenerate artifacts first
    try:
        manager = TemplateManager()
        result = manager.regenerate_claude_code(project_dir)
        if result["changed"]:
            console.print("[dim]Regenerated Claude Code artifacts[/dim]")
            for f in result["changed"]:
                console.print(f"  [success]✓[/success] {f}")
            console.print()
    except FileNotFoundError as e:
        console.print(f"[error]Error:[/error] {e}", style="red")
        raise SystemExit(1) from e

    # ── Provider isolation: inject env block + auth, scrub managed vars ──
    from osprey.cli.claude_code_resolver import (
        detect_managed_policy_conflicts,
        format_managed_policy_conflicts,
        inject_provider_env,
        load_provider_spec,
    )

    # Managed (enterprise) policy settings outrank the process environment AND
    # the --setting-sources project restriction below, so a policy `env` block
    # setting a provider variable would silently redirect the agent to a backend
    # the project did not configure. For a framework driving control systems,
    # refuse to launch rather than start against the wrong provider.
    policy_conflicts = detect_managed_policy_conflicts()
    if policy_conflicts:
        console.print(
            f"[error]✗ Refusing to launch.\n{format_managed_policy_conflicts(policy_conflicts)}[/error]"
        )
        raise SystemExit(1)

    config_path = project_dir / "config.yml"
    cc_config: dict = {}
    if config_path.exists():
        config = yaml.safe_load(config_path.read_text()) or {}
        cc_config = config.get("claude_code", {})
        if not effort:
            effort = cc_config.get("effort")
        # load_provider_spec re-reads config.yml and expands ${VAR} in provider
        # config (e.g. a custom provider's base_url: ${ARGO_PROD_URL}) before
        # resolving — so the proxy below gets a real upstream URL, not a literal.
        spec = load_provider_spec(project_dir)
        if spec:
            if spec.auth_secret_env and not os.environ.get(spec.auth_secret_env):
                console.print(
                    f"[warning]⚠ ${spec.auth_secret_env} not found in environment — "
                    f"provider '{spec.provider}' may not authenticate[/warning]"
                )
            injected = inject_provider_env(os.environ, spec, project_dir=project_dir)
            if injected:
                console.print(f"[dim]Injected: {', '.join(injected)}[/dim]")
            if spec.auth_secret_env and os.environ.get(spec.auth_env_var):
                console.print(f"[dim]Set ${spec.auth_env_var} from ${spec.auth_secret_env}[/dim]")

            # Start translation proxy for OpenAI-compatible providers
            if spec.needs_proxy and spec.upstream_base_url:
                from osprey.infrastructure.proxy.lifecycle import start_proxy

                proxy_port = start_proxy(
                    spec.upstream_base_url,
                    os.environ.get(spec.auth_env_var),
                )
                os.environ["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{proxy_port}"
                console.print(
                    f"[dim]Translation proxy started on :{proxy_port} → {spec.upstream_base_url}[/dim]"
                )

    # Build claude CLI args (claude uses cwd as project root — no --project-dir flag).
    # When claude_code.cli_version is set, build_claude_launch_argv() returns an
    # ``npx -y @anthropic-ai/claude-code@<v>`` prefix instead of bare ``claude``
    # so each project can pin the CLI version (issue #218). ``--no-pin`` opts out
    # of the pin but not the ``--setting-sources project`` provider isolation.
    args = build_claude_launch_argv(cc_config, no_pin=no_pin)
    if resume:
        args.extend(["--resume", resume])
    if print_mode:
        args.append("--print")
    if effort:
        args.extend(["--effort", effort])

    # Claude Code uses the working directory as the project root.
    os.chdir(project_dir)

    # Launch companion web servers (artifact gallery, analytics, etc.)
    started_servers = _launch_companion_servers(project_dir)
    if started_servers:
        console.print("[dim]Companion servers:[/dim]")
        for name, url in started_servers:
            console.print(f"  [success]*[/success] {name}  [dim]{url}[/dim]")
        console.print()

    # Flush all output before the Claude Code TUI takes over the terminal.
    import sys

    console.print(f"[dim]Launching Claude Code in {project_dir}...[/dim]\n")
    sys.stdout.flush()
    sys.stderr.flush()

    # ── Sanitize proxy environment variables ─────────────────────────────────
    # Bun (Claude Code's runtime) reads .env files from every ancestor directory
    # at startup. If any of those files contains a placeholder proxy value
    # (e.g. HTTP_PROXY=http-proxy), Bun fails to parse it and crashes before
    # Claude Code starts. We overwrite invalid proxy values with an empty string
    # so they shadow any stubs Bun might pick up from parent .env files.
    # Valid proxy URLs must start with http:// or https://.
    for _proxy_var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        _proxy_val = os.environ.get(_proxy_var, "")
        if _proxy_val and not (
            _proxy_val.startswith("http://") or _proxy_val.startswith("https://")
        ):
            os.environ[_proxy_var] = ""
            console.print(f"[dim]Cleared invalid proxy env: {_proxy_var}={_proxy_val!r}[/dim]")

    # Launch claude CLI.  Companion servers and the translation proxy run in
    # daemon threads, so the parent process must stay alive — always use
    # subprocess.run (never os.execvp, which replaces the process).
    import subprocess

    raise SystemExit(subprocess.run(args).returncode)
