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
def regen(project, dry_run):
    """Regenerate Claude Code artifacts from config.yml.

    Re-reads config.yml and re-renders all Claude Code integration files
    (.mcp.json, .claude/settings.json, CLAUDE.md, agents). Existing files
    are backed up to osprey-workspace/backup/ before overwriting.

    Prompt overrides (in ``overrides/``) are used instead of framework templates.

    Examples:

    \b
      # Regenerate in current directory
      osprey claude regen

      # Preview changes without writing
      osprey claude regen --dry-run

      # Regenerate for a specific project
      osprey claude regen --project /path/to/project
    """
    from osprey.cli.templates.manager import TemplateManager

    project_dir = Path(project) if project else Path.cwd()

    try:
        manager = TemplateManager()
        result = manager.regenerate_claude_code(project_dir, dry_run=dry_run)
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
        ClaudeCodeModelResolver,
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
    api_providers = config.get("api", {}).get("providers", {})

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
            spec = ClaudeCodeModelResolver.resolve(claude_code_config, api_providers)
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
def chat_claude(project, resume, print_mode, effort):
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
        MANAGED_ENV_VARS,
        ClaudeCodeModelResolver,
        inject_provider_env,
    )

    config_path = project_dir / "config.yml"
    if config_path.exists():
        config = yaml.safe_load(config_path.read_text()) or {}
        cc_config = config.get("claude_code", {})
        if not effort:
            effort = cc_config.get("effort")
        api_providers = config.get("api", {}).get("providers", {})
        spec = ClaudeCodeModelResolver.resolve(cc_config, api_providers)
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
                console.print(
                    f"[dim]Set ${spec.auth_env_var} from ${spec.auth_secret_env}[/dim]"
                )

    # Build claude CLI args
    args = ["claude", "--project-dir", str(project_dir)]
    if resume:
        args.extend(["--resume", resume])
    if print_mode:
        args.append("--print")
    if effort:
        args.extend(["--effort", effort])

    # Replace current process with claude CLI
    console.print(f"[dim]Launching Claude Code in {project_dir}...[/dim]\n")
    os.execvp("claude", args)
