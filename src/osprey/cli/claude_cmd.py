"""Claude Code integration commands.

This module provides the 'osprey claude' command group for managing
Claude Code skill installations.

Commands:
    - claude install: Install a task as a Claude Code skill
    - claude list: List installed skills

Skill Generation:
    Skills can be auto-generated from task frontmatter if the task includes
    a 'skill_description' field. This enables any task to be installed as
    a Claude Code skill without requiring a custom SKILL.md wrapper.

    Frontmatter fields used for skill generation:
    - workflow: Used for skill name (osprey-{workflow})
    - skill_description: Description for Claude to decide when to use the skill
    - allowed_tools: Optional list of allowed tools (defaults to standard set)
"""

import os
import shutil
from pathlib import Path
from typing import Any

import click
import yaml

from osprey.cli.styles import Styles, console

# Default tools for auto-generated skills
DEFAULT_ALLOWED_TOOLS = ["Read", "Glob", "Grep", "Bash", "Edit"]


def parse_task_frontmatter(task: str) -> dict[str, Any]:
    """Parse YAML frontmatter from a task's instructions.md file.

    Args:
        task: Name of the task

    Returns:
        Dictionary of frontmatter fields, empty dict if no frontmatter
    """
    instructions_file = get_tasks_root() / task / "instructions.md"
    if not instructions_file.exists():
        return {}

    content = instructions_file.read_text()

    # Check for frontmatter (starts with ---)
    if not content.startswith("---"):
        return {}

    # Find the closing ---
    end_idx = content.find("---", 3)
    if end_idx == -1:
        return {}

    frontmatter_text = content[3:end_idx].strip()

    try:
        return yaml.safe_load(frontmatter_text) or {}
    except yaml.YAMLError:
        return {}


def get_task_title(task: str) -> str:
    """Extract the title (first H1) from a task's instructions.md file.

    Args:
        task: Name of the task

    Returns:
        The title text, or a formatted version of the task name
    """
    instructions_file = get_tasks_root() / task / "instructions.md"
    if not instructions_file.exists():
        return task.replace("-", " ").title()

    content = instructions_file.read_text()

    # Find first H1 header after frontmatter
    lines = content.split("\n")
    in_frontmatter = False

    for line in lines:
        if line.strip() == "---":
            in_frontmatter = not in_frontmatter
            continue
        if in_frontmatter:
            continue
        if line.startswith("# "):
            return line[2:].strip()

    return task.replace("-", " ").title()


def can_generate_skill(task: str) -> bool:
    """Check if a task can have a skill auto-generated from frontmatter.

    A task is skill-ready if it has a 'skill_description' field in its frontmatter.

    Args:
        task: Name of the task

    Returns:
        True if skill can be auto-generated
    """
    frontmatter = parse_task_frontmatter(task)
    return bool(frontmatter.get("skill_description"))


def generate_skill_content(task: str) -> str:
    """Generate SKILL.md content from task frontmatter.

    Args:
        task: Name of the task

    Returns:
        Generated SKILL.md content

    Raises:
        ValueError: If task doesn't have skill_description in frontmatter
    """
    frontmatter = parse_task_frontmatter(task)

    if not frontmatter.get("skill_description"):
        raise ValueError(f"Task '{task}' does not have 'skill_description' in frontmatter")

    workflow = frontmatter.get("workflow", task)
    skill_name = f"osprey-{workflow}"
    description = frontmatter["skill_description"]
    allowed_tools = frontmatter.get("allowed_tools", DEFAULT_ALLOWED_TOOLS)
    title = get_task_title(task)

    # Format allowed_tools as YAML list or single line
    if isinstance(allowed_tools, list):
        tools_str = ", ".join(allowed_tools)
    else:
        tools_str = str(allowed_tools)

    # Build the SKILL.md content
    skill_content = f"""---
name: {skill_name}
description: >
  {description}
allowed-tools: {tools_str}
---

# {title}

This skill was auto-generated from task frontmatter.

## Instructions

Follow the detailed workflow in [instructions.md](./instructions.md).
"""

    return skill_content


def get_tasks_root() -> Path:
    """Get the root path of the tasks directory."""
    return Path(__file__).parent.parent / "assist" / "tasks"


def get_integrations_root() -> Path:
    """Get the root path of the integrations directory."""
    return Path(__file__).parent.parent / "assist" / "integrations"


def get_available_tasks() -> list[str]:
    """Get list of available tasks from the tasks directory."""
    tasks_dir = get_tasks_root()
    if not tasks_dir.exists():
        return []
    return sorted(
        [d.name for d in tasks_dir.iterdir() if d.is_dir() and (d / "instructions.md").exists()]
    )


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

    Install skills, regenerate artifacts, and launch Claude Code.

    Examples:

    \b
      # Regenerate Claude Code artifacts from config.yml
      osprey claude regen

      # Launch Claude Code with fresh artifacts
      osprey claude chat

      # Install a skill
      osprey claude install pre-commit

      # List installed skills
      osprey claude list

      # Browse available tasks first
      osprey tasks list
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@claude.command(name="install")
@click.argument("task")
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Overwrite existing installation",
)
def install_skill(task: str, force: bool):
    """Install a task as a Claude Code skill.

    Skills are installed to .claude/skills/<task>/ in the current directory.

    Skills can come from two sources:
    1. Custom skill wrappers in integrations/claude_code/<task>/
    2. Auto-generated from task frontmatter (if skill_description is present)

    Examples:

    \b
      # Install pre-commit skill
      osprey claude install pre-commit

      # Force overwrite existing
      osprey claude install pre-commit --force
    """
    available_tasks = get_available_tasks()
    if task not in available_tasks:
        console.print(f"Task '{task}' not found.", style=Styles.ERROR)
        console.print(f"\nAvailable tasks: {', '.join(available_tasks)}")
        console.print("\nRun [command]osprey tasks list[/command] to see all tasks.")
        return

    # Check if Claude Code integration exists for this task
    integration_dir = get_integrations_root() / "claude_code" / task
    has_custom_wrapper = integration_dir.exists() and any(integration_dir.glob("*.md"))
    can_auto_generate = can_generate_skill(task)

    if not has_custom_wrapper and not can_auto_generate:
        console.print(
            f"[warning]⚠[/warning]  No Claude Code skill available for '{task}'",
        )
        console.print("\nTo make this task installable as a skill, add 'skill_description'")
        console.print("to its frontmatter in instructions.md:")
        console.print("\n  [dim]---[/dim]")
        console.print("  [dim]workflow: " + task + "[/dim]")
        console.print("  [dim]skill_description: >-[/dim]")
        console.print("  [dim]  Description of when Claude should use this skill.[/dim]")
        console.print("  [dim]---[/dim]")
        console.print("\nThe task instructions can still be used directly:")
        instructions_path = get_tasks_root() / task / "instructions.md"
        console.print(f"  [path]@{instructions_path}[/path]")
        return

    # Destination directory
    dest_dir = get_claude_skills_dir() / task

    # Check if already installed
    if dest_dir.exists() and any(dest_dir.glob("*.md")) and not force:
        console.print(
            f"[warning]⚠[/warning]  Skill already installed at: {dest_dir.relative_to(Path.cwd())}"
        )
        console.print("    Use [command]--force[/command] to overwrite")
        return

    # Create destination directory
    dest_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold]Installing Claude Code skill: {task}[/bold]\n")

    files_copied = 0

    if has_custom_wrapper:
        # Use custom wrapper: copy skill files (SKILL.md and any other .md files)
        console.print("[dim]Using custom skill wrapper[/dim]\n")
        for source_file in integration_dir.glob("*.md"):
            dest_file = dest_dir / source_file.name
            shutil.copy2(source_file, dest_file)
            console.print(f"  [success]✓[/success] {dest_file.relative_to(Path.cwd())}")
            files_copied += 1
    else:
        # Auto-generate SKILL.md from frontmatter
        console.print("[dim]Auto-generating skill from frontmatter[/dim]\n")
        skill_content = generate_skill_content(task)
        skill_file = dest_dir / "SKILL.md"
        skill_file.write_text(skill_content)
        console.print(
            f"  [success]✓[/success] {skill_file.relative_to(Path.cwd())} [dim](generated)[/dim]"
        )
        files_copied += 1

    # Always copy instructions.md
    instructions_source = get_tasks_root() / task / "instructions.md"
    if instructions_source.exists():
        instructions_dest = dest_dir / "instructions.md"
        shutil.copy2(instructions_source, instructions_dest)
        console.print(f"  [success]✓[/success] {instructions_dest.relative_to(Path.cwd())}")
        files_copied += 1

    # Copy any additional task files (e.g., migrate has versions/, schema.yml)
    task_dir = get_tasks_root() / task
    for item in task_dir.iterdir():
        if item.name == "instructions.md":
            continue  # Already copied
        if item.is_file():
            dest_file = dest_dir / item.name
            shutil.copy2(item, dest_file)
            console.print(f"  [success]✓[/success] {dest_file.relative_to(Path.cwd())}")
            files_copied += 1
        elif item.is_dir():
            dest_subdir = dest_dir / item.name
            if dest_subdir.exists():
                shutil.rmtree(dest_subdir)
            shutil.copytree(item, dest_subdir)
            console.print(
                f"  [success]✓[/success] {dest_subdir.relative_to(Path.cwd())}/ [dim](directory)[/dim]"
            )
            files_copied += 1

    console.print(f"\n[success]✓ Installed {files_copied} files[/success]\n")

    # Show usage hints based on frontmatter
    frontmatter = parse_task_frontmatter(task)
    console.print("[bold]Usage:[/bold]")

    # Try to extract usage hints from skill_description or use defaults
    skill_desc = frontmatter.get("skill_description", "")
    if "commit" in task.lower() or "commit" in skill_desc.lower():
        console.print('  Ask Claude: "Run pre-commit checks"')
        console.print('  Or: "Validate my changes before committing"')
    elif "migrate" in task.lower() or "upgrade" in skill_desc.lower():
        console.print('  Ask Claude: "Upgrade my project to the latest OSPREY version"')
        console.print('  Or: "Help me migrate my OSPREY project"')
    elif "capability" in task.lower():
        console.print('  Ask Claude: "Help me create a new capability"')
        console.print('  Or: "Guide me through building a capability for my Osprey app"')
    elif "test" in task.lower():
        console.print('  Ask Claude: "Help me write tests for this feature"')
        console.print('  Or: "Run the testing workflow"')
    elif "review" in task.lower():
        console.print('  Ask Claude: "Review my code changes"')
        console.print('  Or: "Run an AI code review"')
    else:
        console.print(f'  Ask Claude to help with the "{task}" task')

    console.print()


@claude.command(name="list")
def list_skills():
    """List installed Claude Code skills.

    Shows skills installed in the current project's .claude/skills/ directory,
    as well as tasks available for installation (either with custom wrappers
    or auto-generated from frontmatter).
    """
    installed = get_installed_skills()
    available = get_available_tasks()

    console.print("\n[bold]Claude Code Skills[/bold]\n")

    if installed:
        console.print("[dim]Installed in this project:[/dim]")
        for skill in installed:
            console.print(f"  [success]✓[/success] {skill}")
        console.print()

    # Show available but not installed
    not_installed = [t for t in available if t not in installed]
    if not_installed:
        # Categorize tasks by their skill availability
        with_custom_wrapper = []
        with_auto_generate = []
        without_skill = []

        for task in not_installed:
            integration_dir = get_integrations_root() / "claude_code" / task
            has_custom = integration_dir.exists() and any(integration_dir.glob("*.md"))
            can_auto = can_generate_skill(task)

            if has_custom:
                with_custom_wrapper.append(task)
            elif can_auto:
                with_auto_generate.append(task)
            else:
                without_skill.append(task)

        # Show installable skills (custom + auto-generate)
        installable = with_custom_wrapper + with_auto_generate
        if installable:
            console.print("[dim]Available to install:[/dim]")
            for task in with_custom_wrapper:
                console.print(f"  [info]○[/info] {task}")
            for task in with_auto_generate:
                console.print(f"  [info]○[/info] {task} [dim](auto-generated)[/dim]")
            console.print()
            console.print("Install with: [command]osprey claude install <skill>[/command]\n")

        if without_skill:
            console.print(
                "[dim]Tasks without skill support (use @-mention or add skill_description):[/dim]"
            )
            for task in without_skill:
                console.print(f"  [dim]- {task}[/dim]")
            console.print()
    elif not installed:
        console.print("No skills installed yet.\n")
        console.print("Browse available tasks: [command]osprey tasks list[/command]")
        console.print("Install a skill: [command]osprey claude install <task>[/command]\n")


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
def chat_claude(project, resume, print_mode):
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

    # Replace current process with claude CLI
    console.print(f"[dim]Launching Claude Code in {project_dir}...[/dim]\n")
    os.execvp("claude", args)
