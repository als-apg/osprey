"""CLI subcommands for managing build artifact ownership.

Provides ``osprey scaffold list|claim|diff|unclaim`` commands
for inspecting and customizing the Claude Code build artifacts
that OSPREY generates during ``osprey build`` / ``osprey claude regen``.
"""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any

import click
import yaml

from osprey.cli.styles import console
from osprey.cli.templates.manager import TemplateManager
from osprey.deployment.facility_config import normalize_facility_config
from osprey.services.build_artifacts.catalog import BuildArtifactCatalog
from osprey.services.build_artifacts.ownership import (
    get_user_owned,
    update_config_add_user_owned,
    update_config_remove_user_owned,
    update_manifest_add_user_owned,
    update_manifest_remove_user_owned,
)
from osprey.utils.config import resolve_env_vars


def _load_config(project_dir: Path) -> dict[str, Any]:
    """Load and return config.yml from project_dir."""
    config_file = project_dir / "config.yml"
    if not config_file.exists():
        raise click.ClickException(
            f"No config.yml found in {project_dir}. Are you in an OSPREY project directory?"
        )
    with open(config_file, encoding="utf-8") as f:
        return resolve_env_vars(yaml.safe_load(f) or {})


def _load_facility_config(config_path: str) -> dict[str, Any]:
    """Load and parse a facility-config.yml at an arbitrary path.

    Used by the ``web-terminals`` subcommands, which take an explicit
    ``--config`` path rather than reading a project directory's config.yml (see
    :func:`_load_config`), so no env-var resolution or existence check is applied
    here — ``click.Path(exists=True)`` on the option already guarantees the file
    exists.
    """
    with open(config_path, encoding="utf-8") as f:
        return normalize_facility_config(yaml.safe_load(f) or {})


@click.group(name="scaffold", invoke_without_command=True)
@click.pass_context
def scaffold(ctx):
    """Manage build artifact ownership.

    Framework-managed build artifacts can be claimed per-facility
    for in-place editing. Use the subcommands to inspect, claim, diff,
    and unclaim artifacts.

    Examples:

    \b
      osprey scaffold list                       # Show all artifacts
      osprey scaffold claim agents/channel-finder # Claim for editing
      osprey scaffold diff agents/channel-finder  # Compare yours vs framework
      osprey scaffold unclaim agents/channel-finder # Restore framework management
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@scaffold.command(name="list")
@click.option(
    "--project",
    "-p",
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    default=None,
    help="Project directory (default: current directory)",
)
def list_artifacts(project):
    """List all build artifacts and their ownership status."""
    project_dir = Path(project) if project else Path.cwd()

    try:
        config = _load_config(project_dir)
    except click.ClickException:
        config = {}

    user_owned = get_user_owned(config)
    registry = BuildArtifactCatalog.default()

    framework_managed = []
    owned = []

    for art in registry.all_artifacts():
        if art.canonical_name in user_owned:
            owned.append(art)
        else:
            framework_managed.append(art)

    console.print("\n[bold]Build Artifacts[/bold]\n")

    if framework_managed:
        console.print("  [dim]Framework-managed:[/dim]")
        for art in framework_managed:
            console.print(
                f"    [success]\u2713[/success] {art.canonical_name:<35s} {art.description}"
            )

    if owned:
        console.print("\n  [dim]User-owned:[/dim]")
        for art in owned:
            console.print(f"    [bold]\u2605[/bold] {art.canonical_name:<35s} {art.output_path}")

    console.print()


@scaffold.command(name="claim")
@click.argument("name")
@click.option(
    "--project",
    "-p",
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    default=None,
    help="Project directory (default: current directory)",
)
def claim(name, project):
    """Claim ownership of a framework artifact for in-place editing.

    If the file doesn't exist yet, renders the framework template in-place
    at the canonical output path. If it already exists, just marks it as
    user-owned. Regen will skip user-owned files.

    Examples:

    \b
      osprey scaffold claim agents/channel-finder
      osprey scaffold claim rules/safety
    """
    project_dir = Path(project) if project else Path.cwd()
    registry = BuildArtifactCatalog.default()
    artifact = registry.get(name)

    if artifact is None:
        known = ", ".join(registry.all_names())
        raise click.ClickException(f"Unknown artifact '{name}'. Known artifacts:\n  {known}")

    config = _load_config(project_dir)
    user_owned = get_user_owned(config)

    if name in user_owned:
        raise click.ClickException(
            f"'{name}' is already user-owned. Edit it directly at {artifact.output_path}."
        )

    # Build template context
    manager = TemplateManager()
    from osprey.cli.templates.claude_code import build_claude_code_context

    ctx = build_claude_code_context(manager.template_root, manager.jinja_env, project_dir, config)

    # If file doesn't exist, render the framework template in-place
    output_file = project_dir / artifact.output_path
    if not output_file.exists():
        claude_code_dir = manager.template_root / "claude_code"
        template_file = claude_code_dir / artifact.template_path

        if not template_file.exists():
            raise click.ClickException(f"Template file not found: {artifact.template_path}")

        if template_file.suffix == ".j2":
            template_rel = f"claude_code/{artifact.template_path}"
            template = manager.jinja_env.get_template(template_rel)
            content = template.render(**ctx)
        else:
            content = template_file.read_text(encoding="utf-8")

        if not content.strip():
            console.print(
                "[warning]\u26a0[/warning] Template renders to empty content "
                "(likely a Jinja2 condition is not met for your config).",
                style="yellow",
            )
            if not click.confirm("Create an empty file anyway?"):
                return

        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(content, encoding="utf-8")
        if output_file.suffix == ".py":
            output_file.chmod(output_file.stat().st_mode | 0o755)

        console.print(f"  [success]\u2713[/success] Rendered {name} \u2192 {artifact.output_path}")
    else:
        console.print(f"  [success]\u2713[/success] File already exists at {artifact.output_path}")

    # Update config.yml
    added = update_config_add_user_owned(project_dir, name)
    if added:
        console.print(
            f"  [success]\u2713[/success] Updated config.yml \u2014 scaffold.user_owned += {name}"
        )

    # Update manifest
    update_manifest_add_user_owned(project_dir, manager, ctx, name)

    console.print(f"\n  Edit [path]{artifact.output_path}[/path] \u2014 regen will skip it.\n")


@scaffold.command(name="diff")
@click.argument("name")
@click.option(
    "--project",
    "-p",
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    default=None,
    help="Project directory (default: current directory)",
)
def diff(name, project):
    """Show diff between a framework template and your file.

    Renders the current framework template and compares it against
    your file at the canonical output path using a unified diff.

    Examples:

    \b
      osprey scaffold diff agents/channel-finder
      osprey scaffold diff rules/facility
    """
    project_dir = Path(project) if project else Path.cwd()
    config = _load_config(project_dir)
    user_owned = get_user_owned(config)

    if name not in user_owned:
        raise click.ClickException(
            f"'{name}' is not user-owned in config.yml. Run `osprey scaffold claim {name}` first."
        )

    registry = BuildArtifactCatalog.default()
    artifact = registry.get(name)
    if artifact is None:
        raise click.ClickException(f"Unknown artifact '{name}'.")

    # Read user's file from canonical output path
    user_file = project_dir / artifact.output_path
    if not user_file.exists():
        raise click.ClickException(f"File not found: {artifact.output_path}")
    user_lines = user_file.read_text(encoding="utf-8").splitlines(keepends=True)

    # Render framework template
    manager = TemplateManager()
    from osprey.cli.templates.claude_code import build_claude_code_context

    ctx = build_claude_code_context(manager.template_root, manager.jinja_env, project_dir, config)
    claude_code_dir = manager.template_root / "claude_code"
    template_file = claude_code_dir / artifact.template_path

    if template_file.suffix == ".j2":
        template_rel = f"claude_code/{artifact.template_path}"
        template = manager.jinja_env.get_template(template_rel)
        framework_content = template.render(**ctx)
    else:
        framework_content = template_file.read_text(encoding="utf-8")

    framework_lines = framework_content.splitlines(keepends=True)

    # Generate unified diff
    diff_lines = difflib.unified_diff(
        framework_lines,
        user_lines,
        fromfile=f"framework:{artifact.template_path}",
        tofile=f"yours:{artifact.output_path}",
    )

    output = "".join(diff_lines)
    if output:
        click.echo(output)
    else:
        console.print(
            "[success]\u2713[/success] Your file matches the current framework template "
            "\u2014 no differences."
        )


@scaffold.command(name="unclaim")
@click.argument("name")
@click.option(
    "--project",
    "-p",
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    default=None,
    help="Project directory (default: current directory)",
)
def unclaim(name, project):
    """Release ownership and restore framework management.

    Removes the artifact from the user_owned list in config.yml and
    .osprey-manifest.json. The next ``osprey claude regen`` will
    overwrite the file with the framework template.

    Examples:

    \b
      osprey scaffold unclaim agents/channel-finder
      osprey scaffold unclaim rules/safety
    """
    project_dir = Path(project) if project else Path.cwd()
    config = _load_config(project_dir)
    user_owned = get_user_owned(config)

    if name not in user_owned:
        raise click.ClickException(f"'{name}' is not user-owned in config.yml.")

    # Remove from config.yml
    update_config_remove_user_owned(project_dir, name)

    # Remove from manifest
    update_manifest_remove_user_owned(project_dir, name)

    console.print(f"  [success]\u2713[/success] Released ownership of {name}")
    console.print("\n  Next `osprey claude regen` will overwrite with the framework template.\n")


@scaffold.group(name="web-terminals", invoke_without_command=True)
@click.pass_context
def web_terminals(ctx):
    """Validate and render the ``modules.web_terminals`` deployment stanza.

    Examples:

    \b
      osprey scaffold web-terminals lint --config facility-config.yml
      osprey scaffold web-terminals render --config facility-config.yml -o deploy/
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


def _print_web_terminals_lint_errors(errors: list[Any]) -> None:
    """Print the ``modules.web_terminals lint`` "Errors:" block.

    Shared by ``lint`` (as part of its full errors+warnings report) and
    ``render``'s pre-render lint gate (errors only, no warnings section). Takes
    a list of ``osprey.deployment.web_terminals.lint.Finding`` (not type-hinted
    as such here to keep this module's lazy-import convention for that package).
    """
    console.print("  [dim]Errors:[/dim]")
    for finding in errors:
        console.print(f"    [error]\u2717[/error] [{finding.code}] {finding.message}")


@web_terminals.command(name="lint")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    required=True,
    help="Path to a facility-config.yml to validate.",
)
def web_terminals_lint(config_path):
    """Validate the ``modules.web_terminals`` stanza of a facility config.

    Checks port-family allocation, reserved service names, and duplicate
    users. Exits non-zero if any error-severity finding is reported;
    warnings (e.g. an enabled module with zero configured users) do not
    fail the check, so this is safe to wire into a CI gate.

    Examples:

    \b
      osprey scaffold web-terminals lint --config facility-config.yml
    """
    from osprey.deployment.web_terminals.lint import lint_web_terminals

    config = _load_facility_config(config_path)
    findings = lint_web_terminals(config)

    if not findings:
        console.print("[success]\u2713[/success] modules.web_terminals: no issues found\n")
        return

    errors = [f for f in findings if f.severity == "error"]
    warnings = [f for f in findings if f.severity == "warn"]

    console.print("\n[bold]modules.web_terminals lint[/bold]\n")

    if errors:
        _print_web_terminals_lint_errors(errors)

    if warnings:
        console.print("\n  [dim]Warnings:[/dim]")
        for finding in warnings:
            console.print(f"    [warning]\u26a0[/warning] [{finding.code}] {finding.message}")

    console.print()

    if errors:
        raise SystemExit(1)


@web_terminals.command(name="render")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    required=True,
    help="Path to a facility-config.yml to render.",
)
@click.option(
    "--output",
    "-o",
    "output_dir",
    type=click.Path(file_okay=False, resolve_path=True),
    required=True,
    help="Directory to write the rendered deployment artifacts into.",
)
@click.option(
    "--no-lint",
    is_flag=True,
    default=False,
    help="Skip the lint pre-check (by default, lint errors abort the render).",
)
def web_terminals_render(config_path, output_dir, no_lint):
    """Render the ``modules.web_terminals`` deployment artifacts of a facility config.

    Writes the docker-compose overlay, nginx routing fragment, and static landing
    page into --output, creating the directory (and its nginx/ subdirectory) as
    needed. By default the stanza is linted first and error-severity findings
    abort the render; pass --no-lint to render anyway.

    Examples:

    \b
      osprey scaffold web-terminals render --config facility-config.yml -o deploy/
    """
    from osprey.deployment.web_terminals.render import render_web_terminals

    config = _load_facility_config(config_path)

    if not no_lint:
        from osprey.deployment.web_terminals.lint import lint_web_terminals

        errors = [f for f in lint_web_terminals(config) if f.severity == "error"]
        if errors:
            console.print("\n[bold]modules.web_terminals lint[/bold]\n")
            _print_web_terminals_lint_errors(errors)
            console.print()
            raise click.ClickException(
                "modules.web_terminals has lint errors; fix them or pass --no-lint to render anyway."
            )

    try:
        artifacts = render_web_terminals(config)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    output_path = Path(output_dir)
    written = []
    for relative_path, content in artifacts.items():
        target = output_path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        written.append(target)

    console.print("\n[bold]modules.web_terminals render[/bold]\n")
    for target in written:
        console.print(f"  [success]\u2713[/success] {target}")
    console.print()
