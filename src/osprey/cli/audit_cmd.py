"""Agentic build profile and project safety auditor.

Spawns a Claude agent via the Claude Agent SDK to deeply analyze an OSPREY
build profile or built project directory, producing a structured safety report.
"""

from __future__ import annotations

import asyncio
import re
import tempfile
import uuid
from pathlib import Path

import click

from .styles import Messages, Styles, console

try:
    from rich.panel import Panel
    from rich.table import Table
except ImportError:
    Panel = None  # type: ignore[assignment, misc]
    Table = None  # type: ignore[assignment, misc]

# SDK imports are deferred to runtime to provide a helpful error message
_SDK_AVAILABLE = False
try:
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ResultMessage,
        TextBlock,
        query,
    )

    _SDK_AVAILABLE = True
except ImportError:
    pass


def _detect_target_type(target: str) -> str:
    """Detect whether the target is a profile YAML or a project directory."""
    p = Path(target)
    if p.is_file() and p.suffix in (".yml", ".yaml"):
        return "profile"
    if p.is_dir():
        return "project"
    raise click.BadParameter(f"Target must be a .yml/.yaml profile or a directory, got: {target}")


def _list_files(directory: Path, max_files: int = 500) -> str:
    """List files in a directory, relative to it."""
    files = sorted(p.relative_to(directory) for p in directory.rglob("*") if p.is_file())
    listing = [str(f) for f in files[:max_files]]
    if len(files) > max_files:
        listing.append(f"... and {len(files) - max_files} more files")
    return "\n".join(listing)


def _extract_json(text: str) -> str | None:
    """Extract JSON from agent text output, handling markdown fences."""
    # Try markdown-fenced JSON first
    match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try to find a raw JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return None


async def _run_audit(
    prompt: str,
    model: str,
    cwd: Path,
    budget: float,
    verbose: bool,
) -> tuple[str, float | None, int | None]:
    """Run the audit agent and collect its output.

    Returns:
        Tuple of (collected_text, total_cost, num_turns).
    """
    options = ClaudeAgentOptions(
        model=model,
        cwd=str(cwd),
        permission_mode="bypassPermissions",
        max_turns=30,
        max_budget_usd=budget,
    )

    collected_text: list[str] = []
    total_cost: float | None = None
    num_turns: int | None = None

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    collected_text.append(block.text)
                    if verbose:
                        console.print(f"[dim]{block.text[:200]}...[/dim]")
        elif isinstance(message, ResultMessage):
            total_cost = getattr(message, "total_cost_usd", None)
            num_turns = getattr(message, "num_turns", None)

    return "".join(collected_text), total_cost, num_turns


def _display_report(report, json_output: bool, verbose: bool, cost=None, turns=None):
    """Display the audit report using Rich or JSON."""
    if json_output:
        click.echo(report.model_dump_json(indent=2))
        return

    # Summary panel
    risk_style = {
        "low": Styles.SUCCESS,
        "medium": Styles.WARNING,
        "high": Styles.ERROR,
    }.get(report.overall_risk, Styles.INFO)

    console.print()
    console.print(
        Panel(
            f"[{risk_style}]{report.overall_risk.upper()} RISK[/{risk_style}]\n\n{report.summary}",
            title="Audit Summary",
            border_style=Styles.BORDER_DIM,
        )
    )

    if not report.findings:
        console.print(f"\n  {Messages.success('No findings — clean audit!')}\n")
        return

    # Findings table
    if Table is not None:
        table = Table(border_style=Styles.BORDER_DIM, show_lines=True)
        table.add_column("Severity", width=8)
        table.add_column("Category", width=12)
        table.add_column("Title")
        table.add_column("File", width=30)

        for f in report.findings:
            sev_fmt = {
                "error": Messages.error(f.severity),
                "warning": Messages.warning(f.severity),
                "info": Messages.info(f.severity),
            }.get(f.severity, f.severity)
            table.add_row(sev_fmt, f.category, f.title, f.file_path)

        console.print(table)

    # Detailed findings
    for f in report.findings:
        sev_fmt = {
            "error": Messages.error(f.title),
            "warning": Messages.warning(f.title),
            "info": Messages.info(f.title),
        }.get(f.severity, f.title)

        console.print(f"\n  {sev_fmt}")
        console.print(f"  [dim]{f.explanation}[/dim]")
        console.print(f"  Recommendation: {f.recommendation}")

    # Stats
    errors = sum(1 for f in report.findings if f.severity == "error")
    warnings = sum(1 for f in report.findings if f.severity == "warning")
    infos = sum(1 for f in report.findings if f.severity == "info")
    console.print(f"\n  Findings: {errors} errors, {warnings} warnings, {infos} info")

    if verbose and (cost is not None or turns is not None):
        parts = []
        if cost is not None:
            parts.append(f"Cost: ${cost:.4f}")
        if turns is not None:
            parts.append(f"Turns: {turns}")
        console.print(f"  [dim]{' | '.join(parts)}[/dim]")

    console.print()


@click.command()
@click.argument("target", type=click.Path(exists=True))
@click.option("--build", "build_first", is_flag=True, help="Build profile in temp dir, then audit")
@click.option("--model", default="claude-sonnet-4-6", help="Model for reviewer agent")
@click.option("--budget", default=5.0, type=float, help="Max budget in USD")
@click.option("--verbose", "-v", is_flag=True, help="Show verbose output")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
def audit(
    target: str,
    build_first: bool,
    model: str,
    budget: float,
    verbose: bool,
    json_output: bool,
) -> None:
    """Audit an OSPREY build profile or project for safety risks.

    TARGET is a .yml/.yaml build profile or a built project directory.

    Uses Claude Agent SDK to spawn an AI reviewer that analyzes permissions,
    safety hooks, MCP server configs, overlay files, and lifecycle scripts.

    \b
    Examples:
      osprey audit my-project/           Audit a built project
      osprey audit profile.yml --build   Build then audit
      osprey audit project/ --json       JSON output
    """
    if not _SDK_AVAILABLE:
        console.print(
            f"  {Messages.error('claude-agent-sdk is not installed.')}\n"
            "  Install it with: pip install claude-agent-sdk\n"
            "  Or: uv add claude-agent-sdk"
        )
        raise SystemExit(1)

    from .audit_prompts import AuditReport, build_audit_prompt

    target_type = _detect_target_type(target)
    target_dir = Path(target)

    # Optionally build the profile first
    tmpdir = None
    if build_first:
        if target_type != "profile":
            console.print(
                f"  {Messages.error('--build requires a .yml/.yaml profile, not a directory.')}"
            )
            raise SystemExit(1)

        tmpdir = tempfile.mkdtemp(prefix="osprey-audit-")
        project_name = f"audit-{uuid.uuid4().hex[:8]}"

        if not json_output:
            console.print(f"  Building profile to temp dir: {tmpdir}")

        from .build_cmd import build as build_cmd

        ctx = click.get_current_context()
        ctx.invoke(
            build_cmd,
            project_name=project_name,
            profile=str(target),
            output_dir=tmpdir,
            force=False,
            stream=False,
        )
        target_dir = Path(tmpdir) / project_name
        target_type = "project"

    try:
        if not json_output:
            console.print(f"  Auditing {target_type}: {Messages.path(str(target_dir))}")
            console.print(f"  Model: {model} | Budget: ${budget:.2f}")

        file_listing = _list_files(target_dir)
        prompt = build_audit_prompt(target_type, target_dir, file_listing)

        # Run the agent
        raw_text, cost, turns = asyncio.run(_run_audit(prompt, model, target_dir, budget, verbose))

        # Parse the result
        json_str = _extract_json(raw_text)
        if json_str is None:
            console.print(f"  {Messages.error('Agent did not produce valid JSON output.')}")
            if verbose:
                console.print(f"  [dim]Raw output: {raw_text[:500]}[/dim]")
            raise SystemExit(1)

        try:
            report = AuditReport.model_validate_json(json_str)
        except Exception as e:
            console.print(f"  {Messages.error(f'Failed to parse audit report: {e}')}")
            if verbose:
                console.print(f"  [dim]JSON: {json_str[:500]}[/dim]")
            raise SystemExit(1) from None

        _display_report(report, json_output, verbose, cost=cost, turns=turns)

    finally:
        # Clean up temp dir if we created one
        if tmpdir is not None:
            import shutil

            shutil.rmtree(tmpdir, ignore_errors=True)
