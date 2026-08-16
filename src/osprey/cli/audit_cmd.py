"""Agentic build profile and project safety auditor.

Spawns a Claude agent via the Claude Agent SDK to deeply analyze an OSPREY
build profile or built project directory, producing a structured safety report.
"""

from __future__ import annotations

import asyncio
import re
import tempfile
import uuid
from contextlib import nullcontext
from pathlib import Path

import click
from rich.text import Text

from . import output
from .altitude import lift_gate
from .styles import Styles, data_table, panel

#: The token each severity prints in, spelled once so the findings table and the
#: detail list below it agree on what "error" looks like.
_SEVERITY_STYLES = {
    "error": Styles.ERROR,
    "warning": Styles.WARNING,
    "info": Styles.INFO,
}

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
                        output.note(f"{block.text[:200]}...")
        elif isinstance(message, ResultMessage):
            total_cost = getattr(message, "total_cost_usd", None)
            num_turns = getattr(message, "num_turns", None)

    return "".join(collected_text), total_cost, num_turns


def _display_report(report, json_output: bool, verbose: bool, cost=None, turns=None):
    """Display the audit report using Rich or JSON."""
    if json_output:
        # The one machine seam in this verb: stdout carries the report document
        # and nothing else, so it goes out through click rather than through the
        # renderer. Its key set is pinned by tests/cli/test_json_keyset_capture.py.
        click.echo(report.model_dump_json(indent=2))
        return

    # Summary panel
    risk_style = {
        "low": Styles.SUCCESS,
        "medium": Styles.WARNING,
        "high": Styles.ERROR,
    }.get(report.overall_risk, Styles.INFO)

    # Built as spans rather than as markup: the renderer prints with markup off
    # and Rich carries that into nested renderables, so a "[error]...[/error]"
    # body would show its own tags. Only the verdict wears the token -- a summary
    # painted the same red competes with the line that states the risk.
    summary = Text()
    summary.append(f"{report.overall_risk.upper()} RISK", style=risk_style)
    summary.append(f"\n\n{report.summary}")

    output.report("")
    output.table(panel(summary, title="Audit Summary"))

    if not report.findings:
        output.report("")
        output.report("✓ No findings. The audit came back clean.", style=Styles.SUCCESS)
        output.report("")
        return

    # Findings table
    table = data_table(show_lines=True)
    table.add_column("Severity", width=8)
    table.add_column("Category", width=12)
    table.add_column("Title")
    table.add_column("File", width=30)

    for f in report.findings:
        severity = Text(f.severity, style=_SEVERITY_STYLES.get(f.severity, ""))
        table.add_row(severity, f.category, f.title, f.file_path)

    output.table(table)

    # Detailed findings
    for f in report.findings:
        output.report("")
        output.report(f.title, style=_SEVERITY_STYLES.get(f.severity))
        output.note(f.explanation)
        output.section("", [("Recommendation", f.recommendation)])

    # Stats
    errors = sum(1 for f in report.findings if f.severity == "error")
    warnings = sum(1 for f in report.findings if f.severity == "warning")
    infos = sum(1 for f in report.findings if f.severity == "info")
    output.report("")
    output.report(f"Findings: {errors} errors, {warnings} warnings, {infos} info")

    if verbose and (cost is not None or turns is not None):
        parts = []
        if cost is not None:
            parts.append(f"Cost: ${cost:.4f}")
        if turns is not None:
            parts.append(f"Turns: {turns}")
        output.note(" | ".join(parts))

    output.report("")


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
    safety hooks, MCP server configs, convention directories, and lifecycle
    scripts.

    \b
    Examples:
      osprey audit my-project/           Audit a built project
      osprey audit profile.yml --build   Build then audit
      osprey audit project/ --json       JSON output
    """
    # Every --json verb runs its whole body in machine mode, so a renderer line
    # from anywhere in the stack lands on stderr and stdout carries one document.
    with output.machine_mode() if json_output else nullcontext():
        if verbose:
            # --verbose lifts the altitude gate for this run, so the reviewer's own
            # transcript reaches the terminal alongside the extra detail the flag
            # already prints (the raw answer on a parse failure, cost and turns).
            lift_gate()

        if not _SDK_AVAILABLE:
            output.fail(
                "claude-agent-sdk is not installed",
                "The audit verb reviews the target with an agent, which needs the SDK.",
                "install it with: uv add claude-agent-sdk",
            )
            raise SystemExit(1)

        from .audit_prompts import AuditReport, build_audit_prompt

        target_type = _detect_target_type(target)
        target_dir = Path(target)

        # Optionally build the profile first
        tmpdir = None
        if build_first:
            if target_type != "profile":
                output.fail(
                    "--build needs a .yml or .yaml profile, not a directory",
                    f"The target is a directory: {target}",
                    "drop --build to audit the directory as it stands",
                )
                raise SystemExit(1)

            tmpdir = tempfile.mkdtemp(prefix="osprey-audit-")
            project_name = f"audit-{uuid.uuid4().hex[:8]}"

            if not json_output:
                output.section("", [("Building profile to", tmpdir)])

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
                output.section(
                    "",
                    [
                        ("Auditing", f"{target_type}: {target_dir}"),
                        ("Model", model),
                        ("Budget", f"${budget:.2f}"),
                    ],
                )

            file_listing = _list_files(target_dir)
            prompt = build_audit_prompt(target_type, target_dir, file_listing)

            # Run the agent
            raw_text, cost, turns = asyncio.run(
                _run_audit(prompt, model, target_dir, budget, verbose)
            )

            # Parse the result
            json_str = _extract_json(raw_text)
            if json_str is None:
                output.fail(
                    "The reviewer did not return a report",
                    "Its answer held no JSON document, so there is nothing to read.",
                    "run it again with -v to see what it did say",
                )
                if verbose:
                    output.note(f"Raw output: {raw_text[:500]}")
                raise SystemExit(1)

            try:
                report = AuditReport.model_validate_json(json_str)
            except Exception as e:
                output.fail(
                    "Could not read the audit report",
                    str(e),
                    "run it again with -v to see the document that failed to parse",
                )
                if verbose:
                    output.note(f"JSON: {json_str[:500]}")
                raise SystemExit(1) from None

            _display_report(report, json_output, verbose, cost=cost, turns=turns)

        finally:
            # Clean up temp dir if we created one
            if tmpdir is not None:
                import shutil

                shutil.rmtree(tmpdir, ignore_errors=True)
