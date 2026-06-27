"""Headless agent query command for the OSPREY CLI.

Provides ``osprey query <prompt>`` — a read-only, non-interactive way to send
a prompt to the project's configured agent and report the result.  Writes are
blocked at the SDK level via ``disallowed_tools`` (see
``agent_runner.read_only_disallowed_tools``): the project's MCP write tools
(from ``hook_config.json``), the built-in write/exec/network tools (``Bash``,
``Write``, ``Edit``, ...), and every approval-required MCP action declared in
the server registry (e.g. ``mcp__ariel__entry_create``,
``mcp__python__execute_file``).  This matters because the runner uses
``permission_mode=bypassPermissions``, under which the interactive allow/deny
permission layer is inert, so ``disallowed_tools`` is the sole write guard.

Exit codes
----------
0   Agent run completed; all expected MCP servers connected and no error flag.
1   Agent run completed but verdict failed (server missing, error result, or
    budget cap hit).
2   Infrastructure / usage error that prevented the run from starting at all
    (missing project, SDK not installed, provider not configured).
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import click
import yaml  # type: ignore[import-untyped]

from osprey.agent_runner import (
    EXIT_USAGE,
    SDKWorkflowResult,
    _expected_mcp_servers,
    evaluate_verdict,
    read_only_disallowed_tools,
    run_query,
)
from osprey.cli.project_utils import resolve_project_path


def _is_valid_project(project_dir: Path) -> bool:
    """Return True when *project_dir* looks like a built OSPREY project.

    Requires ``config.yml`` — the file carrying the provider/model config the
    runner unconditionally reads (``resolve_default_model`` and
    ``provider_env_for_project`` both parse it). ``.mcp.json`` is *not* required:
    its absence only means the verdict skips the MCP-server connectivity check.
    Accepting a ``.mcp.json``-only directory would let the run start and then
    crash deep in config resolution — better to fail fast with a usage error.
    """
    return (project_dir / "config.yml").exists()


def _build_json_output(result: SDKWorkflowResult, exit_code: int) -> str:
    """Serialise *result* to the ``--json`` output format."""
    return json.dumps(
        {
            "final_text": "\n".join(result.text_blocks),
            "tool_traces": [
                {
                    "name": t.name,
                    "input": t.input,
                    "result": t.result,
                    "is_error": t.is_error,
                }
                for t in result.tool_traces
            ],
            "mcp_servers": result.mcp_server_status,
            "exit_code": exit_code,
        },
        indent=2,
    )


@click.command()
@click.argument("prompt")
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    default=False,
    help="Emit a JSON object with final_text, tool_traces, mcp_servers, and exit_code.",
)
@click.option(
    "--project",
    "-p",
    default=None,
    metavar="DIR",
    help="Project directory (default: OSPREY_PROJECT env var, then cwd).",
)
@click.pass_context
def query(ctx: click.Context, prompt: str, as_json: bool, project: str | None) -> None:
    """Send PROMPT to the project agent and print the response.

    Runs in read-only mode: write tools are blocked at the SDK level so this
    command is safe to use in CI pipelines and automated workflows.

    Exit codes:

    \b
      0  Agent completed successfully (all expected MCP servers connected).
      1  Agent verdict failed (server missing, error result, or budget cap hit).
      2  Usage error (missing project, SDK not installed, provider not configured).

    Examples:

    \b
      # Query in the current project directory
      $ osprey query "What PVs are used for beam current?"

      # Query a specific project
      $ osprey query --project ~/projects/als-assistant "List vacuum sections"

      # Machine-readable JSON output
      $ osprey query --json "Summarise recent alarms"
    """
    project_dir = resolve_project_path(project)

    if not _is_valid_project(project_dir):
        click.echo(
            "Error: no OSPREY project found. "
            "Run inside an `osprey build` project directory or pass --project.",
            err=True,
        )
        ctx.exit(EXIT_USAGE)
        return

    disallowed_tools = read_only_disallowed_tools(project_dir)
    expected = _expected_mcp_servers(project_dir)

    try:
        result = asyncio.run(run_query(project_dir, prompt, disallowed_tools=disallowed_tools))
    except (ImportError, RuntimeError) as exc:
        # SDK not installed, or run_query wrapped an SDK/connection failure.
        click.echo(f"Error: {exc}", err=True)
        ctx.exit(EXIT_USAGE)
        return
    except (OSError, yaml.YAMLError, ValueError) as exc:
        # Config resolution (read before the SDK run) hit a missing/malformed
        # config.yml, or an unknown/misconfigured provider (the resolver raises
        # ValueError naming the valid providers) — a usage error, not a verdict
        # failure. The exception message is echoed so the actionable hint
        # (e.g. "Built-in providers: …") reaches the user.
        click.echo(f"Error: could not load project configuration: {exc}", err=True)
        ctx.exit(EXIT_USAGE)
        return

    code = evaluate_verdict(result, expected)

    if as_json:
        click.echo(_build_json_output(result, code))
    else:
        click.echo("\n".join(result.text_blocks))

    sys.exit(code)
