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

What it queries is the deployment's ``build/`` — the render, found the way
every repo-scoped verb finds its deployment: walk up from the working directory
to the nearest ``profile.yml``, or start from ``--repo``. Nothing is re-rendered
here, so a profile edited since the last build is reported as a warning and the
query runs against the build as it stands; the one refusal is having nothing
rendered to query at all. That is the same bargain ``osprey chat`` strikes, and
for the same reason: a query is read-only and ends with the command, so being
blocked helps nobody, while not being told would let a script answer from an
older render without ever saying so.

Exit codes
----------
0   Agent run completed; all expected MCP servers connected and no error flag.
1   Agent run completed but verdict failed (server missing, error result, or
    budget cap hit).
2   Infrastructure / usage error that prevented the run from starting at all
    (no build, SDK not installed, provider not configured).
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
    evaluate_verdict,
    expected_mcp_servers,
    read_only_disallowed_tools,
    run_query,
)

from .repo_resolver import find_repo_root, repo_option


def _overlay_repo_env(repo_root: Path) -> None:
    """Load the repo's env chain into ``os.environ``, overriding what is there.

    The SECRETS zone is at the repo root while the render is under ``build/``,
    so the env chain and the ``config.yml`` this verb needs do not live in one
    directory — and the runner's own overlay
    (``osprey.build.claude_code_resolver._env_lookup``) looks beside the config
    it was handed. Without this the provider secret would simply not be found
    and the query would authenticate as whatever the ambient shell exported.

    The chain is loaded in ascending precedence — ``.env.shared`` then ``.env``,
    each with ``override=True`` — so the host-local file wins over the committed
    defaults, and both win over a stale shell export. That is the precedence
    every other launch path applies. Every key is copied, not a declared subset:
    the agent expands ``.mcp.json`` ``${VAR}`` references (Channel Access
    addressing among them) out of this environment, so a narrowed copy would
    silently mis-address the control system rather than fail.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    from osprey.utils.dotenv import chain_files

    for env_file in chain_files(repo_root):
        load_dotenv(env_file, override=True)


def _report_drift(repo_root: Path, build_dir: Path) -> None:
    """Say what the build no longer matches, then let the query proceed.

    Written to stderr rather than stdout: ``osprey query`` is built to be piped
    — ``--json`` especially — and a warning mixed into that stream would break
    the consumer it was meant to inform.

    Raises:
        click.ClickException: When there is no build to query at all. Its exit
            code is remapped to :data:`EXIT_USAGE` by the caller, which owns
            the documented exit-code contract.
    """
    from osprey.deployment.staleness import DriftState, check_drift

    report = check_drift(repo_root)

    # NO_BUILD is the one drift state a query cannot warn its way past, and the
    # config.yml check beside it is the same condition read one file deeper: a
    # manifest with no rendered config beside it is a build in name only, and
    # config.yml is what the runner unconditionally parses for the provider and
    # model (``resolve_default_model``, ``provider_env_for_project``).
    if report.state is DriftState.NO_BUILD or not (build_dir / "config.yml").is_file():
        raise click.ClickException(
            f"no build found at {build_dir} — run `osprey build` first.\n\n"
            "`osprey query` asks the rendered deployment, and there is "
            "nothing rendered yet."
        )

    if report.refuses:
        click.echo(f"WARNING: {report.message}", err=True)
        click.echo("Answering from build/ as it was last rendered.", err=True)

    # Independent of the verdict above: a build can be a faithful render of the
    # current profile and still predate the installed framework.
    if report.version_skew:
        click.echo(f"WARNING: {report.version_skew.message}", err=True)


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
@repo_option
@click.pass_context
def query(ctx: click.Context, prompt: str, as_json: bool, repo: Path | None) -> None:
    """Send PROMPT to this deployment's agent.

    The response is printed. Runs in read-only mode: write tools are blocked at the SDK level so this
    command is safe to use in CI pipelines and automated workflows.

    Asks the deployment enclosing the current directory, answering from its
    build/ as it was last rendered. When profile.yml or a file it points at has
    changed since that build, a warning on stderr says so and the query runs
    anyway.

    Exit codes:

    \b
      0  Agent completed successfully (all expected MCP servers connected).
      1  Agent verdict failed (server missing, error result, or budget cap hit).
      2  Usage error (no build, SDK not installed, provider not configured).

    Examples:

    \b
      $ osprey query "What PVs are used for beam current?"
      $ osprey query --repo ~/als-assistant "List vacuum sections"
      $ osprey query --json "Summarise recent alarms"
    """
    from osprey.deployment.staleness import BUILD_DIRNAME

    repo_root = find_repo_root(repo)
    build_dir = repo_root / BUILD_DIRNAME

    try:
        _report_drift(repo_root, build_dir)
    except click.ClickException as exc:
        # Remapped rather than left to Click's exit 1: "there is nothing to
        # query" is a usage error under this command's documented contract, and
        # a caller distinguishing 1 from 2 must not read it as a failed verdict.
        click.echo(f"Error: {exc.format_message()}", err=True)
        ctx.exit(EXIT_USAGE)
        return

    # Before any provider resolution: the secret lives at the repo root, the
    # provider that needs it is declared in the render.
    _overlay_repo_env(repo_root)

    disallowed_tools = read_only_disallowed_tools(build_dir)
    expected = expected_mcp_servers(build_dir)

    try:
        result = asyncio.run(run_query(build_dir, prompt, disallowed_tools=disallowed_tools))
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
