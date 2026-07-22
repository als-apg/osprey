"""``osprey health`` command — thin CLI wrapper over the health framework.

This module is a thin Click wrapper: it resolves the project, performs the
single ``config.yml`` load, assembles the merged category records (built-in
"core" categories, declarative YAML categories, and facility plugins), runs the
async health suite, and renders the report. All check logic lives in
:mod:`osprey.health`; this file only wires the pieces together.

Design contracts honored here:

* **Single config load.** The CLI loads ``config.yml`` exactly once via
  :func:`osprey.utils.config.get_config_builder` and reports on the outcome
  through a :class:`~osprey.health.core.configuration.ConfigState`. A load
  failure never crashes the command — it degrades into configuration error rows
  while the rest of the report still renders.
* **``--full`` is the sole on_demand gate.** ``--category`` selects which
  categories run but never elevates cost class.
* **Machine-clean ``--json``.** In ``--json`` mode every human-facing line
  (progress spinner, deprecation warning) is routed to a stderr console so
  stdout is a single JSON document that round-trips through :func:`json.loads`.

The module-level ``console`` is intentionally patchable (tests replace
``osprey.cli.health_cmd.console``); ``resolve_project_path`` is imported locally
inside the command so patching it at its source module takes effect.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from osprey.cli.styles import console
from osprey.health.records import build_records, load_config

if TYPE_CHECKING:
    from osprey.health.config import CategoryRecord
    from osprey.health.models import CheckReport

# Loggers that narrate config/registry loading. Osprey's root ``RichHandler``
# writes to stdout, so their chatter (including config-load ERROR blocks) is
# silenced during a run.
_NOISY_LOADER_LOGGERS = ("CONFIG", "registry")

# A level above CRITICAL: nothing a logger or handler emits reaches it, so a
# handler/logger pinned here stays silent for every record — including the
# ERROR-level config-load failure block that ConfigBuilder emits.
_SILENT_LEVEL = logging.CRITICAL + 1


@contextmanager
def _quiet_run_logs(*, as_json: bool, verbose: bool) -> Iterator[None]:
    """Keep incidental log output from corrupting the command's stdout.

    Osprey's root logging handler renders to stdout, so any record it emits
    during the run lands on stdout. ConfigBuilder logs config-load failures at
    ``ERROR``, so merely capping at ``ERROR`` would still leak a Rich error block
    ahead of the report — breaking the ``--json`` machine-clean contract. Both
    paths therefore pin to :data:`_SILENT_LEVEL` (above ``CRITICAL``): ``--json``
    silences every root handler, and the human path (unless ``--verbose``)
    silences the noisy loader loggers. Genuine config failures are never lost —
    the ``configuration`` category reports them as proper rows.
    """
    if as_json:
        handlers = logging.getLogger().handlers
        saved = [(h, h.level) for h in handlers]
        for handler in handlers:
            handler.setLevel(max(handler.level, _SILENT_LEVEL))
        try:
            yield
        finally:
            for handler, level in saved:
                handler.setLevel(level)
        return

    if verbose:
        yield
        return

    saved_levels = {name: logging.getLogger(name).level for name in _NOISY_LOADER_LOGGERS}
    for name in _NOISY_LOADER_LOGGERS:
        logging.getLogger(name).setLevel(_SILENT_LEVEL)
    try:
        yield
    finally:
        for name, level in saved_levels.items():
            logging.getLogger(name).setLevel(level)


def _load_project_env(project_path: Path) -> None:
    """Load the project's ``.env`` into ``os.environ`` with override semantics.

    The ``.env`` file is the source of truth for API keys and facility settings,
    so it overrides any pre-existing process environment. A missing file or a
    missing ``python-dotenv`` is silently ignored.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    dotenv_path = project_path / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path, override=True)


def _validate_categories(
    requested: tuple[str, ...], valid_names: set[str], *, config_ok: bool
) -> tuple[str, ...] | None:
    """Resolve the ``--category`` selection, rejecting unknown names.

    Returns the requested names to run (deduplicated), or ``None`` when none
    were requested (run every category).

    An unknown name is a ``UsageError`` only when the config loaded — then
    ``valid_names`` is the full, authoritative category set. Under a config
    failure only the core categories are known (YAML and plugin categories were
    never parsed), so a non-core name cannot be judged invalid: it passes
    through, matches no record, and the scoped report still renders. This
    preserves the "``--category X`` + broken config → report, no ``UsageError``"
    contract.
    """
    if not requested:
        return None
    if config_ok:
        unknown = [name for name in requested if name not in valid_names]
        if unknown:
            plural = "ies" if len(unknown) > 1 else "y"
            raise click.UsageError(
                f"Unknown health categor{plural}: {', '.join(unknown)}. "
                f"Valid categories: {', '.join(sorted(valid_names))}"
            )
    return tuple(dict.fromkeys(requested))


async def _run_suite(
    records: list[CategoryRecord],
    control_system_config: dict[str, Any],
    *,
    full: bool,
    categories: tuple[str, ...] | None,
    suite_timeout_s: float,
    on_demand_timeout_s: float | None,
) -> CheckReport:
    """Run the merged suite under a :class:`HealthRuntime` async context."""
    from osprey.health.runner import run_health_suite
    from osprey.health.runtime import HealthRuntime

    async with HealthRuntime(control_system_config) as runtime:
        return await run_health_suite(
            records,
            runtime=runtime,
            full=full,
            categories=categories,
            suite_timeout_s=suite_timeout_s,
            on_demand_timeout_s=on_demand_timeout_s,
        )


@click.command()
@click.option(
    "--project",
    "-p",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Project directory (default: current directory or OSPREY_PROJECT env var)",
)
@click.option(
    "--verbose", "-v", is_flag=True, help="Show per-warning and per-error details in the summary"
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit the report as a single JSON document on stdout (machine-readable)",
)
@click.option(
    "--category",
    "categories",
    multiple=True,
    metavar="NAME",
    help="Run only the named category (repeatable). Unknown names are rejected.",
)
@click.option(
    "--full",
    is_flag=True,
    help="Also run on_demand categories (live model chat, pinned CLI download).",
)
@click.option("--basic", "-b", is_flag=True, hidden=True)
def health(
    project: str | None,
    verbose: bool,
    as_json: bool,
    categories: tuple[str, ...],
    full: bool,
    basic: bool,
) -> None:
    """Check the health of your Osprey installation and configuration.

    Runs a suite of diagnostics — configuration validity, file-system layout,
    Python environment, container infrastructure, telemetry store, API
    providers, and the Claude Code CLI — grouped into categories. Cheap
    poll-class categories run by default; costly on_demand categories (live
    model chat completions, pinned-CLI verification) run only with ``--full``.

    Exit codes:

    \b
      0 - all checks passed
      1 - warnings only
      2 - one or more errors (including configuration errors)
      3 - the command itself failed unexpectedly
      130 - interrupted

    Examples:

    \b
      # Poll-class checks for the current project
      $ osprey health

      # Include the on_demand model-chat checks
      $ osprey health --full

      # Only the providers category, as JSON
      $ osprey health --category providers --json

      # A specific project directory
      $ osprey health --project ~/projects/my-agent
    """
    from rich.console import Console

    from osprey.health.config import DEFAULT_SUITE_TIMEOUT_S
    from osprey.health.offload import abandoned_count
    from osprey.health.render import render_json, render_report, run_progress

    from .project_utils import resolve_project_path

    # A dedicated stderr console for out-of-band notices (progress, deprecation,
    # failure messages) that must never touch stdout. In ``--json`` mode it is
    # also the human console so stdout stays a pure JSON document.
    err_console = Console(stderr=True)
    human_console = err_console if as_json else console

    if basic:
        # Deprecation notice is unconditionally stderr — stdout carries only the
        # report (or, under ``--json``, only the JSON document).
        err_console.print(
            "[dim]Warning: --basic is deprecated and has no effect; on_demand checks "
            "are now opt-in via --full.[/dim]"
        )

    try:
        project_path = resolve_project_path(project)
        config_path = project_path / "config.yml"

        with _quiet_run_logs(as_json=as_json, verbose=verbose):
            config_state, expanded, settings, config_ok = load_config(config_path, project_path)

            # Load the project .env after the config load so its values are present
            # in os.environ for the run-time checks (provider canaries, env scan).
            _load_project_env(project_path)

            suite_timeout_s = settings.suite_timeout_s if settings else DEFAULT_SUITE_TIMEOUT_S
            on_demand_timeout_s = settings.on_demand_timeout_s if settings else None

            records, extra_rows = build_records(
                config_state, expanded, settings, config_ok, project_path, suite_timeout_s
            )
            selected = _validate_categories(
                categories, {r.name for r in records}, config_ok=config_ok
            )
            # A config-load failure is a global fault: its ``configuration``
            # error rows (and the resulting exit 2) must surface even when a
            # ``--category`` filter would otherwise scope them out.
            if selected is not None and not config_ok and "configuration" not in selected:
                selected = ("configuration", *selected)

            control_system_config = (expanded or {}).get("control_system", {}) or {}

            with run_progress(console=human_console):
                report = asyncio.run(
                    _run_suite(
                        records,
                        control_system_config,
                        full=full,
                        categories=selected,
                        suite_timeout_s=suite_timeout_s,
                        on_demand_timeout_s=on_demand_timeout_s,
                    )
                )

        # Plugin-load diagnostics are surfaced only on an unfiltered run; a
        # ``--category`` selection keeps the output scoped to what was asked for.
        if selected is None and extra_rows:
            report.results.extend(extra_rows)

        if as_json:
            render_json(report)
        else:
            render_report(report, verbose=verbose, console=console)

        exit_code = report.exit_code

    except click.UsageError:
        raise
    except KeyboardInterrupt:
        human_console.print("\n[yellow]Health check interrupted[/yellow]")
        exit_code = 130
    except Exception as exc:  # noqa: BLE001 - top-level guard: any failure is exit 3
        human_console.print(f"\n[red]Health check failed: {exc}[/red]")
        if verbose:
            human_console.print_exception()
        exit_code = 3

    # A hung sync check leaves a daemon thread running; a normal ``sys.exit`` can
    # then wedge on interpreter teardown. Fall back to ``os._exit`` so an
    # abandoned thread can never block process exit.
    if abandoned_count() > 0:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
    sys.exit(exit_code)
