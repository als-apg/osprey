"""``osprey archive`` — copy the agent record into an append-only tree.

The command copies from a sources tree to a destination tree it is handed and
knows nothing about compose, config or the repo: inside the bundled ``archive``
service both are mounts, run by hand they are whatever the operator names. The
archive never deletes or rewrites a copy; its own staging directory,
``<dest>/.incoming``, is the one place it removes files from.

Exit codes of ``--once``: 0 when the pass completed with no errors, 1 when it
completed with recorded errors, 2 when it could not run. ``--watch`` runs a pass
at once and then every ``--interval`` seconds; it exits 2 after five
consecutive passes that could not run.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import click

from osprey.cli import output

logger = logging.getLogger(__name__)

#: Consecutive passes that could not run before ``--watch`` gives up.
WATCH_FAILURE_CAP = 5

_sleep = time.sleep


def _one_pass(
    sources: Path,
    dest: Path,
    openobserve_url: str | None,
    openobserve_org: str,
    backfill_days: int,
) -> int:
    """Run one pass and report it; return the ``--once`` exit code."""
    from osprey.services.archive.run import ArchiveDestinationError, run_pass
    from osprey.services.archive.telemetry_export import TelemetryExporter

    telemetry = (
        TelemetryExporter(url=openobserve_url, org=openobserve_org, backfill_days=backfill_days)
        if openobserve_url
        else None
    )
    try:
        result = run_pass(sources, dest, telemetry=telemetry)
    except ArchiveDestinationError as exc:
        output.fail("Archive pass could not run.", str(exc))
        return 2
    output.report(
        f"Archived {result.files} files ({result.bytes} bytes) into {result.day}; "
        f"{len(result.errors)} errors"
    )
    if result.errors:
        first = result.errors[0]
        logger.warning("Archive pass error at %s: %s", first["source"], first["error"])
        return 1
    return 0


@click.command("archive")
@click.option(
    "--once/--watch",
    "once",
    default=True,
    help="Run one pass and exit, or keep running a pass every --interval seconds.",
)
@click.option(
    "--sources",
    envvar="OSPREY_ARCHIVE_SOURCES",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Sources tree: <sources>/<kind>/<name>/...",
)
@click.option(
    "--dest",
    envvar="OSPREY_ARCHIVE_DEST",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Existing archive root the day directories are written into.",
)
@click.option(
    "--interval",
    envvar="OSPREY_ARCHIVE_INTERVAL_SECONDS",
    type=click.IntRange(min=60),
    default=86400,
    show_default=True,
    help="Seconds between passes under --watch.",
)
@click.option(
    "--openobserve-url",
    envvar="OSPREY_ARCHIVE_OPENOBSERVE_URL",
    default=None,
    help="Telemetry store to export completed days from; unset, none is exported.",
)
@click.option(
    "--openobserve-org",
    envvar="OSPREY_ARCHIVE_OPENOBSERVE_ORG",
    default="default",
    show_default=True,
    help="Telemetry store organization.",
)
@click.option(
    "--openobserve-backfill-days",
    envvar="OSPREY_ARCHIVE_OPENOBSERVE_BACKFILL_DAYS",
    type=click.IntRange(min=1),
    default=14,
    show_default=True,
    help="Most completed days a pass exports.",
)
def archive(
    once: bool,
    sources: Path,
    dest: Path,
    interval: int,
    openobserve_url: str | None,
    openobserve_org: str,
    openobserve_backfill_days: int,
) -> None:
    """Copy the agent record into an append-only archive.

    Copies transcripts, dispatch run records, artifact stores, plan-queue
    history, the audit ledger and completed days of telemetry into
    <dest>/<YYYY-MM-DD>/ with a sha256 MANIFEST.jsonl. Nothing is deleted or
    rewritten.

    The telemetry store is read with ZO_INGEST_USER_EMAIL and ZO_INGEST_SA_TOKEN
    from the environment.
    """
    if once:
        raise SystemExit(
            _one_pass(sources, dest, openobserve_url, openobserve_org, openobserve_backfill_days)
        )

    could_not_run = 0
    try:
        while True:
            code = _one_pass(
                sources, dest, openobserve_url, openobserve_org, openobserve_backfill_days
            )
            could_not_run = could_not_run + 1 if code == 2 else 0
            if could_not_run >= WATCH_FAILURE_CAP:
                output.fail(f"Archive stopped: {WATCH_FAILURE_CAP} passes in a row could not run.")
                raise SystemExit(2)
            _sleep(interval)
    except KeyboardInterrupt:
        output.report("Stopping the archive.")
