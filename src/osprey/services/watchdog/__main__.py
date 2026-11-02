"""The record entry: judge one submitted prompt in a process of its own.

``python -m osprey.services.watchdog record --project-dir <dir> --submission-file
<path> [--session-key <key>]``

The trigger hook runs this when no web terminal takes the prompt. Without
``--session-key`` the session has no web terminal and the entry only records.
``--session-key`` means the web terminal's intake failed for that session, so the
entry says so in the audit record and, under a hold root from the environment,
holds the session on a flagged verdict. A key with no hold root cannot hold.

The submission file is consumed before anything else can fail. Its prompt reaches
the engines and the hold record, never a log line.

Agent runs for one (watchdog, record session) are bounded across processes by
:class:`~osprey.services.watchdog.runner.LockFileBound`. The process's own lifetime
is bounded by twice the longest watchdog timeout: a pending run may wait out the run
in flight before its own time starts, and a run still undecided at the bound records
``intake_lost``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from osprey.services.watchdog.config import WatchdogDefinition
from osprey.services.watchdog.definitions import load_rendered_config, load_watchdogs
from osprey.services.watchdog.runner import (
    Engine,
    LockFileBound,
    WatchdogRunner,
    unavailable_verdict,
)
from osprey.services.watchdog.verdict import record_verdict
from osprey_connectors.session_hold import hold_root

logger = logging.getLogger(__name__)

_SUBMISSION_FIELDS = ("record_session_id", "submission_id", "prompt")


def _no_notify(_frame: dict[str, Any]) -> None:
    """No browser is listening to this process; the hold record is the truth."""


def _read_submission(path: Path) -> dict[str, str]:
    """Consume the submission file; its fields, or ``ValueError``/``OSError``."""
    try:
        submission = json.loads(path.read_text(encoding="utf-8"))
    finally:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Watchdog submission not removed: %s", type(exc).__name__)
    fields = submission if isinstance(submission, dict) else {}
    for name in _SUBMISSION_FIELDS:
        value = fields.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"submission field {name} is missing or not text")
    return {name: fields[name] for name in _SUBMISSION_FIELDS}


def _record_intake_failed(
    definitions: Iterable[WatchdogDefinition],
    *,
    session_key: str | None,
    record_session_id: str,
    submission_id: str,
) -> None:
    """File that intake could not take this submission, once per watchdog."""
    for defn in definitions:
        record_verdict(
            unavailable_verdict(
                defn,
                session_key=session_key,
                record_session_id=record_session_id,
                submission_id=submission_id,
                error_type="intake_failed",
            )
        )


async def _run(
    definitions: tuple[WatchdogDefinition, ...],
    *,
    submission: Mapping[str, str],
    session_key: str | None,
    root: Path | None,
    config: Mapping[str, Any],
    build_dir: Path,
    engines: Mapping[str, Engine] | None,
) -> None:
    runner = WatchdogRunner(
        definitions,
        hold_root=root,
        notify=_no_notify,
        engines=engines,
        config=config,
        build_dir=build_dir,
        bound=LockFileBound(),
    )
    task = runner.on_prompt(
        session_key=session_key,
        record_session_id=submission["record_session_id"],
        submission_id=submission["submission_id"],
        prompt=submission["prompt"],
        received_at=time.monotonic(),
        can_hold=session_key is not None,
    )
    await asyncio.wait({task}, timeout=2 * max(d.timeout_s for d in definitions))
    await runner.shutdown()


def _record(args: argparse.Namespace, engines: Mapping[str, Engine] | None) -> int:
    try:
        submission = _read_submission(Path(args.submission_file))
    except (OSError, ValueError) as exc:
        logger.error("FATAL: watchdog submission unreadable: %s", type(exc).__name__)
        return 1
    try:
        loaded = load_rendered_config(Path(args.project_dir))
        if loaded is None:
            raise FileNotFoundError(f"no rendered config under {args.project_dir}")
        config, build_dir = loaded
        definitions = load_watchdogs(config, build_dir)
    except Exception as exc:
        logger.error("FATAL: watchdogs not loaded: %s", type(exc).__name__)
        return 1
    if not definitions:
        return 0

    root = hold_root() if args.session_key else None
    session_key = args.session_key if root is not None else None
    if args.session_key:
        _record_intake_failed(
            definitions,
            session_key=session_key,
            record_session_id=submission["record_session_id"],
            submission_id=submission["submission_id"],
        )
    asyncio.run(
        _run(
            definitions,
            submission=submission,
            session_key=session_key,
            root=root,
            config=config,
            build_dir=build_dir,
            engines=engines,
        )
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m osprey.services.watchdog")
    commands = parser.add_subparsers(dest="command", required=True)
    record = commands.add_parser("record", help="judge one submitted prompt, then exit")
    record.add_argument("--project-dir", required=True)
    record.add_argument("--submission-file", required=True)
    record.add_argument("--session-key")
    return parser


def main(argv: list[str] | None = None, *, engines: Mapping[str, Engine] | None = None) -> int:
    """Run the entry; *engines* replaces the engines a build would otherwise construct."""
    return _record(_parser().parse_args(argv), engines)


if __name__ == "__main__":
    sys.exit(main())
