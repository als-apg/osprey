"""Reading a deployment's web-terminal feedback from the command line.

``osprey feedback`` is the retrieval half of the feedback loop. The web
terminal writes every submission as one ``feedback`` entry in the agent
record, and this command group is how a maintainer gets them back out --
``list`` for one line per submission, ``export`` for the whole entries
together with the session steps each one references.

Both verbs read with the operator login, which sees every user's entries,
and wait on a drain first, so a submission still pending in a spool is in the
answer. With no record reachable -- no stack up and no host session holding
one -- the verb takes a lease on the host-local agent-record for as long as it
reads, and the last holder stops that process on the way out.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
from collections.abc import Callable, Iterable, Iterator
from contextlib import ExitStack, contextmanager, nullcontext, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
from rich.text import Text

from osprey.cli.output import fail, machine_mode, note, report, table, warn
from osprey.cli.record_access import Deployment as _Deployment
from osprey.cli.record_access import find_build, operator_login
from osprey.cli.repo_resolver import repo_option
from osprey.cli.styles import Styles, data_table
from osprey_connectors.config import load_project_config
from osprey_connectors.logger import get_logger
from osprey_connectors.workspace import rendered_config_path

logger = get_logger("feedback")

#: How much of a submission's excerpt the listing shows before cutting it.
_EXCERPT_WIDTH = 56

#: Stands in for a table cell the entry does not fill in.
_UNKNOWN = "-"

#: Export field naming the session steps an entry references, resolved.
_RESOLVED_FIELD = "entries"


@dataclass(frozen=True)
class _Record:
    """One ``feedback`` entry as the record answered it.

    :ivar item: The entry's JSON as the read API returns it (``key``, ``ts``,
        ``person``, ``body`` and the ids).
    """

    item: dict[str, Any]

    @property
    def header(self) -> dict[str, Any]:
        body = self.item.get("body")
        return body if isinstance(body, dict) else {}

    @property
    def user(self) -> str:
        person = self.item.get("person")
        return person.strip() if isinstance(person, str) else ""


@contextmanager
def _feedback_session(repo: Path | None, *, announce: bool = True) -> Iterator[_Deployment]:
    """Resolve the deployment repo, then run one feedback verb against it.

    :param repo: Value of ``--repo``, or ``None`` to walk up from the working
        directory.
    :param announce: Print the banner naming the repo. Off for a verb whose
        stdout is the data, where a banner would corrupt a redirect.
    :yields: The repo root and the loaded rendered config.
    :raises click.ClickException: If no deployment repo encloses the search
        start, or the config cannot be read.
    :raises click.Abort: If the repo has no build, or the verb failed.
    """
    action = click.get_current_context().info_name or "feedback"
    build = find_build(
        repo, "Feedback is read from the agent record the rendered config describes."
    )
    if build is None:
        raise click.Abort()
    repo_root, config_path = build

    if announce:
        report(f"Web-terminal feedback: {action} in {repo_root}")

    try:
        yield _Deployment(root=repo_root, config=load_project_config(str(config_path)))
    except (click.Abort, click.ClickException):
        raise
    except KeyboardInterrupt:
        warn("Operation cancelled by user")
        raise click.Abort() from None
    except Exception as exc:
        cause = str(exc)
        if os.environ.get("DEBUG"):
            import traceback

            cause = f"{cause}\n{traceback.format_exc()}"
        fail(f"{action} failed", cause)
        raise click.Abort() from None


@contextmanager
def _record_client(deployment: _Deployment) -> Iterator[Any]:
    """Yield an operator record client, holding a host-local record while it reads.

    With ``OSPREY_AGENT_RECORD_URL`` set, that record is read as it is.
    Otherwise a one-shot :func:`~osprey.agent_record.host.ensure_host_record`
    lease is held for the block: a running stack or host session answers on
    the same loopback port and is reused, and when nothing runs the host-local
    process is started and, as the last holder, stopped again on the way out.
    """
    from osprey.agent_record.client import URL_ENV, RecordClient
    from osprey.agent_record.host import HostRecordRefused, ensure_host_record

    # Feedback reads with the operator login in either mode: it alone sees
    # every user's entries, so a host lease's own login is never used here.
    operator = operator_login(deployment.root)
    if operator is None:
        raise click.Abort()
    login, token = operator
    url = (os.environ.get(URL_ENV) or "").strip()
    if url:
        yield RecordClient(url, login, token)
        return
    with ExitStack() as stack:
        try:
            record = stack.enter_context(
                ensure_host_record(
                    deployment.root,
                    deployment.config,
                    config_path=rendered_config_path(deployment.root),
                )
            )
        except HostRecordRefused as exc:
            fail("The agent record is not available", str(exc))
            raise click.Abort() from None
        yield RecordClient(record.url, login, token)


def _read_feedback(client: Any) -> list[_Record]:
    """Every ``feedback`` entry the operator sees, after draining every spool."""
    from osprey.agent_record.contract import KIND_FEEDBACK, Drain

    answer = client.entries(KIND_FEEDBACK, wait=Drain())
    if not answer.available:
        fail("The agent record did not answer", str(answer.unavailable))
        raise click.Abort()
    if not answer.complete:
        warn(
            "The agent record answered before every pending submission was in it"
            + (f" ({answer.reason})" if answer.reason else "")
            + "; run again for the rest."
        )
    if answer.more:
        warn("Only the newest submissions are shown; the record holds more.")
    data = answer.data if isinstance(answer.data, list) else []
    return [_Record(item) for item in data if isinstance(item, dict)]


def _submitted_moment(record: _Record) -> dt.datetime | None:
    """When a submission was made, in UTC: its ``created_at``, else the entry's ``ts``."""
    for raw in (record.header.get("created_at"), record.item.get("ts")):
        if isinstance(raw, str) and raw.strip():
            with suppress(ValueError):
                stamped = dt.datetime.fromisoformat(raw.strip())
                return stamped if stamped.tzinfo is not None else stamped.replace(tzinfo=dt.UTC)
    return None


def _submitted_at(record: _Record) -> str:
    """Render the UTC minute a submission was made in, for one table row."""
    moment = _submitted_moment(record)
    if moment is None:
        return _UNKNOWN
    return moment.astimezone(dt.UTC).strftime("%Y-%m-%d %H:%M")


def _record_user(record: _Record) -> str:
    """Name who sent a submission: the record's person, else what the entry claims."""
    if record.user:
        return record.user
    identity = record.header.get("identity")
    if isinstance(identity, str) and identity.strip():
        return identity.strip()
    return _UNKNOWN


def _excerpt(header: dict[str, Any]) -> str:
    """Render the start of a submission's message for one table row."""
    raw = header.get("excerpt")
    if not isinstance(raw, str) or not raw.strip():
        return _UNKNOWN
    collapsed = " ".join(raw.split())
    if len(collapsed) <= _EXCERPT_WIDTH:
        return collapsed
    return collapsed[: _EXCERPT_WIDTH - 3].rstrip() + "..."


def _record_id(record: _Record) -> str:
    return str(record.header.get("id") or record.item.get("key") or _UNKNOWN)


#: Sorts before every real submission, so an entry with no readable time ends
#: up at the bottom of the listing instead of the top.
_UNDATED = dt.datetime.min.replace(tzinfo=dt.UTC)


def _newest_first(records: Iterable[_Record]) -> list[_Record]:
    """Order entries by submission time, most recent first."""
    return sorted(
        records,
        key=lambda record: (_submitted_moment(record) or _UNDATED, _record_id(record)),
        reverse=True,
    )


def _render_records(records: list[_Record]) -> None:
    """Print the listing table: one row per submission."""
    listing = data_table()
    listing.add_column("ID", style=Styles.ACCENT, no_wrap=True)
    listing.add_column("User")
    listing.add_column("Submitted (UTC)", no_wrap=True)
    listing.add_column("Channel")
    # Left wrappable on purpose. Pinning this column's width instead makes it
    # outrank the others, and a narrow terminal then truncates the user and
    # channel names -- the two columns a reader is scanning for.
    listing.add_column("Excerpt")
    for record in records:
        header = record.header
        listing.add_row(
            Text(_record_id(record)),
            Text(_record_user(record)),
            Text(_submitted_at(record)),
            Text(str(header.get("channel") or _UNKNOWN)),
            Text(_excerpt(header)),
        )
    table(listing)


def _indented(text: str) -> str:
    """Indent an already-serialized record by one level inside its array."""
    return "\n".join(f"  {line}" for line in text.splitlines())


class _JsonArrayWriter:
    """Write records to a text sink as one JSON array, a record at a time.

    The array's punctuation is written around each record as it arrives, so
    the result is one valid JSON document without ever holding more than a
    single record in memory.
    """

    def __init__(self, write: Callable[[str], object]) -> None:
        self._write = write
        #: Records written so far -- read afterwards for the summary line.
        self.count = 0

    def __call__(self, record: dict[str, Any]) -> None:
        """Append one record, opening the array on the first."""
        self._write("[\n" if self.count == 0 else ",\n")
        self._write(_indented(json.dumps(record, indent=2, default=str)))
        self.count += 1

    def close(self) -> None:
        """Close the array, writing an empty one when nothing arrived.

        Tolerant on purpose: this runs on the way out of a failed export too,
        and a sink that is already broken -- a closed pipe, a full disk -- must
        not replace the failure the caller is about to report with its own.
        """
        try:
            self._write("[]\n" if self.count == 0 else "\n]\n")
        except OSError as exc:
            logger.debug("Could not close the export document: %s", exc)


def _silence_stdout() -> None:
    """Point stdout's file descriptor at the null device.

    Catching :class:`BrokenPipeError` is only half of surviving ``| head``.
    Python's stdout buffer still holds whatever had not been written when the
    pipe went away, and the interpreter flushes that buffer on the way out --
    onto the pipe that is already gone. The flush happens outside any
    ``except`` this module can write, so it prints "Exception ignored on
    flushing sys.stdout" and turns a clean exit into 120. Re-pointing the
    descriptor gives that last flush somewhere harmless to land.

    Does nothing when stdout has no descriptor to re-point, which is how it
    behaves under a test runner that captures into a buffer.
    """
    try:
        target = sys.stdout.fileno()
    except (AttributeError, OSError, ValueError):
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, target)
    finally:
        os.close(devnull)


@contextmanager
def _export_sink(output_path: Path | None) -> Iterator[Callable[[str], object]]:
    """Yield the function an export writes its text through."""
    if output_path is None:
        # ALLOWLISTED raw click.echo, not a renderer primitive: with no
        # --output, stdout IS the export document, byte for byte, the way a
        # --json verb's stdout is one document.
        yield lambda text: click.echo(text, nl=False)
        return
    with output_path.open("w", encoding="utf-8") as handle:
        yield handle.write


def _plural(count: int, noun: str) -> str:
    """Render ``count`` with its noun, pluralized by adding an s."""
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _exported(client: Any, record: _Record) -> tuple[dict[str, Any], int]:
    """One submission as exported, and how many of its references did not resolve.

    The entry's body, with its record ``key``, ``person`` and ``ts``, and under
    :data:`_RESOLVED_FIELD` every referenced entry resolved through
    ``full_entry``. A reference the record no longer answers for is exported
    as ``{"key": ..., "missing": true}``.
    """
    out: dict[str, Any] = dict(record.header)
    for field in ("key", "person", "ts"):
        out.setdefault(field, record.item.get(field))
    refs = record.header.get("refs")
    resolved: list[Any] = []
    missing = 0
    for ref in refs if isinstance(refs, list) else ():
        if not isinstance(ref, str):
            continue
        answer = client.full_entry(ref)
        if answer.available and isinstance(answer.data, dict):
            resolved.append(answer.data)
        else:
            missing += 1
            resolved.append({"key": ref, "missing": True})
    out[_RESOLVED_FIELD] = resolved
    return out, missing


def _export_records(client: Any, output_path: Path | None) -> None:
    """Read every submission, resolve its references, and write the array."""
    records = _newest_first(_read_feedback(client))
    missing = 0
    with _export_sink(output_path) as write:
        writer = _JsonArrayWriter(write)
        try:
            for record in records:
                exported, unresolved = _exported(client, record)
                missing += unresolved
                writer(exported)
        except BrokenPipeError:
            # Whoever was reading stdout stopped reading -- `| head` is the
            # ordinary way that happens; there is nothing to report.
            logger.debug("Export ended early: the reader closed the pipe")
            _silence_stdout()
            return
        finally:
            writer.close()
    where = f"to {output_path}" if output_path is not None else "to standard output"
    line = f"Exported {_plural(writer.count, 'feedback submission')} {where}."
    if missing:
        line += f" {_plural(missing, 'referenced session step')} could not be resolved."
    report(line)


@click.group()
def feedback() -> None:
    """Read the feedback this deployment's web terminals collected.

    People using a web terminal can send feedback from the terminal itself.
    Each submission is kept in the agent record, beside the session it was
    sent from, and these verbs are how you read it back: 'list' for one line
    per submission, 'export' for the whole submission including the session
    steps it references.

    Every user's submissions are read. When no stack and no host session is
    up, the agent record is started on this machine for the read and stopped
    again afterwards.

    Every verb reads the deployment repo you are standing in; pass --repo to
    read another one.

    Examples:

    \b
      $ osprey feedback list
      $ osprey feedback export --output feedback.json
      $ osprey feedback export > feedback.json

    Run 'osprey feedback VERB --help' for the options a single verb takes.
    """


@feedback.command("list")
@repo_option
def list_records(repo: Path | None) -> None:
    """Show one line per feedback submission, newest first.

    Each row names the submission id, who sent it, when it was sent (UTC),
    which way it went out (stored locally, opened as a GitHub issue, or sent
    by email) and the start of the message. Reading a whole message, with the
    session it came from, is what 'export' is for.

    Examples:

    \b
      $ osprey feedback list
      $ osprey feedback list --repo ../other-deployment
    """
    with _feedback_session(repo) as deployment, _record_client(deployment) as client:
        records = _read_feedback(client)
        if not records:
            note("No feedback has been sent from this deployment yet.")
            return
        _render_records(_newest_first(records))


def _file_to_write(_ctx: click.Context, _param: click.Parameter, value: Path | None) -> Path | None:
    """Refuse an ``--output`` that nothing could be written to.

    Click's own checks read the text the operator typed and only stat a path
    that already exists, so two ordinary mistakes reach the command: an empty
    ``--output ""``, which arrives here as ``.``, and a file named inside a
    directory that is not there. Caught at parse time, the run stops before a
    single record read has started; caught at the write, it would stop
    after every submission had been read.
    """
    if value is None:
        return value
    if value.is_dir():
        raise click.BadParameter(f"{value} is a directory; --output names a file to write.")
    if not value.parent.is_dir():
        raise click.BadParameter(f"there is no directory {value.parent} to write {value.name} in.")
    return value


@feedback.command("export")
@repo_option
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    callback=_file_to_write,
    help="Write the export here instead of to stdout",
)
def export(repo: Path | None, output: Path | None) -> None:
    """Export every submission with its session steps, as one JSON array.

    Each object is one submission's whole entry: who sent it, when, from
    which session and which version of the app, how it went out and how large
    it was, whatever the submission was truncated at, the context and
    scrollback that were attached -- and, under 'entries', every session step
    it references, read back from the record. A step the record no longer
    holds is exported as {"key": ..., "missing": true}.

    With no --output the export goes to stdout and nothing else does, so it
    can be redirected straight into a file; errors and the closing summary go
    to standard error. With --output they all go to the terminal and the file
    holds the export.

    Examples:

    \b
      $ osprey feedback export --output feedback.json
      $ osprey feedback export > feedback.json
      $ osprey feedback export --repo ../other-deployment -o out.json
    """
    # Resolved before the verb runs: a relative path means what the operator
    # typed it against.
    output_path = output.resolve() if output is not None else None

    # No banner when stdout is the document, and machine mode with it, so a
    # line printed from anywhere in the stack lands on standard error rather
    # than in the middle of the JSON.
    with (
        _feedback_session(repo, announce=output_path is not None) as deployment,
        machine_mode() if output_path is None else nullcontext(),
        _record_client(deployment) as client,
    ):
        _export_records(client, output_path)
