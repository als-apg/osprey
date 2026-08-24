"""Reading a deployment's web-terminal feedback store from the command line.

``osprey feedback`` is the retrieval half of the feedback loop. The web
terminal writes one record per submission into the deployment's feedback
store, and this command group is how a maintainer gets those records back
out -- ``list`` for the headers, ``export`` for the headers together with
their session context. The store is never exposed over HTTP and there is no
triage UI, so this is the only way to read it.

Both readers face the same wire format. Records live one JSON document per
file, and a store is read by concatenating those files: from inside an
ephemeral container mounted read-only on a user's agent-data volume, or
straight off the host filesystem for a single-user deployment. Either way
what arrives is a run of JSON objects with no framing between them,
delivered in chunks that respect neither document nor character boundaries.
:func:`iter_json_documents` is the one parser both readers use for that
stream, and :class:`FeedbackReadError` is how it refuses a stream it cannot
trust.
"""

from __future__ import annotations

import codecs
import datetime as dt
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager, nullcontext, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, NamedTuple

import click
from rich.text import Text

from osprey.cli.output import fail, machine_mode, note, report, table, warn
from osprey.cli.repo_resolver import find_repo_root, repo_option
from osprey.cli.styles import Styles, data_table
from osprey.deployment.compose_generator import resolve_project_name, resolve_user_volume_names
from osprey.deployment.runtime_helper import get_runtime_command, runtime_env
from osprey.deployment.web_terminals.lifecycle import _require_running_runtime
from osprey.deployment.web_terminals.personas import as_dict, resolve_personas
from osprey.utils.config import load_project_config
from osprey.utils.logger import get_logger
from osprey.utils.workspace import agent_data_base_dir, anchored_path

logger = get_logger("feedback")

#: The four characters JSON counts as whitespace between documents. Anything
#: else between two records is a parse error, not a separator.
_JSON_WHITESPACE = " \t\n\r"

#: The store's two file families, mirroring the naming contract in
#: :mod:`osprey.interfaces.web_terminal.feedback_store`. They are spelled out
#: here rather than imported because the container reader has to hand them to
#: ``find`` inside the container as literal text anyway.
_HEADER_GLOB = "fb-*.json"
_CONTEXT_GLOB = "ctx-*.json"

#: The feedback store's directory, relative to the agent-data root.
_STORE_DIRNAME = "feedback"

#: Volume-name suffix of the volume a user's agent data lives on. Its sibling
#: ``-claude-config`` volume carries live session state and is never mounted.
_AGENT_DATA_SUFFIX = "-agent-data"

#: Where a volume is mounted inside the ephemeral reader container.
_MOUNT_DESTINATION = "/data"

#: The reader script's exit code for "this volume has no feedback store yet".
#: Benign: a user who has never submitted anything. Every other non-zero exit
#: -- including ``find``'s own read errors -- is a real failure and is reported
#: as one, never folded into this code.
_NO_STORE_EXIT = 3

#: Read size for a piped store read. Chunk boundaries are arbitrary as far as
#: :func:`iter_json_documents` is concerned.
_READ_CHUNK_BYTES = 65536

#: How long a reader container gets to exit once its output pipe has been
#: closed, before it is killed. Closing the pipe ends the read, so this only
#: bounds a child that ignores it -- reaching this timeout is a hung container,
#: not a slow store.
_CHILD_EXIT_TIMEOUT_SECONDS = 30

#: Header field a pruned record carries: its session context was deleted to
#: keep the store inside its size budget.
_PRUNED_FLAG = "context_pruned"

#: Export-only field marking a header whose context never arrived and that
#: carries no :data:`_PRUNED_FLAG` to explain the absence.
_MISSING_FLAG = "context_missing"


class FeedbackReadError(Exception):
    """A feedback store could not be read.

    Raised when the byte stream coming back from a store is not a run of JSON
    objects: invalid UTF-8, a malformed document, a top-level document that is
    not an object, or a stream that ends mid-document or mid-character.

    Every message names the byte offset the trouble starts at, counted from
    the first byte of the stream. A store is read by concatenating files, so
    that offset is what lets an operator find the record that went bad.
    """


def _byte_len(text: str) -> int:
    """Return the UTF-8 byte length of ``text``, without encoding pure ASCII.

    Offsets in this module are byte offsets into the stream while the parser
    works in decoded text, so every offset step is a conversion. The ASCII
    fast path keeps that from re-encoding the whole stream a second time.
    """
    return len(text) if text.isascii() else len(text.encode("utf-8"))


def iter_json_documents(chunks: Iterable[bytes]) -> Iterator[dict[str, Any]]:
    """Yield each JSON object from a stream of concatenated JSON documents.

    The stream is parsed incrementally: each document is yielded as soon as
    its last byte has been seen, and the bytes behind it are dropped, so a
    caller can write records out while the producer is still emitting them
    and neither side has to hold the whole store in memory.

    :param chunks: The stream's byte chunks, in order. Chunk boundaries are
        arbitrary -- one may fall inside a document, or inside a multi-byte
        UTF-8 character.
    :returns: Each top-level document, in stream order.
    :raises FeedbackReadError: On invalid UTF-8, a malformed document, a
        top-level document that is not an object (the store only ever holds
        objects), or a stream ending mid-document or mid-character. The
        message names the byte offset.

    A parse failure part-way through the stream is indistinguishable from a
    document that is merely still arriving -- ``{"a": tru`` and ``{"a": xyz``
    fail identically -- so a failure is treated as "not finished yet" until
    the stream ends, and is reported once there is no more input to wait for.
    Documents that parsed before the damage have already been yielded.
    """
    decoder = json.JSONDecoder()
    incremental = codecs.getincrementaldecoder("utf-8")()
    buffer = ""
    buffer_offset = 0  # Byte offset of buffer[0] within the stream.
    consumed_bytes = 0  # Bytes handed to the incremental decoder so far.

    def _decode(chunk: bytes, *, final: bool) -> str:
        """Decode one chunk, reporting a bad byte at its stream offset."""
        nonlocal consumed_bytes
        consumed_bytes += len(chunk)
        try:
            return incremental.decode(chunk, final)
        except UnicodeDecodeError as exc:
            # exc.object is the decoder's pending bytes plus this chunk, all of
            # which are already counted in consumed_bytes.
            offset = consumed_bytes - len(exc.object) + exc.start
            detail = "ends mid-character" if final else f"is not valid UTF-8: {exc.reason}"
            raise FeedbackReadError(
                f"Feedback stream {detail} (invalid UTF-8 at byte offset {offset})"
            ) from exc

    def _drain(*, final: bool) -> Iterator[dict[str, Any]]:
        """Yield every complete document now in the buffer, dropping each."""
        nonlocal buffer, buffer_offset
        while True:
            stripped = buffer.lstrip(_JSON_WHITESPACE)
            if len(stripped) != len(buffer):
                buffer_offset += _byte_len(buffer[: len(buffer) - len(stripped)])
                buffer = stripped
            if not buffer:
                return
            try:
                document, end = decoder.raw_decode(buffer)
            except ValueError as exc:
                if not final:
                    return  # Still arriving; wait for the next chunk.
                position = getattr(exc, "pos", 0)
                raise FeedbackReadError(
                    f"Feedback stream ends with an incomplete or malformed JSON document "
                    f"at byte offset {buffer_offset}: {getattr(exc, 'msg', str(exc))} "
                    f"(failing at byte offset {buffer_offset + _byte_len(buffer[:position])})"
                ) from exc
            if not isinstance(document, dict):
                raise FeedbackReadError(
                    f"Feedback stream holds a {type(document).__name__} document at byte "
                    f"offset {buffer_offset}, not a JSON object; a feedback store only "
                    f"ever holds objects"
                )
            buffer_offset += _byte_len(buffer[:end])
            buffer = buffer[end:]
            yield document

    for chunk in chunks:
        buffer += _decode(chunk, final=False)
        yield from _drain(final=False)
    buffer += _decode(b"", final=True)
    yield from _drain(final=True)


def _stream_chunks(stream: IO[bytes] | None) -> Iterator[bytes]:
    """Yield ``stream``'s bytes in fixed-size chunks until it is exhausted.

    :param stream: An open binary stream, or ``None`` (which yields nothing).
    """
    if stream is None:
        return
    while True:
        chunk = stream.read(_READ_CHUNK_BYTES)
        if not chunk:
            return
        yield chunk


@dataclass(frozen=True)
class VolumeTarget:
    """One agent-data volume this deployment can be asked to read.

    :ivar user: The web-terminal user the volume belongs to.
    :ivar volume: The volume's real runtime name, confirmed to exist.
    :ivar image: The container image the volume is read with.
    :ivar on_roster: ``False`` for a volume left behind by a user who is no
        longer in ``modules.web_terminals.users``.
    """

    user: str
    volume: str
    image: str
    on_roster: bool


@dataclass(frozen=True)
class VolumeError:
    """One volume that could not be read, and why.

    Reading is per-volume: one unreadable volume is reported and the rest of
    the deployment is still read, so a single broken user never hides everyone
    else's feedback.
    """

    user: str
    volume: str
    message: str


@dataclass(frozen=True)
class VolumeHeaders:
    """The record headers read off one user's agent-data volume."""

    user: str
    volume: str
    headers: list[dict[str, Any]]


@dataclass(frozen=True)
class VolumeReadResult:
    """Everything one pass over a deployment's feedback volumes produced.

    :ivar targets: The volumes that were mounted, in roster order followed by
        off-roster leftovers. Hand these to :func:`stream_volume_contexts` to
        read the matching session contexts without listing volumes twice.
    :ivar volumes: The headers each target held.
    :ivar errors: Volumes that could not be read, or roster users whose volume
        does not exist.
    """

    targets: list[VolumeTarget]
    volumes: list[VolumeHeaders]
    errors: list[VolumeError]


def _store_read_script(glob: str) -> str:
    """Build the in-container shell command that concatenates a store.

    ``find ... -exec cat {} +`` rather than a shell glob: a store with
    thousands of records would overflow the shell's argument list.
    """
    store = f"{_MOUNT_DESTINATION}/{_STORE_DIRNAME}"
    return (
        f"[ -d {store} ] || exit {_NO_STORE_EXIT}; "
        f'find {store} -maxdepth 1 -name "{glob}" -exec cat {{}} +'
    )


def _store_read_argv(runtime: str, target: VolumeTarget, glob: str) -> list[str]:
    """Build the argv that reads one volume's store in an ephemeral container.

    The mount is the long ``--mount`` form, not ``-v host:container``, whose
    colon-separated fields break on a name containing a colon, and it is
    ``readonly``: this command only ever reads.
    """
    return [
        runtime,
        "run",
        "--rm",
        "--mount",
        f"type=volume,source={target.volume},destination={_MOUNT_DESTINATION},readonly",
        target.image,
        "sh",
        "-c",
        _store_read_script(glob),
    ]


def _decode_stderr(raw: bytes | str | None) -> str:
    """Render a child process's error output as text fit for a message."""
    if not raw:
        return ""
    text = raw if isinstance(raw, str) else raw.decode("utf-8", errors="replace")
    return " ".join(text.split())


def _read_failed(runtime: str, target: VolumeTarget, code: int, raw: bytes | str | None) -> str:
    """Phrase a failed store read, naming the volume and the child's output."""
    message = (
        f"Could not read feedback from volume {target.volume!r} "
        f"(user {target.user!r}): `{runtime} run` exited {code}."
    )
    detail = _decode_stderr(raw)
    return f"{message} {detail}" if detail else message


def _list_agent_data_volumes(
    runtime: str, project: str, *, env: dict[str, str] | None
) -> dict[str, str]:
    """List this deployment's agent-data volumes as ``{user: volume_name}``.

    The label filter scopes the listing to exactly this compose project -- the
    same value :func:`osprey.deployment.runtime_helper.runtime_env` pins as
    ``COMPOSE_PROJECT_NAME`` -- so a sibling deployment's volumes can never be
    mistaken for this one's. ``-claude-config`` volumes are ignored: they hold
    live session state, not feedback.

    :raises RuntimeError: If the listing itself fails. An empty listing from a
        failed command would read as "this deployment has no volumes".
    """
    argv = [
        runtime,
        "volume",
        "ls",
        "--filter",
        f"label=com.docker.compose.project={project}",
        "--format",
        "{{.Name}}",
    ]
    logger.debug("Running command:\n    %s", " ".join(argv))
    result = subprocess.run(argv, capture_output=True, text=True, env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Could not list this deployment's web-terminal volumes: "
            f"`{runtime} volume ls` exited {result.returncode}. "
            f"{_decode_stderr(result.stderr)}".strip()
        )
    prefix = f"{project}_"
    found: dict[str, str] = {}
    for line in result.stdout.splitlines():
        name = line.strip()
        if not name.startswith(prefix):
            continue
        rest = name[len(prefix) :]
        if not rest.endswith(_AGENT_DATA_SUFFIX):
            continue
        user = rest[: -len(_AGENT_DATA_SUFFIX)]
        if user:
            found[user] = name
    return found


def _verified_volumes(
    config: dict[str, Any], *, runtime: str, env: dict[str, str] | None
) -> tuple[list[VolumeTarget], list[VolumeError]]:
    """Resolve which volumes to read, confirming each one exists first.

    Roster users' volume names are derived deterministically from the config,
    and every derived name is then asserted against the runtime's own listing
    before anything is mounted. That check is the whole point of this helper:
    ``--mount type=volume`` creates a volume that does not exist, so mounting
    an underived-but-unverified name would fabricate an empty store and leave
    a stray volume on the host instead of reporting the real state. A roster
    user with no volume is an error for that user alone.

    Volumes the listing turns up for users who are no longer on the roster are
    read too -- their feedback outlived their account -- using the first image
    the roster resolves, since a departed user's persona may be gone.

    :param config: The deployment's rendered config.
    :param runtime: The container runtime binary, ``docker`` or ``podman``.
    :param env: Environment for the runtime subprocess.
    :returns: ``(targets, errors)``.
    """
    project = resolve_project_name(config)
    listed = _list_agent_data_volumes(runtime, project, env=env)
    listed_names = set(listed.values())

    web_terminals = as_dict(as_dict(config.get("modules")).get("web_terminals"))
    facility_prefix = str(as_dict(config.get("facility")).get("prefix") or "")
    personas = resolve_personas(
        web_terminals, as_dict(config.get("registry")), facility_prefix, strict=False
    )

    targets: list[VolumeTarget] = []
    errors: list[VolumeError] = []
    roster_users: set[str] = set()
    for entry in personas:
        user = str(entry.get("name") or "")
        if not user:
            continue
        roster_users.add(user)
        _, volume = resolve_user_volume_names(config, user)
        image = str(entry.get("image") or "")
        if volume not in listed_names:
            errors.append(
                VolumeError(
                    user,
                    volume,
                    f"User {user!r} has no feedback volume: {volume!r} is not among the "
                    f"volumes {runtime} lists for this deployment. It is not being "
                    "mounted, because mounting a name that does not exist would create "
                    "an empty volume rather than read a store. A user's volume is "
                    "created when their web terminal container first starts — a "
                    "deployment that was never brought up with `osprey up` has none.",
                )
            )
            continue
        if not image:
            errors.append(
                VolumeError(
                    user,
                    volume,
                    f"No container image resolves for user {user!r}, so their feedback "
                    f"volume {volume!r} cannot be read.",
                )
            )
            continue
        targets.append(VolumeTarget(user=user, volume=volume, image=image, on_roster=True))

    fallback_image = next((str(entry["image"]) for entry in personas if entry.get("image")), "")
    for user, volume in sorted(listed.items()):
        if user in roster_users:
            continue
        if not fallback_image:
            errors.append(
                VolumeError(
                    user,
                    volume,
                    f"Volume {volume!r} holds feedback from {user!r}, who is no longer on "
                    "the roster, but this deployment resolves no container image to read "
                    "it with.",
                )
            )
            continue
        targets.append(
            VolumeTarget(user=user, volume=volume, image=fallback_image, on_roster=False)
        )
    return targets, errors


def _read_volume_headers(
    runtime: str, target: VolumeTarget, *, env: dict[str, str] | None
) -> tuple[list[dict[str, Any]], VolumeError | None]:
    """Read one volume's record headers in an ephemeral container."""
    argv = _store_read_argv(runtime, target, _HEADER_GLOB)
    logger.debug("Running command:\n    %s", " ".join(argv))
    result = subprocess.run(argv, capture_output=True, env=env, check=False)
    if result.returncode == _NO_STORE_EXIT:
        return [], None

    # Parse before judging the exit code: a read that failed part-way still
    # wrote out the records it managed to read, and those are real submissions.
    headers: list[dict[str, Any]] = []
    parse_failure: str | None = None
    try:
        # Appended one at a time on purpose: retaining what parsed before the
        # damage is the point, and it must not rest on how far a bulk extend
        # happens to get through an iterator that raises part-way.
        for document in iter_json_documents([result.stdout]):
            headers.append(document)
    except FeedbackReadError as exc:
        parse_failure = (
            f"The feedback store on volume {target.volume!r} (user {target.user!r}) "
            f"could not be read in full: {exc}"
        )

    if result.returncode != 0:
        # The exit code says more about why than a truncated stream does, so it
        # is the failure reported when both happened.
        return headers, VolumeError(
            target.user,
            target.volume,
            _read_failed(runtime, target, result.returncode, result.stderr),
        )
    if parse_failure is not None:
        return headers, VolumeError(target.user, target.volume, parse_failure)
    return headers, None


def read_volume_headers(
    config: dict[str, Any], *, env: dict[str, str] | None = None
) -> VolumeReadResult:
    """Read every feedback record header this deployment's volumes hold.

    One ephemeral read-only container per volume. Nothing is mounted until the
    runtime has confirmed the volume exists (see :func:`_verified_volumes`),
    and the ``-claude-config`` volumes are never mounted at all.

    :param config: The deployment's rendered config.
    :param env: Environment for the runtime subprocesses. Defaults to the
        deployment's own runtime environment.
    :returns: The headers per volume, the targets they were read from, and one
        entry in ``errors`` per volume that could not be read.
    :raises RuntimeError: If the container runtime daemon is unreachable, or
        the volume listing fails. Both would otherwise present as a deployment
        with no feedback in it.
    """
    _require_running_runtime(config)
    runtime = get_runtime_command(config)[0]
    run_env = runtime_env(config) if env is None else env
    targets, errors = _verified_volumes(config, runtime=runtime, env=run_env)

    volumes: list[VolumeHeaders] = []
    for target in targets:
        headers, error = _read_volume_headers(runtime, target, env=run_env)
        volumes.append(VolumeHeaders(user=target.user, volume=target.volume, headers=headers))
        if error is not None:
            errors.append(error)
    return VolumeReadResult(targets=targets, volumes=volumes, errors=errors)


def _raise_or_report(on_error: Callable[[VolumeError], None] | None, error: VolumeError) -> None:
    """Hand ``error`` to the caller's sink, or raise when there is none."""
    if on_error is None:
        raise FeedbackReadError(error.message)
    on_error(error)


def _stream_one_volume(
    runtime: str,
    target: VolumeTarget,
    *,
    env: dict[str, str] | None,
    on_error: Callable[[VolumeError], None] | None,
) -> Iterator[dict[str, Any]]:
    """Yield one volume's session-context documents as they arrive.

    The child's output is read straight off a pipe rather than collected, so a
    caller can write each paired record out while the container is still
    emitting -- one container start, and memory bounded by the largest single
    document. Its error output goes to a temporary file instead of a second
    pipe, so a chatty ``find`` can never fill a pipe buffer and deadlock the
    read.
    """
    argv = _store_read_argv(runtime, target, _CONTEXT_GLOB)
    logger.debug("Running command:\n    %s", " ".join(argv))
    with tempfile.TemporaryFile() as error_output:
        try:
            process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=error_output, env=env)
        except OSError as exc:
            _raise_or_report(
                on_error,
                VolumeError(
                    target.user,
                    target.volume,
                    f"Could not start `{runtime} run` to read the session contexts on "
                    f"volume {target.volume!r} (user {target.user!r}): {exc}",
                ),
            )
            return
        failure: str | None = None
        killed = False
        try:
            try:
                yield from iter_json_documents(_stream_chunks(process.stdout))
            except FeedbackReadError as exc:
                failure = (
                    f"The session contexts on volume {target.volume!r} (user "
                    f"{target.user!r}) could not be read in full: {exc}"
                )
        finally:
            # Closing the pipe is what tells a still-running child to stop, so
            # the wait that follows is bounded rather than open-ended: a child
            # that ignores the broken pipe is killed instead of hanging the CLI.
            if process.stdout is not None:
                process.stdout.close()
            try:
                returncode = process.wait(timeout=_CHILD_EXIT_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                logger.info("Reader container for %s outlived its read; killing it", target.volume)
                process.kill()
                killed = True
                returncode = process.wait()
        # A kill this function issued is not a failed read: the stream reached
        # its end, and the signal in the exit status is this code's own doing.
        # Reporting it would tell an operator their store is unreadable when
        # every record already came back.
        #
        # Where a parse failure and a bad exit status coincide, this reader
        # reports the parse failure while :func:`_read_volume_headers` reports
        # the exit code. That difference is deliberate: a streamed read has
        # already handed the caller every document before the damage, so the
        # byte offset naming where the stream broke is the more useful of the
        # two, while a batch read returns a list either way and the exit code
        # is the better account of why the read stopped.
        if failure is None and not killed and returncode not in (0, _NO_STORE_EXIT):
            error_output.seek(0)
            failure = _read_failed(runtime, target, returncode, error_output.read())
        if failure is not None:
            _raise_or_report(on_error, VolumeError(target.user, target.volume, failure))


def stream_volume_contexts(
    config: dict[str, Any],
    targets: Iterable[VolumeTarget],
    *,
    env: dict[str, str] | None = None,
    on_error: Callable[[VolumeError], None] | None = None,
) -> Iterator[dict[str, Any]]:
    """Stream the session-context documents behind a deployment's records.

    ``targets`` are the volumes :func:`read_volume_headers` already verified
    and mounted, so this reads no listing of its own and repeats no preflight.

    :param config: The deployment's rendered config.
    :param targets: Volumes to read, in the order they should be read.
    :param env: Environment for the runtime subprocesses. Defaults to the
        deployment's own runtime environment.
    :param on_error: Called with each volume that could not be read, so the
        rest of the export still runs. When it is ``None``, such a failure
        raises :class:`FeedbackReadError` instead of passing unnoticed.
    :returns: Each context document, in target order.
    """
    runtime = get_runtime_command(config)[0]
    run_env = runtime_env(config) if env is None else env
    for target in targets:
        yield from _stream_one_volume(runtime, target, env=run_env, on_error=on_error)


def local_feedback_dir(config: dict[str, Any] | None, repo_root: Path | str) -> Path:
    """Locate a deployment's feedback store on the host filesystem.

    This is how a single-user or development deployment is read: its agent data
    lives in a directory rather than a volume, so no container is involved. The
    directory is the one ``agent_data.base_dir`` names, anchored at the
    deployment repo -- not at the current directory -- so the answer is the same
    wherever the command was run from.

    :param config: The deployment's rendered config.
    :param repo_root: The deployment repo the relative form is anchored at.
    :returns: The store directory, which need not exist.
    :raises click.ClickException: If ``agent_data.base_dir`` starts with ``~``.
        Such a path is expanded against the *container's* home directory, which
        is on the claude-config volume -- the one volume these readers never
        mount -- so the records are somewhere this command cannot reach, and
        saying so beats reading an empty directory and reporting no feedback.
    """
    base = agent_data_base_dir(config)
    if base.startswith("~"):
        raise click.ClickException(
            f"This deployment's agent_data.base_dir is {base!r}, and feedback cannot be "
            "read from a home-relative path. Inside a web terminal, '~' expands against "
            "the container's home directory, which lives on that user's claude-config "
            "volume rather than their agent-data one, so the records are not where this "
            "command reads. Give agent_data.base_dir a path relative to the deployment "
            "repo (for example 'var/agent_data') or an absolute one, then rebuild."
        )
    return anchored_path(base, Path(repo_root).resolve()).resolve() / _STORE_DIRNAME


def _local_file_chunks(directory: Path, glob: str) -> Iterator[bytes]:
    """Stream the store's matching files as one run of concatenated bytes.

    The container reader gets exactly this shape out of ``find ... -exec cat``,
    so both readers hand :func:`iter_json_documents` the same thing. A store
    that does not exist yields nothing: a deployment nobody has sent feedback
    from has no store directory, which is zero records rather than an error.

    A file that disappears between the listing and the open is skipped rather
    than raised: the store prunes old session contexts as new submissions
    arrive, so a record vanishing mid-read is routine and says nothing about
    the records still there. A header left without its context that way is
    reported downstream, by :func:`pair_records`, as a record whose context is
    gone -- which is exactly what happened. Any other read failure (a
    permission problem, a bad disk) is still raised.
    """
    if not directory.is_dir():
        logger.debug("No feedback store at %s", directory)
        return
    for path in sorted(directory.glob(glob)):
        try:
            with path.open("rb") as handle:
                yield from _stream_chunks(handle)
        except FileNotFoundError:
            logger.debug("Feedback record %s vanished mid-read (pruned)", path)
            continue
        except OSError as exc:
            raise FeedbackReadError(f"Could not read feedback file {path}: {exc}") from exc


def read_local_headers(directory: Path) -> list[dict[str, Any]]:
    """Read every record header from a store on the host filesystem.

    :param directory: The store directory, from :func:`local_feedback_dir`.
    :returns: Each header, ordered by filename -- which, because a record id
        begins with its timestamp, is submission order.
    :raises FeedbackReadError: If the store does not parse.
    """
    return list(iter_json_documents(_local_file_chunks(directory, _HEADER_GLOB)))


def stream_local_contexts(directory: Path) -> Iterator[dict[str, Any]]:
    """Stream the session contexts from a store on the host filesystem.

    Incremental, like the container reader: each document is yielded as soon as
    it is complete, so an export writes records out while the store is still
    being read.

    :param directory: The store directory, from :func:`local_feedback_dir`.
    :returns: Each context document, ordered by filename.
    :raises FeedbackReadError: If the store does not parse.
    """
    return iter_json_documents(_local_file_chunks(directory, _CONTEXT_GLOB))


@dataclass(frozen=True)
class PairingSummary:
    """What one pairing pass did, for the caller to report afterwards.

    :ivar paired: Records exported with their session context attached.
    :ivar pruned: Headers exported without one, where the header itself says
        the context was pruned to keep the store inside its size budget.
    :ivar missing: Headers exported without one and without that explanation.
    :ivar orphans: Context documents dropped because no header claimed them --
        the header is written last, so a crash mid-submission leaves one.
    """

    paired: int
    pruned: int
    missing: int
    orphans: int


def _merged_record(header: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Fold a context document into its header, header fields winning.

    The context's own ``id`` is what matched the two, and is already the
    header's; every other field it carries joins the record under its own name.
    The header is the record's metadata, so on the (contract-breaking) chance
    that both documents carry the same field, the header's value stands.
    """
    record = dict(header)
    for key, value in context.items():
        if key != "id" and key not in record:
            record[key] = value
    return record


def pair_records(
    headers: Iterable[dict[str, Any]],
    contexts: Iterable[dict[str, Any]],
    sink: Callable[[dict[str, Any]], None],
) -> PairingSummary:
    """Join each record header to its session context and emit the result.

    Pairing is by the ``id`` carried *inside* both documents: a store is read
    by concatenating files, so filenames do not survive the trip. Contexts are
    consumed as they arrive and each completed record goes straight to
    ``sink``, so an export writes incrementally and holds only the headers in
    memory -- never the session bodies, which are the large part.

    A context no header claims is dropped: the header is written last, so an
    interrupted submission leaves a context behind with nothing to describe it.
    Once the contexts run out, the headers that never found one are emitted
    anyway, flagged, because the submission itself is still worth reading.

    That last step also runs when the context stream *fails* part-way, before
    the failure is passed on. A submission whose header was read is a real
    submission whether or not its session could be fetched, and an export that
    dropped every unpaired header on the way out would hand back a document
    holding less than the run had already read.

    :param headers: Every record header in the store.
    :param contexts: The context documents, in any order, ideally streaming.
    :param sink: Called with each finished record, in completion order.
    :returns: The counts, for a summary line after the export.
    :raises FeedbackReadError: Whatever ``contexts`` raises, re-raised after
        the unpaired headers have been emitted.
    """
    pending: dict[str, dict[str, Any]] = {}
    unpairable: list[dict[str, Any]] = []
    for header in headers:
        record_id = header.get("id")
        if isinstance(record_id, str) and record_id:
            pending[record_id] = header
        else:
            unpairable.append(header)

    paired = 0
    orphans = 0
    pruned = 0
    missing = 0

    def _emit_unpaired() -> None:
        """Write out every header no context arrived for, flagged."""
        nonlocal pruned, missing
        for header in [*pending.values(), *unpairable]:
            record = dict(header)
            if record.get(_PRUNED_FLAG):
                pruned += 1
            else:
                record[_MISSING_FLAG] = True
                missing += 1
            sink(record)
        pending.clear()
        unpairable.clear()

    try:
        for context in contexts:
            context_id = context.get("id")
            header = pending.pop(context_id, None) if isinstance(context_id, str) else None
            if header is None:
                orphans += 1
                logger.debug("Dropped feedback context %r: no header carries that id", context_id)
                continue
            sink(_merged_record(header, context))
            paired += 1
    except BaseException:
        # Best-effort on the way out: the failure already travelling is the one
        # worth reporting, and a sink that fails here -- the same closed pipe,
        # the same full disk -- must not take its place and leave the real
        # trouble reported nowhere.
        with suppress(Exception):
            _emit_unpaired()
        raise
    _emit_unpaired()
    return PairingSummary(paired=paired, pruned=pruned, missing=missing, orphans=orphans)


# ---------------------------------------------------------------------------
# The ``osprey feedback`` verbs
# ---------------------------------------------------------------------------

#: The rendered config every verb reads, relative to the deployment repo root.
#: ``build/`` holds the rendered project, so this is the file the running stack
#: was started from.
_RENDERED_CONFIG = "config.yml"

#: How much of a submission's excerpt the listing shows before cutting it.
_EXCERPT_WIDTH = 56

#: Stands in for a table cell the record does not fill in.
_UNKNOWN = "-"


class _Deployment(NamedTuple):
    """The deployment one verb reads: its repo root and its rendered config."""

    root: Path
    config: dict[str, Any]


@dataclass(frozen=True)
class _Record:
    """One submission header, with the user whose store it came out of.

    ``user`` is empty for a single-user deployment, whose one store belongs to
    nobody in particular.
    """

    user: str
    header: dict[str, Any]


@dataclass(frozen=True)
class _Store:
    """What one pass over a deployment's feedback turned up.

    :ivar records: Every header that could be read.
    :ivar errors: Workspaces that could not be read.
    :ivar targets: The verified volumes the headers came from, empty for a
        single-user deployment. Carried so an export streams the matching
        session contexts off the same volumes without listing them again.
    """

    records: list[_Record]
    errors: list[VolumeError]
    targets: list[VolumeTarget]


def _rendered_config_path(repo_root: Path) -> Path:
    """Where the running stack's config lives in a deployment repo."""
    from osprey.deployment.staleness import BUILD_DIRNAME

    return repo_root / BUILD_DIRNAME / _RENDERED_CONFIG


@contextmanager
def _feedback_session(repo: Path | None, *, announce: bool = True) -> Iterator[_Deployment]:
    """Resolve the deployment repo, then run one feedback verb against it.

    Unlike the roster verbs, this does not change into the repo root: both
    readers are handed the repo root and the config they act on, and the
    container runtime's environment is pinned from that config rather than
    from the working directory.

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
    repo_root = find_repo_root(repo)
    config_path = _rendered_config_path(repo_root)

    if not config_path.is_file():
        fail(
            f"No build found in {repo_root}",
            f"{config_path} does not exist. Feedback is read from the deployment "
            "the rendered config describes.",
            "run `osprey build` first",
        )
        raise click.Abort()

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


def _has_roster(config: dict[str, Any]) -> bool:
    """Report whether this deployment runs a web terminal per person.

    That is what decides where feedback lives: a roster means one store per
    user, each on that user's own volume, and no roster means one store in
    this deployment's agent-data directory on the host.
    """
    web_terminals = as_dict(as_dict(config.get("modules")).get("web_terminals"))
    roster = web_terminals.get("users")
    return isinstance(roster, list) and bool(roster)


def _read_store(deployment: _Deployment) -> _Store:
    """Read every submission header this deployment holds."""
    config = deployment.config
    if _has_roster(config):
        result = read_volume_headers(config)
        return _Store(
            records=[
                _Record(user=volume.user, header=header)
                for volume in result.volumes
                for header in volume.headers
            ],
            errors=result.errors,
            targets=result.targets,
        )
    headers = read_local_headers(local_feedback_dir(config, deployment.root))
    return _Store(
        records=[_Record(user="", header=header) for header in headers],
        errors=[],
        targets=[],
    )


def _absent_workspaces(store: _Store) -> list[VolumeError]:
    """The failures that left nothing behind: no header of theirs was read.

    A failed read is not the same as an empty one. The container reader keeps
    whatever parsed before a read died, so a workspace can fail *and* still
    have contributed records -- and those records' session contexts live in a
    second read that has every chance of succeeding.
    """
    read_users = {record.user for record in store.records}
    return [error for error in store.errors if error.user not in read_users]


def _partial_workspaces(store: _Store) -> list[VolumeError]:
    """The failures that still gave up records -- read in part, not lost."""
    read_users = {record.user for record in store.records}
    return [error for error in store.errors if error.user in read_users]


def _stream_contexts(
    deployment: _Deployment, store: _Store, on_error: Callable[[VolumeError], None]
) -> Iterator[dict[str, Any]]:
    """Stream the session contexts behind ``store``'s headers.

    A workspace that failed *and* produced no headers is skipped rather than
    opened again: a second container against it would spend a container start
    to produce the same error twice. One that failed part-way through is still
    read, because the records it did give up have contexts worth attaching.
    """
    if not _has_roster(deployment.config):
        return stream_local_contexts(local_feedback_dir(deployment.config, deployment.root))
    lost = {error.volume for error in _absent_workspaces(store)}
    targets = [target for target in store.targets if target.volume not in lost]
    return stream_volume_contexts(deployment.config, targets, on_error=on_error)


def _report_read_errors(errors: Iterable[VolumeError]) -> None:
    """Report each workspace that could not be read, one line apiece."""
    for error in errors:
        warn(f"No feedback read for {error.user}", error.message)


def _require_something_read(store: _Store) -> None:
    """Refuse to report an empty deployment when every read failed.

    "No feedback" and "nothing could be read" look identical from the outside
    and mean opposite things, so a run that read nothing at all stops here.
    """
    if store.errors and not store.records:
        fail(
            "Could not read any feedback",
            "Every workspace in this deployment failed to read, so whether there "
            "is feedback in it is unknown.",
            "fix the errors above, then run the command again",
        )
        raise click.Abort()


def _record_millis(record_id: Any) -> int | None:
    """Read the UTC milliseconds a record id starts with, if it has any.

    ``str.isdigit`` is true for characters ``int`` will not accept -- ``"²"``
    among them -- so the ASCII check is what keeps one malformed id in a store
    from raising out of the listing and taking every other record with it.
    """
    if not isinstance(record_id, str):
        return None
    parts = record_id.split("-")
    if len(parts) < 3 or not (parts[1].isascii() and parts[1].isdigit()):
        return None
    return int(parts[1])


def _submitted_moment(header: dict[str, Any]) -> dt.datetime | None:
    """When a submission was made, in UTC, from whichever source it has.

    Two sources, in order. ``created_at`` is the stamp the web terminal writes
    into every header. Failing that -- an unreadable stamp, or a record from
    before the field existed -- the record id carries the same instant, because
    the store names every record ``fb-<utc-ms>-<random>``.

    :returns: The instant, or ``None`` when neither source can be read.
    """
    raw = header.get("created_at")
    if isinstance(raw, str) and raw.strip():
        with suppress(ValueError):
            stamped = dt.datetime.fromisoformat(raw.strip())
            return stamped if stamped.tzinfo is not None else stamped.replace(tzinfo=dt.UTC)
    millis = _record_millis(header.get("id"))
    if millis is None:
        return None
    try:
        return dt.datetime.fromtimestamp(millis / 1000, dt.UTC)
    except (OSError, OverflowError, ValueError):
        return None


def _submitted_at(header: dict[str, Any]) -> str:
    """Render the UTC minute a submission was made in, for one table row."""
    moment = _submitted_moment(header)
    if moment is None:
        return _UNKNOWN
    return moment.astimezone(dt.UTC).strftime("%Y-%m-%d %H:%M")


def _record_user(record: _Record) -> str:
    """Name who sent a submission: the store's owner, else what it claims."""
    if record.user:
        return record.user
    identity = record.header.get("identity")
    if isinstance(identity, str) and identity.strip():
        return identity.strip()
    if isinstance(identity, dict):
        for key in ("name", "user", "login", "email"):
            value = identity.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
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


#: Sorts before every real submission, so a record with no readable time ends
#: up at the bottom of the listing instead of the top.
_UNDATED = dt.datetime.min.replace(tzinfo=dt.UTC)


def _newest_first(records: Iterable[_Record]) -> list[_Record]:
    """Order records by submission time, most recent first."""
    return sorted(
        records,
        key=lambda record: (
            _submitted_moment(record.header) or _UNDATED,
            str(record.header.get("id") or ""),
        ),
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
            Text(str(header.get("id") or _UNKNOWN)),
            Text(_record_user(record)),
            Text(_submitted_at(header)),
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

    def __init__(self, write: Callable[[str], None]) -> None:
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
def _export_sink(output_path: Path | None) -> Iterator[Callable[[str], None]]:
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


def _export_summary(
    summary: PairingSummary,
    count: int,
    output_path: Path | None,
    absent: Iterable[str] = (),
    partial: Iterable[str] = (),
) -> str:
    """Phrase what the export wrote, for the line that closes the run.

    :param absent: Users whose workspace gave up nothing at all.
    :param partial: Users whose workspace failed but still gave up records.
    """
    where = f"to {output_path}" if output_path is not None else "to standard output"
    parts = [f"{summary.paired} with session context"]
    if summary.pruned:
        parts.append(f"{summary.pruned} whose context was dropped for space")
    if summary.missing:
        parts.append(f"{summary.missing} with no context")
    line = f"Exported {_plural(count, 'feedback record')} {where} ({', '.join(parts)})."
    if summary.orphans:
        line += (
            f" {_plural(summary.orphans, 'stored context')} had no matching submission "
            "in this read and were skipped."
        )
    missing_users = sorted(set(absent))
    if missing_users:
        line += (
            f" {_plural(len(missing_users), 'workspace')} could not be read and "
            f"{'is' if len(missing_users) == 1 else 'are'} not in this export: "
            f"{', '.join(missing_users)}."
        )
    partial_users = sorted(set(partial) - set(missing_users))
    if partial_users:
        line += (
            f" {_plural(len(partial_users), 'workspace')} "
            f"{'was' if len(partial_users) == 1 else 'were'} only partly readable: "
            f"{', '.join(partial_users)}."
        )
    return line


def _export_store(deployment: _Deployment, output_path: Path | None) -> None:
    """Read, pair and write out every submission, reporting what happened."""
    store = _read_store(deployment)
    _report_read_errors(store.errors)
    _require_something_read(store)

    stream_errors: list[VolumeError] = []
    contexts = _stream_contexts(deployment, store, stream_errors.append)
    read_failure: str | None = None
    summary: PairingSummary | None = None
    with _export_sink(output_path) as write:
        writer = _JsonArrayWriter(write)
        try:
            summary = pair_records([record.header for record in store.records], contexts, writer)
        except FeedbackReadError as exc:
            read_failure = str(exc)
        except BrokenPipeError:
            # Whoever was reading stdout stopped reading -- `| head` is the
            # ordinary way that happens. The export did its job for as long as
            # there was somebody to write to; there is nothing to report.
            logger.debug("Export ended early: the reader closed the pipe")
            _silence_stdout()
            return
        finally:
            # Closed either way, so a store that stopped being readable still
            # leaves a whole JSON document holding what was read.
            writer.close()

    _report_read_errors(stream_errors)
    if read_failure is not None or summary is None:
        fail(
            "The export stopped part-way",
            f"{_plural(writer.count, 'record')} {'was' if writer.count == 1 else 'were'} "
            f"written before the store stopped being readable. {read_failure}",
            "the export holds what was read; check the store, then run again",
        )
        raise click.Abort()
    report(
        _export_summary(
            summary,
            writer.count,
            output_path,
            absent=[error.user for error in _absent_workspaces(store)],
            partial=[error.user for error in [*_partial_workspaces(store), *stream_errors]],
        )
    )


@click.group()
def feedback() -> None:
    """Read the feedback this deployment's web terminals collected.

    People using a web terminal can send feedback from the terminal itself.
    Each submission is stored together with the session it was sent from, in
    the sender's own workspace, and these verbs are how you read it back:
    'list' for one line per submission, 'export' for the whole record
    including that session.

    A multi-user deployment keeps one store per person. Reading it starts a
    short-lived container on each person's workspace, so the deployment's
    container runtime has to be up. A single-user deployment keeps one store
    on this machine, read straight off disk.

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

    A workspace that cannot be read is reported and the rest of the deployment
    is still listed. If nothing at all could be read, this stops instead of
    reporting a deployment with no feedback in it.

    Examples:

    \b
      $ osprey feedback list
      $ osprey feedback list --repo ../other-deployment
    """
    with _feedback_session(repo) as deployment:
        store = _read_store(deployment)
        _report_read_errors(store.errors)
        _require_something_read(store)
        if not store.records:
            note("No feedback has been sent from this deployment yet.")
            return
        _render_records(_newest_first(store.records))


def _file_to_write(ctx: click.Context, param: click.Parameter, value: Path | None) -> Path | None:
    """Refuse an ``--output`` that nothing could be written to.

    Click's own checks read the text the operator typed and only stat a path
    that already exists, so two ordinary mistakes reach the command: an empty
    ``--output ""``, which arrives here as ``.``, and a file named inside a
    directory that is not there. Caught at parse time, the run stops before a
    single reader container has started; caught at the write, it would stop
    after the whole store had been read.
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
    """Export every submission with its session, as one JSON array.

    Each object is one submission's whole record: who sent it, when, from
    which session and which version of the app, how it went out and how large
    it was, whatever the submission was truncated at -- and the session that
    was captured with it. Records are written as they are read, so a large
    store never has to fit in memory.

    With no --output the export goes to stdout and nothing else does, so it
    can be redirected straight into a file; errors and the closing summary go
    to standard error. With --output they all go to the terminal and the file
    holds the export.

    A workspace that cannot be read is reported and named in the closing
    summary: as left out of the export when nothing of it could be read, and
    as only partly readable when it gave up some of its submissions before
    failing. The rest of the deployment is written either way. If nothing at
    all could be read, this stops instead of writing an empty export.

    A submission whose session was dropped to keep the store inside its size
    limit carries 'context_pruned'; one missing its session for any other
    reason is marked 'context_missing'. The submission itself is exported
    either way.

    Examples:

    \b
      $ osprey feedback export --output feedback.json
      $ osprey feedback export > feedback.json
      $ osprey feedback export --repo ../other-deployment -o out.json
    """
    # Resolved before the verb runs: a relative path means what the operator
    # typed it against. Click has already refused a directory or an
    # unwritable destination by here, so no reader container is ever started
    # for an export that had nowhere to go.
    output_path = output.resolve() if output is not None else None

    # No banner when stdout is the document, and machine mode with it, so a
    # line printed from anywhere in the stack lands on standard error rather
    # than in the middle of the JSON.
    with _feedback_session(repo, announce=output_path is not None) as deployment:
        with machine_mode() if output_path is None else nullcontext():
            _export_store(deployment, output_path)
