"""Files the operator attaches to a chat message, kept for the agent to read.

The composer uploads each file as the operator picks it, before the message is
sent. The message then names the stored paths, and the agent opens them with its
Read tool. Two routes::

    POST   /api/chat/{chat_id}/attachments
        Body: the file's bytes. ``X-Osprey-Attachment-Name`` carries the file's
        name, percent-encoded; ``X-Osprey-Attachment-Batch`` names the message
        the file belongs to. Answers 201 with ``{name, path, size, type}``.
        ``path`` is what the agent is given; ``type`` is what the content was
        found to be, which is not always what the name says.

    DELETE /api/chat/{chat_id}/attachments/{stored}
        Removes a file the operator took back before sending. 204 whether or
        not it was still there.

A refusal answers ``{"detail": {"error": <slug>, "message": <text>}}``, the shape
the other chat routes use:

* 400 ``invalid_chat_id``, ``invalid_batch``, ``invalid_name``, ``empty_file``
* 413 ``file_too_large``, ``chat_full``, ``uploads_full``
* 415 ``unsupported_type``
* 422 ``too_many_files``
* 503 ``store_write_failed``

Files are stored under ``<agent-data root>/uploads/<chat_id>/`` as
``<16 hex>-<name>``, the name reduced to ``[A-Za-z0-9._-]``. Nothing serves them
back over HTTP. The gate in front of this router authenticates the caller and
refuses a cross-origin write (:class:`~osprey.interfaces.common_middleware.WebAuthMiddleware`);
this module adds no credential of its own.

A file that was sent with a message is kept: the conversation names its path,
and a replay or a later turn may read it again. One that was never sent is a
leftover, and is deleted once it is :data:`UNSENT_RETENTION_S` old. Which is
which is recorded per chat in a manifest (:data:`SENT_MANIFEST`, a dotfile
beside the uploads) that :func:`mark_sent` writes when ``POST /api/chat``
accepts a prompt naming the file. One document per chat rather than a marker
per file: a message's files are marked together in one atomic write, so a
crash never leaves a message half-marked, and the sweep reads one small file
per chat instead of a marker beside every upload. The sweep
(:func:`sweep_uploads`) runs at startup and before an upload, at most once an
hour; there is no timer. It stays inside the uploads root, never follows a
symlink, and never deletes a sent file.
"""

from __future__ import annotations

import asyncio
import codecs
import contextlib
import logging
import os
import re
import secrets
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, overload
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from starlette.requests import ClientDisconnect

from osprey.interfaces.web_terminal._json_store import read_json_object, write_json_atomic
from osprey.interfaces.web_terminal.operator_session import resolve_agent_data_root
from osprey.interfaces.web_terminal.session_key import is_posture_key
from osprey.utils.workspace import CHAT_UPLOADS_SUBDIR

if TYPE_CHECKING:
    from typing import BinaryIO

logger = logging.getLogger(__name__)
router = APIRouter()

#: The largest single file accepted, in bytes.
MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024
#: The files one message may carry.
MAX_FILES_PER_MESSAGE = 10
#: The bytes one chat's uploads may hold, counting uploads still streaming in.
MAX_CHAT_ATTACHMENT_BYTES = 200 * 1024 * 1024
#: The bytes every chat's uploads may hold together. Chat ids are minted by the
#: client, so without this bound the per-chat cap would bound nothing.
MAX_UPLOADS_BYTES = 2 * 1024 * 1024 * 1024
#: How long a file that was never sent with a message is kept.
UNSENT_RETENTION_S = 24 * 60 * 60
#: How long after one sweep the next upload may run another.
_SWEEP_INTERVAL_S = 60 * 60
#: The per-chat record of the stored names that went out with a message.
SENT_MANIFEST = ".sent.json"
#: The line the composer writes for each attached file, under its header.
_ATTACHMENT_LINE = re.compile(r"^- (?P<path>\S+) \(")

#: The directory under the agent-data root that holds every chat's uploads.
UPLOADS_DIR = CHAT_UPLOADS_SUBDIR
#: The request headers that name the file and the message it belongs to.
NAME_HEADER = "x-osprey-attachment-name"
BATCH_HEADER = "x-osprey-attachment-batch"

_READ_CHUNK_BYTES = 64 * 1024
#: The longest name stem kept. The extension is always kept.
_MAX_STEM_CHARS = 100
_MAX_NAME_HEADER_CHARS = 1024
#: The random prefix, a dash, then a sanitised name of at most 130 characters that
#: does not start with a dot. The same grammar :func:`_name_parts` stores under.
_STORED_NAME = re.compile(r"[0-9a-f]{16}-[A-Za-z0-9_-][A-Za-z0-9._-]{0,129}")
_BATCH_ID = re.compile(r"[A-Za-z0-9-]{8,64}")
_SEPARATORS = re.compile(r"[\\/]")
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]")

#: Extension (lower-case, no dot) -> (the kind of content the name claims, the
#: MIME type a file of that name is typed as). The kind decides how the content
#: is checked. For an image the type is the one the content proves, so a ``.jpg``
#: holding PNG bytes is typed ``image/png``.
_ALLOWED: dict[str, tuple[str, str]] = {
    "png": ("image", "image/png"),
    "jpg": ("image", "image/jpeg"),
    "jpeg": ("image", "image/jpeg"),
    "gif": ("image", "image/gif"),
    "webp": ("image", "image/webp"),
    "pdf": ("pdf", "application/pdf"),
    "txt": ("text", "text/plain"),
    "log": ("text", "text/plain"),
    "csv": ("text", "text/csv"),
    "json": ("text", "application/json"),
    "yaml": ("text", "application/yaml"),
    "yml": ("text", "application/yaml"),
    "md": ("text", "text/markdown"),
    "py": ("text", "text/x-python"),
}

_IMAGE_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)

#: How a document that opens as markup begins. A web page is not a log, whatever
#: its name says, so text that opens this way is refused.
_MARKUP_OPENINGS = (
    "<!doctype",
    "<html",
    "<head",
    "<body",
    "<script",
    "<style",
    "<svg",
    "<iframe",
    "<meta",
    "<title",
    "<?xml",
)

_MB = 1024 * 1024
_CHAT_REFUSED = "This chat cannot take attachments."
_BATCH_REFUSED = "The upload names no message to file it under."
_NAME_REFUSED = "The file name cannot be used."
_TYPE_REFUSED = "This file type cannot be attached. Images, PDFs and plain-text files can."
_SIZE_REFUSED = f"The file is larger than {MAX_ATTACHMENT_BYTES // _MB} MB."
_CHAT_FULL_REFUSED = "This chat has reached its limit for attachments."
_UPLOADS_FULL_REFUSED = "The server has reached its limit for attachments."
_COUNT_REFUSED = f"A message can carry {MAX_FILES_PER_MESSAGE} files."
_EMPTY_REFUSED = "The file is empty."
_STORE_REFUSED = "The file could not be saved."


def _refused(status: int, error: str, message: str) -> HTTPException:
    """The refusal every route here answers with: a status, a slug and a sentence."""
    return HTTPException(status_code=status, detail={"error": error, "message": message})


class _Ledger:
    """This process's bookkeeping for the attachment store of one app.

    ``chat_bytes`` holds, per chat, the bytes its uploads occupy: the files on
    disk plus the bytes of uploads still streaming in. Counting the streaming
    bytes is what stops two uploads racing each other from both passing the
    chat's cap. A chat is measured from disk the first time this process sees
    it, and ``total_bytes`` is every chat's uploads together, measured the same
    way once. ``message_files`` counts the files of each message, keyed by
    ``(chat_id, batch)``; the least recently used entries are dropped past a
    bound, so a long-running server does not grow without limit.
    """

    _MESSAGES_KEPT = 512

    def __init__(self) -> None:
        self.chat_bytes: dict[str, int] = {}
        self.total_bytes: int | None = None
        self.message_files: OrderedDict[tuple[str, str], int] = OrderedDict()
        #: When the last sweep ran (:func:`time.monotonic`), or ``None`` for never.
        self.swept_at: float | None = None

    def message_count(self, message: tuple[str, str]) -> int:
        return self.message_files.get(message, 0)

    def add_file(self, message: tuple[str, str]) -> None:
        self.message_files[message] = self.message_files.get(message, 0) + 1
        self.message_files.move_to_end(message)
        while len(self.message_files) > self._MESSAGES_KEPT:
            self.message_files.popitem(last=False)

    def drop_file(self, message: tuple[str, str]) -> None:
        remaining = self.message_files.get(message, 0) - 1
        if remaining > 0:
            self.message_files[message] = remaining
        else:
            self.message_files.pop(message, None)

    def add_bytes(self, chat_id: str, count: int) -> None:
        """Count *count* bytes (negative to uncount) against the chat and the whole tree."""
        self.chat_bytes[chat_id] += count
        if self.total_bytes is not None:
            self.total_bytes += count

    def uncount(self, chat_id: str, size: int) -> None:
        """Take a deleted file's *size* off whatever has been measured so far."""
        if chat_id in self.chat_bytes:
            self.add_bytes(chat_id, -size)
        elif self.total_bytes is not None:
            self.total_bytes -= size


def _ledger(app: Any) -> _Ledger:
    """The app's ledger, created on first use."""
    ledger = getattr(app.state, "chat_attachment_ledger", None)
    if ledger is None:
        ledger = _Ledger()
        app.state.chat_attachment_ledger = ledger
    return ledger


def _measure(chat_dir: Path) -> int:
    """The bytes of the files stored in *chat_dir*. Dotfiles are uploads still being written."""
    if not chat_dir.is_dir():
        return 0
    total = 0
    for entry in chat_dir.iterdir():
        if entry.name.startswith("."):
            continue
        with contextlib.suppress(OSError):
            total += entry.stat().st_size
    return total


def _chat_total(ledger: _Ledger, chat_id: str, chat_dir: Path) -> int:
    """The bytes *chat_id*'s uploads hold, measured from disk the first time it is asked for."""
    total = ledger.chat_bytes.get(chat_id)
    if total is None:
        total = _measure(chat_dir)
        ledger.chat_bytes[chat_id] = total
    return total


def _uploads_total(ledger: _Ledger, uploads_dir: Path) -> int:
    """The bytes every chat's uploads hold, measured from disk the first time it is asked for."""
    if ledger.total_bytes is None:
        total = 0
        if uploads_dir.is_dir():
            for chat_dir in uploads_dir.iterdir():
                total += _measure(chat_dir)
        ledger.total_bytes = total
    return ledger.total_bytes


def _uploads_dir(request: Request) -> Path:
    """The directory holding every chat's uploads, under the app's agent-data root."""
    return Path(resolve_agent_data_root(request.app)) / UPLOADS_DIR


def _sent_names(chat_dir: Path) -> set[str]:
    """The stored names *chat_dir*'s manifest says went out with a message."""
    document = read_json_object(chat_dir / SENT_MANIFEST) or {}
    names = document.get("sent")
    if not isinstance(names, list):
        return set()
    return {name for name in names if isinstance(name, str)}


def _attached_names(chat_dir: Path, prompt: str) -> set[str]:
    """The stored names under *chat_dir* that *prompt*'s attachment block names.

    Only a path inside this chat's own directory, with a name of the stored
    grammar, counts: a prompt that names another chat's file, or any other
    path, marks nothing.
    """
    names = set()
    for line in prompt.splitlines():
        match = _ATTACHMENT_LINE.match(line)
        if match is None:
            continue
        path = Path(match.group("path"))
        if path.parent == chat_dir and _STORED_NAME.fullmatch(path.name):
            names.add(path.name)
    return names


def mark_sent(app: Any, chat_id: str, prompt: str) -> None:
    """Record the files *prompt* carries as sent, so the sweep keeps them.

    Called once ``POST /api/chat`` has accepted *prompt* for *chat_id*. A prompt
    that names no file of this chat writes nothing. The manifest is replaced
    whole under the app's lock, so two messages of one chat sent close together
    both land in it.
    """
    if not is_posture_key(chat_id):
        return
    chat_dir = Path(resolve_agent_data_root(app)) / UPLOADS_DIR / chat_id
    names = _attached_names(chat_dir, prompt)
    if not names:
        return
    with _manifest_lock(app):
        sent = _sent_names(chat_dir) | names
        try:
            write_json_atomic(chat_dir / SENT_MANIFEST, {"sent": sorted(sent)})
        except OSError:
            logger.warning("Could not record the sent files of chat %s", chat_id, exc_info=True)


def _manifest_lock(app: Any) -> threading.Lock:
    """The lock every manifest write of this app takes."""
    lock = getattr(app.state, "chat_attachment_manifest_lock", None)
    if lock is None:
        lock = threading.Lock()
        app.state.chat_attachment_manifest_lock = lock
    return lock


def _sweep_chat(chat_dir: Path, ledger: _Ledger, cutoff: float) -> None:
    """Delete *chat_dir*'s never-sent files older than *cutoff*, and its stale part files."""
    sent = _sent_names(chat_dir)
    try:
        entries = list(os.scandir(chat_dir))
    except OSError:
        return
    for entry in entries:
        try:
            # Never through a symlink: neither read, nor deleted, nor counted.
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                continue
            stat = entry.stat(follow_symlinks=False)
        except OSError:
            continue
        name = entry.name
        if name == SENT_MANIFEST or name in sent or stat.st_mtime >= cutoff:
            continue
        is_leftover = name.startswith(".") and name.endswith(".part")
        if not is_leftover and _STORED_NAME.fullmatch(name) is None:
            continue
        try:
            os.unlink(entry.path)
        except OSError:
            continue
        if not is_leftover:
            ledger.uncount(chat_dir.name, stat.st_size)


def sweep_uploads(uploads_dir: Path, ledger: _Ledger, *, now: float | None = None) -> None:
    """Delete every never-sent upload older than :data:`UNSENT_RETENTION_S`.

    Walks only the chat directories directly under *uploads_dir*, by name, and
    never follows a symlink into or out of the tree. Each file deleted comes off
    the ledger's measurements, so the caps see the tree as it now is. *now* is
    the wall clock to age files against; the ledger notes the sweep so an upload
    does not run another within :data:`_SWEEP_INTERVAL_S`.
    """
    ledger.swept_at = time.monotonic()
    cutoff = (time.time() if now is None else now) - UNSENT_RETENTION_S
    try:
        chats = [
            entry
            for entry in os.scandir(uploads_dir)
            if is_posture_key(entry.name)
            and not entry.is_symlink()
            and entry.is_dir(follow_symlinks=False)
        ]
    except OSError:
        return
    for chat in chats:
        _sweep_chat(Path(chat.path), ledger, cutoff)


def sweep_uploads_at_startup(app: Any) -> None:
    """The startup sweep: before any upload is counted, so the caps measure the swept tree."""
    try:
        sweep_uploads(Path(resolve_agent_data_root(app)) / UPLOADS_DIR, _ledger(app))
    except Exception:
        logger.warning("The upload sweep did not run", exc_info=True)


def _sweep_if_due(ledger: _Ledger, uploads_dir: Path) -> None:
    """Run the sweep before an upload when the last one is more than an hour old."""
    if ledger.swept_at is not None and time.monotonic() - ledger.swept_at < _SWEEP_INTERVAL_S:
        return
    sweep_uploads(uploads_dir, ledger)


@overload
def _batch(request: Request, *, required: Literal[True]) -> str: ...
@overload
def _batch(request: Request, *, required: Literal[False]) -> str | None: ...
def _batch(request: Request, *, required: bool) -> str | None:
    """The message id the client names, checked against the closed id grammar."""
    raw = request.headers.get(BATCH_HEADER, "").strip()
    if not raw:
        if required:
            raise _refused(400, "invalid_batch", _BATCH_REFUSED)
        return None
    if _BATCH_ID.fullmatch(raw) is None:
        raise _refused(400, "invalid_batch", _BATCH_REFUSED)
    return raw


def _name_parts(request: Request) -> tuple[str, str]:
    """The stem and the lower-case extension of the uploaded name, made safe to store.

    Only the last path segment of the name is kept, so no name can climb out of
    its directory. Every character outside ``[A-Za-z0-9._-]`` becomes ``_``. A
    name that is empty once cleaned, or starts with a dot, is refused.
    """
    raw = request.headers.get(NAME_HEADER, "")
    if not raw or len(raw) > _MAX_NAME_HEADER_CHARS:
        raise _refused(400, "invalid_name", _NAME_REFUSED)
    leaf = _UNSAFE_CHARS.sub("_", _SEPARATORS.split(unquote(raw))[-1])
    if not leaf or leaf.startswith("."):
        raise _refused(400, "invalid_name", _NAME_REFUSED)
    stem, dot, ext = leaf.rpartition(".")
    if not dot:
        stem, ext = leaf, ""
    return stem[:_MAX_STEM_CHARS], ext.lower()


def _declared_length(request: Request) -> int | None:
    """The Content-Length the client declared, or ``None`` when it declared none it can read."""
    raw = request.headers.get("content-length")
    try:
        length = int(raw) if raw is not None else None
    except ValueError:
        return None
    return length if length is not None and length >= 0 else None


def _image_type(head: bytes) -> str | None:
    """The image type the leading bytes prove, or ``None``."""
    for signature, mime in _IMAGE_SIGNATURES:
        if head.startswith(signature):
            return mime
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    return None


def _is_plain_text(handle: BinaryIO) -> bool:
    """Whether the whole file is plain text: UTF-8, no NUL byte, and not a web page."""
    decoder = codecs.getincrementaldecoder("utf-8")()
    opening = ""
    try:
        for chunk in iter(lambda: handle.read(_READ_CHUNK_BYTES), b""):
            if b"\x00" in chunk:
                return False
            text = decoder.decode(chunk)
            if not opening:
                opening = text.lstrip("\ufeff \t\r\n").lower()[:16]
        decoder.decode(b"", final=True)
    except UnicodeDecodeError:
        return False
    return not opening.startswith(_MARKUP_OPENINGS)


def _content_type(part: Path, kind: str, declared: str) -> str | None:
    """The MIME type *part*'s content proves for a file of *kind*, or ``None`` when it proves none.

    An image is checked by its leading bytes and a PDF by its header. Text is
    checked over the whole file, since the first kilobyte of a binary can read
    as text.
    """
    with part.open("rb") as handle:
        if kind == "image":
            return _image_type(handle.read(16))
        if kind == "pdf":
            return declared if handle.read(5) == b"%PDF-" else None
        return declared if _is_plain_text(handle) else None


@router.post("/api/chat/{chat_id}/attachments", status_code=201)
async def upload_chat_attachment(chat_id: str, request: Request) -> Response:
    """Store one file the operator attached to a message that has not been sent yet.

    The body is read in chunks and refused as soon as it passes the per-file
    cap or the chat's cap, so an oversized upload is never buffered whole. The
    file is written under a temporary name and moved into place only once its
    content has been checked; a refused or interrupted upload leaves nothing.
    """
    if not is_posture_key(chat_id):
        raise _refused(400, "invalid_chat_id", _CHAT_REFUSED)
    batch = _batch(request, required=True)
    stem, ext = _name_parts(request)
    if ext not in _ALLOWED:
        raise _refused(415, "unsupported_type", _TYPE_REFUSED)
    kind, declared = _ALLOWED[ext]
    name = f"{stem}.{ext}"

    ledger = _ledger(request.app)
    message = (chat_id, batch)
    if ledger.message_count(message) >= MAX_FILES_PER_MESSAGE:
        raise _refused(422, "too_many_files", _COUNT_REFUSED)
    declared_length = _declared_length(request)
    if declared_length is not None and declared_length > MAX_ATTACHMENT_BYTES:
        raise _refused(413, "file_too_large", _SIZE_REFUSED)
    uploads_dir = _uploads_dir(request)
    chat_dir = uploads_dir / chat_id
    # Leftovers go first, so the caps measure the tree as it is kept.
    _sweep_if_due(ledger, uploads_dir)
    total = _chat_total(ledger, chat_id, chat_dir)
    # Measured now, before any byte is counted, so an upload in flight is never missed.
    everything = _uploads_total(ledger, uploads_dir)
    if declared_length is not None and total + declared_length > MAX_CHAT_ATTACHMENT_BYTES:
        raise _refused(413, "chat_full", _CHAT_FULL_REFUSED)
    if declared_length is not None and everything + declared_length > MAX_UPLOADS_BYTES:
        raise _refused(413, "uploads_full", _UPLOADS_FULL_REFUSED)

    token = secrets.token_hex(8)
    part = chat_dir / f".{token}.part"
    final = chat_dir / f"{token}-{name}"
    written = 0
    mime = ""
    stored = False
    ledger.add_file(message)
    try:
        try:
            chat_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            fd = os.open(part, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except OSError:
            raise _refused(503, "store_write_failed", _STORE_REFUSED) from None
        with os.fdopen(fd, "wb") as out:
            try:
                async for chunk in request.stream():
                    if written + len(chunk) > MAX_ATTACHMENT_BYTES:
                        raise _refused(413, "file_too_large", _SIZE_REFUSED)
                    if ledger.chat_bytes[chat_id] + len(chunk) > MAX_CHAT_ATTACHMENT_BYTES:
                        raise _refused(413, "chat_full", _CHAT_FULL_REFUSED)
                    if _uploads_total(ledger, uploads_dir) + len(chunk) > MAX_UPLOADS_BYTES:
                        raise _refused(413, "uploads_full", _UPLOADS_FULL_REFUSED)
                    ledger.add_bytes(chat_id, len(chunk))
                    written += len(chunk)
                    out.write(chunk)
            except OSError:
                raise _refused(503, "store_write_failed", _STORE_REFUSED) from None
        if written == 0:
            raise _refused(400, "empty_file", _EMPTY_REFUSED)
        # The whole file is read for text; off the loop, so no other chat waits on it.
        detected = await asyncio.to_thread(_content_type, part, kind, declared)
        if detected is None:
            raise _refused(415, "unsupported_type", _TYPE_REFUSED)
        mime = detected
        try:
            os.replace(part, final)
        except OSError:
            raise _refused(503, "store_write_failed", _STORE_REFUSED) from None
        stored = True
    except ClientDisconnect:
        # The client let go mid-upload; there is nobody left to answer.
        return Response(status_code=204)
    finally:
        if not stored:
            ledger.add_bytes(chat_id, -written)
            ledger.drop_file(message)
            with contextlib.suppress(OSError):
                part.unlink()

    return JSONResponse(
        status_code=201,
        content={"name": name, "path": str(final.absolute()), "size": written, "type": mime},
    )


@router.delete("/api/chat/{chat_id}/attachments/{stored}", status_code=204)
async def remove_chat_attachment(chat_id: str, stored: str, request: Request) -> Response:
    """Remove a file the operator took back before sending it.

    Only a stored name under the chat's own uploads directory can be named, so
    nothing outside that directory is reachable. A name that is not there is a
    204 as well, which makes a repeated removal harmless.
    """
    if not is_posture_key(chat_id):
        raise _refused(400, "invalid_chat_id", _CHAT_REFUSED)
    if _STORED_NAME.fullmatch(stored) is None:
        raise _refused(400, "invalid_name", _NAME_REFUSED)
    batch = _batch(request, required=False)

    target = _uploads_dir(request) / chat_id / stored
    try:
        size = target.stat().st_size if target.is_file() else None
    except OSError:
        size = None
    if size is None:
        return Response(status_code=204)
    try:
        target.unlink()
    except OSError:
        raise _refused(503, "store_write_failed", _STORE_REFUSED) from None

    ledger = _ledger(request.app)
    if chat_id in ledger.chat_bytes:
        ledger.add_bytes(chat_id, -size)
    elif ledger.total_bytes is not None:
        ledger.total_bytes -= size
    if batch is not None:
        ledger.drop_file((chat_id, batch))
    return Response(status_code=204)
