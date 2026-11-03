"""The verbs: ``list``, ``show``, ``accept`` and ``reject``, and the entry point."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from osprey.deployment.web_terminals.roster import (
    PRIVILEGE_SETUP_TOOL,
    USERNAME_CHARSET_RE,
    effective_image_source,
)
from osprey.integrations.canopy import PUSH_SUBPATH
from osprey.integrations.canopy.library import (
    REPORT_JSON,
    REPORT_MD,
    SCREEN_SUFFIX,
    Judged,
    judge,
    library_digest,
    scan,
)

from . import container, grammar
from . import deployment as deployment_module
from .container import (
    CHECK_SCRIPT,
    EMIT_SCRIPT,
    LIST_SCRIPT,
    STAGE_MOUNT,
    STREAM_SCRIPT,
    VOLUME_MOUNT,
)
from .deployment import LIBRARY_PATHSPEC, STAGE_SUBPATH, Deployment
from .grammar import (
    ID_PATTERN,
    MAX_ASSET_BYTES,
    MAX_FILES,
    MAX_LIST_BUNDLES,
    MAX_NOTE_CHARS,
    MAX_REQUEST_BYTES,
    MAX_TOTAL_BYTES,
    MAX_VERDICT_BYTES,
    has_forbidden_char,
    is_reserved_segment,
    safe_text,
)
from .refusals import (
    CHECK_EXIT,
    EXIT_BAD_LAYOUT,
    EXIT_BAD_REASON,
    EXIT_BAD_REQUEST,
    EXIT_BAD_RID,
    EXIT_BAD_SOURCE_DIR,
    EXIT_BAD_STAGE,
    EXIT_BAD_TARGET_DIR,
    EXIT_BAD_VERDICT,
    EXIT_BUNDLE_MISSING,
    EXIT_CASE_DIR,
    EXIT_CASE_FILE,
    EXIT_CLOSURE_REFUSED,
    EXIT_COLLISION,
    EXIT_CONTAINER_FAILED,
    EXIT_CONTAINER_START,
    EXIT_CONTAINER_TIMEOUT,
    EXIT_DIGEST_MISMATCH,
    EXIT_EMIT_DIGEST,
    EXIT_FILES_MISMATCH,
    EXIT_LIBRARY_COPY,
    EXIT_NO_STAGE,
    EXIT_OK,
    EXIT_REPLACED_EXISTS,
    EXIT_REVIEWER,
    EXIT_STREAM_CAP,
    EXIT_STREAM_LINK,
    EXIT_USAGE,
    EXIT_WRITE_FAILED,
    Refusal,
)

FORMAT = "canopy-push/1"
REQUEST_NAME = "request.json"
#: Written into a checked stage by ``show``: what the checks could not run.
CHECK_NOTES_NAME = "check-notes.json"
#: The finding code CANOPY's check returns when no staged spec covers a screen.
SPEC_UNAVAILABLE = "spec_unavailable"
FILES_DIRNAME = "files"
REQUEST_KEYS = (
    "format",
    "rid",
    "source_id",
    "source_dir",
    "rev",
    "target_dir",
    "note",
    "created",
    "files",
)
RID_RE = re.compile(r"[0-9a-f]{16}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
LIBRARY_SEGMENT = "library"
SCREENS_DIRNAME = "screens"
ROOT_DIR = SCREENS_DIRNAME + "/"
STAGE_MAX_AGE_S = 14 * 24 * 3600
#: Where the bytes an accept replaced are kept, below ``var/canopy-push/<user>/``, for 14 days.
REPLACED_DIRNAME = "replaced"
FILE_MODE = 0o644
DIR_MODE = 0o755
#: The privilege a ``--reviewer``'s render must hold: the agent's deployment-editing
#: setup tool, which is what makes a tier the deployment's admin, whatever it is named.
REVIEWER_PRIVILEGE = PRIVILEGE_SETUP_TOOL

#: What is printed once, at the top of every ``show``.
BANNER = "filed from {user}'s volume: an ask approval is not proof the user filed it"
#: The framing line of the digest: the digest covers these three things and nothing else.
DIGEST_FRAME = "osprey-canopy-push-digest/1"


def _plain(value: Any) -> str:
    """``value`` as text for a line :func:`_out` or :func:`_say` will make safe: no escaping here."""
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=True)


def _say(message: str) -> None:
    sys.stderr.write("push: " + safe_text(message) + "\n")


def _out(line: str = "") -> None:
    sys.stdout.write(safe_text(line) + "\n")


# --- list -------------------------------------------------------------------------------------


def _list(deployment: Deployment, host_ids: tuple[int, int]) -> int:
    roster = deployment.roster
    if not roster:
        _out("no web-terminal roster: nothing to list")
        return EXIT_OK
    for user in roster:
        volume = deployment.volume_of(user)
        if not deployment_module._volume_exists(deployment.runtime, volume):
            _out(f"{user}: no volume")
            continue
        image, _payload = deployment.image_of(user)
        argv = container.container_argv(
            deployment.runtime,
            "list",
            nonce=secrets.token_hex(4),
            image=image,
            script=LIST_SCRIPT,
            args=[
                f"{VOLUME_MOUNT}/{'/'.join(PUSH_SUBPATH)}",
                str(MAX_LIST_BUNDLES),
                str(MAX_REQUEST_BYTES),
            ],
            volume=volume,
            host_ids=host_ids,
        )
        done = container._run_container(deployment.runtime, argv)
        container._started(done, deployment.runtime)
        if done.returncode != 0:
            raise Refusal(
                EXIT_CONTAINER_FAILED, f"{user}: the list container exited {done.returncode}"
            )
        try:
            listed = json.loads(done.stdout[:MAX_VERDICT_BYTES])
            bundles = listed["bundles"]
        except (ValueError, KeyError, TypeError):
            raise Refusal(
                EXIT_BAD_VERDICT, f"{user}: the list container printed no listing"
            ) from None
        for bundle in bundles:
            _out(_list_line(user, bundle))
        if listed.get("truncated"):
            _out(f"{user}: more than {MAX_LIST_BUNDLES} directories; only the first were read")
    return EXIT_OK


def _list_line(user: str, bundle: Any) -> str:
    name = bundle.get("name") if isinstance(bundle, dict) else None
    if not isinstance(name, str) or not RID_RE.fullmatch(name):
        return f"{user}: (ignored: {_plain(name)})"
    if "request" not in bundle:
        return f"{user}/{name}: (unreadable: {_plain(bundle.get('error', 'no request.json'))})"
    try:
        request = json.loads(bundle["request"])
    except ValueError:
        return f"{user}/{name}: (unreadable: request.json is not JSON)"
    if not isinstance(request, dict):
        return f"{user}/{name}: (unreadable: request.json is not an object)"
    noted = request.get("note")
    note = noted if isinstance(noted, str) else ""
    return (
        f"{user}/{name}  created={_plain(request.get('created'))}  "
        f"{_plain(request.get('source_id'))} -> {_plain(request.get('target_dir'))}  "
        f"note={_plain(note[:80])}"
    )


# --- show: stream -------------------------------------------------------------------------------


def _parse_target(target: str) -> tuple[str, str]:
    user, sep, rid = target.partition("/")
    if not sep or not RID_RE.fullmatch(rid):
        raise Refusal(EXIT_BAD_RID, f"{target!r} is not <user>/<rid> with a 16-hex rid")
    return user, rid


def _stream_bundle(deployment: Deployment, user: str, volume: str, rid: str) -> dict[str, bytes]:
    """The bundle's members, read once from the stream: ``{member path: bytes}``."""
    image, _payload = deployment.image_of(user)
    argv = container.container_argv(
        deployment.runtime,
        "show-stream",
        nonce=secrets.token_hex(4),
        image=image,
        script=STREAM_SCRIPT,
        args=[
            f"{VOLUME_MOUNT}/{'/'.join(PUSH_SUBPATH)}/{rid}",
            str(MAX_FILES),
            str(MAX_TOTAL_BYTES),
            str(MAX_ASSET_BYTES),
        ],
        volume=volume,
        host_ids=(os.getuid(), os.getgid()),
    )
    name = container._container_name(argv)
    try:
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as exc:
        raise Refusal(EXIT_CONTAINER_START, f"cannot start {deployment.runtime}: {exc}") from None
    timed_out = threading.Event()

    def on_timeout() -> None:
        timed_out.set()
        proc.kill()
        container._kill(deployment.runtime, name)

    timer = threading.Timer(container.CONTAINER_TIMEOUT_S, on_timeout)
    timer.start()
    members: dict[str, bytes] = {}
    breach: str | None = None
    tar_error: str | None = None
    total = 0
    try:
        assert proc.stdout is not None
        try:
            with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
                for member in tar:
                    breach = _member_breach(member, len(members), total)
                    if breach is not None:
                        break
                    handle = tar.extractfile(member)
                    data = handle.read() if handle is not None else b""
                    if len(data) != member.size:
                        breach = f"{member.name}: short read"
                        break
                    members[member.name] = data
                    total += len(data)
        except tarfile.TarError as exc:
            tar_error = str(exc)
        if breach is not None:
            proc.kill()
            container._kill(deployment.runtime, name)
        _stdout, stderr = proc.communicate()
    finally:
        timer.cancel()
    if timed_out.is_set():
        timer.join()  # the kill of the container runs on the timer's thread; wait for it
        raise Refusal(EXIT_CONTAINER_TIMEOUT, f"container {name} timed out")
    if breach is not None:
        raise Refusal(EXIT_STREAM_CAP, f"{user}/{rid}: {breach}")
    said = _script_error(stderr)
    if proc.returncode in (125, 126, 127):
        raise Refusal(
            EXIT_CONTAINER_START, f"{deployment.runtime} could not start the container: {said}"
        )
    if proc.returncode == 5:
        raise Refusal(
            EXIT_BUNDLE_MISSING,
            f"bundle evicted from {user}'s volume (newest-8 limit); ask the user to re-push",
        )
    if proc.returncode == 3:
        raise Refusal(EXIT_STREAM_LINK, f"{user}/{rid}: refused in the volume: {said}")
    if proc.returncode == 4:
        raise Refusal(EXIT_STREAM_CAP, f"{user}/{rid}: over a cap in the volume: {said}")
    if proc.returncode != 0:
        raise Refusal(
            EXIT_CONTAINER_FAILED,
            f"{user}/{rid}: the stream container exited {proc.returncode}: {said}",
        )
    if tar_error is not None:
        raise Refusal(EXIT_BAD_VERDICT, f"{user}/{rid}: the stream is not a tar: {tar_error}")
    return members


def _member_breach(member: tarfile.TarInfo, count: int, total: int) -> str | None:
    """Why ``member`` may not be read, or ``None``: caps and kinds, checked before any byte of it."""
    if not member.isreg():
        return f"{member.name}: not a regular file"
    if member.size > MAX_ASSET_BYTES:
        return f"{member.name}: over the per-file cap"
    if count + 1 > MAX_FILES:
        return "too many files"
    if total + member.size > MAX_TOTAL_BYTES:
        return "bundle too large"
    return None


def _script_error(stderr: bytes) -> str:
    lines = [line for line in stderr.decode("utf-8", "replace").splitlines() if line.strip()]
    if not lines:
        return ""
    try:
        return str(json.loads(lines[-1]).get("error", lines[-1]))
    except (ValueError, AttributeError):
        return lines[-1]


# --- show: validate -----------------------------------------------------------------------------


@dataclass
class Request:
    """The validated ``request.json`` and the files it covers (paths below ``source_dir``)."""

    rid: str
    source_id: str
    source_dir: str
    target_dir: str
    note: str
    created: str
    rev: str
    raw: bytes
    files: dict[str, bytes] = field(default_factory=dict)

    @property
    def top(self) -> str:
        """The pushed screen's id at the destination."""
        rel = self.source_id[len(self.source_dir) + 1 :] if self.source_dir else self.source_id
        return f"{self.target_dir}/{rel}"

    def destination(self, path: str) -> str:
        return f"{ROOT_DIR}{self.target_dir}/{path}"


def _clean_member_name(name: str) -> list[str]:
    parts = name.split("/")
    if name.startswith("/") or any(part in ("", ".", "..") for part in parts):
        raise Refusal(EXIT_BAD_LAYOUT, f"member {name!r} is not a clean relative path")
    return parts


def _validate_layout(members: Mapping[str, bytes]) -> tuple[bytes, dict[str, bytes]]:
    """``(request.json bytes, {path below files/: bytes})``; anything else in the bundle is a refusal."""
    request: bytes | None = None
    files: dict[str, bytes] = {}
    for name, data in members.items():
        parts = _clean_member_name(name)
        if parts == [REQUEST_NAME]:
            request = data
        elif parts[0] == FILES_DIRNAME and len(parts) > 1:
            files["/".join(parts[1:])] = data
        else:
            raise Refusal(
                EXIT_BAD_LAYOUT,
                f"member {name!r} is neither {REQUEST_NAME} nor under {FILES_DIRNAME}/",
            )
    if request is None:
        raise Refusal(EXIT_BAD_LAYOUT, f"the bundle holds no {REQUEST_NAME}")
    if not files:
        raise Refusal(EXIT_BAD_LAYOUT, "the bundle holds no files")
    return request, files


def _validate_request(raw: bytes, rid: str, files: dict[str, bytes]) -> Request:
    if len(raw) > MAX_REQUEST_BYTES:
        raise Refusal(
            EXIT_BAD_REQUEST, f"{REQUEST_NAME} is {len(raw)} bytes; at most {MAX_REQUEST_BYTES}"
        )
    try:
        doc = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, ValueError):
        raise Refusal(EXIT_BAD_REQUEST, f"{REQUEST_NAME} is not ASCII JSON") from None
    if not isinstance(doc, dict) or tuple(doc) != REQUEST_KEYS:
        raise Refusal(
            EXIT_BAD_REQUEST,
            f"{REQUEST_NAME} does not hold exactly the keys {', '.join(REQUEST_KEYS)}",
        )
    if doc["format"] != FORMAT:
        raise Refusal(
            EXIT_BAD_REQUEST, f"{REQUEST_NAME} format is {_plain(doc['format'])!r}, not {FORMAT!r}"
        )
    if doc["rid"] != rid:
        raise Refusal(
            EXIT_BAD_REQUEST, f"{REQUEST_NAME} names rid {_plain(doc['rid'])!r}, not {rid!r}"
        )
    for key in ("source_id", "source_dir", "target_dir", "note", "created", "rev"):
        if not isinstance(doc[key], str):
            raise Refusal(EXIT_BAD_REQUEST, f"{REQUEST_NAME} {key} is not a string")
    if not _HEX64.fullmatch(doc["rev"]):
        raise Refusal(EXIT_BAD_REQUEST, f"{REQUEST_NAME} rev is not a sha256")
    if len(doc["created"]) > 20:
        raise Refusal(EXIT_BAD_REQUEST, f"{REQUEST_NAME} created is not a timestamp")
    if len(doc["note"]) > MAX_NOTE_CHARS or has_forbidden_char(doc["note"]):
        raise Refusal(
            EXIT_BAD_REQUEST, f"{REQUEST_NAME} note is too long or holds a forbidden character"
        )
    source_id, source_dir, target_dir = doc["source_id"], doc["source_dir"], doc["target_dir"]
    if not ID_PATTERN.fullmatch(source_id):
        raise Refusal(EXIT_BAD_SOURCE_DIR, f"source_id {_plain(source_id)!r} is not a screen id")
    if source_dir:
        if not ID_PATTERN.fullmatch(source_dir) or not source_id.startswith(source_dir + "/"):
            raise Refusal(
                EXIT_BAD_SOURCE_DIR,
                f"source_dir {_plain(source_dir)!r} is not a directory holding {_plain(source_id)!r}",
            )
    elif "/" in source_id:
        raise Refusal(EXIT_BAD_SOURCE_DIR, "an empty source_dir is allowed for a top-level id only")
    if (
        not ID_PATTERN.fullmatch(target_dir)
        or target_dir.split("/")[0] == LIBRARY_SEGMENT
        or any(is_reserved_segment(segment) for segment in target_dir.split("/"))
    ):
        raise Refusal(
            EXIT_BAD_TARGET_DIR,
            f"target_dir {_plain(target_dir)!r} is not a directory a push may land in",
        )
    listed = doc["files"]
    if not isinstance(listed, list) or any(
        not isinstance(entry, dict) or set(entry) != {"path", "size", "sha256"} for entry in listed
    ):
        raise Refusal(
            EXIT_BAD_REQUEST, f"{REQUEST_NAME} files is not a list of {{path, size, sha256}}"
        )
    claimed = {entry["path"]: (entry["size"], entry["sha256"]) for entry in listed}
    actual = {path: (len(data), hashlib.sha256(data).hexdigest()) for path, data in files.items()}
    if claimed != actual or [entry["path"] for entry in listed] != sorted(claimed):
        raise Refusal(
            EXIT_FILES_MISMATCH, f"{REQUEST_NAME} files do not equal the members of the bundle"
        )
    rel = source_id[len(source_dir) + 1 :] if source_dir else source_id
    if rel + SCREEN_SUFFIX not in files:
        raise Refusal(
            EXIT_FILES_MISMATCH, f"the pushed screen {_plain(source_id)!r} is not among the members"
        )
    return Request(
        rid, source_id, source_dir, target_dir, doc["note"], doc["created"], doc["rev"], raw, files
    )


def _check_case_against_library(request: Request, judged: Judged) -> None:
    """Refuse a destination, or an ancestor of one, that differs only in case from a library path."""
    existing = set(judged.paths)
    existing |= {path[:at] for path in judged.paths for at, char in enumerate(path) if char == "/"}
    folded = {name.casefold(): name for name in existing}
    for path in sorted(request.files):
        destination = request.destination(path)
        names = [destination[:at] for at, char in enumerate(destination) if char == "/"] + [
            destination
        ]
        for name in names:
            if name in existing:
                continue
            spelled = folded.get(name.casefold())
            if spelled is None:
                continue
            code = EXIT_CASE_FILE if name == destination else EXIT_CASE_DIR
            raise Refusal(
                code,
                f"{name!r} differs only in case from the existing library path {spelled!r}; "
                "use that spelling or another name",
            )


# --- show: stage ------------------------------------------------------------------------------------


def _write_file(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, FILE_MODE)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
    os.chmod(path, FILE_MODE)


def _fix_modes(root: Path) -> None:
    """Every directory below ``root`` 0755 and every file 0644, whatever the umask was."""
    os.chmod(root, DIR_MODE)
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames:
            os.chmod(Path(dirpath) / name, DIR_MODE)
        for name in filenames:
            os.chmod(Path(dirpath) / name, FILE_MODE)


def _copy_library(library: Path, into: Path) -> None:
    """A host copy of ``library`` at ``into``: regular files 0644, directories 0755, a symlink refused."""
    into.mkdir(parents=True)
    os.chmod(into, DIR_MODE)
    for entry in scan(library):
        if entry["kind"] == "symlink":
            raise Refusal(
                EXIT_LIBRARY_COPY,
                f"the library holds a symlink at {entry['path']}; remove it first",
            )
        if entry["kind"] == "other":
            raise Refusal(
                EXIT_LIBRARY_COPY,
                f"the library holds a special file at {entry['path']}; remove it first",
            )
        if Path(entry["path"]).name == ".DS_Store":
            continue
        target = into / entry["path"]
        if entry["kind"] == "dir":
            target.mkdir(parents=True, exist_ok=True)
            os.chmod(target, DIR_MODE)
        else:
            _write_file(target, (library / entry["path"]).read_bytes())


def _stage(deployment: Deployment, user: str, request: Request, nonce: str) -> Path:
    """Write the fresh stage ``var/canopy-push/<user>/<rid>.<nonce>/`` and return it."""
    stage = deployment.root / STAGE_SUBPATH / user / f"{request.rid}.{nonce}"
    stage.mkdir(parents=True)
    os.chmod(stage, DIR_MODE)
    for path, data in request.files.items():
        _write_file(stage / "overlay" / request.destination(path), data)
    _write_file(stage / REQUEST_NAME, request.raw)
    _copy_library(deployment.library, stage / "current")
    _copy_library(deployment.library, stage / "would-be")
    for path, data in request.files.items():
        target = stage / "would-be" / request.destination(path)
        if target.exists():
            target.unlink()
        _write_file(target, data)
    _fix_modes(stage)
    return stage


def _rejected_pairs(deployment: Deployment) -> set[tuple[str, str]]:
    """Every ``(volume_user, rid)`` the push log records a rejection for."""
    return {
        (str(entry.get("volume_user")), str(entry.get("rid")))
        for entry in deployment_module._log_entries(deployment)
        if entry.get("decision") == "rejected"
    }


def _remove_tree(path: Path) -> None:
    if path.is_symlink():
        path.unlink()
    else:
        shutil.rmtree(path)


def _prune_stages(deployment: Deployment, now: float | None = None) -> None:
    """Remove stages older than 14 days and stages of a rejected ``(user, rid)``; nothing else."""
    now = time.time() if now is None else now
    rejected = _rejected_pairs(deployment)
    root = deployment.root / STAGE_SUBPATH
    if not root.is_dir():
        return
    for user_dir in sorted(root.iterdir()):
        if not user_dir.is_dir() or user_dir.is_symlink():
            continue
        for stage in sorted(user_dir.iterdir()):
            rid = stage.name.split(".", 1)[0]
            if not RID_RE.fullmatch(rid):
                continue
            try:
                age = now - stage.lstat().st_mtime
            except OSError:
                continue
            if age > STAGE_MAX_AGE_S or (user_dir.name, rid) in rejected:
                _remove_tree(stage)


def _promote_stage(fresh: Path, rid: str) -> Path:
    """Rename the checked stage to ``<rid>/``, removing any previous stage of that rid."""
    final = fresh.parent / rid
    if final.exists() or final.is_symlink():
        _remove_tree(final)
    os.rename(fresh, final)
    return final


# --- show: check -----------------------------------------------------------------------------------


def _check_in_container(
    deployment: Deployment, user: str, stage: Path, request: Request, host_ids: tuple[int, int]
) -> dict[str, Any]:
    image, payload = deployment.image_of(user)
    argv = container.container_argv(
        deployment.runtime,
        "show-check",
        nonce=secrets.token_hex(4),
        image=image,
        script=CHECK_SCRIPT,
        args=[STAGE_MOUNT, request.target_dir, request.top],
        stage=stage,
        payload_dir=payload,
        host_ids=host_ids,
    )
    done = container._run_container(deployment.runtime, argv)
    container._started(done, deployment.runtime)
    if done.returncode != 0:
        said = _script_error(done.stderr)
        raise Refusal(
            EXIT_CONTAINER_FAILED, f"the check container exited {done.returncode}: {said}"
        )
    try:
        verdict = json.loads(done.stdout[:MAX_VERDICT_BYTES])
        refusals = verdict["refusals"]
        ok = verdict["ok"]
    except (ValueError, KeyError, TypeError):
        raise Refusal(EXIT_BAD_VERDICT, "the check container printed no verdict") from None
    if not ok or refusals:
        lines = [f"{item.get('code')}: {_plain(item.get('message'))}" for item in refusals]
        for item in refusals:
            lines.extend(
                f"  {key}: {_plain(item[key])}"
                for key in ("errors", "extra", "missing", "doc", "nodeId", "ref")
                if item.get(key) not in (None, [], "")
            )
        code = (
            CHECK_EXIT.get(str(refusals[0].get("code")), EXIT_CLOSURE_REFUSED)
            if refusals
            else EXIT_BAD_VERDICT
        )
        raise Refusal(code, "the staged push failed its checks:", *lines)
    if not isinstance(verdict.get("wouldBe"), dict) or not isinstance(verdict.get("current"), dict):
        raise Refusal(EXIT_BAD_VERDICT, "the check container printed no reports")
    notes = spec_unavailable_notes(verdict)
    if notes is None:
        raise Refusal(EXIT_BAD_VERDICT, "the check container printed a malformed specUnavailable")
    return {**verdict, "specUnavailable": notes}


def spec_unavailable_notes(holder: Any) -> list[dict[str, str]] | None:
    """The ``specUnavailable`` entries of a verdict or a stage's check notes.

    Each entry is ``{"screen": <id>, "message": <what is missing>}``: a screen
    the checks ran without the IR schema and the prop lint for. Absent means
    none (a verdict from a CANOPY that predates the staged spec reports none);
    anything but a list of such objects is ``None``, malformed.
    """
    if not isinstance(holder, Mapping):
        return None
    raw = holder.get("specUnavailable", [])
    if not isinstance(raw, list):
        return None
    notes = []
    for item in raw:
        if (
            not isinstance(item, Mapping)
            or not isinstance(item.get("screen"), str)
            or not isinstance(item.get("message"), str)
        ):
            return None
        notes.append({"screen": item["screen"], "message": item["message"]})
    return notes


def spec_unavailable_record(notes: Sequence[Mapping[str, str]]) -> dict[str, list[str]] | None:
    """What the push log records of ``notes``: the screens and the distinct messages, or ``None``."""
    if not notes:
        return None
    return {
        "screens": sorted({note["screen"] for note in notes}),
        "messages": sorted({note["message"] for note in notes}),
    }


def spec_unavailable_line(notes: Sequence[Mapping[str, str]]) -> str | None:
    """The one ``spec_unavailable:`` line ``show`` and ``accept`` print, or ``None`` when the spec was there."""
    record = spec_unavailable_record(notes)
    if record is None:
        return None
    screens = record["screens"]
    return (
        f"{SPEC_UNAVAILABLE}: the checks ran without the IR schema and the prop lint for "
        f"{len(screens)} screen(s) ({', '.join(screens)}): {'; '.join(record['messages'])}"
    )


def _write_check_notes(stage: Path, notes: Sequence[Mapping[str, str]]) -> None:
    """Keep what the checks could not run in the stage, for ``accept`` to print and log."""
    body = json.dumps({"specUnavailable": list(notes)}, sort_keys=True, ensure_ascii=True)
    _write_file(stage / CHECK_NOTES_NAME, body.encode("ascii"))


def _read_check_notes(stage: Path) -> list[dict[str, str]]:
    """The ``specUnavailable`` notes ``show`` kept in ``stage``.

    A stage written before the notes existed has none, and reads as none.
    :raises Refusal: the notes file is there but is not notes.
    """
    path = stage / CHECK_NOTES_NAME
    if not path.is_file():
        return []
    try:
        notes = spec_unavailable_notes(json.loads(path.read_bytes().decode("ascii")))
    except (UnicodeDecodeError, ValueError):
        notes = None
    if notes is None:
        raise Refusal(EXIT_BAD_STAGE, f"{path} is not check notes; run show again")
    return notes


# --- show: print -------------------------------------------------------------------------------------


def framed_digest(request: Request, current_digest: str) -> str:
    """sha256 over the sorted ``(path, length, sha256)`` triples of the overlay, ``request.json`` and the library digest."""
    body = [DIGEST_FRAME]
    for path in sorted(request.files):
        data = request.files[path]
        body.append(f"{request.destination(path)}\t{len(data)}\t{hashlib.sha256(data).hexdigest()}")
    body.append(f"{REQUEST_NAME}\t{len(request.raw)}\t{hashlib.sha256(request.raw).hexdigest()}")
    body.append(f"library\t{current_digest}")
    return hashlib.sha256(("\n".join(body) + "\n").encode("utf-8")).hexdigest()


_ROW_KEY_ORDER = ("top", "topScreen", "screen", "instance", "nodeId", "nodeType", "kind")


def _row_line(row: Any) -> str:
    if not isinstance(row, dict):
        return _plain(row)
    keys = [k for k in _ROW_KEY_ORDER if k in row] + sorted(
        k for k in row if k not in _ROW_KEY_ORDER
    )
    return "  ".join(f"{key}={_plain(row[key])}" for key in keys)


def _row_is_of(row: Any, ids: set[str], paths: set[str]) -> bool:
    if isinstance(row, str):
        return row in ids
    if not isinstance(row, dict):
        return False
    return (
        any(row.get(key) in ids for key in ("screen", "topScreen", "top", "id"))
        or row.get("path") in paths
    )


def _row_strings(row: Any) -> set[str]:
    if isinstance(row, str):
        return {row}
    if isinstance(row, dict):
        return {value for value in row.values() if isinstance(value, str)}
    return set()


def _string_leaves(doc: Any) -> list[tuple[str, str]]:
    """Every string leaf of ``doc`` with its RFC 6901 pointer, in document order."""
    out: list[tuple[str, str]] = []
    stack: list[tuple[str, Any]] = [("", doc)]
    while stack:
        pointer, value = stack.pop()
        if isinstance(value, str):
            out.append((pointer, value))
        elif isinstance(value, dict):
            for key, child in reversed(list(value.items())):
                stack.append((f"{pointer}/{key.replace('~', '~0').replace('/', '~1')}", child))
        elif isinstance(value, list):
            for index, child in reversed(list(enumerate(value))):
                stack.append((f"{pointer}/{index}", child))
    return out


def _row_keys(report: Mapping[str, Any]) -> list[str]:
    return [key for key in report if key not in ("generator", "files", "provenance")]


def _print_show(
    user: str, request: Request, judged: Judged, verdict: Mapping[str, Any], stage: Path
) -> None:
    would_be, current = verdict["wouldBe"], verdict["current"]
    _out(BANNER.format(user=user))
    _out(
        f"request: {user}/{request.rid}  source={request.source_id}  source_dir={request.source_dir!r}  "
        f"target_dir={request.target_dir}  created={request.created}  rev={request.rev}"
    )
    _out(f"note: {request.note}")
    _out(f"stage: {stage}")
    spec_line = spec_unavailable_line(verdict["specUnavailable"])
    if spec_line is not None:
        _out(spec_line)
    _out()
    _out("== files ==")
    for path in sorted(request.files):
        data = request.files[path]
        _out(
            f"  {request.destination(path)}  {len(data)} bytes  sha256={hashlib.sha256(data).hexdigest()}"
        )
    pushed_paths = {request.destination(path) for path in request.files}
    pushed_ids = {
        p[len(ROOT_DIR) : -len(SCREEN_SUFFIX)] for p in pushed_paths if p.endswith(SCREEN_SUFFIX)
    }
    printed: set[str] = set()
    for key in _row_keys(would_be):
        rows = [row for row in would_be.get(key) or [] if _row_is_of(row, pushed_ids, pushed_paths)]
        _out()
        _out(f"== {key} of the pushed screens ({len(rows)}) ==")
        for row in rows:
            printed |= _row_strings(row)
            _out("  " + _row_line(row))
    _out()
    _out("== deltas: would-be minus current (both recomputed) ==")
    for key in _row_keys(would_be):
        after = Counter(
            json.dumps(row, sort_keys=True, ensure_ascii=True) for row in would_be.get(key) or []
        )
        before = Counter(
            json.dumps(row, sort_keys=True, ensure_ascii=True) for row in current.get(key) or []
        )
        added, removed = after - before, before - after
        if not added and not removed:
            continue
        _out(f"  {key}: +{sum(added.values())} -{sum(removed.values())}")
        for line in sorted(added):
            _out("    + " + _row_line(json.loads(line)))
        for line in sorted(removed):
            _out("    - " + _row_line(json.loads(line)))
    _out()
    _out("== destinations ==")
    collisions = 0
    for path in sorted(request.files):
        destination = request.destination(path)
        if destination in judged.paths:
            collisions += 1
            _out(f"  {destination}  COLLIDES with an existing library file")
        else:
            _out(f"  {destination}  new")
    if collisions:
        _out(
            f"  {collisions} destination(s) collide: accept needs --replace; the replaced bytes are kept "
            f"under var/canopy-push/{user}/replaced/ for 14 days, and the lasting copy is git history, "
            "so commit the library first"
        )
    _out()
    _out("== free text of the pushed screens (JSON pointer: value) ==")
    for path in sorted(request.files):
        if not path.endswith(SCREEN_SUFFIX):
            continue
        try:
            doc = json.loads(request.files[path].decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            continue
        for pointer, value in _string_leaves(doc):
            if value in printed:
                continue
            _out(f"  {request.destination(path)} {pointer}: {value}")
    _out()
    _out(f"digest: {framed_digest(request, judged.digest)}")


# --- show ---------------------------------------------------------------------------------------


def _show(deployment: Deployment, target: str, host_ids: tuple[int, int]) -> int:
    user, rid = _parse_target(target)
    volume = deployment_module._check_user(deployment, user)
    deployment_module._check_fresh(deployment)
    with deployment_module._SharedLibraryLock(deployment.library):
        members = _stream_bundle(deployment, user, volume, rid)
        raw, files = _validate_layout(members)
        request = _validate_request(raw, rid, files)
        grammar._check_grammar(files)
        judged = judge(scan(deployment.library))
        _check_case_against_library(request, judged)
        _prune_stages(deployment)
        fresh = _stage(deployment, user, request, secrets.token_hex(4))
    try:
        verdict = _check_in_container(deployment, user, fresh, request, host_ids)
        _write_check_notes(fresh, verdict["specUnavailable"])
    except Refusal:
        _remove_tree(fresh)
        raise
    final = _promote_stage(fresh, rid)
    _print_show(user, request, judged, verdict, final)
    return EXIT_OK


# --- accept and reject: the attestation and the log --------------------------------------------------------


def _check_reviewer(deployment: Deployment, reviewer: str) -> None:
    """Refuse unless ``reviewer`` is a roster user whose render may edit the deployment.

    The render must hold :data:`REVIEWER_PRIVILEGE`, read off its rendered
    config, so the check follows the capability and not the tier's name. An
    attestation, not authentication.
    """
    if (
        not USERNAME_CHARSET_RE.fullmatch(reviewer)
        or reviewer not in deployment.roster
        or REVIEWER_PRIVILEGE not in deployment.privileges_of(reviewer)
    ):
        raise Refusal(
            EXIT_REVIEWER,
            f"--reviewer {reviewer!r} is not a roster user whose render holds "
            f"{REVIEWER_PRIVILEGE} (the reviewer is an attestation, not authentication)",
        )


def _utc_stamp() -> str:
    """The replaced-bytes directory's suffix: the accept's UTC second."""
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


# --- accept: the stage ---------------------------------------------------------------------------------


def _load_stage(deployment: Deployment, user: str, rid: str) -> tuple[Path, Request]:
    """The checked stage ``var/canopy-push/<user>/<rid>/`` and its request, hashed back from the overlay."""
    stage = deployment.root / STAGE_SUBPATH / user / rid
    overlay = stage / "overlay"
    if (
        stage.is_symlink()
        or not (stage / REQUEST_NAME).is_file()
        or not overlay.is_dir()
        or not (stage / "would-be").is_dir()
    ):
        raise Refusal(
            EXIT_NO_STAGE, f"no checked stage for {user}/{rid} at {stage}; run show first"
        )
    raw = (stage / REQUEST_NAME).read_bytes()
    try:
        target_dir = json.loads(raw.decode("ascii"))["target_dir"]
    except (UnicodeDecodeError, ValueError, KeyError, TypeError):
        raise Refusal(
            EXIT_BAD_STAGE, f"{stage / REQUEST_NAME} is not a request; run show again"
        ) from None
    if not isinstance(target_dir, str) or not ID_PATTERN.fullmatch(target_dir):
        raise Refusal(EXIT_BAD_STAGE, f"{stage / REQUEST_NAME} names no target_dir; run show again")
    prefix = f"{ROOT_DIR}{target_dir}/"
    files: dict[str, bytes] = {}
    for dirpath, dirnames, filenames in os.walk(overlay, followlinks=False):
        for name in (*dirnames, *filenames):
            path = Path(dirpath) / name
            rel = path.relative_to(overlay).as_posix()
            if path.is_symlink() or not (path.is_dir() or path.is_file()):
                raise Refusal(
                    EXIT_BAD_STAGE, f"{rel} in the stage is not a regular entry; run show again"
                )
            if name in filenames:
                if not rel.startswith(prefix):
                    raise Refusal(
                        EXIT_BAD_STAGE,
                        f"{rel} in the stage is outside screens/{target_dir}/; run show again",
                    )
                files[rel[len(prefix) :]] = path.read_bytes()
    request = _validate_request(raw, rid, files)
    grammar._check_grammar(files)
    return stage, request


def _overlay_digest(judged: Judged, request: Request) -> str:
    """The library digest of the current listing with the overlay applied, in the listing's own order."""
    files = dict(judged.files)
    for path, data in request.files.items():
        files[request.destination(path)] = hashlib.sha256(data).hexdigest()
    prefix = len(ROOT_DIR)
    ordered = sorted(
        files.items(), key=lambda item: item[0][prefix:].encode("utf-8", "surrogatepass")
    )
    return library_digest("".join(f"{sha}  {path}\n" for path, sha in ordered).encode("utf-8"))


def _check_collisions(request: Request, judged: Judged, replace: bool) -> list[str]:
    """The destinations that exist in the library; refuse those whose bytes differ unless ``replace``."""
    existing = dict(judged.files)
    replacing: list[str] = []
    differing: list[str] = []
    for path in sorted(request.files):
        destination = request.destination(path)
        if destination not in existing:
            continue
        replacing.append(destination)
        if existing[destination] != hashlib.sha256(request.files[path]).hexdigest():
            differing.append(destination)
    if differing and not replace:
        raise Refusal(
            EXIT_COLLISION,
            f"{len(differing)} destination(s) collide with an existing library file; "
            "repeat with --replace to replace them (the replaced bytes are kept 14 days, "
            "the lasting copy is git history):",
            *(f"  {destination}" for destination in differing),
        )
    return replacing


# --- accept: the emit container ---------------------------------------------------------------------------


def _emit_report(
    deployment: Deployment,
    user: str,
    stage: Path,
    request: Request,
    judged: Judged,
    host_ids: tuple[int, int],
) -> tuple[dict[str, Any], str]:
    """``(report, markdown)`` emitted by CANOPY over the stage's ``would-be/``, provenance carried over."""
    image, payload = deployment.image_of(user)
    argv = container.container_argv(
        deployment.runtime,
        "accept-emit",
        nonce=secrets.token_hex(4),
        image=image,
        script=EMIT_SCRIPT,
        args=[f"{STAGE_MOUNT}/would-be"],
        stage=stage,
        payload_dir=payload,
        host_ids=host_ids,
    )
    done = container._run_container(deployment.runtime, argv)
    container._started(done, deployment.runtime)
    if done.returncode != 0:
        said = _script_error(done.stderr)
        raise Refusal(EXIT_CONTAINER_FAILED, f"the emit container exited {done.returncode}: {said}")
    try:
        emitted = json.loads(done.stdout[:MAX_VERDICT_BYTES])
        report, markdown = emitted["json"], emitted["md"]
        if not isinstance(report, dict) or not isinstance(markdown, str):
            raise TypeError
    except (ValueError, KeyError, TypeError):
        raise Refusal(EXIT_BAD_VERDICT, "the emit container printed no report") from None
    files = report.get("files")
    digest = files.get("digest") if isinstance(files, dict) else None
    expected = _overlay_digest(judged, request)
    if digest != expected:
        raise Refusal(
            EXIT_EMIT_DIGEST,
            f"the emitted report covers {_plain(digest)}, not the current listing with the overlay "
            f"applied ({expected}); the stage does not match the library: run show again",
        )
    stored = _stored_report(deployment.library)
    if "provenance" in stored:
        report["provenance"] = stored["provenance"]
    return report, markdown


def _stored_report(library: Path) -> dict[str, Any]:
    try:
        stored = json.loads((library / REPORT_JSON).read_bytes())
    except (OSError, ValueError):
        return {}
    return stored if isinstance(stored, dict) else {}


def _report_bytes(report: Mapping[str, Any], markdown: str) -> tuple[bytes, bytes]:
    """``(json bytes, markdown bytes)`` in the form ``canopy_mcp.report --write`` writes them."""
    try:
        return (
            (json.dumps(report, indent=2, ensure_ascii=True) + "\n").encode("ascii"),
            markdown.encode("ascii"),
        )
    except (UnicodeEncodeError, TypeError, ValueError):
        raise Refusal(
            EXIT_BAD_VERDICT, "the emitted report is not ASCII JSON and markdown"
        ) from None


# --- accept: the writes --------------------------------------------------------------------------------------


def _create_replaced_dir(deployment: Deployment, user: str, rid: str) -> Path:
    """``var/canopy-push/<user>/replaced/<rid>.<utc>/``, created with an exclusive ``mkdir``."""
    parent = deployment.root / STAGE_SUBPATH / user / REPLACED_DIRNAME
    parent.mkdir(parents=True, exist_ok=True)
    os.chmod(parent, DIR_MODE)
    path = parent / f"{rid}.{_utc_stamp()}"
    try:
        os.mkdir(path, DIR_MODE)
    except FileExistsError:
        raise Refusal(
            EXIT_REPLACED_EXISTS,
            f"the replaced-bytes directory {path} already exists; retry in a second",
        ) from None
    os.chmod(path, DIR_MODE)
    return path


def _copy_replaced(library: Path, into: Path, paths: Sequence[str]) -> None:
    """Copy the bytes of every library file about to be replaced into the replaced-bytes directory."""
    for path in paths:
        _write_file(into / path, (library / path).read_bytes())


def _place_file(root: Path, rel: str, data: bytes) -> None:
    """Write ``data`` at ``root/rel`` by tmp+replace through ``O_NOFOLLOW`` directory fds.

    Every directory on the way is created or forced to 0755, the file is
    written to ``.<name>.<random>.tmp`` beside its destination, fchmod'ed to
    0644, fsynced and renamed over the destination. A failure leaves no tmp.
    """
    parts = rel.split("/")
    fds = [os.open(root, os.O_RDONLY | os.O_NOFOLLOW | os.O_DIRECTORY)]
    try:
        for name in parts[:-1]:
            with contextlib.suppress(FileExistsError):
                os.mkdir(name, DIR_MODE, dir_fd=fds[-1])
            fds.append(os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_DIRECTORY, dir_fd=fds[-1]))
            os.fchmod(fds[-1], DIR_MODE)
        dfd = fds[-1]
        tmp = f".{parts[-1]}.{secrets.token_hex(4)}.tmp"
        fd = os.open(
            tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, FILE_MODE, dir_fd=dfd
        )
        try:
            try:
                os.write(fd, data)
                os.fchmod(fd, FILE_MODE)
                os.fsync(fd)
            finally:
                os.close(fd)
            os.rename(tmp, parts[-1], src_dir_fd=dfd, dst_dir_fd=dfd)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp, dir_fd=dfd)
            raise
        with contextlib.suppress(OSError):
            os.fsync(dfd)
    finally:
        for fd in reversed(fds):
            os.close(fd)


def _prune_replaced(deployment: Deployment, now: float | None = None) -> None:
    """Remove replaced-bytes directories older than 14 days; nothing else, and only ``accept`` calls this."""
    now = time.time() if now is None else now
    root = deployment.root / STAGE_SUBPATH
    if not root.is_dir():
        return
    for user_dir in sorted(root.iterdir()):
        replaced = user_dir / REPLACED_DIRNAME
        if user_dir.is_symlink() or replaced.is_symlink() or not replaced.is_dir():
            continue
        for path in sorted(replaced.iterdir()):
            if not RID_RE.fullmatch(path.name.split(".", 1)[0]):
                continue
            try:
                age = now - path.lstat().st_mtime
            except OSError:
                continue
            if age > STAGE_MAX_AGE_S:
                _remove_tree(path)


# --- accept: the epilogue ------------------------------------------------------------------------------------


def _epilogue(
    deployment: Deployment,
    user: str,
    request: Request,
    reviewer: str,
    digest: str,
    *,
    has_repo: bool,
) -> None:
    """What the admin does next: compare, commit with the path-limited form, build."""
    root = shlex.quote(str(deployment.root))
    if not has_repo:
        _out(
            "this deployment has no repository of its own; put it under version control, "
            "or the replaced panels are kept 14 days only"
        )
    _out(
        f"accepted {user}/{request.rid} -> {ROOT_DIR}{request.target_dir}/ "
        f"(reviewer {reviewer}, digest {digest})"
    )
    _out(f"library: {deployment.library}")
    _out(f"pushed (compare with: git -C {root} diff --cached -- {LIBRARY_PATHSPEC}):")
    for path in sorted(request.files):
        _out(f"  {request.destination(path)}")
    _out(f"  {REPORT_MD}")
    _out(f"  {REPORT_JSON}")
    _out(
        "commit the library with the path-limited form; an uncommitted developer-avenue change "
        "in the library is committed with it:"
    )
    message = shlex.quote(
        f"canopy: accept {user}/{request.rid} ({request.target_dir}), reviewer {reviewer}, digest {digest}"
    )
    _out(f"  git -C {root} add -- {LIBRARY_PATHSPEC}")
    _out(f"  git -C {root} status --ignored -- {LIBRARY_PATHSPEC}     # nothing may be ignored")
    _out(f"  git -C {root} commit -m {message} -- {LIBRARY_PATHSPEC}")
    if effective_image_source(deployment.web_terminals) == "local":
        _out("then build: run osprey build, then osprey up")
    else:
        _out(
            "then build: push to the deployment remote, let the pipeline build and publish the "
            "images, then osprey up"
        )
    _out(
        "until osprey build runs, osprey up and osprey restart refuse on source drift; "
        "osprey up --as-built restarts the stack as last built"
    )


# --- accept -----------------------------------------------------------------------------------------------------


def _accept(deployment: Deployment, args: argparse.Namespace, host_ids: tuple[int, int]) -> int:
    user, rid = _parse_target(args.target)
    deployment_module._check_user(deployment, user)
    _check_reviewer(deployment, args.reviewer)
    if not _HEX64.fullmatch(args.digest):
        raise Refusal(EXIT_DIGEST_MISMATCH, "--digest is not the 64-hex digest show printed")
    with deployment_module._ExclusiveLibraryLock(deployment.library):
        deployment_module._check_fresh(deployment)
        stage, request = _load_stage(deployment, user, rid)
        spec_notes = _read_check_notes(stage)
        judged = judge(scan(deployment.library))
        digest = framed_digest(request, judged.digest)
        if digest != args.digest:
            raise Refusal(
                EXIT_DIGEST_MISMATCH,
                f"the digest of the staged push against the current library is {digest}, not "
                f"{args.digest}; the library or the push changed since show: run show again and "
                "review it anew",
            )
        replacing = _check_collisions(request, judged, args.replace)
        _check_case_against_library(request, judged)
        report, markdown = _emit_report(deployment, user, stage, request, judged, host_ids)
        json_bytes, md_bytes = _report_bytes(report, markdown)
        replaced = [
            *replacing,
            *(name for name in (REPORT_JSON, REPORT_MD) if (deployment.library / name).is_file()),
        ]
        files = [
            {"path": request.destination(path), "sha256": hashlib.sha256(data).hexdigest()}
            for path, data in sorted(request.files.items())
        ]
        replaced_dir = _create_replaced_dir(deployment, user, rid)
        entry = deployment_module._log_entry(
            user,
            rid,
            target_dir=request.target_dir,
            reviewer=args.reviewer,
            digest=digest,
            decision="accepted",
            files=files,
            replaced=replaced,
            replaced_dir=str(replaced_dir),
            spec_unavailable=spec_unavailable_record(spec_notes),
        )
        deployment_module._append_log(deployment, entry)
        try:
            _copy_replaced(deployment.library, replaced_dir, replaced)
            for path in sorted(request.files):
                _place_file(deployment.library, request.destination(path), request.files[path])
            _place_file(deployment.library, REPORT_MD, md_bytes)
            _place_file(deployment.library, REPORT_JSON, json_bytes)
        except OSError as exc:
            raise Refusal(
                EXIT_WRITE_FAILED,
                f"writing into {deployment.library} failed: {type(exc).__name__}: {exc}",
                f"the accept is logged ({user}/{rid}); the library is now refused as stale, "
                "and that refusal prints the recovery",
            ) from None
        spec_line = spec_unavailable_line(spec_notes)
        if spec_line is not None:
            _out(spec_line)
        _epilogue(
            deployment,
            user,
            request,
            args.reviewer,
            digest,
            has_repo=(deployment.root / ".git").exists(),
        )
        _prune_replaced(deployment)
    return EXIT_OK


# --- reject -----------------------------------------------------------------------------------------------------


def _reject(deployment: Deployment, args: argparse.Namespace) -> int:
    user, rid = _parse_target(args.target)
    deployment_module._check_user(deployment, user)
    _check_reviewer(deployment, args.reviewer)
    reason = args.reason
    if not reason.strip() or len(reason) > MAX_NOTE_CHARS or has_forbidden_char(reason):
        raise Refusal(
            EXIT_BAD_REASON,
            f"--reason must be non-empty, at most {MAX_NOTE_CHARS} characters and free of "
            "control, format and surrogate characters",
        )
    target_dir, files = _staged_summary(deployment, user, rid)
    entry = deployment_module._log_entry(
        user,
        rid,
        target_dir=target_dir,
        reviewer=args.reviewer,
        digest=None,
        decision="rejected",
        files=files,
        replaced=[],
        reason=reason,
    )
    deployment_module._append_log(deployment, entry)
    _out(f"rejected {user}/{rid}: {reason}")
    _out("logged; the next show removes the stage of this request, if one exists")
    return EXIT_OK


def _staged_summary(
    deployment: Deployment, user: str, rid: str
) -> tuple[str | None, list[dict[str, str]]]:
    """``(target_dir, files)`` of the checked stage's request when there is one, else ``(None, [])``."""
    try:
        doc = json.loads((deployment.root / STAGE_SUBPATH / user / rid / REQUEST_NAME).read_bytes())
    except (OSError, ValueError):
        return None, []
    if not isinstance(doc, dict) or not isinstance(doc.get("target_dir"), str):
        return None, []
    target_dir = doc["target_dir"]
    files = [
        {"path": f"{ROOT_DIR}{target_dir}/{item['path']}", "sha256": str(item.get("sha256"))}
        for item in doc.get("files") or []
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    ]
    return target_dir, files


# --- entry point ------------------------------------------------------------------------------------


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m osprey.integrations.canopy.push",
        description="Review CANOPY push requests on the host before they enter the screen library.",
    )
    parser.add_argument(
        "--repo", help="the deployment repo (default: the one enclosing the working directory)"
    )
    parser.add_argument(
        "--library", help="review against this library instead of data/canopy/library"
    )
    verbs = parser.add_subparsers(dest="verb", required=True)
    verbs.add_parser("list", help="list every user's pending push requests")
    show = verbs.add_parser("show", help="validate, stage and print one request")
    show.add_argument("target", help="<user>/<rid>")
    accept = verbs.add_parser("accept", help="write a shown request into the library")
    accept.add_argument("target", help="<user>/<rid>")
    accept.add_argument(
        "--reviewer", required=True, help="the admin who reviewed it (an attestation)"
    )
    accept.add_argument("--digest", required=True, help="the digest show printed")
    accept.add_argument(
        "--replace", action="store_true", help="replace the library files the push collides with"
    )
    reject = verbs.add_parser("reject", help="log the rejection of a request")
    reject.add_argument("target", help="<user>/<rid>")
    reject.add_argument(
        "--reviewer", required=True, help="the admin who reviewed it (an attestation)"
    )
    reject.add_argument("--reason", required=True, help="why it is rejected")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: the verb's exit code; every refusal prints its lines to stderr."""
    try:
        args = _parse(argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else EXIT_USAGE
    host_ids = (os.getuid(), os.getgid())
    try:
        deployment_module._refuse_root()
        deployment = deployment_module._load_deployment(args.repo, args.library)
        deployment_module._refuse_non_owner(deployment.library)
        with deployment_module._Lock(deployment):
            if args.verb == "list":
                return _list(deployment, host_ids)
            if args.verb == "accept":
                return _accept(deployment, args, host_ids)
            if args.verb == "reject":
                return _reject(deployment, args)
            return _show(deployment, args.target, host_ids)
    except Refusal as exc:
        for line in exc.lines:
            _say(line)
        return exc.code
    except OSError as exc:
        _say(f"{type(exc).__name__}: {exc}")
        return EXIT_CONTAINER_FAILED
