"""The deployment the verbs act on: its checks, its locks, its freshness and its push log."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import pwd
import shlex
import stat
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osprey.cli.repo_resolver import RepoNotFoundError, find_repo_root
from osprey.deployment.compose_generator import resolve_user_volume_names
from osprey.deployment.runtime_helper import get_runtime_command
from osprey.deployment.web_terminals.roster import (
    USERNAME_CHARSET_RE,
    as_dict,
    rendered_configs,
    resolve_cards,
    roster_user_names,
    tier_privileges,
)
from osprey.integrations.canopy import LIBRARY_SUBPATH, PAYLOAD_DIRNAME
from osprey.integrations.canopy.library import library_is_fresh, regeneration_command
from osprey_connectors.config import load_project_config

from .refusals import (
    EXIT_BAD_USER,
    EXIT_BUSY,
    EXIT_CONTAINER_START,
    EXIT_NO_DEPLOYMENT,
    EXIT_NO_LIBRARY,
    EXIT_NO_VOLUME,
    EXIT_NOT_ON_ROSTER,
    EXIT_NOT_OWNER,
    EXIT_ROOT_UID,
    EXIT_RUNTIME_MISSING,
    EXIT_RUNTIME_PODMAN,
    EXIT_STALE,
    Refusal,
)

#: Where the stages, the lock and the log live below the deployment.
STAGE_SUBPATH = Path("var/canopy-push")
LOCK_SUBPATH = Path("var/canopy/push.lock")
LOG_SUBPATH = Path("var/canopy/push-log.jsonl")
#: The library path the epilogue's git commands name, as the deployment repo holds it.
LIBRARY_PATHSPEC = LIBRARY_SUBPATH.as_posix()


# --- the deployment --------------------------------------------------------------------------


@dataclass(frozen=True)
class Deployment:
    """What the verbs need of the deployment they act on."""

    root: Path
    config: dict[str, Any]
    library: Path
    runtime: str

    @property
    def web_terminals(self) -> dict[str, Any]:
        return as_dict(as_dict(self.config.get("modules")).get("web_terminals"))

    @property
    def roster(self) -> list[str]:
        return roster_user_names(self.web_terminals)

    def volume_of(self, user: str) -> str:
        return str(resolve_user_volume_names(self.config, user)[1])

    def card_of(self, user: str) -> dict[str, Any]:
        # Strict, as the render is: the image named here is the one the render
        # gave this user's card, and a roster the render would refuse names none.
        try:
            entries = resolve_cards(
                self.web_terminals, as_dict(self.config.get("registry")), strict=True
            )
        except ValueError as exc:
            raise Refusal(EXIT_NOT_ON_ROSTER, f"{user} resolves to no render: {exc}") from None
        for entry in entries:
            if entry["name"] == user:
                return entry
        raise Refusal(EXIT_NOT_ON_ROSTER, f"{user} resolves to no render")

    def privileges_of(self, user: str) -> tuple[str, ...]:
        """The deployment-editing privileges the render ``user`` runs holds.

        Read off that render's own rendered ``config.yml``, as lint's privilege
        belt reads it; a render that is unrendered or unreadable holds none.
        """
        render = self.card_of(user).get("render")
        if not isinstance(render, str):
            return ()
        document = rendered_configs(self.config, self.root).get(render)
        if document is None:
            return ()
        return tier_privileges(document)

    def image_of(self, user: str) -> tuple[str, str]:
        """``(image, payload directory inside it)`` of the render ``user`` runs."""
        entry = self.card_of(user)
        return entry["image"], f"{entry['container_project_dir']}/{PAYLOAD_DIRNAME}"


def _effective_uid() -> int:
    return os.geteuid()


def _refuse_root() -> None:
    if _effective_uid() == 0:
        raise Refusal(
            EXIT_ROOT_UID,
            "run push review as the deployment's owner; add that user to the docker group "
            "instead of using sudo",
        )


def _load_deployment(repo: str | None, library: str | None) -> Deployment:
    try:
        root = find_repo_root(Path(repo) if repo else None)
    except RepoNotFoundError as exc:
        raise Refusal(EXIT_NO_DEPLOYMENT, f"no deployment: {exc.message}") from None
    config_path = root / "build" / "config.yml"
    if not config_path.is_file():
        raise Refusal(EXIT_NO_DEPLOYMENT, f"no build in {root}: {config_path} does not exist")
    try:
        config = load_project_config(config_path)
    except Exception as exc:
        raise Refusal(EXIT_NO_DEPLOYMENT, f"cannot read {config_path}: {exc}") from None
    try:
        runtime = get_runtime_command(config)[0]
    except RuntimeError as exc:
        raise Refusal(EXIT_RUNTIME_MISSING, f"no container runtime: {exc}") from None
    if runtime != "docker":
        raise Refusal(EXIT_RUNTIME_PODMAN, "push review requires docker in this release")
    return Deployment(root, config, Path(library) if library else root / LIBRARY_SUBPATH, runtime)


def _refuse_non_owner(library: Path) -> None:
    try:
        status = os.lstat(library)
    except FileNotFoundError:
        raise Refusal(EXIT_NO_LIBRARY, f"no library at {library}") from None
    if not stat.S_ISDIR(status.st_mode):
        raise Refusal(EXIT_NO_LIBRARY, f"{library} is not a directory")
    if status.st_uid != _effective_uid():
        raise Refusal(
            EXIT_NOT_OWNER, f"run as the deployment's owner (uid {status.st_uid} owns {library})"
        )


def _check_user(deployment: Deployment, user: str) -> str:
    """``user``'s agent-data volume name, once it matched the charset, is on the roster and exists."""
    if not USERNAME_CHARSET_RE.fullmatch(user):
        raise Refusal(EXIT_BAD_USER, f"{user!r} is not a roster user name")
    if user not in deployment.roster:
        raise Refusal(EXIT_NOT_ON_ROSTER, f"{user} is not on the web-terminal roster")
    volume = deployment.volume_of(user)
    if not _volume_exists(deployment.runtime, volume):
        raise Refusal(EXIT_NO_VOLUME, f"{user} has no agent-data volume ({volume})")
    return volume


def _volume_exists(runtime: str, volume: str) -> bool:
    try:
        done = subprocess.run(
            [runtime, "volume", "inspect", volume], capture_output=True, check=False, timeout=60
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise Refusal(EXIT_CONTAINER_START, f"cannot run {runtime} volume inspect: {exc}") from None
    return done.returncode == 0


class _Lock:
    """``var/canopy/push.lock``, held exclusively for the run (``EXIT_BUSY`` when another holds it)."""

    def __init__(self, deployment: Deployment):
        self.path = deployment.root / LOCK_SUBPATH
        self.fd = -1

    def __enter__(self) -> _Lock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(self.fd)
            raise Refusal(EXIT_BUSY, f"another push review holds {self.path}") from None
        return self

    def __exit__(self, *_exc: object) -> None:
        os.close(self.fd)


class _SharedLibraryLock:
    """A shared ``flock`` on the library directory, so no accept runs while it is read."""

    def __init__(self, library: Path):
        self.library = library
        self.fd = -1

    def __enter__(self) -> _SharedLibraryLock:
        self.fd = os.open(self.library, os.O_RDONLY)
        fcntl.flock(self.fd, fcntl.LOCK_SH)
        return self

    def __exit__(self, *_exc: object) -> None:
        os.close(self.fd)


class _ExclusiveLibraryLock(_SharedLibraryLock):
    """An exclusive ``flock`` on the library directory: no ``show`` and no store write runs meanwhile."""

    def __enter__(self) -> _ExclusiveLibraryLock:
        self.fd = os.open(self.library, os.O_RDONLY)
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        return self


def _check_fresh(deployment: Deployment) -> None:
    """Refuse unless the library's report covers its files; a report refusal also prints the recovery."""
    freshness = library_is_fresh(deployment.library)
    if freshness:
        return
    lines = [f"the library {deployment.library} is not fresh:", *freshness.lines]
    if any(line.startswith(("digest mismatch", "no report")) for line in freshness.lines):
        lines += _recovery_lines(deployment)
    raise Refusal(EXIT_STALE, *lines)


def _recovery_lines(deployment: Deployment) -> list[str]:
    """What an admin does after an interrupted accept: the discard sequence, then the keep sequence."""
    root = shlex.quote(str(deployment.root))
    on_disk = ", ".join(f"{user}/{rid}" for user, rid in _accepts_on_disk(deployment)) or "none"
    return [
        "if an accept was interrupted, recover in one of two ways:",
        f"  discard: remove the added paths and any leftover .*.tmp, restore each changed path from "
        f"the replaced-bytes directory its log entry names ({LOG_SUBPATH}, replaced_dir) or from git, "
        "then run show and accept again",
        f"  discard, blanket: git -C {root} restore --source=HEAD --staged --worktree -- "
        f"{LIBRARY_PATHSPEC} && git -C {root} clean -fd -- {LIBRARY_PATHSPEC}",
        "    WARNING: this discards every uncommitted library change, including the logged accepts "
        f"whose files are on disk and would be lost: {on_disk}",
        f"  keep: review the tree, run {regeneration_command(deployment.library)}, then show "
        "(new digest) and accept; the landed files then equal the overlay and do not collide",
    ]


def _accepts_on_disk(deployment: Deployment) -> list[tuple[str, str]]:
    """Every logged ``(volume_user, rid)`` accept of which at least one file is in the library as logged."""
    found: list[tuple[str, str]] = []
    for entry in _log_entries(deployment):
        if entry.get("decision") != "accepted":
            continue
        pair = (str(entry.get("volume_user")), str(entry.get("rid")))
        files = entry.get("files")
        if pair in found or not isinstance(files, list):
            continue
        for item in files:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                continue
            try:
                data = (deployment.library / item["path"]).read_bytes()
            except OSError:
                continue
            if hashlib.sha256(data).hexdigest() == item.get("sha256"):
                found.append(pair)
                break
    return found


def _log_entries(deployment: Deployment) -> list[dict[str, Any]]:
    """Every JSON object line of the push log, tolerant of anything else."""
    log = deployment.root / LOG_SUBPATH
    entries: list[dict[str, Any]] = []
    try:
        lines = log.read_bytes().splitlines()
    except FileNotFoundError:
        return entries
    for line in lines:
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict):
            entries.append(entry)
    return entries


def _os_user() -> str:
    uid = os.getuid()
    try:
        return pwd.getpwuid(uid).pw_name
    except KeyError:
        return str(uid)


def _log_entry(
    user: str,
    rid: str,
    *,
    target_dir: str | None,
    reviewer: str,
    digest: str | None,
    decision: str,
    files: list[dict[str, str]],
    replaced: list[str],
    **extra: Any,
) -> dict[str, Any]:
    """One push-log entry.

    ``extra`` is ``replaced_dir`` and ``spec_unavailable`` (the screens checked
    without the staged spec and why, or ``None``) for an acceptance, and
    ``reason`` for a rejection.
    """
    return {
        "rid": rid,
        "volume_user": user,
        "target_dir": target_dir,
        "reviewer_attested": reviewer,
        "os_user": _os_user(),
        "sudo_user": os.environ.get("SUDO_USER") or None,
        "digest": digest,
        "decision": decision,
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": files,
        "replaced": replaced,
        **extra,
    }


def _append_log(deployment: Deployment, entry: Mapping[str, Any]) -> None:
    """Append one entry to the push log and ``fsync`` it: the attestation exists before any byte lands."""
    path = deployment.root / LOG_SUBPATH
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(entry, sort_keys=True, ensure_ascii=True) + "\n").encode("ascii")
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        os.write(fd, line)
        os.fsync(fd)
    finally:
        os.close(fd)
