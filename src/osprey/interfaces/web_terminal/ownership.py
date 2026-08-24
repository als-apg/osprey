"""Where a Scaffold Gallery claim is durably recorded, and how it is read back.

A claim is a statement that an artifact is the operator's. That statement has
to be written somewhere it will still be true tomorrow, and where that is
depends entirely on what the gallery is running on top of:

* **Profile** — the project names a profile that this process can reach *and
  that is genuinely editable*. The profile is the source of truth: ``osprey
  build`` copies its convention directories into the project and derives
  ``scaffold.user_owned`` from what it copied. Claims move the artifact there
  (``osprey scaffold claim``, callable as
  :func:`osprey.cli.scaffold_cmd.claim_into_profile`) and the gallery reads and
  saves the profile's copy from then on. Writing the project copy instead would
  be overwritten by the next ``osprey build``. A profile baked into an image is
  not this: see :data:`RENDER_ZONE_READONLY_ENV`.

* **Volume** — a deployed web terminal. The project tree is baked into the
  image, so every write to it is destroyed on the next container recreation,
  and destroyed *silently*: the claim reports success, the gallery shows it as
  user-owned, and then it evaporates. The per-user ``claude-config`` volume is
  the only surface in that container that outlives the image, so ownership
  records and the claimed bodies both live there::

      $CLAUDE_CONFIG_DIR/osprey/scaffold/
          user_owned.json        ownership records, keyed by canonical name
          files/<output_path>    the content each claim is a claim *of*

* **Degraded** — the project names a profile, that profile cannot be reached,
  and there is no volume either. Nothing here is durable, so a claim refuses
  with the same message ``osprey scaffold claim`` gives. Falling back to
  ``config.yml`` would make the gallery a back door around the single-writer
  contract: it would report success, and the next build would erase it.

* **Config** — the project names no profile at all (a pre-profile manifest).
  Ownership goes in ``config.yml``, which is what such a project has always
  used and, on a host, is at least durable.

The mode question is "can this process reach a durable source of truth", and it
must have ONE answer per project. Privilege is not part of it: inside a
container the root entrypoint and the unprivileged server look at the same tree
and must resolve the same surface, which is why an image-baked profile is
skipped outright rather than being decided by ``os.access`` — see
:func:`resolve_ownership` and :data:`RENDER_ZONE_READONLY_ENV`. An operator who
bind-mounts a real, durable profile into a container and wants profile mode
back unsets that marker.

The gallery *reads* the union of build-derived ownership (``config.yml``) and
the per-user store; :func:`merge_user_owned` states which wins.

Artifact names use the same vocabulary as ``scaffold.user_owned``: the full
path below the convention's destination, so a nested command is
``commands/osprey/scan``, never ``commands/scan``. Two spellings for one
artifact would make the union meaningless.
"""

from __future__ import annotations

import json
import logging
import os
import posixpath
import stat
import tempfile
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any

try:  # pragma: no cover - present on every POSIX deployment target
    import fcntl
except ImportError:  # pragma: no cover - Windows has no flock
    fcntl = None  # type: ignore[assignment]

from osprey.services.python_executor.refusal_audit import record_protected_refusal

logger = logging.getLogger(__name__)

#: The per-user claude-config volume mount point, exported by the web-terminal
#: compose service (it is also $HOME inside that container).
CLAUDE_CONFIG_ENV = "CLAUDE_CONFIG_DIR"

#: Set to ``1`` by ``Dockerfile.j2`` on every image the build produces, and by
#: nothing else. Its literal meaning is "the render zone in this image is
#: root-owned"; its consequence for ownership is the stronger statement that
#: the whole project tree — profile included — came out of an image and is
#: RE-CREATED FROM THAT IMAGE on every container recreation.
#:
#: :func:`resolve_ownership` reads it because the container render rewrites the
#: manifest's ``profile_path_abs`` to ``/app/<project>/profile.yml``, a file
#: that genuinely exists in the image. Writability alone therefore cannot tell
#: an editable profile from a baked copy: the app (uid 1000) cannot write the
#: root-owned tree and correctly falls through, while the root entrypoint can
#: and used to resolve ``PROFILE`` — the same project, two answers, decided by
#: which process asked. ``osprey.interfaces.web_terminal.app`` reads the same
#: variable to skip the render-zone writes it would make on a host.
RENDER_ZONE_READONLY_ENV = "OSPREY_RENDER_ZONE_READONLY"

#: Project-relative prefixes the store is allowed to address, matching what the
#: ownership system covers (``ownership_canonical`` in
#: :mod:`osprey.cli.build_persistence`). The store is data on a mounted volume,
#: so its paths decide where a restore writes: without this, an entry naming
#: ``.env`` or ``config.yml`` would have the app overwrite them at startup.
OWNABLE_PREFIXES = (".claude/", "services/")

#: Store layout, namespaced under ``osprey/`` because $CLAUDE_CONFIG_DIR is
#: shared with Claude Code's own state (and is the user's $HOME).
STORE_SUBPATH = ("osprey", "scaffold")
INDEX_FILENAME = "user_owned.json"
CONTENT_DIRNAME = "files"
LOCK_FILENAME = ".user_owned.lock"

SCHEMA_VERSION = 1

CLAIMED = "claimed"
RELEASED = "released"


class OwnershipStoreError(RuntimeError):
    """A durable write did not land, so no ownership record was made.

    Distinct from :class:`~osprey.cli.scaffold_cmd.ScaffoldClaimError`, which
    says a claim is *not allowed*. This one says it was allowed and the volume
    would not take it — an operator-visible failure either way, but not one the
    operator can fix by naming something else.
    """


class OwnershipMode(Enum):
    """Which surface a claim is durably recorded on."""

    PROFILE = "profile"
    VOLUME = "volume"
    DEGRADED = "degraded"
    CONFIG = "config"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def generated_project_paths() -> frozenset[str]:
    """Project paths the build generates, which ownership must never freeze.

    Read from :data:`~osprey.cli.profile_conventions.RESERVED_EXACT_PATHS` so
    there is one table with one owner: a generated file protected there later
    is protected here the moment it is added.
    """
    from osprey.cli.profile_conventions import RESERVED_EXACT_PATHS

    return RESERVED_EXACT_PATHS


def _safe_relative(output_path: str) -> PurePosixPath | None:
    """Validate a project-relative output path before it indexes the store.

    Names reach this module from HTTP request paths, and records reach it from
    a mounted volume that an older OSPREY — or anything else with write access
    to the user's ``$HOME`` — may have written. Neither is trusted, so this is
    the one place a store path is admitted, and every operation that turns a
    path into a file goes through it: read, write, drop, and restore.

    Two rules, and the second is why the restore is safe rather than merely
    contained. A path must sit inside the ownable tree, so a stored copy cannot
    address ``.env`` or ``config.yml``. And it must not be one the build
    generates: ``.claude/hooks/hook_config.json`` is inside ``.claude/`` and
    passes the first rule easily, but it is the write-safety layer's runtime
    configuration, and a copy restored over it at container start would shadow
    every later change to what generates it.
    """
    candidate = PurePosixPath(output_path)
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        logger.warning("Refusing unsafe artifact output path: %r", output_path)
        return None
    if not any(str(candidate).startswith(prefix) for prefix in OWNABLE_PREFIXES):
        logger.warning(
            "Refusing artifact output path outside %s: %r",
            " or ".join(OWNABLE_PREFIXES),
            output_path,
        )
        return None
    if str(candidate) in generated_project_paths():
        logger.warning(
            "Refusing generated artifact output path %r: the build owns this file, and a "
            "stored copy would shadow every later change to what generates it.",
            output_path,
        )
        return None
    return candidate


#: Surface name the restore records its refusals under, so an operator reading
#: ``protected-writes.jsonl`` can tell a body refused at container start from
#: one refused in the gallery. They arrive by different routes and at different
#: privilege levels, and the restore's are the ones worth waking up for.
RESTORE_REFUSAL_SURFACE = "scaffold_restore"


def _audit_restore_refusal(path: str, *, channel: str, reason: str, because: str) -> None:
    """Record one refused restore and say so in the log.

    Every refusal on the restore path goes through here, so none can reach the
    log without also reaching the durable audit record -- a refusal only this
    process ever saw is the one an operator cannot investigate afterwards. The
    call sites decide *what* is refused and how to phrase it; the surface, the
    doubled path and the sentence they share are decided once.

    Args:
        path: Project-relative path the store record aimed at.
        channel: Channel that owns it, or the phrase standing in for one.
        reason: Short slug queried out of ``protected-writes.jsonl``.
        because: The clause completing "Refusing to restore <path> from the
            ownership store: <because>."
    """
    record_protected_refusal(
        surface=RESTORE_REFUSAL_SURFACE,
        target_file=path,
        key_or_path=path,
        channel=channel,
        reason=reason,
    )
    logger.warning("Refusing to restore %s from the ownership store: %s.", path, because)


def _reserved_subtree_channel(project_rel: str) -> str | None:
    """The channel owning a reserved subtree ROOT, or ``None``.

    ``is_reserved_write`` matches ``.claude/rules/**`` with :mod:`fnmatch`,
    which needs the separator to be there: it answers for
    ``.claude/rules/x.md`` and for the empty-suffix ``.claude/rules/``, but not
    for the bare ``.claude/rules``. The bare name is not a curiosity — it is
    the directory itself, which is what a writer names to replace, move or
    remove the whole convention at once, and it is spelled that way by
    ``PurePosixPath`` on every path that has been through ``normpath``.

    Answered here rather than by widening the pattern table, because the table
    is read verbatim by :func:`~osprey.cli.profile_conventions.is_reserved_write`
    for FILE paths and an entry that matched the bare directory would also have
    to keep matching everything under it — two jobs, one string. The channel
    returned is the subtree's own, so the refusal reads identically whether the
    writer named the directory or a file inside it.
    """
    from osprey.cli.profile_conventions import RESERVED_PATH_PATTERNS

    normalized = posixpath.normpath(PurePosixPath(project_rel).as_posix()).casefold()
    for reserved in RESERVED_PATH_PATTERNS:
        if not reserved.pattern.endswith("/**"):
            continue
        root = reserved.pattern[:-3].casefold()
        if normalized == root or normalized.startswith(root + "/"):
            return reserved.channel
    return None


def reserved_write_channel(project_dir: Path, output_path: str) -> str | None:
    """The channel owning the file *output_path* actually addresses, or ``None``.

    Two questions, and the second is the one a path can hide from the first:

    1. Is the path itself protected? That is
       :func:`~osprey.cli.profile_conventions.is_reserved_write`, and any
       non-``None`` answer is a refusal — including the one it gives for a path
       it cannot judge — plus :func:`_reserved_subtree_channel` for the bare
       directory name its ``**`` patterns cannot match.
    2. Where do the bytes land once symlinks are resolved? Every writer in this
       package reaches disk through the filesystem, which follows links: a
       write opens the target and a delete removes what the link points at. So
       a link at an unprotected name (``.claude/agents/x.md`` ->
       ``../rules/safety.md``) is lexically an agent and physically a rule, and
       a resolved path that leaves the project altogether cannot be vouched for
       at all.

    Shared rather than duplicated because the answer must not depend on which
    door the write came through. The gallery asks it before a save, a claim, a
    create and a delete; the restore asks it before installing a stored body as
    root at container start. A second copy of this would be a second policy,
    and the copy that runs as root is the worst place to find out they drifted.

    Args:
        project_dir: Root the path is relative to.
        output_path: Project-relative posix path the writer would write.

    Returns:
        A phrase naming the owning channel, or ``None`` when the path is the
        caller's to write.
    """
    from osprey.cli.profile_conventions import NOT_PROJECT_RELATIVE_CHANNEL, is_reserved_write

    channel = is_reserved_write(output_path) or _reserved_subtree_channel(output_path)
    if channel is not None:
        return channel

    try:
        resolved = (project_dir / output_path).resolve()
        root = project_dir.resolve()
    except OSError:
        # An unresolvable path is one this function cannot vouch for, and
        # "cannot judge" must not read as "writable".
        return NOT_PROJECT_RELATIVE_CHANNEL
    if not resolved.is_relative_to(root):
        return NOT_PROJECT_RELATIVE_CHANNEL

    # Inside the project, but not necessarily at the file the name spelled.
    landed = resolved.relative_to(root).as_posix()
    return is_reserved_write(landed) or _reserved_subtree_channel(landed)


# ── Ownership records ────────────────────────────────────────────────


@dataclass(frozen=True)
class OwnershipRecord:
    """One artifact's entry in the durable store."""

    name: str
    state: str
    output_path: str | None = None
    updated_at: str | None = None

    @property
    def claimed(self) -> bool:
        return self.state == CLAIMED

    @classmethod
    def from_json(cls, name: str, entry: dict[str, Any]) -> OwnershipRecord | None:
        state = entry.get("state")
        if state not in (CLAIMED, RELEASED):
            return None

        def _text(key: str) -> str | None:
            value = entry.get(key)
            return value if isinstance(value, str) else None

        return cls(
            name=name,
            state=state,
            output_path=_text("output_path"),
            updated_at=_text("updated_at"),
        )

    def to_json(self) -> dict[str, Any]:
        entry: dict[str, Any] = {"state": self.state, "updated_at": self.updated_at or _now()}
        if self.output_path:
            entry["output_path"] = self.output_path
        return entry


@dataclass(frozen=True)
class OwnershipStore:
    """Ownership records and claimed bodies on a durable volume.

    Reads never raise. The store lives on a volume that a crashed write or an
    out-of-space container can truncate, and a corrupt file must not take the
    gallery down with it: a gallery that shows build-derived ownership only is
    recoverable, one that 500s on every request is not.

    Writes take an exclusive lock and land atomically, so two requests claiming
    different artifacts at the same moment cannot lose one another's record.
    """

    root: Path

    @property
    def index_path(self) -> Path:
        return self.root / INDEX_FILENAME

    @property
    def content_dir(self) -> Path:
        return self.root / CONTENT_DIRNAME

    # ── Read ──────────────────────────────────────────────────────────

    def read(self) -> dict[str, OwnershipRecord]:
        """Return ownership records by canonical name. Never raises."""
        entries = self._read_document()[0].get("artifacts")
        if not isinstance(entries, dict):
            return {}

        records: dict[str, OwnershipRecord] = {}
        for name, entry in entries.items():
            if not isinstance(name, str) or not isinstance(entry, dict):
                continue
            record = OwnershipRecord.from_json(name, entry)
            if record is not None:
                records[name] = record
        return records

    def read_content(self, output_path: str) -> str | None:
        """Return the durable body for *output_path*, or None if there isn't one.

        The body is read through :meth:`_contained_content_path`, never by
        joining and opening: the content directory lives on the agent-writable
        claude-config volume, so what a store path points AT is as untrusted as
        the path itself.
        """
        relative = _safe_relative(output_path)
        if relative is None:
            return None
        resolved = self._contained_content_path(relative, output_path)
        if resolved is None:
            return None
        try:
            # O_NOFOLLOW closes the window between the resolve above and this
            # open: `resolved` contains no symlink by construction, so the only
            # way its final component is one now is that something replaced it
            # in between -- which, on a volume the agent writes, is a race worth
            # refusing rather than one worth assuming away. It guards the final
            # component only, which is why containment is checked above.
            #
            # O_NONBLOCK closes the other half of that same window. If what got
            # swapped in is a FIFO, a blocking open (or read) waits for a writer
            # that never comes -- and this walk runs in the container entrypoint
            # before the server exists, so the container would hang at start
            # forever. Every step here is written to fail open; an unbounded
            # wait is the one failure mode that is neither open nor closed.
            fd = os.open(resolved, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except OSError:
            return None
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                return None
            # Only now, on a descriptor proven to be a regular file, where
            # blocking cannot mean waiting on a writer -- a short read on a
            # non-blocking regular file would otherwise be a silent truncation.
            os.set_blocking(fd, True)
            with os.fdopen(fd, "r", encoding="utf-8") as handle:
                fd = -1  # fdopen owns it now, and its close is the only one
                return handle.read()
        except (OSError, UnicodeDecodeError):
            return None
        finally:
            if fd >= 0:
                os.close(fd)

    def _contained_content_path(self, relative: PurePosixPath, output_path: str) -> Path | None:
        """The real file a store body names, or ``None`` if it escapes the store.

        The destination side of a restore is already hardened -- a record whose
        write would land on a reserved file is refused. This is the SOURCE side,
        and it was not.

        The content directory is on the per-user claude-config volume, which is
        the agent's own ``$HOME`` and writable by it, while
        :func:`rehydrate` now runs from the container entrypoint as **root**.
        Reading a body with :meth:`~pathlib.Path.read_text` follows symlinks, so
        the agent could plant ``files/.claude/agents/x.md`` as a link onto any
        root-readable file -- or, just as effectively, make an intermediate
        directory the link -- and have root copy that file's bytes into the
        render zone, where the agent can read them. An exfiltration primitive
        built out of a file the agent is allowed to write.

        So the join is resolved and re-checked against the resolved content
        root, and what it lands on has to be a file this store could plausibly
        have written. Each way of failing that is refused the same way a
        reserved destination is -- audited under
        :data:`RESTORE_REFUSAL_SURFACE`, skipped, never raised, one poisoned
        record must not cost the operator the rest of their claimed bodies --
        but under its OWN reason, because they are different things for an
        operator to find in the log and would otherwise be indistinguishable:

        * **escapes the store** -- the resolved path is outside the content
          directory. A symlink, at the final component or at any directory
          above it.
        * **not a regular file** -- inside the store, but a FIFO, device or
          directory. Nothing to exfiltrate, but nothing this store wrote
          either, and a FIFO specifically is a way to stall the reader.
        * **a hard link** -- inside the store, a regular file, and still not
          one this store created: bodies are always written fresh through
          ``NamedTemporaryFile`` + :func:`os.replace`, so a legitimate body has
          exactly one link. More than one means the same inode is reachable
          under a name the store does not control, which is the escape again
          with the symlink removed. (``st_dev`` is deliberately NOT also
          required to match: a legitimate store whose content directory sits on
          its own bind mount would fail that, and refusing a real body is worse
          than the narrow case it would add.)

        Args:
            relative: Store-relative body path, already through
                :func:`_safe_relative`.
            output_path: The record's own spelling, for the audit record.

        Returns:
            The resolved regular file to read, or ``None``.
        """
        root = self.content_dir
        try:
            resolved = (root / relative).resolve()
            root_resolved = root.resolve()
        except OSError:
            return None

        if not resolved.exists():
            return None  # no body kept for this record; not an escape

        if not resolved.is_relative_to(root_resolved):
            return self._refuse_body(
                output_path,
                resolved,
                channel="a file outside the ownership store's own content directory",
                reason="ownership store body escapes the store",
                detail="it resolves outside the store's content directory",
            )

        try:
            st = resolved.stat()
        except OSError:
            return None

        if not stat.S_ISREG(st.st_mode):
            return self._refuse_body(
                output_path,
                resolved,
                channel="something in the ownership store that is not a file",
                reason="ownership store body is not a regular file",
                detail="it is not a regular file",
            )

        if st.st_nlink > 1:
            return self._refuse_body(
                output_path,
                resolved,
                channel="a file the ownership store did not create",
                reason="ownership store body is a hard link",
                detail=f"it has {st.st_nlink} links, so the same inode is reachable elsewhere",
            )

        return resolved

    def _refuse_body(
        self, output_path: str, resolved: Path, *, channel: str, reason: str, detail: str
    ) -> None:
        """Audit one refused store body and return ``None``.

        Shared by every branch of :meth:`_contained_content_path` so the three
        refusals differ only in what they say, never in whether they are
        recorded -- a refusal that reached no audit log is one only this
        process ever saw.
        """
        _audit_restore_refusal(
            output_path, channel=channel, reason=reason, because=f"{detail} ({resolved})"
        )
        return None

    def _read_document(self) -> tuple[dict[str, Any], bool]:
        """Return ``(document, was_corrupt)`` — an unreadable store reads empty."""
        try:
            raw = self.index_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}, False
        except OSError as exc:
            logger.warning("Scaffold ownership store unreadable at %s: %s", self.index_path, exc)
            return {}, False

        try:
            document = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.warning("Scaffold ownership store is corrupt (%s): %s", self.index_path, exc)
            return {}, True

        if not isinstance(document, dict):
            logger.warning("Scaffold ownership store is not an object: %s", self.index_path)
            return {}, True
        return document, False

    # ── Write ─────────────────────────────────────────────────────────

    def claim(self, name: str, output_path: str, content: str | None = None) -> None:
        """Record ownership of *name*, and the body it is a claim of.

        Record and body are written together. A record whose body was not kept
        would survive a container recreation as a claim on the framework's own
        text: the gallery would say "user-owned" and show the operator someone
        else's file.

        *content* is ``None`` for artifacts the gallery does not edit as text —
        a skill or a service is a whole directory — which are tracked by record
        alone.

        Raises:
            OwnershipStoreError: The body was given and could not be kept. The
                record is not written: a claim recorded against a body that is
                missing (an unstorable path) or truncated (a full volume) is
                the exact failure described above, and a loud one now beats a
                silent one at the next container start.
        """
        if content is not None and not self.write_content(output_path, content):
            raise OwnershipStoreError(
                f"Could not persist {output_path} to the ownership store at {self.root}. "
                "Nothing was recorded — see the log for why the write failed."
            )
        record = OwnershipRecord(
            name=name,
            state=CLAIMED,
            output_path=output_path,
            updated_at=_now(),
        )
        with self._mutate() as entries:
            entries[name] = record.to_json()

    def release(self, name: str, output_path: str | None = None) -> None:
        """Record that *name* is no longer user-owned, and drop its body.

        A tombstone, not a deletion. ``config.yml`` is read-only at runtime in
        this mode, so a release that merely forgot the entry would be undone on
        the next read by the build-derived ownership the operator just gave up
        — the button would report success and change nothing.
        """
        with self._mutate() as entries:
            previous = entries.get(name)
            stored = output_path
            if stored is None and isinstance(previous, dict):
                candidate = previous.get("output_path")
                stored = candidate if isinstance(candidate, str) else None
            if stored:
                self._drop_content(stored)
            entries[name] = OwnershipRecord(name=name, state=RELEASED, updated_at=_now()).to_json()

    def write_content(self, output_path: str, content: str) -> bool:
        """Persist one artifact body atomically. Returns whether it landed.

        Atomic for the same reason the index is: a body half-written by a
        crash or a full volume still reads back as a body, and
        :func:`rehydrate` would restore that truncation into the project tree
        as the operator's own file. A body either replaces the previous one or
        leaves it alone.
        """
        relative = _safe_relative(output_path)
        if relative is None:
            return False
        target = self.content_dir / relative
        staged: Path | None = None
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                "w",
                dir=target.parent,
                prefix=f".{target.name}-",
                suffix=".tmp",
                delete=False,
                encoding="utf-8",
            ) as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
                staged = Path(handle.name)
            os.replace(staged, target)
        except OSError as exc:
            logger.warning("Could not persist %s to the ownership store: %s", output_path, exc)
            if staged is not None:
                staged.unlink(missing_ok=True)
            return False
        return True

    def _drop_content(self, output_path: str) -> None:
        relative = _safe_relative(output_path)
        if relative is None:
            return
        (self.content_dir / relative).unlink(missing_ok=True)

    @contextmanager
    def _locked(self):
        """Hold an exclusive lock for one read-modify-write cycle."""
        self.root.mkdir(parents=True, exist_ok=True)
        if fcntl is None:  # pragma: no cover - POSIX-only deployment target
            yield
            return
        with open(self.root / LOCK_FILENAME, "w", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    @contextmanager
    def _mutate(self):
        """Yield the raw entries map for editing, then persist it."""
        with self._locked():
            document, corrupt = self._read_document()
            if corrupt:
                self._quarantine()
                document = {}
            entries = document.get("artifacts")
            if not isinstance(entries, dict):
                entries = {}
            yield entries
            document["artifacts"] = entries
            self._write_document(document)

    def _write_document(self, document: dict[str, Any]) -> None:
        """Replace the index atomically, so a reader never sees half a file."""
        document["version"] = SCHEMA_VERSION
        with tempfile.NamedTemporaryFile(
            "w", dir=self.root, prefix=".user_owned-", suffix=".tmp", delete=False, encoding="utf-8"
        ) as handle:
            json.dump(document, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
            staged = Path(handle.name)
        os.replace(staged, self.index_path)

    def _quarantine(self) -> None:
        """Set a corrupt index aside rather than deleting it.

        The operator gets to see what was there; the gallery gets a store it
        can write. Overwriting in place would destroy the only evidence of how
        the volume was damaged.
        """
        spoiled = self.index_path.with_suffix(".json.corrupt")
        try:
            os.replace(self.index_path, spoiled)
            logger.warning("Moved corrupt scaffold ownership store aside: %s", spoiled)
        except OSError as exc:  # pragma: no cover - only on an unwritable volume
            logger.warning("Could not quarantine corrupt store %s: %s", self.index_path, exc)


# ── What may be claimed ──────────────────────────────────────────────


def refuse_generated_path(name: str, output_path: str) -> None:
    """Refuse ownership of a project path some other channel generates.

    The CLI claim path applies this rule already (``_refuse_generated`` in
    :mod:`osprey.cli.scaffold_cmd`), so the two gallery modes that hand claims
    to it inherit it. The two that record ownership themselves — a container
    writing the volume, a pre-profile project writing ``config.yml`` — do not,
    and without this they are a way around it.

    It matters most in exactly the mode where the CLI is furthest away.
    ``.claude/hooks/hook_config.json`` is the write-safety layer's runtime
    configuration, rendered from the enabled ``mcp_servers`` and
    ``config.control_system.write_tools``; frozen onto a per-user volume it
    would be restored over the freshly generated file at every container start,
    so a rebuilt image would ship a new write-tool set and the container would
    keep running the operator's stale one, silently and indefinitely.

    :func:`_safe_relative` refuses these paths too, so the store cannot hold one
    however a record naming it got there. This is the operator-facing half:
    containment stops the damage silently, and a refusal says why.

    Raises:
        ScaffoldClaimError: If *output_path* is a reserved generated path.
    """
    from osprey.cli.profile_conventions import reserved_path_channel

    if output_path not in generated_project_paths():
        return

    from osprey.cli.scaffold_cmd import ScaffoldClaimError

    channel = reserved_path_channel(output_path) or "the build itself"
    raise ScaffoldClaimError(
        f"'{name}' ({output_path}) is generated, not authored: it is written by {channel}.\n\n"
        "  Owning it would freeze this copy against the build that regenerates it, so it\n"
        "  would go on shadowing every later change to what generates it — and in a\n"
        "  deployed container it would be restored over the freshly generated file on\n"
        "  every start.\n\n"
        "  Change what generates it instead. Nothing was written."
    )


def refuse_unstorable_path(name: str, output_path: str) -> None:
    """Refuse ownership the per-user store has no way to hold.

    Only paths under :data:`OWNABLE_PREFIXES` can be written to the store, so a
    record naming anything else would be a claim with no body behind it — the
    failure :meth:`OwnershipStore.claim` describes, arriving one layer earlier
    where it can still be reported as a refusal rather than a broken write.

    Raises:
        ScaffoldClaimError: If *output_path* is outside the ownable tree.
    """
    if _safe_relative(output_path) is not None:
        return

    from osprey.cli.scaffold_cmd import ScaffoldClaimError

    raise ScaffoldClaimError(
        f"'{name}' ({output_path}) cannot be owned from a deployed container: the\n"
        f"  per-user store holds artifacts under {' and '.join(OWNABLE_PREFIXES)} only.\n\n"
        "  Recording it would report success and keep no copy of the file, so the next\n"
        "  container recreation would bring back the framework's own version and still\n"
        "  call it yours.\n\n"
        "  Carry this one in the profile instead, and rebuild. Nothing was written."
    )


# ── Mode resolution ──────────────────────────────────────────────────


@dataclass(frozen=True)
class Ownership:
    """The surface this project's claims are durably recorded on."""

    mode: OwnershipMode
    store: OwnershipStore | None = None
    profile_root: Path | None = None


def resolve_ownership(project_dir: Path, env: dict[str, str] | None = None) -> Ownership:
    """Decide where claims for *project_dir* durably belong.

    The topology answers this, not the caller's privilege. ``PROFILE`` means
    "the profile repo is the editable source of truth here" — a claim written
    into it survives, because the next build reads it back. Inside an image it
    is none of those things, for root as much as for anyone else.

    Resolution order:

    0. This is a CONTAINER RENDER (:data:`RENDER_ZONE_READONLY_ENV` is ``1``) →
       the profile is skipped entirely, whatever its permissions say, and
       resolution continues at 2. The container render rewrites the manifest's
       ``profile_path_abs`` to a path that *exists in the image*, so the
       writability test below stops being a test of anything: the app, as the
       unprivileged runtime user, cannot write the root-owned tree and falls
       through to the volume, while the root entrypoint can write it and used
       to resolve ``PROFILE`` — never reading the durable store, restoring
       nothing, and leaving the restore path's reserved-path gate unexercised
       on the one caller that runs as root. Two answers for one project,
       decided by who asked. A write to that baked profile would also be a lie
       in its own right: it lives in the container's writable layer and is gone
       on the next recreation, which is exactly what ``DEGRADED`` exists to
       refuse. An operator who bind-mounts a real, durable profile into a
       container and wants profile semantics unsets the variable.
    1. The profile resolves and is writable → PROFILE. It is the source of
       truth wherever it can genuinely be reached — the bare-host case, which
       this rule leaves exactly as it was.
    2. A per-user volume is mounted → VOLUME. Durable across recreation.
    3. The manifest *names* a profile this process cannot reach → DEGRADED.
       Writes refuse. This is the case that must not silently fall back:
       ``config.yml`` in a profile-built project is derived output, so a claim
       recorded there would report success and be erased by the next build.
    4. The manifest names no profile at all → CONFIG. A pre-profile project,
       where config.yml is what ownership has always meant.

    Args:
        project_dir: The render whose claims are being resolved.
        env: Environment to read, for tests. Defaults to ``os.environ``.
    """
    environ = os.environ if env is None else env

    if not is_container_render(environ):
        profile_root = _reachable_profile_root(project_dir)
        if profile_root is not None:
            return Ownership(OwnershipMode.PROFILE, profile_root=profile_root)

    store = _volume_store(environ)
    if store is not None:
        return Ownership(OwnershipMode.VOLUME, store=store)

    if _names_a_profile(project_dir):
        return Ownership(OwnershipMode.DEGRADED)

    return Ownership(OwnershipMode.CONFIG)


def is_container_render(environ: Mapping[str, str] | None = None) -> bool:
    """Whether this process is looking at a project tree baked into an image.

    The ONE reading of :data:`RENDER_ZONE_READONLY_ENV`. Public because it has
    two callers with one meaning: this module skips the image-baked profile,
    and :mod:`osprey.interfaces.web_terminal.app` skips the render-zone writes
    it would make on a host. Spelled once so the two cannot come apart on a
    value like ``"true"`` -- a server that decided it was on a host while the
    gallery decided it was in a container is the split-brain this replaces.

    Args:
        environ: Environment to read. Defaults to ``os.environ``.
    """
    return (os.environ if environ is None else environ).get(
        RENDER_ZONE_READONLY_ENV, ""
    ).strip() == "1"


def _names_a_profile(project_dir: Path) -> bool:
    """Whether the project's manifest records a profile at all.

    Distinguishes "built from a profile I cannot reach from here" from "built
    before profiles existed" — the same key the deploy side reads, so the two
    agree about what a project is.
    """
    try:
        from osprey.cli.templates.manifest import manifest_profile_path

        return manifest_profile_path(project_dir) is not None
    except Exception as exc:  # noqa: BLE001 - the gallery must open regardless
        logger.warning("Could not read the manifest for %s: %s", project_dir, exc)
        return False


def _reachable_profile_root(project_dir: Path) -> Path | None:
    """The profile root this project was built from, if it is reachable here.

    Delegates to ``scaffold_cmd`` rather than re-deriving: "is a profile
    reachable" already has one implementation there, behind the same checks the
    CLI claim path applies (manifest path present, profile file on disk,
    profile root not inside the installed package). A second copy in the web
    layer would be one rule in two places, drifting apart.
    """
    from osprey.cli.scaffold_cmd import profile_root_or_none

    try:
        root = profile_root_or_none(project_dir)
    except Exception as exc:  # noqa: BLE001 - the gallery must open regardless
        logger.warning("Could not resolve the profile for %s: %s", project_dir, exc)
        return None
    if root is None or not os.access(root, os.W_OK):
        # An unwritable profile root is not a place a claim can land; treat it
        # as unreachable so the fallback surfaces rather than a failed write.
        #
        # Writability is a necessary condition, never a sufficient one: on a
        # container render the baked profile is writable BY ROOT and is still
        # not a source of truth, which is why resolve_ownership decides that
        # case before it ever gets here.
        return None
    return root


def _volume_store(environ: Mapping[str, str]) -> OwnershipStore | None:
    """The per-user durable store, when this process has one mounted.

    ``CLAUDE_CONFIG_DIR`` is where the web-terminal compose service mounts the
    per-user volume, so its presence is the mount itself — not a guess about
    what kind of machine this is.
    """
    config_dir = environ.get(CLAUDE_CONFIG_ENV)
    if not config_dir:
        return None
    return OwnershipStore(Path(config_dir).joinpath(*STORE_SUBPATH))


@dataclass(frozen=True)
class ProfileArtifact:
    """One ownable artifact the profile itself supplies.

    Attributes:
        name: Canonical ownership name (``rules/safety``, ``skills/diagnose``).
        output_path: Project-relative path the build copies it to.
        path: Where it lives in the profile.
        is_directory: A whole directory (a skill, a service), not a file.
    """

    name: str
    output_path: str
    path: Path
    is_directory: bool


def profile_artifacts(profile_root: Path | None) -> dict[str, ProfileArtifact]:
    """Every ownable artifact a profile supplies, by canonical name.

    This is what the build's convention scan will register at the next build,
    computed now — so an artifact a claim just moved into the profile is
    already owned as far as the gallery is concerned, instead of disappearing
    from it until someone rebuilds.

    Both the plan and the naming come from the build's own implementation
    rather than a copy: the name an artifact is owned under has to be spelled
    identically on both sides or the union with ``config.yml`` is meaningless.
    """
    if profile_root is None:
        return {}

    try:
        from osprey.cli.build_persistence import ownership_canonical
        from osprey.cli.profile_conventions import plan_convention_copies

        copies = plan_convention_copies(profile_root)
    except Exception as exc:  # noqa: BLE001 - the gallery must open regardless
        logger.warning("Could not read the profile at %s: %s", profile_root, exc)
        return {}

    found: dict[str, ProfileArtifact] = {}
    for copy in copies:
        name = ownership_canonical(copy)
        if name is None:
            continue
        found[name] = ProfileArtifact(
            name=name,
            output_path=copy.destination,
            path=copy.source,
            is_directory=copy.is_directory,
        )
    return found


# ── Reading the union ────────────────────────────────────────────────


def merge_user_owned(
    config_owned: Iterable[str],
    records: dict[str, OwnershipRecord],
) -> list[str]:
    """Merge build-derived ownership with the per-user store.

    PRECEDENCE: the store wins for any name it mentions.

    ``config.yml`` states what the *profile* owned when the image was built:
    facility-wide intent, derived by a scan. ``user_owned.json`` is what *this
    operator* did in this container afterwards — later in time, narrower in
    scope, and typed by a human rather than derived. Later, more specific human
    intent wins.

    The asymmetric alternative — union the claims, ignore the releases — is
    what produces the failure this module exists to prevent. With ``config.yml``
    read-only at runtime, an unclaim would have nowhere to be recorded, so the
    button would report success and change nothing. A source of truth that can
    only add is not a source of truth.
    """
    merged = [
        name
        for name in dict.fromkeys(config_owned)
        if not (name in records and not records[name].claimed)
    ]
    known = set(merged)
    for name, record in records.items():
        if record.claimed and name not in known:
            merged.append(name)
            known.add(name)
    return merged


def rehydrate(
    project_dir: Path,
    store: OwnershipStore,
    records: dict[str, OwnershipRecord],
    render_framework: Callable[[str], str | None],
) -> list[str]:
    """Restore durable claimed bodies into the project tree.

    On a freshly recreated container every claimed file in the tree is the
    image's again; this puts the operator's back, so the running agent reads
    what the operator wrote rather than the framework default they replaced.

    On a container that has been up a while it must not undo work, so a project
    file is overwritten only when it is provably untouched since the image
    built it — which is exactly when it still renders identical to its
    framework template. Anything else is somebody's edit and is left alone.

    Args:
        project_dir: Root of the image-baked project tree.
        store: The durable store to restore from.
        records: Already-read records, so the caller does not pay for a second read.
        render_framework: Renders an artifact's framework template by canonical
            name, or returns ``None`` when it cannot be rendered. Only called
            for files that actually differ, so the settled path costs nothing.

    Returns:
        Canonical names whose body was restored.
    """
    restored: list[str] = []
    for name, record in records.items():
        if not record.claimed or not record.output_path:
            continue
        relative = _safe_relative(record.output_path)
        if relative is None:
            continue

        # The protected set, asked of every record before its body can become a
        # file. ``_safe_relative`` above keeps the store inside the ownable
        # tree, but that tree deliberately contains ``.claude/rules/**``,
        # ``.claude/skills/**`` and ``.claude/agents/**`` — they are what the
        # gallery exists to claim. The store itself is on the agent-writable
        # claude-config volume, so a record naming a reserved path is a way to
        # have this walk install agent-authored instruction text, and under the
        # container entrypoint it would be installed as root into a tree the
        # agent cannot otherwise write. Skipped and audited rather than raised:
        # one poisoned record must not cost the operator every other body they
        # claimed, and a container that will not start is a worse outcome than
        # one that starts with one file left as the image shipped it.
        channel = reserved_write_channel(project_dir, str(relative))
        if channel is not None:
            _audit_restore_refusal(
                str(relative),
                channel=channel,
                reason="reserved path in ownership store",
                because=f"it belongs to {channel}",
            )
            continue

        durable = store.read_content(record.output_path)
        if durable is None:
            continue

        target = project_dir / relative
        if target.is_file():
            try:
                current = target.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if current == durable:
                continue  # already the operator's copy
            if not is_pristine_from_image(name, current, render_framework):
                logger.info(
                    "Leaving %s alone: it was edited after the last gallery save, so the "
                    "durable copy is not the newer version.",
                    relative,
                )
                continue

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(durable, encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not restore %s from the ownership store: %s", relative, exc)
            continue
        restored.append(name)

    return restored


def is_pristine_from_image(
    name: str,
    current: str,
    render_framework: Callable[[str], str | None],
) -> bool:
    """Whether the project's copy still holds what the image baked into it.

    A file that renders identical to its framework template has not been
    touched since the build wrote it. A custom artifact has no template; it is
    in the tree only because something in this container put it there, so it is
    never pristine.

    This is the single test for "is the tree copy somebody's work", and both
    directions of the tree/store relationship have to use it or they contradict
    each other: :func:`rehydrate` declines to overwrite a non-pristine tree
    copy, and the gallery's read path has to show that same copy rather than
    the older durable one it just refused to impose.
    """
    try:
        framework = render_framework(name)
    except Exception as exc:  # noqa: BLE001 - a render failure must not clobber an edit
        logger.warning("Could not render %s to check for local edits: %s", name, exc)
        return False
    return framework is not None and framework == current
