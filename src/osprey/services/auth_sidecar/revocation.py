"""Revoked login sessions, kept as digests in one file so a logout survives a restart.

A session cookie is signed, not stateful: once issued it stays cryptographically
valid until its recorded expiry. Logout therefore needs somewhere to record "this
session id is dead", and the verify endpoint consults that record on every
``auth_request`` subrequest.

**The file.** One owner-only JSON file, :data:`REVOCATION_FILE_NAME`, in the
directory :data:`~osprey.services.auth_sidecar.audit.AUDIT_DIR_ENV` names — the
subdirectory compose binds for this service and for nobody else. Its shape is
``{"v": 1, "revoked": {<sha256 hex of the session id>: <expiry epoch>}}``. It is
loaded when the store is built, with every entry at or past its expiry dropped,
and rewritten whole on every logout. A container recreate (``osprey users
passwd``, a decommission re-render, a podman image-drift reconcile) therefore
comes back up refusing every cookie that was logged out before it.

**Why a plain SHA-256.** A session id is ``secrets`` output with 128 bits of
entropy, so there is no dictionary to hash-guess from, and the file is a deny
list of dead sessions: an entry authenticates nobody, it only names a session
that is refused anyway. The raw id never touches the disk. A key would buy
nothing — rotating the session secret already kills every cookie at the
signature check — and would couple this store to that secret. The same argument
as :func:`osprey.interfaces.web_auth._digest`.

**Writes are atomic.** Each write goes to a dot-named temporary file in the
same directory, is chmod ``0600`` and fsynced, and replaces the file in one
``os.replace``. A reader never sees a torn file, and a failed write leaves no
temporary behind.

**Storage failing never costs the decision.** A missing file is a silent fresh
start. An unreadable or unrecognised file starts empty with one warning; a file
whose shape is right but some entries are not keeps every well-formed entry,
because each one is still a true statement that the session was logged out. A
write that fails keeps the revocation in memory, where it is enforced for the
life of the process, and warns once for that process; later logouts keep trying
to write, so a disk that frees up starts persisting again.

**The directory is never created.** One created inside the container would sit
in its writable layer and vanish on the next recreate — the trap
:func:`~osprey.services.auth_sidecar.audit.audit_directory` refuses a relative
value over. A missing directory is a broken bind, and it degrades like any other
failed write.

Password rotation does **not** rely on this store. Invalidation there rests on
the credential-generation tag carried in each unlocked-user entry, which is
recomputed from the current stored hash on every verify and so survives
restarts. Revocation covers logout only.

Memory is bounded by logouts-per-session-lifetime, not by container uptime: an
entry is dropped once the clock passes the expiry that was recorded with it,
since a session that has expired is rejected on its own expiry check and no
longer needs a revocation record. The file is bounded the same way, because
each write carries only unexpired entries.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import os
import re
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["REVOCATION_FILE_NAME", "RevocationStore"]

REVOCATION_FILE_NAME = ".revoked-sessions.json"
"""The revoked-session file's name inside the audit directory.

Dot-named so the archive leaves it alone and nobody reads it as a ledger: a
ledger ends ``.jsonl``, and the archive skips dot-named ``.json`` files there.
"""

REVOCATION_FILE_VERSION = 1
REVOCATION_FILE_MODE = 0o600

_DIGEST_RE = re.compile(r"[0-9a-f]{64}")

_warned = False


def _digest(session_id: str) -> str:
    """The key a session id is stored under: its SHA-256, lowercase hex."""
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()


def _warn(reason: str) -> None:
    """Log a storage failure once per process; later calls are silent."""
    global _warned
    if _warned:
        return
    _warned = True
    logger.warning(
        "Logouts will not survive a restart of this service: the revoked-session file %s %s. "
        "Logged-out sessions stay refused until this process stops.",
        REVOCATION_FILE_NAME,
        reason,
    )


def _is_expiry(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


class RevocationStore:
    """Tracks revoked session ids until the moment they would have expired.

    Expiry timestamps are absolute epoch seconds, matching the per-user expiry
    carried in the session cookie, so callers can pass the cookie's value
    straight through. Callers pass raw session ids; the store keys everything by
    digest.

    :meth:`revoke` writes the file synchronously, on the event loop. That is one
    small file write per logout, which is rare, and it keeps "recorded" and
    "persisted" in the order a restart needs.

    Not thread-safe on its own; the sidecar's async request handlers touch it
    from a single event loop, and every method completes without awaiting.
    """

    def __init__(
        self, directory: Path | None = None, clock: Callable[[], float] = time.time
    ) -> None:
        """Create a store, loading the file in *directory* when one is named.

        Never raises: an unusable file or directory degrades to an empty store.

        Args:
            directory: The directory the revoked-session file lives in. ``None``
                keeps the store in memory only.
            clock: Returns the current time as absolute epoch seconds. Injectable
                so tests can advance time without sleeping.
        """
        self._clock = clock
        self._path = None if directory is None else directory / REVOCATION_FILE_NAME
        self._revoked: dict[str, float] = self._load()

    @property
    def path(self) -> Path | None:
        """The revoked-session file, or ``None`` for a memory-only store."""
        return self._path

    def revoke(self, session_id: str, expires_at: float) -> None:
        """Record a session id as revoked until ``expires_at``.

        Re-revoking a known id keeps the later of the two expiries, so a
        refreshed cookie cannot shorten an existing record. The whole unexpired
        map is then written to the file, when the store has one.

        Args:
            session_id: The session id from the cookie being logged out.
            expires_at: Absolute epoch seconds at which that session expires on
                its own. Passing an already-past value is harmless: the session
                is expired, :meth:`is_revoked` reports ``False`` for it, and the
                entry is dropped on the next sweep.
        """
        self.purge_expired()
        key = _digest(session_id)
        existing = self._revoked.get(key)
        if existing is None or expires_at > existing:
            self._revoked[key] = expires_at
        if self._path is not None:
            try:
                self._persist(self._path)
            except OSError as exc:
                _warn(f"cannot be written ({exc.strerror or exc})")

    def is_revoked(self, session_id: str) -> bool:
        """Whether ``session_id`` was revoked and has not yet reached its expiry.

        Reports ``False`` once the recorded expiry has passed, whether or not the
        entry has been swept yet — the answer never depends on sweep timing. A
        caller reaching that point has an expired session anyway and rejects it on
        the expiry check. Dropping a lapsed entry here does not write: the next
        :meth:`revoke` writes the pruned map.
        """
        key = _digest(session_id)
        expires_at = self._revoked.get(key)
        if expires_at is None:
            return False
        if self._clock() >= expires_at:
            del self._revoked[key]
            return False
        return True

    def purge_expired(self) -> int:
        """Drop every entry whose recorded expiry has passed.

        Called on each :meth:`revoke`, which is what keeps the store bounded by
        logouts-per-lifetime. Exposed so a caller can sweep on its own schedule.

        Returns:
            How many entries were dropped.
        """
        now = self._clock()
        stale = [key for key, expires_at in self._revoked.items() if now >= expires_at]
        for key in stale:
            del self._revoked[key]
        return len(stale)

    def __len__(self) -> int:
        """How many entries are held, including any not yet swept."""
        return len(self._revoked)

    def _load(self) -> dict[str, float]:
        """The unexpired, well-formed entries of the file; ``{}`` when there are none."""
        if self._path is None:
            return {}
        try:
            text = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            _warn(f"cannot be read ({exc})")
            return {}
        try:
            payload = json.loads(text)
        except ValueError:
            _warn("is not JSON")
            return {}
        if not isinstance(payload, dict):
            _warn("is not a JSON object")
            return {}
        if payload.get("v") != REVOCATION_FILE_VERSION:
            _warn(f"is not version {REVOCATION_FILE_VERSION}")
            return {}
        entries = payload.get("revoked")
        if not isinstance(entries, dict):
            _warn("holds no revoked-session object")
            return {}
        now = self._clock()
        loaded: dict[str, float] = {}
        malformed = 0
        for key, expires_at in entries.items():
            if not (isinstance(key, str) and _DIGEST_RE.fullmatch(key) and _is_expiry(expires_at)):
                malformed += 1
                continue
            if now < expires_at:
                loaded[key] = float(expires_at)
        if malformed:
            _warn(f"holds {malformed} malformed entries, which were skipped")
        return loaded

    def _persist(self, path: Path) -> None:
        """Replace *path* with the current map, atomically and owner-only."""
        fd, tmp = tempfile.mkstemp(
            dir=path.parent, prefix=f"{REVOCATION_FILE_NAME}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(
                    {"v": REVOCATION_FILE_VERSION, "revoked": self._revoked},
                    handle,
                    sort_keys=True,
                )
                handle.flush()
                with contextlib.suppress(OSError):
                    os.fsync(handle.fileno())
            os.chmod(tmp, REVOCATION_FILE_MODE)
            os.replace(tmp, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
