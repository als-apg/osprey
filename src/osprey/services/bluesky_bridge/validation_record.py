"""Content-hash-keyed store of PASSING plan-validation records.

The bridge is otherwise stateless (see ``runs.py``'s module docstring for the
run registry's own rationale) — this module is a bounded, explicit exception:
a session-tier plan file must be validated (``plan_validation.py``'s
:func:`~osprey.services.bluesky_bridge.plan_validation.validate_plan`)
before anything downstream will treat it as real, and re-validating on every
load/launch check would mean re-running the stage-3 dry-run subprocess on
every request. Instead, task 2.3's validate route calls
:func:`~osprey.services.bluesky_bridge.plan_validation.hash_plan_body` once and
records a PASS here by that hash; task 2.4's session-layer load gate and task
2.5's launch gate re-hash the file's *current* on-disk content the same way
and ask whether that hash has a passing record.

Only a passing validation is ever recorded — a failed validation records
nothing, so an unknown hash and a previously-failed hash are indistinguishable
to :meth:`ValidationRecordStore.has_passing_record` (both answer ``False``).
This also means an edit to a previously-passing file changes its content hash
and silently drops back to "no record" until re-validated — exactly the
re-check tasks 2.4/2.5 need.

In-memory only, like ``runs.py``'s ``RunRegistry``: a bridge restart loses
every record, so a session plan that passed validation before a restart must
be re-validated before it can be loaded or launched again.
"""

from __future__ import annotations

from threading import Lock


class ValidationRecordStore:
    """Thread-safe in-memory set of content hashes with a PASSING validation.

    Mirrors ``runs.py``'s ``RunRegistry``: a single :class:`threading.Lock`
    guards the underlying set, since the bridge serves concurrent requests.
    """

    def __init__(self) -> None:
        self._passing_hashes: set[str] = set()
        self.lock = Lock()

    def record(self, content_hash: str) -> None:
        """Mark ``content_hash`` as having a passing validation.

        Callers must only call this for a hash whose validation actually
        passed (see :class:`~osprey.services.bluesky_bridge.plan_validation.ValidationResult`.
        ``passed``) — recording a failing hash would defeat the "unknown or
        failed both read as no record" contract :meth:`has_passing_record`
        relies on. Recording an already-recorded hash is a harmless no-op.
        """
        with self.lock:
            self._passing_hashes.add(content_hash)

    def has_passing_record(self, content_hash: str) -> bool:
        """Whether ``content_hash`` has a recorded passing validation.

        Returns `False` for both a hash that was never recorded and a hash
        whose only validation attempt failed (failures are never recorded —
        see :meth:`record`), which are indistinguishable by design.
        """
        with self.lock:
            return content_hash in self._passing_hashes


# Module-level singleton: the bridge process's one validation-record store.
# Reachable via `from .validation_record import validation_records` — task
# 2.3's validate route records into it, tasks 2.4/2.5's gates query it.
validation_records = ValidationRecordStore()
