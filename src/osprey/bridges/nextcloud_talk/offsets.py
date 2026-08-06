"""Persisted per-room poll offsets for the Talk long-poll loop.

Talk's chat API is cursor-based: each poll passes the room's ``lastKnownMessageId``
and receives only what arrived after it. That cursor is the bridge's entire memory
of "where am I in this room", so it has to outlive the process — a restart that
forgot it would either replay the whole room's history or skip everything posted
while the bridge was down. This store is that memory: a flat
``room token -> lastKnownMessageId`` map on the same bridge-owned ``/data`` volume
as the core dedup/history stores (see
:attr:`~osprey.bridges.nextcloud_talk.config.NextcloudBridgeConfig.offsets_path`).

Two properties, and deliberately nothing else:

  * **A room absent from the store has never been polled.** :meth:`get` returns
    ``None`` for it, which the poller reads as a first-ever start and uses to skip
    the room's backlog. Every later start finds a real offset and resumes.
  * **Advance is monotonic.** :meth:`advance` never moves an offset backward; a
    lower id is silently ignored, not an error. This is what makes crash recovery
    safe: a crash mid-batch re-fetches the batch, so ``advance`` legitimately sees
    ids it has already passed (and, with one thread per room, out-of-order calls
    are cheap to tolerate rather than to forbid). If a regression could land, the
    persisted cursor would rewind behind an already-handled message and the room
    would replay history — duplicate suppression lives in the engine's dedup
    claims, but only for as long as those claims exist.

*When* to advance is the poller's decision (only after ``handle_event`` has
returned for every message in the fetched batch), and it stays there — this store
owns durability and monotonicity, nothing about batch or cursor policy.

Persistence, atomicity and the single lock that makes the store safe to share
across room threads all come from
:class:`~osprey.bridges.core.JsonFileStore`.
"""

from __future__ import annotations

from osprey.bridges.core import JsonFileStore


class OffsetStore(JsonFileStore):
    """Crash-safe ``room -> lastKnownMessageId`` map. See the module docstring."""

    def _coerce(self, loaded: dict) -> dict:
        """Keep only well-formed ``str -> int`` entries from the loaded file.

        A malformed entry is dropped rather than repaired: an offset we cannot
        trust is indistinguishable from never having polled the room, and
        ``None`` (skip history) is the safe reading — inventing a number could
        skip past unhandled messages. ``bool`` is rejected explicitly because it
        is an ``int`` subclass, so a stray ``true`` would otherwise become the
        offset ``1`` and replay the room from its second message.
        """
        return {
            room: value
            for room, value in loaded.items()
            if isinstance(room, str)
            and isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0
        }

    def get(self, room: str) -> int | None:
        """The room's persisted offset, or ``None`` if it has never been polled."""
        with self._lock:
            offset: int | None = self._data.get(room)
            return offset

    def advance(self, room: str, message_id: int) -> None:
        """Move the room's offset forward to ``message_id`` and persist it.

        A no-op when ``message_id`` is not ahead of the stored offset (no write,
        no error). Check and write happen under the inherited lock, so two room
        threads racing on the same room cannot interleave into a regression.
        """
        with self._lock:
            current = self._data.get(room)
            if current is not None and message_id <= current:
                return
            self._data[room] = message_id
            self._flush_locked()
