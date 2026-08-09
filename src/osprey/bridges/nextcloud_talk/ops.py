"""Nextcloud Talk's :class:`~osprey.bridges.core.ChannelOps` implementation.

:class:`NextcloudTalkOps` is I/O only. Every ordering, dedup, retry, and crash-recovery
decision stays in :mod:`osprey.bridges.core`; what lives here is Talk's wire format and
the wording the room actually sees. This module carries the **posting** members
(:meth:`~NextcloudTalkOps.post_ack`, :meth:`~NextcloudTalkOps.post_answer`,
:meth:`~NextcloudTalkOps.post_queued`, :meth:`~NextcloudTalkOps.post_giveup`,
:meth:`~NextcloudTalkOps.post_superseded`), inbound file downloads
(:meth:`~NextcloudTalkOps.download_inputs`), outbound artifact delivery
(:meth:`~NextcloudTalkOps.deliver_files`), and the pure
:meth:`~NextcloudTalkOps.coalesce_key`. The two inbound members
(:meth:`~NextcloudTalkOps.parse_event`, :meth:`~NextcloudTalkOps.resolve_reply_context`)
join them on the same class as thin delegations to
:mod:`osprey.bridges.nextcloud_talk.events`, which is where Talk's wire format is read.

**Every post-claim member reads the persisted ``entry`` and nothing else.** None of them
touches a live wire event, which is what makes one implementation serve the live path, the
retry drain, and startup reconcile identically — on the latter two the wire event is long
gone, and a member that reached for it would work in testing and fail after a restart. The
entry keys these members read are the ``nc_``-prefixed ones the claim stamped (see
:data:`NC_ROOM` / :data:`NC_MESSAGE_ID`) plus the engine's own ``history_key`` /
``sender_id``. The two inbound members are the exception by definition: they run *before*
the claim exists, on the wire event itself, and produce the entry the rest read back.

The failure contracts are **asymmetric and load-bearing** — the engine's crash-safety
ordering is built on them, so inverting one here silently breaks delivery rather than
failing a test:

=========================  ==============================================================
member                     on failure
=========================  ==============================================================
``parse_event``            return ``None`` for anything the wire hands it, with the ONE
                           documented exception that ``RoomTypeUnresolved`` propagates —
                           see :mod:`osprey.bridges.nextcloud_talk.events`
``resolve_reply_context``  swallow — reply context is enrichment; the question is
                           dispatched with or without the quote
``post_ack``               swallow — a missing "on it" must never cost the user an answer
``download_inputs``        swallow per file — each failure becomes a skip note the agent
                           can relay, in its own provenance bucket
``post_answer``            **raise** — the raise IS the redelivery signal; swallowing
                           silently drops the answer the run already paid for
``deliver_files``          swallow — degrade to text-only; the answer has already landed
                           and a failed upload must never un-deliver it
``post_queued``            may raise — the engine parked the entry before calling, so a
                           lost notice never loses the request
``post_giveup``            **raise** — the engine marks the entry terminal only after the
                           notice lands; lost silence is worse than a duplicate notice
``post_superseded``        swallow — the coalesce CAS already committed
``coalesce_key``           pure; never raises, never does I/O
=========================  ==============================================================

Wording is deliberately dull and never leaks machine detail: no exception text, no run id,
no ``give_up_reason``, no ``result["error"]``. A user in a control room needs to know
whether their question is answered, waiting, or abandoned — the diagnosis belongs in the
bridge log, which is where it stays.
"""

from __future__ import annotations

import base64
import logging
import re
from collections.abc import Mapping, Sequence
from typing import Any

import httpx

from osprey.bridges.core import (
    KNOWN_EXTENSIONS,
    MAX_DELIVERED_DOC_BYTES,
    MAX_DELIVERED_IMAGE_BYTES,
    ChannelOps,
    InboundEvent,
    InputDownload,
    ReplyContext,
    artifact_descriptors,
    ext_for_mime,
    fetch_artifact,
    safe_label,
)

from .client import MAX_DOWNLOAD_BYTES, REQUEST_TIMEOUT, DavTooLargeError, TalkClient, upload_dir
from .config import NextcloudBridgeConfig
from .events import QUOTED_FILES_KEY
from .events import parse_event as parse_talk_message
from .events import resolve_reply_context as resolve_talk_reply_context
from .rooms import RoomDirectory

logger = logging.getLogger(__name__)

# --- entry keys this module reads ------------------------------------------
#
# The adapter's half of the persisted entry, stamped by the claim. All ``nc_``-prefixed
# and therefore disjoint from ``osprey.bridges.core.RESERVED_ENTRY_KEYS`` (a collision
# there is a silent overwrite in one direction or the other, so the prefix is the whole
# safety mechanism).

NC_ROOM = "nc_room"
"""Talk room token to post into. Without it nothing can be delivered at all."""

NC_MESSAGE_ID = "nc_message_id"
"""Numeric Talk id of the message that started this exchange, used as the ``replyTo``
target so notices attach to the question in a busy room. Note this is Talk's own integer
id, NOT the engine's room-scoped ``message_id`` string."""

NC_FILES = "nc_files"
"""References to the files attached to the message ITSELF (the own-provenance bucket).

The quoted message's references ride under
:data:`~osprey.bridges.nextcloud_talk.events.QUOTED_FILES_KEY`, which is imported rather
than re-spelled here so the reader and the writer of that key cannot drift apart."""

# --- limits ----------------------------------------------------------------

MAX_MESSAGE_CHARS = 32_000
"""Talk's per-message character ceiling. A longer answer is split across messages rather
than truncated — an answer that silently loses its tail is worse than several messages."""

# --- room-facing wording ---------------------------------------------------

ACK_TEXT = "On it — working on this now."
"""Immediate "your question was heard" signal, posted before the long dispatch."""

EMPTY_ANSWER_TEXT = "The run finished without producing any text."
"""Posted when a completed run carries no answer text. Something must always land, or a
successful-but-silent run is indistinguishable from a broken bridge."""

ERROR_TEXT = (
    "I couldn't complete that request. Nothing was changed. Please try again, or ask an "
    "operator to check the bridge log."
)
"""Posted for any terminal status that is not ``completed``. The machine-readable failure
never appears in the room."""

QUEUED_TEXT = (
    "Queued — the service I need is unavailable right now. I'll answer here once it "
    "runs, so there's no need to re-send this."
)
"""First-park notice. A re-park is silent (the engine posts this only once)."""

GIVEUP_TEXT = (
    "I couldn't run this automatically and I've stopped retrying, so this question will "
    "not be answered. Please re-send it later, or ask an operator to look into it."
)
"""Abandonment notice. Honest about the outcome and silent about the reason."""

SUPERSEDED_TEXT = "Superseded by a newer question in this conversation — I won't answer this one."
"""Posted as a reply to the older queued message that was dropped."""

# --- agent-facing skip notes ------------------------------------------------
#
# Unlike the wording above these never reach the room: they ride the dispatch payload so
# the AGENT can tell the user which attachment it did not get and why. Still no machine
# detail — the exception goes to the log.

DEFAULT_INPUT_FILENAME = "attachment"
"""Stand-in name for an inbound file whose reference carries neither name nor id."""

NO_PATH_REASON = "couldn't be fetched (no file path was recorded)"
"""Skip reason for a reference with no DAV path — nothing could locate the file."""

TOO_LARGE_REASON = f"too large to fetch (over {MAX_DOWNLOAD_BYTES // (1024 * 1024)} MiB)"
"""Skip reason for a file over the download cap, quoting the same constant the cap comes
from so the two cannot drift."""

UNAVAILABLE_REASON = "couldn't be fetched from the file store"
"""Skip reason for a deleted, moved, or unreachable file (a 404, a transport failure)."""


def _room(entry: Mapping[str, Any]) -> str:
    """The room token to post into, or ``""`` when the entry carries none."""
    room = entry.get(NC_ROOM)
    return room if isinstance(room, str) else ""


TOKEN_FIELD = "token"
"""Field on a raw Talk chat message naming the room it belongs to.

Talk stamps it on every message object in an OCS chat response, and
:meth:`TalkClient.poll_messages` passes those objects through unmodified — so it is the
server's own answer to "which room did this come from", not something the adapter infers.
:meth:`NextcloudTalkOps.parse_event` reads it because the engine hands the member a bare
message with no wrapper carrying the room."""


def _event_room(event: Any) -> str:
    """The room token a raw Talk message came from, or ``""`` when it names none.

    Tolerant on purpose: this runs on whatever the network handed the poller, and
    :meth:`NextcloudTalkOps.parse_event` must answer a malformed message with ``None``
    rather than an exception. A blank token counts as absent — no Talk room is addressed
    by whitespace.
    """
    token = event.get(TOKEN_FIELD) if isinstance(event, Mapping) else None
    return token.strip() if isinstance(token, str) else ""


def _reply_target(entry: Mapping[str, Any]) -> int | None:
    """The ``replyTo`` message id, or ``None`` to post without quoting.

    A missing or malformed id degrades to a plain room message rather than failing the
    post: an un-anchored notice still reaches the user, while a raise here would cost
    them the answer entirely. ``bool`` is rejected explicitly — it is an ``int`` subclass
    but never a message id.
    """
    message_id = entry.get(NC_MESSAGE_ID)
    if isinstance(message_id, int) and not isinstance(message_id, bool):
        return message_id
    return None


def _chunk(text: str, limit: int = MAX_MESSAGE_CHARS) -> list[str]:
    """Split ``text`` into chunks of at most ``limit`` characters.

    Cuts at the last newline inside each window so a split lands between lines where it
    can, and hard-splits a single line longer than ``limit`` where it cannot.
    ``"".join(_chunk(text)) == text`` always holds: this splits, it never reflows,
    trims, or re-wraps, so markdown passes through byte-for-byte.

    Args:
        text: The answer text, assumed non-empty.
        limit: Maximum characters per chunk.

    Returns:
        The chunks in posting order; empty only for empty input.
    """
    chunks: list[str] = []
    rest = text
    while len(rest) > limit:
        cut = rest.rfind("\n", 0, limit)
        # rfind returning -1 (no newline) or 0 (the window is one long line preceded by
        # a bare newline) leaves no boundary worth preferring — take the full window.
        cut = limit if cut <= 0 else cut + 1  # keep the newline with the chunk it ends
        chunks.append(rest[:cut])
        rest = rest[cut:]
    if rest:
        chunks.append(rest)
    return chunks


def _refs(entry: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    """The inbound file references stored under ``key``.

    Tolerant by design: this runs on a JSON round-trip of whatever was persisted, and a
    malformed store must cost attachments, never the dispatch.

    Args:
        entry: The persisted entry.
        key: :data:`NC_FILES` or
            :data:`~osprey.bridges.nextcloud_talk.events.QUOTED_FILES_KEY`.

    Returns:
        The reference mappings, with anything of the wrong shape dropped.
    """
    refs = entry.get(key)
    if not isinstance(refs, list):
        return []
    return [ref for ref in refs if isinstance(ref, Mapping)]


def _input_filename(ref: Mapping[str, Any]) -> str:
    """The filename to ship for one inbound file.

    Falls back through Talk's ``name`` then its file ``id`` to
    :data:`DEFAULT_INPUT_FILENAME`, so a file always has *something* the agent can refer
    to in its answer. This is a payload label, never a storage path, so
    :func:`~osprey.bridges.core.safe_label`'s CR/LF strip and length bound are the whole
    sanitization needed.
    """
    name = ref.get("name")
    file_id = ref.get("id")
    return safe_label(
        name if isinstance(name, str) else None,
        safe_label(file_id if isinstance(file_id, str) else None, DEFAULT_INPUT_FILENAME),
    )


_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9._-]")
"""Everything replaced with ``_`` in an upload filename. An allowlist rather than a
denylist: the stem comes from worker-supplied data, and a ``/`` or ``..`` reaching a DAV
path would address a file outside the artifact tree."""


def _segment(value: Any) -> str:
    """Reduce a worker-supplied name to one safe DAV filename stem.

    Args:
        value: Candidate name; anything that is not a string is treated as absent.

    Returns:
        The sanitized stem, or ``""`` when nothing usable survives (a name made only of
        dots and separators, for instance).
    """
    text = safe_label(value if isinstance(value, str) else None, "")
    return _UNSAFE_NAME_CHARS.sub("_", text).strip("._")[:120]


def _upload_stem(label: Any, fallback: Any) -> str:
    """Build the filename stem an artifact is uploaded under, before its extension.

    Args:
        label: Preferred name (the worker's ``filename``, when it sent one).
        fallback: Name to use when ``label`` yields nothing usable — the artifact id, for
            a descriptor carrying no ``filename`` and for a bare id string from an older
            worker, which has none to carry.

    Returns:
        One safe path segment, never empty.
    """
    return _segment(label) or _segment(fallback) or "artifact"


def _disambiguate(stem: str, suffix: str) -> str:
    """Insert ``suffix`` into ``stem``, ahead of any trailing dot-suffix.

    Purely lexical, so no mime is needed: ``plot.png`` becomes ``plot-<suffix>.png``
    rather than ``plot.png-<suffix>``, which both reads as a filename in the room and
    leaves :func:`_upload_name`'s already-suffixed check able to recognize the
    extension.

    ``_segment`` strips the leading dot run, so the head is never empty. The TAIL can
    be: ``_segment`` strips before it truncates at 120 characters, so a long name cut
    exactly at a dot ends in one, and the partition then yields ``""``. That costs a
    trailing dot in the filename and nothing else — the result is still one non-empty
    segment that is neither ``.`` nor ``..``, which is all ``dav_mkcol_put`` requires.
    """
    head, dot, tail = stem.rpartition(".")
    return f"{head}-{suffix}{dot}{tail}" if dot else f"{stem}-{suffix}"


_UPLOAD_EXTENSIONS = KNOWN_EXTENSIONS | {".png"}
"""Every extension :func:`_deliver_one` can append. ``.png`` is unioned in because it
is the fixed extension of the image path and is deliberately absent from the core mime
allowlist (see the LOAD-BEARING note at the image-bucket call site).

With this set a name can need SEVERAL strips: ``plot.png.pdf`` loses ``.pdf`` and then
``.png`` before it can be compared against ``plot``."""

_UPLOAD_EXTENSIONS_BY_LENGTH = tuple(sorted(_UPLOAD_EXTENSIONS, key=len, reverse=True))
""":data:`_UPLOAD_EXTENSIONS`, longest first — the order :func:`_name_key` strips in.

Longest-match is made STRUCTURAL here rather than left to chance. As the set stands no
member is a suffix of another, so at most one can ever match and the order is
immaterial; but that is a property of today's contents, not of the algorithm. Add one
compound suffix (``.tar.gz``, which ends with a hypothetical ``.gz``) and a set-iterating
loop would strip whichever it happened to reach first — a key that differs between
runs, and a collision check that silently stops being reproducible. Sorting by length
means the longest match always wins, so the invariant is enforced by construction and a
future addition cannot quietly break it."""


def _name_key(stem: str) -> str:
    """The key on which two artifacts are judged to collide.

    Deliberately NOT the stem itself. :func:`_upload_name` does not double an
    extension the stem already carries, so a stem of ``plot.png`` and a stem of
    ``plot`` both upload as ``plot.png`` once the bytes are served as PNG — two
    different stems, one filename.

    Every trailing KNOWN extension is stripped, in a loop, after case-folding.
    Each part of that matters:

    * *known* rather than "any trailing dot-suffix", or ``report.v2`` and
      ``report.v2.pdf`` key differently (``report`` vs ``report.v2``) and still
      collide on ``report.v2.pdf``. It also stops ``report.draft`` and
      ``report.final`` from being disambiguated for no reason;
    * *in a loop*, because one strip is not enough: ``plot.png.pdf`` has to lose
      both before it can be compared with ``plot``. The loop walks
      :data:`_UPLOAD_EXTENSIONS_BY_LENGTH`, so the longest match wins by
      construction rather than by luck of set iteration order;
    * *case-folded*, because ``_upload_name``'s own check is case-insensitive, so
      ``a.PDF`` and ``a`` collide once ``.pdf`` is appended.

    Two artifacts whose extensions would in the end have differed still key alike and
    one is disambiguated needlessly. That is the deliberate direction: the extension
    is not known until after the fetch, and over-separating costs a longer filename
    while under-separating costs an overwritten file.
    """
    key = stem.lower()
    while True:
        for extension in _UPLOAD_EXTENSIONS_BY_LENGTH:
            if key.endswith(extension) and len(key) > len(extension):
                key = key[: -len(extension)]
                break
        else:
            return key


def _unique_stems(descriptors: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Map each artifact id in one delivery to the stem it is uploaded under.

    The worker names an artifact by BASENAME, so a run whose steps each wrote their own
    ``plot.png`` produces several descriptors carrying the SAME ``filename``. All of a
    run's artifacts share one DAV directory, so uploading them under one name would
    overwrite each earlier file and share the survivor once per artifact — a silently
    lost plot, reported as delivered. Every artifact after the first to claim a name
    therefore gets a slice of its id inserted, and an ordinal after that if even the
    slice collides, so no two artifacts can reach one path.

    Collision is judged on :func:`_name_key`, not on the stem: two stems that differ
    can still yield ONE filename, so comparing stems would let exactly the overwrite
    this function exists to prevent back through.

    Settled over the whole delivery at once and BEFORE any fetch, which is possible
    because the stem — unlike the extension — does not depend on the served
    ``Content-Type``. That keeps a STEM independent of fetch order and of which fetches
    happened to fail, so a redelivery of the same run reproduces exactly the same stems,
    which is what the upload's idempotency rests on. Two conditions on that, both real:

    * only the STEM is reproduced — the extension is settled per artifact from the
      served ``Content-Type``, so a delivery whose conversion falls back differently
      still writes a different filename than the one before it;
    * assignment is ORDER-DEPENDENT by design (first to claim a key keeps the bare
      stem), so reproducibility holds only while the descriptor list arrives in the
      same order. It does: the list is rebuilt by ``artifact_descriptors`` from the
      run's own ``artifacts``, preserving the worker's order. A caller that sorted or
      de-duplicated the descriptors between deliveries would break this, which is why
      the list is passed through untouched.

    Args:
        descriptors: Every descriptor this delivery will attempt, in worker order.
            An entry naming no artifact is skipped, as is a repeat of an id already seen.

    Returns:
        ``{artifact_id: stem}``, with no two ids sharing a stem.
    """
    stems: dict[str, str] = {}
    taken: set[str] = set()
    for descriptor in descriptors:
        artifact_id = descriptor.get("artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id or artifact_id in stems:
            continue
        stem = _upload_stem(descriptor.get("filename"), artifact_id)
        if _name_key(stem) in taken:
            stem = _disambiguate(stem, _segment(artifact_id)[:8] or "artifact")
            base, ordinal = stem, 2
            while _name_key(stem) in taken:
                stem = _disambiguate(base, str(ordinal))
                ordinal += 1
        taken.add(_name_key(stem))
        stems[artifact_id] = stem
    return stems


def _upload_name(stem: str, extension: str) -> str:
    """Give an upload stem its extension.

    Args:
        stem: The stem this artifact was assigned by :func:`_unique_stems`.
        extension: Extension to ensure, including the dot. Derived from the mime rather
            than from any worker-supplied name, so a worker cannot choose it.

    Returns:
        A single safe path segment carrying ``extension``, CASE-INSENSITIVELY: a stem
        that already ends in it is left alone rather than doubled, so ``("a.PDF",
        ".pdf")`` returns ``a.PDF`` — which carries the extension without literally
        ending in the string passed. :func:`_name_key` folds case for exactly this
        reason, so two stems differing only in an extension's case are still seen as
        one name.
    """
    if extension and stem.lower().endswith(extension.lower()):
        return stem
    # The worker's filename is predicted alongside delivered_mime, so a name carrying a
    # DIFFERENT extension is a prediction the delivery contradicted ("data.png" for a
    # render that never happened). Replace it rather than stack it — the extension still
    # comes only from the mime.
    #
    # Only a KNOWN extension is replaced, and this must stay in lockstep with what
    # _name_key strips. If this stripped more than _name_key does, two stems that
    # _unique_stems judged distinct could still converge on one filename here — the
    # overwrite _unique_stems exists to prevent, reintroduced one step later. A generic
    # "short alnum suffix" rule does exactly that: report.draft and report.final key
    # apart, then both upload as report.pdf.
    root, dot, suffix = stem.rpartition(".")
    if dot and root and f".{suffix}".lower() in _UPLOAD_EXTENSIONS:
        stem = root
    return f"{stem}{extension}"


class NextcloudTalkOps:
    """Talk's platform I/O behind the ``ChannelOps`` seam.

    Thread-safe and free of per-dispatch state: the engine shares ONE instance across
    every room thread and the drain thread, and each member derives everything it needs
    from the ``entry`` it is handed.
    """

    def __init__(
        self,
        cfg: NextcloudBridgeConfig,
        client: TalkClient | None = None,
        http: httpx.Client | None = None,
        rooms: RoomDirectory | None = None,
    ):
        """Wire the adapter to its config, its two HTTP clients, and the room cache.

        Every collaborator is explicit and injectable, and each has a default derived from
        ``cfg``, so a test that exercises one member need not name the other three.

        Args:
            cfg: The bridge config. The posting members address rooms from the entry
                rather than from config; ``cfg`` is held for the members that genuinely
                need it (the bot identity behind the mention filter and the DAV tree).
            client: Talk client to use instead of one built from ``cfg``. Tests inject a
                :class:`httpx.MockTransport`-backed client this way.
            http: Client for the WORKER's artifact byte route — a different host, auth
                scheme, and timeout budget from Talk's, so deliberately not the Talk
                client's. ``trust_env`` comes from config for the same reason it does
                there: a proxy inherited from a dev shell must not mount itself in front
                of the worker.
            rooms: The room-type cache :meth:`parse_event` consults. **A real deployment
                MUST pass the very directory the room pollers hold.** Only
                :meth:`RoomDirectory.resolve_once` — which is poller-side — populates a
                directory, so a private one this class built for itself would stay
                permanently empty: every :meth:`parse_event` call would raise
                :class:`RoomTypeUnresolved`, every poll cycle would leave its offset
                unadvanced, and the bridge would defer every message forever without
                ever losing one. The default exists so the nine members that never parse
                stay constructible in one line; it is not a production wiring.
        """
        self._cfg = cfg
        self._client = client if client is not None else TalkClient(cfg)
        self._http = (
            http
            if http is not None
            else httpx.Client(timeout=REQUEST_TIMEOUT, trust_env=cfg.core.trust_env)
        )
        self._rooms = rooms if rooms is not None else RoomDirectory(self._client)

    # --- inbound parse members ---------------------------------------------
    #
    # Both delegate to :mod:`osprey.bridges.nextcloud_talk.events`, which owns Talk's
    # wire format. Nothing is parsed twice: what these add is the seam's arity, and for
    # ``parse_event`` the room the module function needs and the engine does not supply.

    def parse_event(self, event: Any) -> InboundEvent | None:
        """Parse one raw Talk chat message into an :class:`InboundEvent`.

        **Where the room comes from.** The module-level
        :func:`~osprey.bridges.nextcloud_talk.events.parse_event` takes the polled-from
        room explicitly, because it decides both the dedup scope and where replies go.
        The seam does not: the engine calls this with exactly one argument, and the
        poller hands :meth:`BridgeRuntime.handle_event` a bare Talk message with no
        wrapper carrying the room. So the room is read from the message's own
        :data:`TOKEN_FIELD`.

        That is the same authority, not a weaker substitute. Talk stamps ``token`` on
        every message object in an OCS chat response and
        :meth:`TalkClient.poll_messages` returns those objects unmodified, so the field
        *is* the server's statement of which room it answered for — and it is the room's
        canonical token, where the poller's own string is operator-typed config, so
        keying dedup and history on it is if anything the more consistent of the two.

        A message that names no usable room is **ignored**. Without a room there is no
        dedup key to claim it under and no destination to answer into, so there is
        nothing a later attempt could do differently either; ``None`` is both the safe
        answer and the contract (this member never raises on bad input).

        Args:
            event: One raw message object from :meth:`TalkClient.poll_messages`.

        Returns:
            The parsed event, or ``None`` to ignore the message — including when it
            carries no usable room token.

        Raises:
            RoomTypeUnresolved: If the room's type has not resolved yet. Deliberately
                **not** caught: it means a caller reached past the poller-side hold, and
                it propagates so the poll cycle leaves its persisted offset unadvanced
                and the message is re-fetched once the type is known. Swallowing it here
                would discard the message with no claim taken and the cursor already
                past it.
        """
        room = _event_room(event)
        if not room:
            # The id only, never the payload: message text is user content and this line
            # would otherwise copy a control-room conversation into the bridge log.
            logger.warning(
                "ignoring Talk message %r with no usable room token",
                event.get("id") if isinstance(event, Mapping) else None,
            )
            return None
        return parse_talk_message(event, room, self._cfg, self._rooms)

    def resolve_reply_context(self, event: InboundEvent) -> ReplyContext | None:
        """Resolve the quoted message a Talk reply points at, or ``None``.

        Free of I/O and never raises — Talk ships the quote inline as ``parent`` on the
        wire event, so this is pure parsing of :attr:`InboundEvent.raw`. See
        :func:`~osprey.bridges.nextcloud_talk.events.resolve_reply_context` for what is
        read and why the quoted text keeps a mention the question's own text loses.
        """
        return resolve_talk_reply_context(event)

    # --- posting members ---------------------------------------------------

    def _require_room(self, entry: Mapping[str, Any]) -> str:
        """The room token to post into.

        Raises:
            ValueError: If the entry carries no room token. Callers that must not raise
                catch this along with everything else; the members that must raise let it
                through, which is correct — an entry with no destination cannot be
                delivered by any later attempt either, and the retry queue's give-up
                ceiling is what eventually stops it.
        """
        room = _room(entry)
        if not room:
            raise ValueError(f"entry carries no {NC_ROOM}: nothing can be posted")
        return room

    def post_ack(self, entry: Mapping[str, Any]) -> None:
        """Post the immediate "on it" reply. Never raises.

        Best-effort by contract: the engine calls this as the first thing after a won
        claim, before the long dispatch, so a Talk hiccup here must cost the user a
        courtesy message and nothing more.
        """
        try:
            self._client.post_message(
                self._require_room(entry), ACK_TEXT, reply_to=_reply_target(entry)
            )
        except Exception:
            logger.warning(
                "ack post failed for %s; dispatch continues",
                entry.get(NC_MESSAGE_ID),
                exc_info=True,
            )

    def post_answer(self, entry: Mapping[str, Any], result: Mapping[str, Any]) -> None:
        """Post the terminal answer text into the room. **Raises** on failure.

        Long answers are split across messages (Talk caps one message at
        :data:`MAX_MESSAGE_CHARS`) and posted in order, stopping at the first failure so
        the raise reaches the engine. The engine then re-queues the entry and the drain
        re-delivers from the first chunk, which may duplicate chunks that already landed:
        that is at-least-once delivery, accepted deliberately. Silently dropping the
        tail of an answer would be the alternative, and it is far worse — a truncated
        answer reads as a complete one.

        Args:
            entry: The persisted entry; supplies the room and the reply target.
            result: The terminal result. Only ``status`` and ``text_output`` are read —
                ``error`` and the rest stay out of the room.

        Raises:
            ValueError: If the entry carries no room token.
            Exception: Whatever the Talk client raises on a failed post.
        """
        room = self._require_room(entry)
        reply_to = _reply_target(entry)
        for index, chunk in enumerate(_answer_chunks(result)):
            # The first message quotes the question so a multi-message answer stays
            # anchored to it in a busy room; the rest follow as a plain continuation
            # rather than repeating the quote on every chunk.
            self._client.post_message(room, chunk, reply_to=reply_to if index == 0 else None)

    def post_queued(self, entry: Mapping[str, Any], result: Mapping[str, Any]) -> None:
        """Post the first-park "your question is waiting" notice.

        May raise: the engine parks the entry BEFORE calling this and swallows the raise,
        so a lost notice costs a notification and never the request.

        Args:
            entry: The persisted entry.
            result: The retryable failure that triggered the park. Deliberately unread —
                it carries machine failure detail, and the user's answer to "what now?"
                is the same whatever the cause.
        """
        self._client.post_message(
            self._require_room(entry), QUEUED_TEXT, reply_to=_reply_target(entry)
        )

    def post_giveup(self, entry: Mapping[str, Any]) -> None:
        """Post the honest abandonment notice. **Raises** on failure.

        The engine marks the entry terminal only after this returns, so a raise leaves it
        queued for another attempt next cycle. A user who is never told their question
        was abandoned waits forever; a duplicate notice is a much smaller harm.

        The machine-side ``give_up_reason`` stays in store meta and never reaches the
        room.
        """
        self._client.post_message(
            self._require_room(entry), GIVEUP_TEXT, reply_to=_reply_target(entry)
        )

    def post_superseded(self, entry: Mapping[str, Any]) -> None:
        """Post the one-line "a newer question replaced this" note. Never raises.

        Posted as a reply to the superseded message, so the note lands on the question it
        is about rather than looking like a comment on the newest one. The coalesce CAS
        has already committed by the time the engine calls this, so a failure here must
        not disturb the pass.
        """
        try:
            self._client.post_message(
                self._require_room(entry), SUPERSEDED_TEXT, reply_to=_reply_target(entry)
            )
        except Exception:
            logger.warning(
                "superseded note failed for %s; the entry is superseded regardless",
                entry.get(NC_MESSAGE_ID),
                exc_info=True,
            )

    # --- inbound file downloads --------------------------------------------

    def download_inputs(self, entry: Mapping[str, Any]) -> InputDownload:
        """Fetch the inbound files this exchange referenced. Never raises.

        Reads the references **from the persisted entry and nothing else** — the message's
        own attachments from :data:`NC_FILES`, the quoted message's from
        :data:`~osprey.bridges.nextcloud_talk.events.QUOTED_FILES_KEY`. That is the whole
        point of the member: by the time the retry drain or a post-restart reconcile calls
        it, the wire event is long gone, so an implementation that reached for one would
        work in testing and silently lose every attachment after a restart.

        The two provenance buckets stay separate end to end. The engine reports own
        attachments at the payload's top level and quoted ones inside ``reply_to``, so a
        user can tell whether the file *they* attached or the one they *quoted* was
        dropped — mixing them would misattribute the loss.

        No capability or budget negotiation happens here: every referenced file is
        downloaded, and the engine's single memoized probe decides whether the bytes
        actually ship. Duplicating that decision here would probe the pair from the
        adapter, which it must never do.

        Args:
            entry: The persisted entry.

        Returns:
            The two buckets of files and skip notes. Exactly ``InputDownload()`` when the
            entry references no files at all, which keeps the dispatch payload
            byte-identical to the text-only path.
        """
        own_refs = _refs(entry, NC_FILES)
        quoted_refs = _refs(entry, QUOTED_FILES_KEY)
        if not own_refs and not quoted_refs:
            return InputDownload()
        files, skipped = self._download_refs(own_refs)
        quoted_files, quoted_skipped = self._download_refs(quoted_refs)
        return InputDownload(
            files=files,
            skipped=skipped,
            quoted_files=quoted_files,
            quoted_skipped=quoted_skipped,
        )

    def _download_refs(
        self, refs: Sequence[Mapping[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Download one bucket's references. Never raises.

        Args:
            refs: The reference mappings for one provenance bucket.

        Returns:
            ``(files, skipped)`` — worker-payload dicts for what was fetched, and
            ``{"filename", "reason"}`` notes for what was not. Every reference lands in
            exactly one of the two, in the order Talk listed them, so a failure is
            reported rather than silently absent.
        """
        files: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        for ref in refs:
            filename = _input_filename(ref)
            path = ref.get("path")
            # Blank counts as absent: no file can live at a whitespace-only path, so
            # asking the server about it would waste a round trip and then report the
            # honest-but-misleading "not in the file store". The parse step drops
            # path-less references already, so this is defence in depth.
            if not isinstance(path, str) or not path.strip():
                skipped.append({"filename": filename, "reason": NO_PATH_REASON})
                continue
            try:
                data = self._client.dav_download(path, max_bytes=MAX_DOWNLOAD_BYTES)
            except DavTooLargeError:
                logger.warning("inbound file %r is over the download cap; skipping", path)
                skipped.append({"filename": filename, "reason": TOO_LARGE_REASON})
                continue
            except Exception:
                # A deleted or moved file is a plain 404, and the room may be
                # unreachable — both are one lost attachment, never a lost dispatch.
                logger.warning("could not download inbound file %r", path, exc_info=True)
                skipped.append({"filename": filename, "reason": UNAVAILABLE_REASON})
                continue
            mimetype = ref.get("mimetype")
            files.append(
                {
                    "filename": filename,
                    "mime": mimetype if isinstance(mimetype, str) else "",
                    "content_b64": base64.b64encode(data).decode("ascii"),
                    # Fresh input, as opposed to a replayed prior artifact: the worker
                    # budgets these against its fresh-file caps.
                    "ingest": True,
                }
            )
        return files, skipped

    # --- outbound artifact delivery ----------------------------------------

    def deliver_files(
        self, entry: Mapping[str, Any], result: Mapping[str, Any]
    ) -> Mapping[str, str]:
        """Upload the run's artifacts into the room and share them. Never raises.

        Each artifact is fetched from the worker's byte route, uploaded into the bot's own
        DAV tree under :func:`~osprey.bridges.nextcloud_talk.client.upload_dir`, and
        shared into the room as a member-only share. Artifacts are independent: one that
        cannot be fetched, uploaded, or shared costs only itself.

        **Returns ``{}`` unconditionally — including on complete success. This is not an
        oversight.** The engine treats a returned URL as re-fetchable by an
        *unauthenticated* GET, because that is how it re-injects a prior artifact into a
        later run. A Talk room share is authenticated — readable by the room's members and
        nobody else — so a URL returned here would 401 for the engine and break
        re-injection in a way that surfaces much later and far from its cause. Returning
        nothing makes re-injection fall back to the worker byte route, which always works.
        The alternative, minting a world-readable public link, is deliberately not done:
        nothing from a control-room conversation is made public. Do not "fix" this to
        return the share URL.

        Best-effort by contract: the answer text has already landed via
        :meth:`post_answer`, and no upload failure may un-deliver it. Every step is
        accounted for — :func:`~osprey.bridges.core.fetch_artifact` returns ``None``
        rather than raising, and the layout check and each upload/share are caught — so no
        path out of here raises.

        Args:
            entry: The persisted entry; supplies the destination room.
            result: The terminal result; supplies ``run_id`` and ``artifacts``.

        Returns:
            An empty mapping, always.
        """
        room = _room(entry)
        # The live path takes the run id from the result; the drain's re-attached
        # delivery has it persisted on the entry as well.
        run_id = result.get("run_id") or entry.get("run_id")
        if not room or not isinstance(run_id, str) or not run_id:
            logger.debug("no room or run id to deliver artifacts for; text only")
            return {}
        artifacts = result.get("artifacts")
        descriptors = artifact_descriptors(artifacts if isinstance(artifacts, list) else None)
        if not descriptors:
            return {}
        try:
            directory = upload_dir(room, run_id)
        except ValueError:
            # A room token or run id that is not a single path segment. Text-only is the
            # right degradation here; writing outside the artifact tree is not.
            logger.warning(
                "no upload path for room %r run %r; text only", room, run_id, exc_info=True
            )
            return {}

        # One naming pass over every descriptor, before any fetch: this run's artifacts
        # all land in ONE directory, and the worker names them by basename, so two steps
        # that each wrote plot.png would otherwise overwrite one another. See
        # _unique_stems. The stem can be settled this early precisely because — unlike
        # the extension — it does not depend on the served Content-Type, which is what
        # decides the delivery path once the bytes are in hand.
        stems = _unique_stems(descriptors)

        for descriptor in descriptors:
            artifact_id = descriptor.get("artifact_id")
            if not isinstance(artifact_id, str) or artifact_id not in stems:
                continue  # artifact_descriptors already drops these; belt for a future shape
            self._deliver_one(room, directory, run_id, descriptor, stem=stems[artifact_id])
        return {}

    def _deliver_one(
        self,
        room: str,
        directory: str,
        run_id: str,
        descriptor: Mapping[str, Any],
        *,
        stem: str,
    ) -> None:
        """Fetch, upload, and share one artifact. Never raises.

        The descriptor only *names* the artifact: its ``delivered_mime`` is a
        prediction made before anything was rendered, so an artifact whose
        conversion failed arrives as its original bytes under a different
        Content-Type. Routing on the prediction would drop exactly those
        artifacts, silently (osprey #503) — so the delivered bytes decide the
        path, and the descriptor only supplies a preferred name.

        Only the EXTENSION is settled here, and only because it depends on the
        served ``Content-Type``, which is not known until the bytes are in hand.
        The stem arrives already settled from :func:`_unique_stems`, which can
        decide it before any fetch precisely because it does NOT depend on the
        served type — so an artifact served as something other than it announced
        still keeps the worker's own stem.

        Args:
            room: Destination room token.
            directory: DAV directory this run's artifacts go into.
            run_id: Run the artifact belongs to.
            descriptor: Normalized artifact descriptor — ``artifact_id`` plus any
                ``filename`` / ``delivered_mime`` hints the worker sent.
            stem: The filename stem assigned to this artifact for this delivery —
                already sanitized, and unique across the whole delivery (see
                :func:`_unique_stems`).
        """
        artifact_id = descriptor["artifact_id"]
        fetched = fetch_artifact(
            self._http,
            self._cfg.core,
            run_id,
            artifact_id,
            max_bytes=MAX_DELIVERED_DOC_BYTES,
        )
        if fetched is None:
            # fetch_artifact logged why (transport or oversize).
            return
        if fetched.is_png:
            # Talk renders a PNG inline, and an oversize one renders broken —
            # worse than not posting it. Documents get the larger budget.
            if len(fetched.data) > MAX_DELIVERED_IMAGE_BYTES:
                logger.warning(
                    "image artifact %s oversize (%d bytes), not delivered to room %s",
                    artifact_id,
                    len(fetched.data),
                    room,
                )
                return
            extension = ".png"
        else:
            predicted = descriptor.get("delivered_mime")
            extension = ext_for_mime(
                fetched.content_type or (predicted if isinstance(predicted, str) else None)
            )
        name = _upload_name(stem, extension)
        try:
            path = self._client.dav_mkcol_put(directory, name, fetched.data)
            self._client.share_to_room(path, room)
        except Exception:
            # A file that uploaded but failed to share is left where it is on purpose:
            # both halves are idempotent, so a redelivery of the same run re-shares it,
            # and cleaning up would need another call that can fail just as easily.
            logger.warning(
                "artifact %s not delivered to room %s; text only",
                artifact_id,
                room,
                exc_info=True,
            )

    # --- pure members ------------------------------------------------------

    def coalesce_key(self, entry: Mapping[str, Any]) -> Sequence[str]:
        """Group queued entries by conversation and sender.

        When dispatch is down and several questions pile up, the drain keeps only the
        newest entry per key and supersedes the rest. Keying on
        ``[history_key, sender_id]`` collapses one person's repeated asking in one room
        while never collapsing two different people's questions into one.

        Args:
            entry: The persisted entry.

        Returns:
            ``[history_key, sender_id]``, or ``()`` — the engine's "stands alone, never
            coalesce" sentinel — when either is missing. Standing alone is the
            fail-closed direction: a wrong group silently discards someone's question.
        """
        history_key = entry.get("history_key")
        sender_id = entry.get("sender_id")
        if not isinstance(history_key, str) or not history_key:
            return ()
        if not isinstance(sender_id, str) or not sender_id:
            return ()
        return [history_key, sender_id]


# NOT dead code, and not to be "cleaned up": this is the whole static conformance check
# for the seam. Assigning a NextcloudTalkOps to a ChannelOps makes mypy verify every
# member's FULL signature — parameter names, types, arity, return type — which a runtime
# ``isinstance`` protocol check cannot do (it only looks for the names). Nothing runs it
# and nothing constructs at import; the value is entirely in type-check time.
def _static_conformance(ops: NextcloudTalkOps) -> ChannelOps:
    return ops


def _answer_chunks(result: Mapping[str, Any]) -> list[str]:
    """The messages :meth:`NextcloudTalkOps.post_answer` should post, in order.

    A non-completed status becomes the channel's fixed error wording — the result's own
    ``error`` string is internal and never posted. A completed run with no text still
    gets a message, so "it worked but said nothing" never looks like a dead bridge.

    Args:
        result: The terminal result.

    Returns:
        One or more messages, each within :data:`MAX_MESSAGE_CHARS`. Never empty.
    """
    if result.get("status") != "completed":
        return [ERROR_TEXT]
    text = result.get("text_output")
    if not isinstance(text, str) or not text.strip():
        return [EMPTY_ANSWER_TEXT]
    return _chunk(text)
