"""Gather, once, everything a feedback report might want to say.

A feedback submission is rendered twice — once at the full record budget kept
on disk, once at the trimmed budget carried into the report body — and both
renderings describe the *same* moment. Reading the transcript twice would not
only cost twice (``read_session`` fans out into every subagent file, so each
pass is ``1 + N`` file reads); it would risk the two renderings disagreeing,
because the session keeps writing while the request is in flight. So the raw
material is collected exactly once per request into :class:`Material`, and the
render tiers work from that struct.

Two things in here look like over-caution and are not:

* **The session id is validated as a UUID before anything touches it.** It
  arrives from the browser and lands in ``transcript_dir / f"{session_id}.jsonl"``
  with no sanitising of its own, so anything that is not the canonical UUID
  spelling is refused here rather than resolved on disk.

* **Timestamps are parsed before they are compared.** Claude Code transcripts
  stamp ``2026-08-20T18:04:44.210Z``; :class:`~osprey.stores.artifact_store.ArtifactEntry`
  stamps ``2026-08-20T18:04:44.210000+00:00``. Those two spellings denote the
  same instant but do not order together as text (``Z`` is 0x5A, ``+`` is 0x2B),
  so a string comparison silently sorts every artifact before every event.

* **The budget is counted and cut in UTF-8 bytes, never characters.** The
  clipboard payload has to stay under GitHub's issue-body limit, which is a
  byte limit; a session full of non-ASCII text is up to four times longer in
  bytes than in characters, so a character-counted budget silently overshoots.
  Slicing is done on a character boundary (see :func:`_tail_within_bytes`) and
  encoding uses ``surrogatepass``, because a lone surrogate that arrived
  through JSON would otherwise raise on ``str.encode`` mid-render.
"""

from __future__ import annotations

import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from osprey.mcp_server.workspace.transcript_reader import TranscriptReader
    from osprey.stores.artifact_store import ArtifactEntry, ArtifactStore

CONTEXT_OK = "ok"
"""Session context was read in full."""

CONTEXT_TRANSCRIPT_NOT_FOUND = "transcript_not_found"
"""A session id was supplied but no transcript exists for it (dead session)."""

CONTEXT_NO_SESSION = "no_session"
"""No session id was supplied — there is nothing to read context from."""

BUNDLE_TRIM_MARKER = "[…older content trimmed]"
"""Left in a section where older content was dropped to fit the budget."""

TAGGED_ARTIFACTS_HEADER = "artifacts tagged to this session (titles and ids only)"
"""Header of the artifact block whose entries carry this session's tag."""

UNTAGGED_ARTIFACTS_HEADER = (
    "artifacts created on this deployment during this session's time window "
    "(not session-tagged; may include other terminal tabs or the chat panel)"
)
"""Header of the weaker, time-windowed artifact block.

The wording is a claim made to whoever reads the report: this block is *not*
"this session's artifacts" and must not be read as such. It is spelled here in
plain ASCII (a typographic apostrophe would not survive a verbatim comparison)
and pinned verbatim by this module's own tests. The dialog's help popover makes
the same promise to the operator in its own words and pins its own constant —
the two surfaces are deliberately independent, so neither imports the other's
string.
"""

NO_WINDOW_ARTIFACTS_HEADER = "session-tagged artifacts only (no usable session window)"
"""Header used instead when no time window could be derived (see
:func:`select_artifacts`) — the untagged block is omitted rather than guessed at."""

PAYLOAD_CAP_BYTES = 60_000
"""Whole clipboard payload ceiling, in UTF-8 bytes.

Sized to stay under GitHub's 64 KB issue-body limit with room to spare, so a
pasted report is never silently cut in half by the forge.
"""

CONTEXT_FLOOR_BYTES = 8_000
"""Bytes of the payload always reserved for session context.

Without a floor, a long enough report would push the context out entirely and
the operator would be told they attached a session they did not attach.
"""

PAYLOAD_SEPARATOR = "\n\n---\n\n# session context\n\n"
"""What sits between the operator's own words and the composed context.

It renders as a rule and a heading in a GitHub issue body, so the reader can
see at a glance where the human report stops and the machine dump starts.
"""

TEXT_TRUNCATION_MARKER = "[text truncated — full text is in the Local record <id>]"
"""Marker appended to a report too long for the clipboard payload.

``<id>`` is a literal placeholder that :func:`assemble_payload` substitutes when
it is given a ``record_id``. The substitution happens *here*, before the cap is
applied — a caller that replaced it afterwards would add the record id's bytes
to a payload already measured at the ceiling.
"""

_STATUS_NOTES = {
    CONTEXT_TRANSCRIPT_NOT_FOUND: "no transcript found for this session — text and metadata only",
    CONTEXT_NO_SESSION: "no session id — text and metadata only",
}


@dataclass
class Material:
    """The raw material of one feedback submission, collected once.

    Every render tier reads from here and none of them read from disk. Fields
    are the tiers in order: deployment ``metadata``, the tool-call/agent
    ``events`` log, the ``chat`` turns, the terminal ``scrollback`` tail, and
    the artifact index (``artifact_entries``, unfiltered — the window is
    applied by :func:`select_artifacts`, which the render tier calls).

    ``context_status`` is one of :data:`CONTEXT_OK`,
    :data:`CONTEXT_TRANSCRIPT_NOT_FOUND` or :data:`CONTEXT_NO_SESSION`, and is
    reported back to the operator so a report that carries only the typed text
    and the deployment metadata says so instead of looking merely empty.
    """

    metadata: dict[str, Any] = field(default_factory=dict)
    context_status: str = CONTEXT_OK
    events: list[dict[str, Any]] = field(default_factory=list)
    chat: list[dict[str, Any]] = field(default_factory=list)
    scrollback: str = ""
    artifact_entries: list[ArtifactEntry] = field(default_factory=list)
    session_id: str | None = None


def validate_session_id(session_id: Any) -> str:
    """Return ``session_id`` if it is a canonical session UUID, else raise.

    The id is client-supplied and is used directly as a transcript filename,
    so only the canonical lowercase hyphenated spelling is accepted: the URN
    form, the braced form, bare hex and an uppercased UUID all round-trip
    through :class:`uuid.UUID` but would name a different file (or no file) on
    a case-sensitive filesystem.

    Raises:
        ValueError: if ``session_id`` is not a canonical UUID string.
    """
    if not isinstance(session_id, str):
        raise ValueError(f"session_id must be a UUID string, got {type(session_id).__name__}")
    try:
        parsed = uuid.UUID(session_id)
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"session_id is not a UUID: {session_id!r}") from exc
    if str(parsed) != session_id:
        raise ValueError(f"session_id is not in canonical UUID form: {session_id!r}")
    return session_id


def parse_timestamp(value: Any) -> datetime | None:
    """Parse an ISO-8601 timestamp into an aware datetime, or return ``None``.

    Handles both spellings that meet here: the ``Z`` suffix Claude Code writes
    into transcripts and the ``+00:00`` suffix ``datetime.isoformat()`` writes
    into the artifact index. A timestamp without a zone is read as UTC, so the
    result is always comparable with the rest.

    Returns ``None`` — never raises — for the empty string, ``None``, a
    non-string, or anything unparseable. ``read_session`` synthesizes
    ``timestamp: ""`` for empty subagent files, so undated events are ordinary
    input here, not an error.
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def collect_material(
    reader: TranscriptReader,
    store: ArtifactStore,
    session_id: str | None,
    scrollback: str,
    metadata: dict[str, Any],
) -> Material:
    """Read every source once and return the material to render from.

    ``reader`` and ``store`` are injected rather than constructed here: the
    route builds them as ``TranscriptReader(app.state.project_cwd)`` and
    ``ArtifactStore(app.state.workspace_dir)``, and a reader rooted anywhere
    else silently reports every session as missing.

    Three shapes come out of this:

    * **A live session** — one call each to ``find_transcript_by_id``,
      ``read_session_by_id``, ``read_chat_history_by_id`` and ``list_entries``.
      ``list_entries`` is called with no ``session_filter``: the store's filter
      matches "this session or untagged" and so ignores the time window, which
      is the whole point of :func:`select_artifacts`.

    * **A dead session** (``find_transcript_by_id`` returns ``None``) — status
      :data:`CONTEXT_TRANSCRIPT_NOT_FOUND` and no further reads at all. This
      never raises; a session that ended, or a transcript that was cleaned up,
      still produces a report carrying the typed text and the metadata, which
      is exactly what the operator is told the report contains.

    * **No session id** — status :data:`CONTEXT_NO_SESSION`, metadata only.
      The artifact index is skipped along with the transcript: with no session
      there is neither a tag to match nor a set of events to derive a window
      from, so every entry in the store would be equally (un)related to this
      report and listing them would be noise, not context.
    """
    material = Material(metadata=dict(metadata), scrollback=scrollback, session_id=session_id)

    if session_id is None:
        material.context_status = CONTEXT_NO_SESSION
        return material

    if reader.find_transcript_by_id(session_id) is None:
        material.context_status = CONTEXT_TRANSCRIPT_NOT_FOUND
        return material

    material.context_status = CONTEXT_OK
    material.events = reader.read_session_by_id(session_id)
    material.chat = reader.read_chat_history_by_id(session_id)
    material.artifact_entries = store.list_entries()
    return material


def select_artifacts(
    entries: list[ArtifactEntry],
    session_id: str | None,
    event_timestamps: list[Any],
) -> tuple[list[ArtifactEntry], list[ArtifactEntry], bool]:
    """Split the artifact index into this session's artifacts and its neighbours.

    Returns ``(tagged, windowed, window_ok)``:

    * ``tagged`` — entries whose ``session_id`` matches, i.e. artifacts this
      session demonstrably produced.
    * ``windowed`` — *untagged* entries (``session_id == ""``) stamped inside
      the session's time window. Untagged is the normal case, not the odd one:
      a terminal session started without ``OSPREY_SESSION_ID`` tags nothing, so
      matching on the tag alone finds nothing for it. These entries are a
      weaker claim than ``tagged`` — they may belong to another terminal tab or
      to the chat panel — and callers must present them as such.
    * ``window_ok`` — whether a window could be derived at all. ``False`` means
      ``windowed`` is empty because nothing was datable, not because nothing
      matched, and the rendered section should say so.

    The window is ``min()``/``max()`` over the ``event_timestamps`` that parse.
    It is deliberately not ``events[0]``/``events[-1]``: ``read_session`` sorts
    by raw timestamp *string* and synthesizes ``""`` for empty subagent files,
    so the first event is routinely an undated one that would collapse the
    window to nothing.
    """
    tagged = [e for e in entries if session_id and e.session_id == session_id]

    parsed_events = [ts for ts in (parse_timestamp(t) for t in event_timestamps) if ts is not None]
    if not parsed_events:
        return tagged, [], False

    start, end = min(parsed_events), max(parsed_events)
    windowed = []
    for entry in entries:
        if entry.session_id != "":
            continue
        stamp = parse_timestamp(entry.timestamp)
        if stamp is not None and start <= stamp <= end:
            windowed.append(entry)
    return tagged, windowed, True


def _utf8_len(text: str) -> int:
    """Length of ``text`` in UTF-8 bytes, tolerating lone surrogates."""
    return len(text.encode("utf-8", "surrogatepass"))


def _tail_within_bytes(text: str, budget: int) -> str:
    """Return the longest tail of ``text`` that fits ``budget`` UTF-8 bytes.

    The cut lands on a character boundary: the byte slice is taken first and
    any continuation bytes (``0b10xxxxxx``) left at its head are dropped, so a
    multi-byte character is never split. Slicing by *characters* instead would
    have to guess how many bytes each one costs.
    """
    if budget <= 0:
        return ""
    data = text.encode("utf-8", "surrogatepass")
    if len(data) <= budget:
        return text
    tail = data[len(data) - budget :]
    start = 0
    while start < len(tail) and tail[start] & 0xC0 == 0x80:
        start += 1
    return tail[start:].decode("utf-8", "surrogatepass")


def _fit_pieces(pieces: list[str], budget: int, *, keep_tail: bool) -> tuple[list[str], bool]:
    """Fit whole ``pieces`` into ``budget`` bytes, dropping the oldest first.

    ``keep_tail`` says where the newest content sits: at the end of the list
    (chronological sections) or at its head (the chat tier, which renders
    newest-first). Whichever end holds the *older* content is the end that gets
    dropped, and :data:`BUNDLE_TRIM_MARKER` is spliced in at that end so the
    gap is visible in the report.

    Returns ``(pieces_including_marker, dropped)``.
    """
    if sum(_utf8_len(piece) for piece in pieces) <= budget:
        return list(pieces), False

    marker = BUNDLE_TRIM_MARKER + "\n"
    available = budget - _utf8_len(marker)
    if available < 0:
        return [], True

    kept: list[str] = []
    for piece in reversed(pieces) if keep_tail else pieces:
        cost = _utf8_len(piece)
        if cost > available:
            break
        available -= cost
        kept.append(piece)

    if keep_tail:
        kept.reverse()
        return [marker, *kept], True
    return [*kept, marker], True


class _BundleWriter:
    """Accumulates bundle sections until the byte budget is spent.

    Tier order is a *priority* order, not a set of quotas: each section takes
    what it needs from what is left, so an oversized event log legitimately
    crowds out the artifact inventory below it. Sections that cannot fit even
    their own header are dropped whole rather than rendered as a bare heading.
    """

    def __init__(self, budget_bytes: int) -> None:
        self._parts: list[str] = []
        self._remaining = max(0, budget_bytes)
        self.truncated = False

    def _header(self, title: str) -> str:
        # One blank line between sections, and never two: a chat turn already
        # ends with one, so an unconditional separator would double it.
        needs_gap = bool(self._parts) and not self._parts[-1].endswith("\n\n")
        separator = "\n" if needs_gap else ""
        return f"{separator}## {title}\n"

    def _commit(self, header: str, pieces: list[str]) -> None:
        self._parts.append(header)
        self._parts.extend(pieces)
        self._remaining -= _utf8_len(header) + sum(_utf8_len(piece) for piece in pieces)

    def add_section(self, title: str, pieces: list[str], *, keep_tail: bool = True) -> None:
        """Render ``pieces`` under ``title``, trimming oldest-first to fit."""
        if not pieces:
            return
        header = self._header(title)
        body_budget = self._remaining - _utf8_len(header)
        if body_budget <= 0:
            self.truncated = True
            return
        kept, dropped = _fit_pieces(pieces, body_budget, keep_tail=keep_tail)
        if not kept:
            self.truncated = True
            return
        self._commit(header, kept)
        self.truncated = self.truncated or dropped

    def add_text_section(self, title: str, text: str) -> None:
        """Render free text under ``title``, keeping its tail.

        Unlike :meth:`add_section` this cuts *inside* a line when it has to:
        terminal scrollback can be one enormous unbroken line, and dropping it
        whole would report nothing at all. A partial first line is then dropped
        so the section still starts on a line boundary.
        """
        if not text:
            return
        body = text if text.endswith("\n") else text + "\n"
        header = self._header(title)
        body_budget = self._remaining - _utf8_len(header)
        if body_budget <= 0:
            self.truncated = True
            return

        if _utf8_len(body) <= body_budget:
            self._commit(header, [body])
            return

        marker = BUNDLE_TRIM_MARKER + "\n"
        tail = _tail_within_bytes(body, body_budget - _utf8_len(marker))
        if not tail:
            self.truncated = True
            return
        break_at = tail.find("\n")
        if 0 <= break_at < len(tail) - 1:
            tail = tail[break_at + 1 :]
        self._commit(header, [marker, tail])
        self.truncated = True

    def result(self) -> str:
        """The assembled bundle, guaranteed encodable as UTF-8.

        Everything above this line counts bytes with ``surrogatepass`` so a
        lone surrogate cannot raise mid-render — but the *material* is read off
        disk (chat turns, tool and agent names, artifact titles), and a browser
        that stringified an unpaired UTF-16 half writes ``\\ud800`` straight
        into a transcript. Handing that on unrepaired moved the failure to the
        response renderer, which raises **after** the caller has written its
        record: the store ended up holding a submission the request then 500'd
        on. Substituting here is also the one place it is safe to do: the
        replacement is never longer than what it replaces, so the budget the
        sections were trimmed to still holds.
        """
        return "".join(self._parts).encode("utf-8", "replace").decode("utf-8")


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


def first_line(text: str, limit: int = 200) -> str:
    """First line of ``text``, stripped of control characters and capped.

    Two callers with the same hazard: the error summary rendered into a bundle,
    and the excerpt the route stores in a record header — which ``osprey
    feedback list`` prints straight into a maintainer's terminal. The text
    comes from a browser, so a report whose first line is ``\\x1b[2J`` would
    clear the screen of whoever is triaging it. A tab survives; everything else
    in Unicode category ``C`` (controls, format characters, surrogates,
    unassigned) does not.

    The strip happens *before* the cap, so escape bytes cannot spend the
    caller's budget either.
    """
    stripped = text.strip()
    if not stripped:
        return ""
    line = stripped.splitlines()[0]
    cleaned = "".join(ch for ch in line if ch == "\t" or unicodedata.category(ch)[0] != "C")
    return cleaned[:limit]


def _format_metadata(metadata: dict[str, Any]) -> list[str]:
    return [f"- {key}: {value}\n" for key, value in metadata.items()]


def _format_event(event: dict[str, Any]) -> str:
    """One line per event: when, what, which tool or agent, and any error.

    Tool arguments are deliberately left out — they are unbounded and are the
    part of a session most likely to carry values the operator did not mean to
    publish. An error's first result line is kept because a report that says
    only "a tool failed" is not worth sending.
    """
    stamp = event.get("timestamp") or "(undated)"
    kind = event.get("type") or "event"
    if kind == "tool_call":
        name = event.get("full_tool_name") or event.get("tool") or event.get("tool_name") or "?"
        line = f"{stamp}  {kind}  {name}"
        if event.get("is_error"):
            summary = first_line(str(event.get("result_summary") or ""))
            line = f"{line}  ERROR: {summary}" if summary else f"{line}  ERROR"
    else:
        name = event.get("agent_type") or event.get("agent_id") or ""
        line = f"{stamp}  {kind}  {name}".rstrip()
    return line + "\n"


def _format_turn(turn: dict[str, Any]) -> str:
    role = turn.get("role") or "?"
    stamp = turn.get("timestamp") or "(undated)"
    return f"[{role} {stamp}]\n{turn.get('content') or ''}\n\n"


def _format_artifacts(entries: list[ArtifactEntry]) -> list[str]:
    """Render an artifact block as ids and titles only, oldest line first.

    Nothing else travels: a description, filename or metadata dict can quote
    the data the artifact was made from, and the inventory is here to say
    *what exists*, not to carry it into a public issue.
    """
    oldest_first = sorted(
        entries, key=lambda e: parse_timestamp(e.timestamp) or datetime.min.replace(tzinfo=UTC)
    )
    return [f"- {entry.id}  {entry.title.strip() or '(untitled)'}\n" for entry in oldest_first]


def render_bundle(material: Material, budget_bytes: int) -> tuple[str, bool]:
    """Render ``material`` into at most ``budget_bytes`` UTF-8 bytes.

    Returns ``(text, truncated)``. The same material and budget always render
    to byte-identical output: the tiers are rendered in a fixed order, nothing
    is read from the clock or the filesystem here, and every ordering is total.

    The tiers, in priority order, are (1) deployment metadata, (2) the event
    log, (3) chat history newest turn first, (4) the terminal scrollback tail
    and (5) the artifact inventory. Each takes what it needs from the budget
    that is left, and each is trimmed by dropping its *oldest* content — the
    end of a session is what a report is usually about.

    When ``material.context_status`` is not :data:`CONTEXT_OK` the four context
    tiers are replaced by a one-line note saying why, so a report carrying only
    the deployment metadata says so rather than looking merely uneventful.
    """
    writer = _BundleWriter(budget_bytes)
    writer.add_section("deployment", _format_metadata(material.metadata), keep_tail=False)

    if material.context_status != CONTEXT_OK:
        note = _STATUS_NOTES.get(
            material.context_status, f"session context unavailable ({material.context_status})"
        )
        writer.add_section("session context", [note + "\n"], keep_tail=False)
        return writer.result(), writer.truncated

    writer.add_section(
        f"event log ({_count(len(material.events), 'event')})",
        [_format_event(event) for event in material.events],
    )
    writer.add_section(
        f"chat history ({_count(len(material.chat), 'turn')}, newest first)",
        [_format_turn(turn) for turn in reversed(material.chat)],
        keep_tail=False,
    )
    writer.add_text_section("terminal scrollback (tail)", material.scrollback)

    tagged, windowed, window_ok = select_artifacts(
        material.artifact_entries,
        material.session_id,
        [event.get("timestamp", "") for event in material.events],
    )
    if window_ok:
        writer.add_section(TAGGED_ARTIFACTS_HEADER, _format_artifacts(tagged))
        writer.add_section(UNTAGGED_ARTIFACTS_HEADER, _format_artifacts(windowed))
    else:
        writer.add_section(NO_WINDOW_ARTIFACTS_HEADER, _format_artifacts(tagged))

    return writer.result(), writer.truncated


def _head_within_bytes(text: str, budget: int) -> str:
    """Return the longest *head* of ``text`` that fits ``budget`` UTF-8 bytes.

    The mirror of :func:`_tail_within_bytes`: the cut is walked back off any
    continuation byte so a multi-byte character is never split.
    """
    if budget <= 0:
        return ""
    data = text.encode("utf-8", "surrogatepass")
    if len(data) <= budget:
        return text
    end = budget
    while end > 0 and data[end] & 0xC0 == 0x80:
        end -= 1
    return data[:end].decode("utf-8", "surrogatepass")


def clamp_text_len(value: Any, cap: int = PAYLOAD_CAP_BYTES) -> int:
    """Read a client-supplied text length as a byte count in ``0..cap``.

    The bundle endpoint takes this number from the browser to decide how much
    of the payload the typed text will claim, so it is exactly the number an
    attacker (or a bug) would push out of range to lift the cap: a negative
    value would *add* budget, a huge one would starve the context, and a
    non-numeric one would raise inside the handler. Anything unusable — ``None``,
    an absent field, a negative number, a non-numeric string, a list — reads as
    ``0``, which costs the caller nothing but a slightly larger bundle.
    """
    try:
        parsed = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return max(0, min(parsed, cap))


def _max_text_bytes(cap: int, floor: int, separator: str) -> int:
    """Most of the payload the report itself may claim.

    The one place this cut is computed. :func:`payload_bundle_budget` needs it
    to size what is left, and :func:`assemble_payload` needs the same number to
    decide whether the report has to be truncated at all — derived twice, the
    two would disagree the first time one of them was edited, and the payload
    would come out either over the cap or short of the context ``floor``.
    """
    return max(0, cap - _utf8_len(separator) - floor)


def payload_bundle_budget(
    text_bytes: int,
    cap: int = PAYLOAD_CAP_BYTES,
    floor: int = CONTEXT_FLOOR_BYTES,
    separator: str = PAYLOAD_SEPARATOR,
) -> int:
    """Bytes left for the context bundle once ``text_bytes`` of report are placed.

    This is the *single* budget formula. :func:`assemble_payload` and
    :func:`bundle_for_copy` both call it rather than each deriving their own,
    because the standalone Copy button must produce a bundle byte-identical to
    the one inside the send payload — two formulas that agree today would drift
    the first time one of them is edited.

    A report longer than ``cap - separator - floor`` claims only that much: the
    rest of it is truncated, so the context still gets its ``floor``.
    """
    claimed = min(max(0, text_bytes), _max_text_bytes(cap, floor, separator))
    return max(0, cap - _utf8_len(separator) - claimed)


def _truncate_text(text: str, budget: int, record_id: str | None) -> str:
    """Cut ``text`` to ``budget`` bytes, keeping its head and marking the cut.

    The head is what survives, unlike every section of the bundle: a report's
    first sentences are the report, while a session's *last* events are the
    interesting ones.
    """
    marker = TEXT_TRUNCATION_MARKER
    if record_id:
        marker = marker.replace("<id>", record_id)
    marker_piece = "\n" + marker
    head_budget = budget - _utf8_len(marker_piece)
    if head_budget <= 0:
        # Absurd parameters only: not even the marker fits. Say as much of it
        # as there is room for rather than silently returning the raw head.
        return _head_within_bytes(marker, budget)
    return _head_within_bytes(text, head_budget) + marker_piece


def assemble_payload(
    text: str,
    material: Material,
    cap: int = PAYLOAD_CAP_BYTES,
    floor: int = CONTEXT_FLOOR_BYTES,
    separator: str = PAYLOAD_SEPARATOR,
    record_id: str | None = None,
) -> tuple[str, str, bool]:
    """Build the clipboard payload: the report, then the composed context.

    Returns ``(payload, bundle, text_truncated)``. ``payload`` is what the
    browser writes to the clipboard verbatim — the frontend does no size
    arithmetic of its own, and **this is the only place the cap is applied**.
    ``len(payload.encode("utf-8")) <= cap`` holds by construction: the report
    claims at most ``cap - separator - floor`` bytes and the bundle is rendered
    into exactly what is left.

    ``record_id`` is substituted into :data:`TEXT_TRUNCATION_MARKER` when the
    caller already knows it. The send route does: it allocates the id with
    ``new_record_id()`` *before* composing and hands the same id to
    ``write_record`` afterwards, precisely so the marker, the digest stored in
    the header and the bytes the operator pasted all describe one string.
    Left out, the marker keeps its literal ``<id>`` placeholder — which
    is right for the record-less bundle endpoint, but a caller must **not**
    substitute the id afterwards: that would add bytes to a payload already
    measured at the ceiling.

    The separator is omitted when there is no context at all, so a report sent
    without a session does not carry an empty "session context" heading.
    """
    max_text = _max_text_bytes(cap, floor, separator)
    text_bytes = _utf8_len(text)

    if text_bytes <= max_text:
        text_part, text_truncated = text, False
    else:
        text_part, text_truncated = _truncate_text(text, max_text, record_id), True

    budget = payload_bundle_budget(text_bytes, cap=cap, floor=floor, separator=separator)
    bundle, _ = render_bundle(material, budget)

    payload = text_part + separator + bundle if bundle else text_part
    return payload, bundle, text_truncated


def bundle_for_copy(
    material: Material,
    text_len: Any,
    cap: int = PAYLOAD_CAP_BYTES,
    floor: int = CONTEXT_FLOOR_BYTES,
    separator: str = PAYLOAD_SEPARATOR,
) -> str:
    """Render the context bundle alone, sized as if ``text_len`` bytes preceded it.

    This backs the standalone "Copy session context" button, which writes no
    record. ``text_len`` is the UTF-8 **byte** count of whatever is in the text
    box (the browser must measure it with ``TextEncoder``, never
    ``String.length``) and is clamped by :func:`clamp_text_len` before it can
    influence the budget.

    The result is byte-identical to the bundle section of
    :func:`assemble_payload` for the same material and text length — both size
    themselves through :func:`payload_bundle_budget`. That equality is what
    lets the operator copy the context, keep typing, and still send a report
    whose context matches what they pasted.
    """
    budget = payload_bundle_budget(
        clamp_text_len(text_len, cap=cap), cap=cap, floor=floor, separator=separator
    )
    bundle, _ = render_bundle(material, budget)
    return bundle
