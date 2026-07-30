"""Shared fixtures for the bridge-engine suite: the ``ChannelOps`` recording double.

:class:`RecordingChannelOps` implements the full :class:`~osprey.bridges.core.ports.ChannelOps`
protocol and records every call — plus any engine-side event a test :meth:`~RecordingChannelOps.mark`\\ s
into the same timeline (a dedup claim, a dispatcher run) — so ordering invariants read as one
assertion::

    ops = RecordingChannelOps()
    ...
    ops.assert_order("claim", "post_ack", "dispatch", "post_answer", "history_append")

Import it directly (``from tests.bridges.conftest import RecordingChannelOps``) when a test
needs custom construction, or use the ``channel_ops`` fixture for a fresh default instance.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest

from osprey.bridges.core.ports import InboundEvent, InputDownload, ReplyContext

# --- canned-return knob types -------------------------------------------------
# Every knob is "a constant, or a callable over the member's own argument". The
# callable form is what lets ONE double answer differently per event or per entry:
# with constants alone, two wire events necessarily parse to the same
# ``message_id``, dedup collapses them into one claim, and a multi-room test cannot
# express two rooms doing different things at once.
ParseKnob = InboundEvent | None | Callable[[Any], InboundEvent | None]
ReplyContextKnob = ReplyContext | None | Callable[[InboundEvent], ReplyContext | None]
InputsKnob = InputDownload | None | Callable[[Mapping[str, Any]], InputDownload]
FileUrlsKnob = Mapping[str, str] | None | Callable[[Mapping[str, Any]], Mapping[str, str]]
CoalesceKnob = str | Sequence[str] | Callable[[Mapping[str, Any]], str | Sequence[str]]


@dataclass(frozen=True)
class RecordedCall:
    """One timeline entry: the op (protocol member or ``mark``\\ ed engine event) and a
    plain dict of whatever arguments/details the call carried."""

    op: str
    details: Mapping[str, Any] = field(default_factory=dict)


class RecordingChannelOps:
    """A ``ChannelOps`` test double that records call order and returns canned values.

    Construction knobs (all optional). Each takes **either a constant or a callable of
    that member's own argument**, resolved at call time — so a knob may also be
    reassigned mid-test, and a callable knob sees state the test has changed since:

    ``parse_result``
        What :meth:`parse_event` returns (default ``None`` = ignore the event).
        Callable form: ``(event) -> InboundEvent | None``.
    ``reply_context``
        What :meth:`resolve_reply_context` returns (default ``None`` = not a reply).
        Callable form: ``(event: InboundEvent) -> ReplyContext | None``.
    ``inputs``
        What :meth:`download_inputs` returns (default empty :class:`InputDownload`).
        Callable form: ``(entry) -> InputDownload``.
    ``file_urls``
        What :meth:`deliver_files` returns (default ``{}``). Callable form:
        ``(entry) -> Mapping[str, str]`` — the entry only, not the run result.
    ``coalesce``
        What :meth:`coalesce_key` returns (default ``""`` = never coalesce).
        Callable form: ``(entry) -> str | Sequence[str]``.

    The callable form is what makes one shared instance usable by a multi-room test:
    ``parse_result=lambda raw: make_event(message_id=f"{raw['room']}:{raw['id']}")``
    gives every wire event a distinct id, so dedup treats them as distinct messages
    instead of collapsing them into a single claim.

    Failure injection: :meth:`fail_next` queues an exception for a member's NEXT call —
    the call is recorded first, then raises — so drain tests can express "the first
    ``post_answer`` fails, the retry succeeds" without a custom double.
    """

    def __init__(
        self,
        *,
        parse_result: ParseKnob = None,
        reply_context: ReplyContextKnob = None,
        inputs: InputsKnob = None,
        file_urls: FileUrlsKnob = None,
        coalesce: CoalesceKnob = "",
    ) -> None:
        self.calls: list[RecordedCall] = []
        self.parse_result = parse_result
        self.reply_context = reply_context
        self.inputs: InputsKnob = inputs if inputs is not None else InputDownload()
        # A constant is copied once here, as it always was; a callable is kept intact
        # and its result copied per call instead (see :meth:`deliver_files`).
        self.file_urls: FileUrlsKnob = file_urls if callable(file_urls) else dict(file_urls or {})
        self.coalesce = coalesce
        self._failures: dict[str, deque[BaseException]] = {}

    # -- recording machinery -------------------------------------------------

    def mark(self, op: str, **details: Any) -> None:
        """Record an ENGINE-side event (``claim``, ``dispatch``, ``history_append``, ...)
        into the same timeline as the protocol calls, so cross-boundary ordering
        ("claim precedes dispatch") is assertable alongside channel I/O. Wire it into
        store/dispatcher stubs: ``dispatcher.run = lambda *a, **k: ops.mark("dispatch")``."""
        self._record(op, details)

    def fail_next(self, op: str, exc: BaseException) -> None:
        """Queue ``exc`` to be raised by the next call to ``op`` (after recording it).
        Multiple queued failures pop FIFO; once drained, calls succeed again."""
        self._failures.setdefault(op, deque()).append(exc)

    def _record(self, op: str, details: Mapping[str, Any]) -> None:
        self.calls.append(RecordedCall(op, dict(details)))
        queued = self._failures.get(op)
        if queued:
            raise queued.popleft()

    # -- timeline inspection -------------------------------------------------

    @property
    def names(self) -> list[str]:
        """Every recorded op name, in call order."""
        return [call.op for call in self.calls]

    def of(self, op: str) -> list[Mapping[str, Any]]:
        """The recorded details of every call to ``op``, in order."""
        return [call.details for call in self.calls if call.op == op]

    def count(self, op: str) -> int:
        """How many times ``op`` was called."""
        return len(self.of(op))

    def assert_order(self, *ops: str) -> None:
        """Assert ``ops`` occur as a subsequence of the recorded timeline (greedy
        left-to-right match, so repeated ops are handled naturally). Raises
        ``AssertionError`` naming the first op that could not be matched, with the
        full timeline for diagnosis.

        Because the match is a greedy subsequence, this CANNOT express "X only ever
        after Y": ``['history_append', 'post_answer', 'history_append']`` passes
        ``assert_order('post_answer', 'history_append')`` despite an append preceding
        the post. Use :meth:`assert_never_before` for those invariants."""
        position = 0
        names = self.names
        for op in ops:
            try:
                position = names.index(op, position) + 1
            except ValueError:
                raise AssertionError(
                    f"expected {op!r} (in order {list(ops)!r}) but it does not occur "
                    f"after position {position} in recorded timeline {names!r}"
                ) from None

    def assert_never_before(self, op: str, other: str) -> None:
        """Assert NO occurrence of ``op`` precedes the FIRST occurrence of ``other``
        (which must occur). The strict form of "op only ever after other": unlike
        :meth:`assert_order`'s greedy subsequence matching, a stray earlier ``op`` —
        exactly the shape an engine bug that appended history on a failed attempt
        would produce — fails here, with the full timeline for diagnosis."""
        names = self.names
        assert other in names, f"expected {other!r} in recorded timeline {names!r}"
        first_other = names.index(other)
        assert op not in names[:first_other], (
            f"{op!r} occurred before the first {other!r} in recorded timeline {names!r}"
        )

    # -- ChannelOps protocol members ------------------------------------------
    # Each canned return resolves its knob the same way: call it with this member's
    # argument if it is callable, otherwise hand the constant back untouched.

    def parse_event(self, event: Any) -> InboundEvent | None:
        self._record("parse_event", {"event": event})
        knob = self.parse_result
        return knob(event) if callable(knob) else knob

    def resolve_reply_context(self, event: InboundEvent) -> ReplyContext | None:
        self._record("resolve_reply_context", {"event": event})
        knob = self.reply_context
        return knob(event) if callable(knob) else knob

    def post_ack(self, entry: Mapping[str, Any]) -> None:
        self._record("post_ack", {"entry": entry})

    def download_inputs(self, entry: Mapping[str, Any]) -> InputDownload:
        self._record("download_inputs", {"entry": entry})
        knob = self.inputs
        return knob(entry) if callable(knob) else knob

    def post_answer(self, entry: Mapping[str, Any], result: Mapping[str, Any]) -> None:
        self._record("post_answer", {"entry": entry, "result": result})

    def deliver_files(
        self, entry: Mapping[str, Any], result: Mapping[str, Any]
    ) -> Mapping[str, str]:
        self._record("deliver_files", {"entry": entry, "result": result})
        knob = self.file_urls
        return dict(knob(entry) if callable(knob) else knob or {})

    def post_queued(self, entry: Mapping[str, Any], result: Mapping[str, Any]) -> None:
        self._record("post_queued", {"entry": entry, "result": result})

    def post_giveup(self, entry: Mapping[str, Any]) -> None:
        self._record("post_giveup", {"entry": entry})

    def post_superseded(self, entry: Mapping[str, Any]) -> None:
        self._record("post_superseded", {"entry": entry})

    def coalesce_key(self, entry: Mapping[str, Any]) -> str | Sequence[str]:
        self._record("coalesce_key", {"entry": entry})
        knob = self.coalesce
        return knob(entry) if callable(knob) else knob


@pytest.fixture
def channel_ops() -> RecordingChannelOps:
    """A fresh default recording double (parse ignores, no reply, no files)."""
    return RecordingChannelOps()
