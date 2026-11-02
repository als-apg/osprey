"""An in-memory agent record for the bridge-engine suite, as one bridge sees it.

:class:`MemoryRecord` takes the bridge's ``bridge_turn`` writes (the spool writer's
``write(kind, body, **ids)`` signature) and answers its ``bridge_history`` reads in
the record client's wire shape: newest turn first, each with its run's facts. A
turn's answer is the run's own, which a dispatch worker records — :meth:`answer`
stands in for that. Like the real reader, a turn naming a run the record does not
hold yet makes the read incomplete.

:class:`RecordHistory` is the real
:class:`~osprey.bridges.core.history.ConversationHistory` over a
:class:`MemoryRecord`, plus two test conveniences: :meth:`RecordHistory.seed` (an
earlier answered turn, run answer included) and :meth:`RecordHistory.recent` (what a
read answers, uncut). With ``ops`` it marks its reads and writes into a
:class:`~tests.bridges.conftest.RecordingChannelOps` timeline.

The real path — the spool writer, a running agent-record service and its drain —
is covered by ``tests/bridges/test_history_record.py``.
"""

from __future__ import annotations

import copy
import itertools
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from osprey.agent_record.client import ClientAnswer
from osprey.agent_record.contract import KIND_BRIDGE_TURN
from osprey.bridges.core.history import ConversationHistory, HistoryRead

__all__ = ["MemoryRecord", "RecordHistory"]

_T0 = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)


class MemoryRecord:
    """The record one bridge reads and writes (see the module docstring).

    ``unavailable`` set makes every read answer unavailable with that reason;
    ``incomplete`` set makes every read answer incomplete with it.
    """

    def __init__(self) -> None:
        self.turns: list[dict[str, Any]] = []
        self.runs: dict[str, str | None] = {}
        self.unavailable: str | None = None
        self.incomplete: str | None = None
        self.reads: list[dict[str, Any]] = []
        self._ids = itertools.count(1)

    # --- the writer side ---------------------------------------------------
    def write(self, kind: str, body: Mapping[str, Any], **ids: Any) -> str:
        assert kind == KIND_BRIDGE_TURN, kind
        n = next(self._ids)
        entry_id = f"turn-{n}"
        self.turns.append(
            {
                "key": f"sp:teams-bridge:{entry_id}",
                "kind": kind,
                "ts": (_T0 + timedelta(seconds=n)).isoformat(),
                "run_id": ids.get("run_id"),
                "body": copy.deepcopy(dict(body)),
            }
        )
        return entry_id

    def answer(self, run_id: str, text: str | None) -> None:
        """The dispatch worker recording run *run_id*, whose final answer is *text*."""
        self.runs[run_id] = text

    def bodies(self, conversation: str | None = None) -> list[dict[str, Any]]:
        """Every recorded turn body, oldest first, optionally of one conversation."""
        return [
            t["body"]
            for t in self.turns
            if conversation is None or t["body"].get("conversation") == conversation
        ]

    # --- the reader side ---------------------------------------------------
    def bridge_history(
        self,
        *,
        conversation: str | None = None,
        limit: int | None = None,
        wait: Any = None,
    ) -> ClientAnswer:
        self.reads.append({"conversation": conversation, "limit": limit, "wait": wait})
        if self.unavailable is not None:
            return ClientAnswer.failed(self.unavailable)
        if self.incomplete is not None:
            return ClientAnswer(data=[], complete=False, reason=self.incomplete)
        turns = [
            t
            for t in reversed(self.turns)
            if conversation is None or t["body"].get("conversation") == conversation
        ]
        head = turns[:limit] if limit is not None else turns
        rows: list[dict[str, Any]] = []
        complete, reason = True, ""
        for turn in head:
            run = None
            run_id = turn["run_id"]
            if run_id:
                if run_id in self.runs:
                    run = {"run_id": run_id, "status": "done", "answer": self.runs[run_id]}
                else:
                    complete, reason = False, f"run {run_id} has no accept or terminal entry yet"
            rows.append({"turn": copy.deepcopy(turn), "run": run})
        return ClientAnswer(
            data=rows, complete=complete, more=len(turns) > len(head), reason=reason
        )


class RecordHistory(ConversationHistory):
    """:class:`ConversationHistory` over a :class:`MemoryRecord` (module docstring)."""

    def __init__(self, record: MemoryRecord | None = None, ops: Any = None, **kwargs: Any):
        self.record = record if record is not None else MemoryRecord()
        self._ops = ops
        super().__init__(self.record, write=self.record.write, **kwargs)

    def _mark(self, name: str, **details: Any) -> None:
        if self._ops is not None:
            self._ops.mark(name, **details)

    def read(self, key: str) -> HistoryRead:
        read = super().read(key)
        self._mark("history_read", key=key, turns=len(read.turns), record=read.record)
        return read

    def append(self, key: str, question: str, **kwargs: Any) -> str | None:
        self._mark("history_append", key=key, question=question)
        return super().append(key, question, **kwargs)

    def append_failed(self, key: str, question: str, asked_by=None, **kwargs: Any):
        self._mark("history_append_failed", key=key, question=question)
        return super().append_failed(key, question, asked_by, **kwargs)

    # --- test conveniences -------------------------------------------------
    def seed(
        self,
        key: str,
        question: str,
        answer: str,
        *,
        run_id: str,
        artifacts: Sequence[Mapping[str, Any]] | None = None,
        asked_by: Mapping[str, Any] | None = None,
    ) -> None:
        """An earlier answered turn: its run's answer recorded, then the turn."""
        self.record.answer(run_id, answer)
        ConversationHistory.append(
            self, key, question, run_id=run_id, artifacts=artifacts, asked_by=asked_by
        )

    def recent(self, key: str) -> list[dict[str, Any]]:
        """What a read of *key* answers, uncut, without marking the timeline."""
        return ConversationHistory.read(self, key).turns
