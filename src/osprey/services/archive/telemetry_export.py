"""Export completed days of the telemetry store's logs and traces into the archive.

A UTC day ``D`` is exported once, by the first pass at or after ``D+1 01:00Z``
(an hour for late OTLP batches), for every completed day not yet recorded in a
manifest, walking back at most ``backfill_days``. Per type (``logs``,
``traces``) every stream is read hour by hour, a page of hits at a time, and
written one hit per line to ``<pass-day>/openobserve/<D>/<type>/<stream>.jsonl``.
Every stream of a day is staged before any is placed, and the day is recorded
only when all of them were read; a failure leaves the day for the next pass.

Credentials come only from the environment — ``ZO_INGEST_USER_EMAIL`` and
``ZO_INGEST_SA_TOKEN``, the ingest service account the deployment provisions —
never from a flag, whose value would be visible in the process list.
"""

from __future__ import annotations

import base64
import json
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from osprey.services.archive.manifest import ArchiveState

if TYPE_CHECKING:
    from osprey.services.archive.run import ArchiveTree

EMAIL_ENV = "ZO_INGEST_USER_EMAIL"
TOKEN_ENV = "ZO_INGEST_SA_TOKEN"
TELEMETRY_TYPES: tuple[str, ...] = ("logs", "traces")
PAGE_SIZE = 1000
LATE_BATCH_GRACE = timedelta(hours=1)
DEFAULT_BACKFILL_DAYS = 14


class TelemetryExportError(Exception):
    """One day could not be exported."""


def pending_days(now: datetime, exported: set[str], backfill_days: int) -> list[date]:
    """The completed days not yet exported, oldest first, at most *backfill_days* back."""
    latest = (now.astimezone(UTC) - LATE_BATCH_GRACE).date() - timedelta(days=1)
    days = [latest - timedelta(days=back) for back in range(max(backfill_days, 0))]
    return sorted(d for d in days if d.isoformat() not in exported)


def _micros(moment: datetime) -> int:
    return int(moment.timestamp() * 1_000_000)


def _stream_filename(stream: str) -> str:
    return stream.replace("/", "_").lstrip(".") + ".jsonl"


@dataclass
class TelemetryExporter:
    """Reads the telemetry store's search API as the ingest service account."""

    url: str
    org: str = "default"
    backfill_days: int = DEFAULT_BACKFILL_DAYS
    environ: Mapping[str, str] = field(default_factory=lambda: os.environ)
    transport: httpx.BaseTransport | None = None
    timeout: float = 60.0

    def _authorization(self) -> str:
        email = self.environ.get(EMAIL_ENV)
        token = self.environ.get(TOKEN_ENV)
        if not email or not token:
            raise TelemetryExportError(
                f"{EMAIL_ENV} and {TOKEN_ENV} must be set to read the telemetry store"
            )
        encoded = base64.b64encode(f"{email}:{token}".encode()).decode("ascii")
        return f"Basic {encoded}"

    def _client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self.url.rstrip("/"),
            headers={"Authorization": self._authorization()},
            timeout=self.timeout,
            transport=self.transport,
        )

    def _streams(self, client: httpx.Client, kind: str) -> list[str]:
        response = client.get(f"/api/{self.org}/streams", params={"type": kind})
        response.raise_for_status()
        body = response.json()
        return sorted(
            str(entry["name"])
            for entry in (body.get("list") or [])
            if isinstance(entry, dict) and entry.get("name")
        )

    def _hits(self, client: httpx.Client, kind: str, stream: str, day: date) -> Iterator[Any]:
        midnight = datetime.combine(day, time(0), tzinfo=UTC)
        quoted = stream.replace('"', '""')
        for hour in range(24):
            start = midnight + timedelta(hours=hour)
            end = start + timedelta(hours=1)
            offset = 0
            while True:
                payload = {
                    "query": {
                        "sql": f'SELECT * FROM "{quoted}" ORDER BY _timestamp ASC',
                        "start_time": _micros(start),
                        "end_time": _micros(end),
                        "from": offset,
                        "size": PAGE_SIZE,
                    }
                }
                response = client.post(
                    f"/api/{self.org}/_search", params={"type": kind}, json=payload
                )
                response.raise_for_status()
                hits = response.json().get("hits") or []
                yield from hits
                if len(hits) < PAGE_SIZE:
                    break
                offset += PAGE_SIZE

    def _export_day(
        self, client: httpx.Client, tree: ArchiveTree, day: date
    ) -> tuple[dict[str, int], list[tuple[Path, int, str, str]]]:
        rows: dict[str, int] = {}
        staged: list[tuple[Path, int, str, str]] = []
        try:
            for kind in TELEMETRY_TYPES:
                for stream in self._streams(client, kind):
                    count = 0

                    def _lines(kind: str = kind, stream: str = stream) -> Iterator[bytes]:
                        nonlocal count
                        for hit in self._hits(client, kind, stream, day):
                            count += 1
                            yield (json.dumps(hit, sort_keys=True) + "\n").encode("utf-8")

                    temp, size, sha = tree.stage(_lines())
                    rel = f"openobserve/{day.isoformat()}/{kind}/{_stream_filename(stream)}"
                    staged.append((temp, size, sha, rel))
                    rows[f"{kind}/{stream}"] = count
        except BaseException:
            for temp, *_ in staged:
                tree.discard(temp)
            raise
        return rows, staged

    @staticmethod
    def _place_day(
        tree: ArchiveTree,
        label: str,
        rows: dict[str, int],
        staged: list[tuple[Path, int, str, str]],
    ) -> None:
        """Place a staged day and record it; a failure discards what is still staged.

        The day line is written last, so a day that fails part-way stays
        unrecorded and the next pass exports it again.
        """
        pending = list(staged)
        try:
            while pending:
                temp, size, sha, rel = pending[0]
                path = tree.place(temp, rel)
                pending.pop(0)
                tree.append(
                    {
                        "kind": "file",
                        "source": rel,
                        "path": path,
                        "size": size,
                        "sha256": sha,
                        "source_mtime_ns": None,
                        "archived_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                    }
                )
            tree.append({"kind": "telemetry_day", "day": label, "rows": rows})
        except BaseException:
            for temp, *_ in pending:
                tree.discard(temp)
            raise

    def export(
        self, tree: ArchiveTree, state: ArchiveState, *, now: datetime
    ) -> tuple[list[str], list[dict[str, str]]]:
        """Export every pending day; return the days recorded and the errors met."""
        days = pending_days(now, state.telemetry_days, self.backfill_days)
        if not days:
            return [], []
        exported: list[str] = []
        errors: list[dict[str, str]] = []
        try:
            client = self._client()
        except TelemetryExportError as exc:
            return [], [{"source": "openobserve", "error": str(exc)}]
        with client:
            for day in days:
                label = day.isoformat()
                try:
                    rows, staged = self._export_day(client, tree, day)
                except (httpx.HTTPError, OSError, ValueError) as exc:
                    errors.append({"source": f"openobserve/{label}", "error": str(exc)})
                    continue
                try:
                    self._place_day(tree, label, rows, staged)
                except OSError as exc:
                    errors.append({"source": f"openobserve/{label}", "error": str(exc)})
                    continue
                state.telemetry_days.add(label)
                exported.append(label)
        return exported, errors
