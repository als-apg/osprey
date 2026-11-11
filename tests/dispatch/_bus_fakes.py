"""A fake dispatch worker for the bus suites: canned answers, every request recorded."""

from __future__ import annotations

import asyncio
from typing import Any

from osprey.dispatch.bus import protocol
from osprey.dispatch.bus.link import WorkerLink


class FakeWorker:
    """A worker's request handlers, recording what each was asked."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.boot_nonce = 1.0
        self.capabilities = ["input_files", "prior_answers"]
        self.in_flight: set[str] = set()
        self.calls: list[tuple[bytes, dict[str, Any]]] = []
        self.reject: Exception | None = None
        self.statuses: dict[str, dict[str, Any]] = {}
        self.rows: list[dict[str, Any]] = []
        self.blob = b"\x89PNG\r\n\x1a\n" + bytes(range(256))
        self.hold: asyncio.Event | None = None
        self.link: WorkerLink | None = None
        self._count = 0

    def handlers(self) -> dict[bytes, Any]:
        return {kind: self._handler(kind) for kind in protocol.REQUESTS}

    def _handler(self, kind: bytes):
        async def handle(body: dict[str, Any], _blob: bytes | None):
            self.calls.append((kind, body))
            if self.hold is not None:
                await self.hold.wait()
            if self.reject is not None:
                raise self.reject
            return await getattr(self, kind.decode().lower())(body)

        return handle

    def hello(self) -> dict[str, Any]:
        return {
            "boot_nonce": self.boot_nonce,
            "capabilities": self.capabilities,
            "in_flight": sorted(self.in_flight),
        }

    def asked(self, kind: bytes) -> int:
        return sum(1 for k, _ in self.calls if k == kind)

    async def dispatch(self, _body):
        self._count += 1
        run_id = f"{self.name}-run-{self._count}"
        self.in_flight.add(run_id)
        return {"status": "accepted", "run_id": run_id}, None

    async def status(self, body):
        if body["run_id"] not in self.statuses:
            raise protocol.BusError(404, "not found")
        return self.statuses[body["run_id"]], None

    async def cancel(self, body):
        if body["run_id"] in self.in_flight:
            return {"run_id": body["run_id"], "cancelled": True}, None
        return {"run_id": body["run_id"], "cancelled": False, "reason": "no task handle"}, None

    async def runs(self, _body):
        return self.rows, None

    async def artifacts(self, _body):
        return [{"artifact_id": "a1"}], None

    async def artifact(self, _body):
        return {"mime": "image/png", "filename": "plot.png"}, self.blob
