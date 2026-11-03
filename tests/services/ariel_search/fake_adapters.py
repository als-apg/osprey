"""Facility adapters with no logbook behind them, for tests that ingest entries they build."""

from __future__ import annotations

from collections.abc import AsyncIterator

from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.ingestion.base import FacilityAdapter

#: The one origin ``HttpStubAdapter``'s pictures are served from.
ORIGINS = frozenset({("https", "h.example", 443)})


class StubAdapter(FacilityAdapter):
    """A source whose entries the tests hand to ``ingest_one`` directly; it never yields."""

    @property
    def source_system_name(self) -> str:
        return "Stub"

    async def fetch_entries(self, *_args, **_kwargs) -> AsyncIterator:  # type: ignore[override]
        for entry in ():  # never yields
            yield entry


class HttpStubAdapter(StubAdapter):
    """An http source whose pictures live on ``h.example``; it yields ``entries``."""

    def __init__(self, config: ARIELConfig, entries: list[dict] | None = None) -> None:
        super().__init__(config)
        self.entries = entries or []

    def attachment_origins(self) -> frozenset:
        return ORIGINS

    async def fetch_entries(self, *_args, **_kwargs) -> AsyncIterator:  # type: ignore[override]
        for entry in list(self.entries):
            yield dict(entry)
