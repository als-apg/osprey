"""An example facility adapter that declares entry fields, for tests.

The shipped adapters keep the built-in entry form, so nothing in the source
tree exercises a facility that asks its authors for more. The service, the web
routes and the MCP tools all need the same such facility to test against, and
three suites each building their own would drift apart. It is spelled once,
here, and every consumer imports it explicitly rather than inheriting it from a
``conftest.py`` it might not sit under.

The example declares three fields:

* ``book`` — a required ``select`` with the choices ``ops`` and ``physics``;
* ``day`` — a ``date``;
* ``scan`` — a ``dynamic_select`` whose choices depend on ``day`` and are read
  from a per-test table.

Everything the adapter is told is recorded on its :class:`ExampleState`: each
``create_entry`` request and each options call. The state also holds the
toggles a test flips — whether the adapter writes, whether the options call
fails or hangs, and whether a ``logbook`` select is declared as well.

:class:`DictRepository` is the part of the ARIEL repository that writing and
publishing touch, kept in a dict, so an entry written by one service can be
read back by a later publish — through the same or another service instance.

The ``example_entry_fields`` fixture patches
``osprey.services.ariel_search.ingestion.get_adapter`` to return one shared
adapter for the test, the same injection point the existing create, publish and
route tests patch; ``dict_repository`` is an empty :class:`DictRepository`. A
test module enables them by importing ``example_entry_fields_fixture`` and
``dict_repository_fixture`` — the fixtures are registered under the short
names, so the import does not shadow the test parameters.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import pytest

from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.exceptions import IngestionError
from osprey.services.ariel_search.ingestion.base import FacilityAdapter
from osprey.services.ariel_search.models import FacilityEntryCreateRequest
from osprey.services.ariel_search.search.base import ParameterDescriptor

EXAMPLE_SOURCE_SYSTEM = "Example Logbook"
"""The example adapter's source system name — not ``Generic JSON``, so a write
through the service is a facility write that awaits re-ingestion."""

OPTIONS_OK = "ok"
OPTIONS_FAIL = "fail"
OPTIONS_HANG = "hang"
_OPTIONS_MODES = (OPTIONS_OK, OPTIONS_FAIL, OPTIONS_HANG)

BOOK_OPTIONS: list[dict[str, str]] = [
    {"value": "ops", "label": "Operations"},
    {"value": "physics", "label": "Physics"},
]
LOGBOOK_OPTIONS: list[dict[str, str]] = [
    {"value": "control-room", "label": "Control Room"},
    {"value": "maintenance", "label": "Maintenance"},
]


def example_config() -> ARIELConfig:
    """A minimal ARIEL config the example adapter and a service can be built from."""
    return ARIELConfig.from_dict(
        {
            "database": {"uri": "postgresql://test"},
            "ingestion": {"adapter": "generic_json", "source_url": "/tmp/test.json"},
        }
    )


@dataclass
class ExampleState:
    """What one test sets on the example adapter, and what the adapter records.

    Attributes:
        supports_write: Whether the adapter writes entries.
        options_mode: ``ok`` answers from ``options_table``; ``fail`` raises
            :class:`IngestionError`; ``hang`` never returns.
        declare_logbook: Whether a ``logbook`` select is declared after the
            three example fields.
        options_table: The ``scan`` choices, keyed by the ``day`` value the
            options call receives; a day absent from the table has no choices.
        created: Every request ``create_entry`` was given, in order.
        options_calls: Every options call as ``(name, values)``, in order.
    """

    supports_write: bool = True
    options_mode: str = OPTIONS_OK
    declare_logbook: bool = False
    options_table: dict[Any, list[dict[str, str]]] = field(default_factory=dict)
    created: list[FacilityEntryCreateRequest] = field(default_factory=list)
    options_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)


def example_descriptors(*, declare_logbook: bool = False) -> list[ParameterDescriptor]:
    """The example's entry fields, in form order, as a fresh list."""
    descriptors = [
        ParameterDescriptor(
            name="book",
            label="Book",
            description="Which book the entry is filed in",
            param_type="select",
            default="ops",
            options=[dict(option) for option in BOOK_OPTIONS],
            section="Entry",
            required=True,
        ),
        ParameterDescriptor(
            name="day",
            label="Day",
            description="The shift day the entry is about",
            param_type="date",
            default=None,
            section="Entry",
        ),
        ParameterDescriptor(
            name="scan",
            label="Scan",
            description="The scan taken on that day",
            param_type="dynamic_select",
            default=None,
            section="Entry",
            depends_on=("day",),
        ),
    ]
    if declare_logbook:
        descriptors.append(
            ParameterDescriptor(
                name="logbook",
                label="Logbook",
                description="The facility logbook the entry is written to",
                param_type="select",
                default="control-room",
                options=[dict(option) for option in LOGBOOK_OPTIONS],
                section="Entry",
            )
        )
    return descriptors


class ExampleEntryFieldsAdapter(FacilityAdapter):
    """A facility adapter declaring the example entry fields, driven by its state."""

    def __init__(self, config: ARIELConfig, state: ExampleState | None = None) -> None:
        super().__init__(config)
        self.state = state if state is not None else ExampleState()

    @property
    def source_system_name(self) -> str:
        return EXAMPLE_SOURCE_SYSTEM

    @property
    def supports_write(self) -> bool:
        return self.state.supports_write

    @property
    def requires_write_auth(self) -> bool:
        return False

    async def fetch_entries(
        self,
        since: Any = None,  # noqa: ARG002 - facility adapter contract; the example has no upstream to read
        until: Any = None,  # noqa: ARG002 - facility adapter contract; the example has no upstream to read
        limit: int | None = None,  # noqa: ARG002 - facility adapter contract; the example has no upstream to read
    ) -> AsyncIterator[Any]:
        return
        yield

    async def create_entry(self, request: FacilityEntryCreateRequest) -> str:
        if not self.state.supports_write:
            raise NotImplementedError(f"{EXAMPLE_SOURCE_SYSTEM} adapter does not write entries")
        self.state.created.append(request)
        return f"example-{len(self.state.created)}"

    def get_entry_field_descriptors(self) -> list[ParameterDescriptor]:
        return example_descriptors(declare_logbook=self.state.declare_logbook)

    async def get_entry_field_options(
        self, name: str, values: dict[str, Any]
    ) -> list[dict[str, str]]:
        self.state.options_calls.append((name, dict(values)))
        if self.state.options_mode == OPTIONS_FAIL:
            raise IngestionError("example options lookup failed", EXAMPLE_SOURCE_SYSTEM)
        if self.state.options_mode == OPTIONS_HANG:
            await asyncio.Event().wait()
        if self.state.options_mode not in _OPTIONS_MODES:
            raise ValueError(f"unknown options_mode {self.state.options_mode!r}")
        if name != "scan":
            return []
        return [dict(option) for option in self.state.options_table.get(values.get("day"), [])]


class DictRepository:
    """The write-and-read-back half of the ARIEL repository, kept in a dict."""

    def __init__(self) -> None:
        self.entries: dict[str, dict[str, Any]] = {}

    async def upsert_entry(self, entry: dict[str, Any]) -> None:
        self.entries[entry["entry_id"]] = dict(entry)

    async def get_entry(self, entry_id: str) -> dict[str, Any] | None:
        entry = self.entries.get(entry_id)
        return dict(entry) if entry is not None else None


@pytest.fixture(name="example_entry_fields")
def example_entry_fields_fixture(monkeypatch: pytest.MonkeyPatch) -> ExampleEntryFieldsAdapter:
    """One example adapter for this test, returned by every ``get_adapter`` call."""
    adapter = ExampleEntryFieldsAdapter(example_config())
    monkeypatch.setattr(
        "osprey.services.ariel_search.ingestion.get_adapter",
        lambda _config: adapter,
    )
    return adapter


@pytest.fixture(name="dict_repository")
def dict_repository_fixture() -> DictRepository:
    """An empty dict-backed repository for this test."""
    return DictRepository()
