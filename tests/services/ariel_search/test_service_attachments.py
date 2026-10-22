"""Tests for entry attachments on the facility write path.

An adapter that declares ``supports_attachments`` receives the entry's files in
``request.attachment_paths`` and publishes them inside ``create_entry``; one that
does not receives no paths, and the files stay in ARIEL. The adapter here is a
small in-memory logbook, so each test can say what the facility holds after the
write and what it hands back on the read-back.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from osprey.services.ariel_search.database.repository import SchemaFacts
from osprey.services.ariel_search.exceptions import AttachmentPublishError, IngestionError
from osprey.services.ariel_search.ingestion.base import FacilityAdapter
from osprey.services.ariel_search.models import (
    EntryAttachment,
    FacilityEntryCreateRequest,
    SyncStatus,
)
from osprey.services.ariel_search.service import ARIELSearchService
from tests.fixtures.ariel_entry_fields import (  # noqa: F401 - fixtures used by name
    DictRepository,
    dict_repository_fixture,
    example_config,
)

pytestmark = pytest.mark.asyncio

FACILITY = "Example Files Logbook"

PLOT = EntryAttachment(filename="orbit.png", mime_type="image/png", data=b"\x89PNG plot")
NOTE = EntryAttachment(filename="note.txt", mime_type="text/plain", data=b"a note")


class FileLogbook(FacilityAdapter):
    """A facility logbook in memory that can take files with an entry."""

    def __init__(self, *, takes_files: bool) -> None:
        super().__init__(example_config())
        self.takes_files = takes_files
        self.requests: list[FacilityEntryCreateRequest] = []
        self.received: list[dict[str, bytes]] = []
        self.fail_with: Exception | None = None
        self.readback: list[dict[str, Any]] = []

    @property
    def source_system_name(self) -> str:
        return FACILITY

    @property
    def supports_write(self) -> bool:
        return True

    @property
    def requires_write_auth(self) -> bool:
        return False

    @property
    def supports_attachments(self) -> bool:
        return self.takes_files

    async def fetch_entries(
        self,
        since: Any = None,  # noqa: ARG002 - facility adapter contract; the read-back list is fixed per test
        until: Any = None,  # noqa: ARG002 - facility adapter contract; the read-back list is fixed per test
        limit: int | None = None,  # noqa: ARG002 - facility adapter contract; the read-back list is fixed per test
    ) -> AsyncIterator[Any]:
        for entry in self.readback:
            yield entry

    async def create_entry(self, request: FacilityEntryCreateRequest) -> str:
        self.requests.append(request)
        # The files are read while the call runs: the paths need only live that long.
        self.received.append({Path(p).name: Path(p).read_bytes() for p in request.attachment_paths})
        if self.fail_with is not None:
            raise self.fail_with
        return f"fac-{len(self.requests)}"


@pytest.fixture(name="repository")
def repository_fixture(dict_repository: DictRepository) -> DictRepository:
    """The dict repository with plain native attachment storage recorded."""
    dict_repository.schema_facts = AsyncMock(return_value=SchemaFacts(False, False))
    dict_repository.store_attachment = AsyncMock()
    return dict_repository


def _service(
    repository: DictRepository, adapter: FileLogbook, monkeypatch: pytest.MonkeyPatch
) -> ARIELSearchService:
    monkeypatch.setattr(
        "osprey.services.ariel_search.ingestion.get_adapter", lambda _config: adapter
    )
    monkeypatch.setattr(
        "osprey.services.ariel_search.enhancement.qmd_export.mirror_entry_best_effort",
        lambda *_args: True,
    )
    return ARIELSearchService(config=example_config(), pool=MagicMock(), repository=repository)


def _facility_record(entry_id: str, attachments: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "entry_id": entry_id,
        "source_system": FACILITY,
        "timestamp": None,
        "author": "op",
        "raw_text": "Orbit\n\nDrift",
        "attachments": attachments,
        "metadata": {},
        "created_at": None,
        "updated_at": None,
    }


async def test_opting_adapter_receives_every_file_as_a_path(
    repository: DictRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each file reaches the adapter as a readable path under its own name."""
    adapter = FileLogbook(takes_files=True)
    service = _service(repository, adapter, monkeypatch)

    result = await service.create_entry(
        FacilityEntryCreateRequest(subject="Orbit", details="Drift"), attachments=[PLOT, NOTE]
    )

    assert adapter.received == [{"orbit.png": PLOT.data, "note.txt": NOTE.data}]
    assert result.attachments_published is True
    assert result.attachment_count == 2


async def test_staged_paths_are_gone_after_the_publish(
    repository: DictRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The staged copies live only while the adapter runs."""
    adapter = FileLogbook(takes_files=True)
    service = _service(repository, adapter, monkeypatch)

    await service.create_entry(
        FacilityEntryCreateRequest(subject="Orbit", details="Drift"), attachments=[PLOT]
    )

    (path,) = adapter.requests[0].attachment_paths
    assert not Path(path).exists()
    assert not Path(path).parent.exists()


async def test_same_named_files_each_reach_the_adapter(
    repository: DictRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two files sharing a name are staged apart, so neither overwrites the other."""
    adapter = FileLogbook(takes_files=True)
    service = _service(repository, adapter, monkeypatch)
    other = EntryAttachment(filename="orbit.png", mime_type="image/png", data=b"second")

    await service.create_entry(
        FacilityEntryCreateRequest(subject="Orbit", details="Drift"), attachments=[PLOT, other]
    )

    paths = adapter.requests[0].attachment_paths
    assert [Path(p).name for p in paths] == ["orbit.png", "orbit.png"]
    assert len(set(paths)) == 2


async def test_a_filename_with_directories_is_staged_by_its_base_name(
    repository: DictRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A submitted name never places a staged file outside the staging directory."""
    adapter = FileLogbook(takes_files=True)
    service = _service(repository, adapter, monkeypatch)
    sneaky = EntryAttachment(filename="../../etc/passwd", mime_type=None, data=b"x")

    await service.create_entry(
        FacilityEntryCreateRequest(subject="Orbit", details="Drift"), attachments=[sneaky]
    )

    assert adapter.received == [{"passwd": b"x"}]


async def test_non_opting_adapter_never_sees_paths_and_files_stay_in_ariel(
    repository: DictRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An adapter that does not declare support gets no paths; ARIEL keeps the files."""
    adapter = FileLogbook(takes_files=False)
    service = _service(repository, adapter, monkeypatch)

    result = await service.create_entry(
        FacilityEntryCreateRequest(subject="Orbit", details="Drift"), attachments=[PLOT]
    )

    assert adapter.requests[0].attachment_paths == []
    assert result.attachments_published is False
    assert result.attachment_count == 1
    stored = repository.entries["fac-1"]["attachments"]
    assert [item["filename"] for item in stored] == ["orbit.png"]
    assert "published" not in stored[0]


async def test_published_files_get_stand_ins_until_the_facility_record_arrives(
    repository: DictRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no read-back yet, ARIEL shows its own copies, marked as published."""
    adapter = FileLogbook(takes_files=True)
    service = _service(repository, adapter, monkeypatch)

    result = await service.create_entry(
        FacilityEntryCreateRequest(subject="Orbit", details="Drift"), attachments=[PLOT]
    )

    assert result.sync_status == SyncStatus.PENDING_SYNC
    (item,) = repository.entries["fac-1"]["attachments"]
    assert item["filename"] == "orbit.png"
    assert item["published"] is True
    repository.store_attachment.assert_awaited_once()


async def test_facility_read_back_wins_and_no_stand_in_is_written(
    repository: DictRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A read-back carrying the facility's files leaves ARIEL no copy of its own to keep."""
    adapter = FileLogbook(takes_files=True)
    facility_files = [{"url": "https://logbook.example/files/1", "filename": "orbit.png"}]
    adapter.readback = [_facility_record("fac-1", facility_files)]
    service = _service(repository, adapter, monkeypatch)

    result = await service.create_entry(
        FacilityEntryCreateRequest(subject="Orbit", details="Drift"), attachments=[PLOT]
    )

    assert result.sync_status == SyncStatus.SYNCED
    assert repository.entries["fac-1"]["attachments"] == facility_files
    repository.store_attachment.assert_not_awaited()
    assert result.attachment_count == 1


async def test_read_back_without_files_keeps_the_stand_ins(
    repository: DictRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A facility record that lists no files yet does not cost ARIEL its copies."""
    adapter = FileLogbook(takes_files=True)
    adapter.readback = [_facility_record("fac-1", [])]
    service = _service(repository, adapter, monkeypatch)

    await service.create_entry(
        FacilityEntryCreateRequest(subject="Orbit", details="Drift"), attachments=[PLOT]
    )

    (item,) = repository.entries["fac-1"]["attachments"]
    assert item["published"] is True


@pytest.mark.parametrize(
    "error",
    [
        IngestionError("upload of orbit.png refused", source_system=FACILITY),
        AttachmentPublishError("upload of orbit.png refused", source_system=FACILITY),
    ],
    ids=["ingestion-error", "attachment-publish-error"],
)
async def test_a_failed_file_publish_fails_the_whole_submit(
    error: IngestionError,
    repository: DictRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The error says the entry may exist without its files; ARIEL keeps nothing."""
    adapter = FileLogbook(takes_files=True)
    adapter.fail_with = error
    service = _service(repository, adapter, monkeypatch)

    with pytest.raises(AttachmentPublishError) as raised:
        await service.create_entry(
            FacilityEntryCreateRequest(subject="Orbit", details="Drift"), attachments=[PLOT]
        )

    text = str(raised.value)
    assert "upload of orbit.png refused" in text
    assert "may already exist" in text
    assert FACILITY in text
    assert "without all of its files" in text
    assert repository.entries == {}
    repository.store_attachment.assert_not_awaited()


async def test_a_failed_text_only_publish_is_not_reworded(
    repository: DictRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A publish that carried no files keeps its own error."""
    adapter = FileLogbook(takes_files=True)
    adapter.fail_with = IngestionError("logbook down", source_system=FACILITY)
    service = _service(repository, adapter, monkeypatch)

    with pytest.raises(IngestionError) as raised:
        await service.create_entry(FacilityEntryCreateRequest(subject="Orbit", details="Drift"))

    assert not isinstance(raised.value, AttachmentPublishError)


async def _seed_native_entry(repository: DictRepository) -> None:
    await repository.upsert_entry(
        {
            "entry_id": "ariel-abc",
            "source_system": "ARIEL Web",
            "timestamp": None,
            "author": "op",
            "raw_text": "Orbit\n\nDrift",
            "attachments": [
                {
                    "url": "/api/attachments/att-0123456789ab",
                    "type": "image/png",
                    "filename": "orbit.png",
                },
                {"url": "https://elsewhere.example/x.png", "filename": "x.png"},
            ],
            "metadata": {"tags": []},
        }
    )
    repository.get_attachment_original = AsyncMock(
        return_value={"filename": "orbit.png", "mime_type": "image/png", "data": PLOT.data}
    )


async def test_republish_carries_the_entrys_native_files(
    repository: DictRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Publishing an ARIEL entry to an opting logbook sends the files ARIEL holds for it."""
    adapter = FileLogbook(takes_files=True)
    service = _service(repository, adapter, monkeypatch)
    await _seed_native_entry(repository)

    result = await service.publish_entry("ariel-abc")

    assert adapter.received == [{"orbit.png": PLOT.data}]
    repository.get_attachment_original.assert_awaited_once_with("att-0123456789ab")
    assert result.attachments_published is True


async def test_republish_to_a_non_opting_logbook_sends_no_files(
    repository: DictRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A logbook that takes no files gets the text alone, as before."""
    adapter = FileLogbook(takes_files=False)
    service = _service(repository, adapter, monkeypatch)
    await _seed_native_entry(repository)

    result = await service.publish_entry("ariel-abc")

    assert adapter.requests[0].attachment_paths == []
    repository.get_attachment_original.assert_not_awaited()
    assert result.attachments_published is False
    assert result.attachment_count == 0


async def test_republish_with_a_missing_native_copy_publishes_nothing(
    repository: DictRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file ARIEL cannot read back stops the publish before the logbook is written."""
    adapter = FileLogbook(takes_files=True)
    service = _service(repository, adapter, monkeypatch)
    await _seed_native_entry(repository)
    repository.get_attachment_original = AsyncMock(return_value=None)

    with pytest.raises(IngestionError, match="orbit.png"):
        await service.publish_entry("ariel-abc")

    assert adapter.requests == []


def test_staging_fits_a_name_no_filesystem_accepts(tmp_path):
    """A NUL byte or an over-long name is cut to a valid one, keeping the extension."""
    from osprey.services.ariel_search.models import EntryAttachment
    from osprey.services.ariel_search.service import _stage_files

    long_name = "é" * 200 + ".png"
    paths = _stage_files(
        tmp_path,
        [
            EntryAttachment(filename="sc\0an.h5", mime_type=None, data=b"a"),
            EntryAttachment(filename=long_name, mime_type="image/png", data=b"b"),
        ],
    )

    first, second = (Path(p) for p in paths)
    assert first.name == "scan.h5" and first.read_bytes() == b"a"
    assert second.suffix == ".png" and len(second.name.encode()) <= 255
    assert second.read_bytes() == b"b"
