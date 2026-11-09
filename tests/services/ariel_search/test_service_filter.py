"""Service-level dimension filter and offset plumbing (C-MODOUT).

The service hands ``entry_filter=`` and ``offset=`` only to modules whose
descriptor opted in, widens ``max_results`` for a module that does not page
itself, and post-filters every module's results with ``EntryFilter.matches``
whatever the module did with the filter.

Every module here is a fake registered through a patched registry.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.filters import BUILTIN_DIMENSIONS, EntryFilter
from osprey.services.ariel_search.models import ARIELSearchRequest
from osprey.services.ariel_search.search.base import ModuleOutput, SearchToolDescriptor
from osprey.services.ariel_search.service import ARIELSearchService
from tests.services.ariel_search.repo_fakes import attach_fake_fts

ALICE = {"entry_id": "E1", "author": "Alice", "source_system": "als", "raw_text": "a"}
BOB = {"entry_id": "E2", "author": "Bob", "source_system": "als", "raw_text": "b"}
CAROL = {"entry_id": "E3", "author": "Carol", "source_system": "als", "raw_text": "c"}


class _FakeInput(BaseModel):
    query: str


class _Recorder:
    """An async module ``execute`` that records its kwargs and returns fixed rows."""

    def __init__(self, result: Any) -> None:
        self.calls: list[dict[str, Any]] = []
        self.result = result

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.result

    @property
    def kwargs(self) -> dict[str, Any]:
        assert len(self.calls) == 1
        return self.calls[0]


def _descriptor(execute: Any, *, entry_filter: bool = False, offset: bool = False) -> Any:
    return SearchToolDescriptor(
        name="third_search",
        description="a third-party module",
        search_mode="third",
        args_schema=_FakeInput,
        execute=execute,
        format_result=lambda *a, **k: {},
        accepts_entry_filter=entry_filter,
        accepts_offset=offset,
    )


def _registry(
    descriptor: SearchToolDescriptor, *, enhancement_modules: tuple[str, ...] = ()
) -> MagicMock:
    def _get(name: str) -> Any:
        if name != descriptor.search_mode:
            return None
        module = MagicMock()
        module.get_tool_descriptor.return_value = descriptor
        return module

    registry = MagicMock()
    registry.list_ariel_search_modules.return_value = [descriptor.search_mode]
    registry.get_ariel_search_module.side_effect = _get
    registry.list_ariel_enhancement_modules.return_value = list(enhancement_modules)
    return registry


def _service(*, has_copy_state: bool = False, **extra: Any) -> ARIELSearchService:
    config = ARIELConfig.from_dict(
        {
            "database": {"uri": "postgresql://localhost:5432/test"},
            "search_modules": {"third": {"enabled": True}},
            **extra,
        }
    )
    pool = MagicMock()
    pool.close = AsyncMock()
    repository = MagicMock()
    repository.health_check = AsyncMock(return_value=(True, "OK"))
    repository.validate_search_model_table = AsyncMock()
    attach_fake_fts(repository, has_v2=False, has_copy_state=has_copy_state)
    return ARIELSearchService(config=config, pool=pool, repository=repository)


def _alice_only() -> EntryFilter:
    return EntryFilter.parse({"author": ["alice"]}, BUILTIN_DIMENSIONS)


async def _search(descriptor: SearchToolDescriptor, **kwargs: Any) -> Any:
    service = _service()
    with patch("osprey.registry.get_registry", return_value=_registry(descriptor)):
        return await service.search("beam", mode="third", **kwargs)


class TestRequest:
    def test_request_defaults(self) -> None:
        request = ARIELSearchRequest(query="q")
        assert request.entry_filter is None
        assert request.offset == 0

    def test_negative_offset_is_clamped(self) -> None:
        assert ARIELSearchRequest(query="q", offset=-5).offset == 0


class TestPostFilter:
    @pytest.mark.asyncio
    async def test_module_that_ignores_the_filter_is_still_filtered(self) -> None:
        recorder = _Recorder([(ALICE, 0.9), (BOB, 0.8), (CAROL, 0.7)])

        result = await _search(_descriptor(recorder), entry_filter=_alice_only())

        assert [e["entry_id"] for e in result.entries] == ["E1"]
        assert "entry_filter" not in recorder.kwargs

    @pytest.mark.asyncio
    async def test_module_output_rows_are_post_filtered_too(self) -> None:
        recorder = _Recorder(ModuleOutput(entries=[(BOB, 0.9), (ALICE, 0.5)]))

        result = await _search(_descriptor(recorder), entry_filter=_alice_only())

        assert [e["entry_id"] for e in result.entries] == ["E1"]

    @pytest.mark.asyncio
    async def test_no_filter_keeps_every_row(self) -> None:
        recorder = _Recorder([(ALICE, 0.9), (BOB, 0.8)])

        result = await _search(_descriptor(recorder))

        assert [e["entry_id"] for e in result.entries] == ["E1", "E2"]

    @pytest.mark.asyncio
    async def test_image_caption_stale_post_filter_keeps_rows_with_pictures(self) -> None:
        """A stale-caption row a module selected survives the post-filter.

        The module selected ``has_images`` (as the repository reads do), so
        ``matches`` sees the same picture flag as ``to_sql`` and removes
        nothing; a row without that flag is not stale (n/a) and is dropped.
        """
        from tests.services.ariel_search.test_filters import FILTER_DIMS

        service = _service(
            has_copy_state=True,
            enhancement_modules={
                "image_caption": {"enabled": True, "model": {"model_id": "m2"}},
            },
        )
        status = {"image_caption": {"status": "complete", "marker": "m1"}}
        stale = {**ALICE, "has_images": True, "enhancement_status": status}
        no_pictures = {**BOB, "has_images": False, "enhancement_status": status}
        recorder = _Recorder(
            ModuleOutput(entries=[(stale, 1.0), (no_pictures, 0.5)], filter_applied=True)
        )
        flt = EntryFilter.parse({"enhancement": ["image_caption:stale"]}, FILTER_DIMS)
        descriptor = _descriptor(recorder, entry_filter=True)
        registry = _registry(descriptor, enhancement_modules=("image_caption",))
        with patch("osprey.registry.get_registry", return_value=registry):
            result = await service.search("beam", mode="third", entry_filter=flt)

        assert [e["entry_id"] for e in result.entries] == ["E1"]


class TestDescriptorGating:
    @pytest.mark.asyncio
    async def test_unflagged_kwargs_module_gets_no_offset_and_a_widened_page(self) -> None:
        recorder = _Recorder([(ALICE, 0.9)])

        await _search(_descriptor(recorder), entry_filter=_alice_only(), offset=30, max_results=20)

        assert "offset" not in recorder.kwargs
        assert "entry_filter" not in recorder.kwargs
        assert recorder.kwargs["max_results"] == 30 + 20 * 2

    @pytest.mark.asyncio
    async def test_widened_page_is_capped_at_200(self) -> None:
        recorder = _Recorder([])

        await _search(_descriptor(recorder), offset=190, max_results=50)

        assert recorder.kwargs["max_results"] == 200

    @pytest.mark.asyncio
    async def test_unflagged_module_is_sliced_at_the_offset(self) -> None:
        rows = [({**ALICE, "entry_id": f"E{i}"}, 1.0 - i / 100) for i in range(10)]
        recorder = _Recorder(rows)

        result = await _search(_descriptor(recorder), offset=3, max_results=4)

        assert [e["entry_id"] for e in result.entries] == ["E3", "E4", "E5", "E6"]

    @pytest.mark.asyncio
    async def test_flagged_module_gets_filter_offset_and_the_page_size(self) -> None:
        flt = _alice_only()
        recorder = _Recorder(ModuleOutput(entries=[(ALICE, 0.9)], filter_applied=True))

        result = await _search(
            _descriptor(recorder, entry_filter=True, offset=True),
            entry_filter=flt,
            offset=20,
            max_results=10,
        )

        assert recorder.kwargs["entry_filter"] is flt
        assert recorder.kwargs["offset"] == 20
        assert recorder.kwargs["max_results"] == 10
        # A self-paging module's page is not sliced again by the service.
        assert [e["entry_id"] for e in result.entries] == ["E1"]
