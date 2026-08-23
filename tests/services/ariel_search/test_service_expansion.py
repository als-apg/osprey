"""Service-level vocabulary expansion: resolution, dispatch and transparency.

Pins the contract the service owns (FR4): it alone reads ``expand_query``,
resolves it against ``expand_by_default`` and ``expand_modes``, parses the
query once through the descriptor's declared parser, and hands modules
``parsed=``/``query_expansion=`` strictly by what their descriptor opted into.
Modules never see ``expand_query``, and the caller's ``advanced_params`` dict is
never mutated.

Every module here is a fake registered through a patched registry, so these
tests are independent of the built-in descriptors' own migration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.exceptions import (
    PatternError,
    SearchTimeoutError,
    VocabularyError,
)
from osprey.services.ariel_search.models import DiagnosticLevel
from osprey.services.ariel_search.search.base import (
    ExpansionGroup,
    ModuleOutput,
    ParsedKeywordQuery,
    QueryExpansion,
    SearchToolDescriptor,
)
from osprey.services.ariel_search.search.keyword import parse_keyword_query
from osprey.services.ariel_search.service import ARIELSearchService

if TYPE_CHECKING:
    from pathlib import Path

VOCABULARY_YML = """
concepts:
  - canonical: troubleshoot
    kind: shorthand
    forms:
      - t/s
      - ts
  - canonical: beam position monitor
    kind: acronym
    forms:
      - bpm
"""

MALFORMED_VOCABULARY_YML = """
concepts:
  - canonical: troubleshoot
    kind: verb
    forms:
      - ts
"""

ENTRY_ROW = ({"entry_id": "E1", "raw_text": "troubleshoot the beam position monitor"}, 0.9)


class _FakeInput(BaseModel):
    """Minimal args schema; the service never validates against it."""

    query: str


class _Recorder:
    """An async module ``execute`` that records how the service called it."""

    def __init__(self, *, result: Any = None, raises: Exception | None = None) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.result = [ENTRY_ROW] if result is None else result
        self.raises = raises

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))
        if self.raises is not None:
            raise self.raises
        return self.result

    @property
    def kwargs(self) -> dict[str, Any]:
        """The keyword arguments of the single recorded call."""
        assert len(self.calls) == 1, f"expected exactly one call, got {len(self.calls)}"
        return self.calls[0][1]


def _descriptor(
    mode: str,
    execute: Any,
    *,
    accepts_expansion: bool = False,
    query_parser: Any = None,
) -> SearchToolDescriptor:
    """Build a fake module descriptor with the given opt-ins."""
    return SearchToolDescriptor(
        name=f"{mode}_search",
        description=f"fake {mode}",
        search_mode=mode,
        args_schema=_FakeInput,
        execute=execute,
        format_result=lambda *a, **k: {},
        accepts_expansion=accepts_expansion,
        query_parser=query_parser,
    )


def _registry(descriptors: dict[str, SearchToolDescriptor]) -> MagicMock:
    """A registry mock serving exactly the given fake search modules."""

    def _get(name: str) -> Any:
        descriptor = descriptors.get(name)
        if descriptor is None:
            return None
        module = MagicMock()
        module.get_tool_descriptor.return_value = descriptor
        return module

    registry = MagicMock()
    registry.list_ariel_search_modules.return_value = list(descriptors)
    registry.get_ariel_search_module.side_effect = _get
    return registry


def _service(
    tmp_path: Path,
    *,
    vocabulary: dict[str, Any] | None = None,
    yml: str | None = VOCABULARY_YML,
    modes: tuple[str, ...] = ("keyword",),
) -> ARIELSearchService:
    """Build a service whose config carries a real, loaded vocabulary file."""
    config_dict: dict[str, Any] = {
        "database": {"uri": "postgresql://localhost:5432/test"},
        "search_modules": {mode: {"enabled": True} for mode in modes},
    }
    if vocabulary is not None:
        block = dict(vocabulary)
        if yml is not None and "path" not in block:
            path = tmp_path / "vocabulary.yml"
            path.write_text(yml)
            block["path"] = str(path)
        config_dict["vocabulary"] = block

    config = ARIELConfig.from_dict(config_dict)
    pool = MagicMock()
    pool.close = AsyncMock()
    repository = MagicMock()
    repository.health_check = AsyncMock(return_value=(True, "OK"))
    repository.validate_search_model_table = AsyncMock()
    return ARIELSearchService(config=config, pool=pool, repository=repository)


def _enabled(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    """An enabled vocabulary block with optional overrides."""
    block: dict[str, Any] = {"enabled": True}
    block.update(overrides)
    return block


# --- expand_query never reaches a module ------------------------------------


class TestExpandQueryIsServiceOnly:
    """``expand_query`` is resolved by the service and stripped from kwargs."""

    @pytest.mark.asyncio
    async def test_expand_query_never_in_module_kwargs(self, tmp_path: Path) -> None:
        """A module never receives the ``expand_query`` control itself."""
        recorder = _Recorder()
        service = _service(tmp_path, vocabulary=_enabled(tmp_path))
        descriptors = {
            "keyword": _descriptor(
                "keyword",
                recorder,
                accepts_expansion=True,
                query_parser=parse_keyword_query,
            )
        }

        with patch("osprey.registry.get_registry", return_value=_registry(descriptors)):
            await service.search("ts bpm", mode="keyword", advanced_params={"expand_query": True})

        assert "expand_query" not in recorder.kwargs

    @pytest.mark.asyncio
    async def test_advanced_params_dict_is_not_mutated(self, tmp_path: Path) -> None:
        """The caller's own dict survives the search unchanged (identity too)."""
        recorder = _Recorder()
        service = _service(tmp_path, vocabulary=_enabled(tmp_path))
        descriptors = {
            "keyword": _descriptor(
                "keyword",
                recorder,
                accepts_expansion=True,
                query_parser=parse_keyword_query,
            )
        }
        advanced_params: dict[str, Any] = {"expand_query": False, "fuzzy_fallback": True}
        before = dict(advanced_params)

        with patch("osprey.registry.get_registry", return_value=_registry(descriptors)):
            await service.search("ts bpm", mode="keyword", advanced_params=advanced_params)

        assert advanced_params == before
        assert advanced_params["expand_query"] is False
        assert recorder.kwargs["fuzzy_fallback"] is True


# --- the descriptor decides the call shape ----------------------------------


class TestArgumentContract:
    """``parsed=`` and ``query_expansion=`` follow the descriptor, not the mode."""

    @pytest.mark.asyncio
    async def test_declaring_module_receives_parsed_and_expansion(self, tmp_path: Path) -> None:
        """A parser+expansion descriptor gets both, and the parser runs once."""
        recorder = _Recorder()
        parser = MagicMock(side_effect=parse_keyword_query)
        service = _service(tmp_path, vocabulary=_enabled(tmp_path))
        descriptors = {
            "keyword": _descriptor("keyword", recorder, accepts_expansion=True, query_parser=parser)
        }

        with patch("osprey.registry.get_registry", return_value=_registry(descriptors)):
            await service.search("ts bpm", mode="keyword")

        assert parser.call_count == 1
        assert parser.call_args.args == ("ts bpm",)
        kwargs = recorder.kwargs
        assert set(kwargs) == {
            "max_results",
            "start_date",
            "end_date",
            "parsed",
            "query_expansion",
        }
        assert isinstance(kwargs["parsed"], ParsedKeywordQuery)
        assert isinstance(kwargs["query_expansion"], QueryExpansion)

    @pytest.mark.asyncio
    async def test_default_descriptor_module_is_called_exactly_as_today(
        self, tmp_path: Path
    ) -> None:
        """Default flags: no new kwargs, entries still returned, no expansion."""
        recorder = _Recorder()
        service = _service(tmp_path, vocabulary=_enabled(tmp_path))
        descriptors = {"keyword": _descriptor("keyword", recorder)}

        with patch("osprey.registry.get_registry", return_value=_registry(descriptors)):
            result = await service.search(
                "ts bpm", mode="keyword", advanced_params={"expand_query": True}
            )

        assert set(recorder.kwargs) == {"max_results", "start_date", "end_date"}
        assert len(result.entries) == 1
        assert result.expanded_terms == ()

    @pytest.mark.asyncio
    async def test_parser_without_expansion_opt_in_gets_parsed_only(self, tmp_path: Path) -> None:
        """``parsed=`` is passed for a declared parser even with no expansion."""
        recorder = _Recorder()
        service = _service(tmp_path, vocabulary=_enabled(tmp_path))
        descriptors = {
            "keyword": _descriptor("keyword", recorder, query_parser=parse_keyword_query)
        }

        with patch("osprey.registry.get_registry", return_value=_registry(descriptors)):
            await service.search("ts bpm", mode="keyword")

        assert set(recorder.kwargs) == {"max_results", "start_date", "end_date", "parsed"}


# --- resolving whether expansion applies ------------------------------------


class TestExpansionResolution:
    """``expand_query``, ``expand_by_default`` and ``expand_modes``."""

    @staticmethod
    async def _run(
        service: ARIELSearchService,
        recorder: _Recorder,
        *,
        mode: str = "keyword",
        query: str = "ts bpm",
        advanced_params: dict[str, Any] | None = None,
    ) -> Any:
        descriptors = {
            mode: _descriptor(
                mode, recorder, accepts_expansion=True, query_parser=parse_keyword_query
            )
        }
        with patch("osprey.registry.get_registry", return_value=_registry(descriptors)):
            return await service.search(query, mode=mode, advanced_params=advanced_params)

    @pytest.mark.asyncio
    async def test_expand_query_false_suppresses_expansion(self, tmp_path: Path) -> None:
        """An explicit false wins over ``expand_by_default: true``."""
        recorder = _Recorder()
        service = _service(tmp_path, vocabulary=_enabled(tmp_path))

        result = await self._run(service, recorder, advanced_params={"expand_query": False})

        assert "query_expansion" not in recorder.kwargs
        assert result.expanded_terms == ()

    @pytest.mark.asyncio
    async def test_expand_by_default_true_with_flag_unset(self, tmp_path: Path) -> None:
        """Unset ``expand_query`` follows ``expand_by_default: true``."""
        recorder = _Recorder()
        service = _service(tmp_path, vocabulary=_enabled(tmp_path, expand_by_default=True))

        await self._run(service, recorder)

        assert isinstance(recorder.kwargs["query_expansion"], QueryExpansion)

    @pytest.mark.asyncio
    async def test_expand_by_default_false_with_flag_unset(self, tmp_path: Path) -> None:
        """Unset ``expand_query`` follows ``expand_by_default: false``."""
        recorder = _Recorder()
        service = _service(tmp_path, vocabulary=_enabled(tmp_path, expand_by_default=False))

        result = await self._run(service, recorder)

        assert "query_expansion" not in recorder.kwargs
        assert result.expanded_terms == ()

    @pytest.mark.asyncio
    async def test_expand_by_default_false_is_overridden_by_explicit_true(
        self, tmp_path: Path
    ) -> None:
        """An explicit true wins over ``expand_by_default: false``."""
        recorder = _Recorder()
        service = _service(tmp_path, vocabulary=_enabled(tmp_path, expand_by_default=False))

        await self._run(service, recorder, advanced_params={"expand_query": True})

        assert isinstance(recorder.kwargs["query_expansion"], QueryExpansion)

    @pytest.mark.asyncio
    async def test_disabled_vocabulary_never_expands(self, tmp_path: Path) -> None:
        """With no vocabulary block, ``expand_query: true`` is a no-op."""
        recorder = _Recorder()
        service = _service(tmp_path)

        await self._run(service, recorder, advanced_params={"expand_query": True})

        assert "query_expansion" not in recorder.kwargs

    @pytest.mark.asyncio
    async def test_expand_modes_excludes_a_mode(self, tmp_path: Path) -> None:
        """A mode outside ``expand_modes`` is dispatched unexpanded."""
        recorder = _Recorder()
        service = _service(
            tmp_path,
            vocabulary=_enabled(tmp_path, expand_modes=["keyword"]),
            modes=("keyword", "semantic"),
        )

        result = await self._run(service, recorder, mode="semantic")

        assert "query_expansion" not in recorder.kwargs
        assert result.expanded_terms == ()

    @pytest.mark.asyncio
    async def test_expand_modes_includes_the_named_mode(self, tmp_path: Path) -> None:
        """The mode named by ``expand_modes`` still expands."""
        recorder = _Recorder()
        service = _service(
            tmp_path,
            vocabulary=_enabled(tmp_path, expand_modes=["keyword"]),
            modes=("keyword", "semantic"),
        )

        await self._run(service, recorder, mode="keyword")

        assert isinstance(recorder.kwargs["query_expansion"], QueryExpansion)


# --- what text expansion matches over ---------------------------------------


class TestMatchingText:
    """Keyword matches over ``search_text`` + phrases; others over the query."""

    @staticmethod
    async def _expansion_for(
        service: ARIELSearchService,
        query: str,
        *,
        query_parser: Any,
        mode: str = "keyword",
    ) -> QueryExpansion:
        recorder = _Recorder()
        descriptors = {
            mode: _descriptor(mode, recorder, accepts_expansion=True, query_parser=query_parser)
        }
        with patch("osprey.registry.get_registry", return_value=_registry(descriptors)):
            await service.search(query, mode=mode)
        expansion = recorder.kwargs["query_expansion"]
        assert isinstance(expansion, QueryExpansion)
        return expansion

    @pytest.mark.asyncio
    async def test_field_filter_value_does_not_expand(self, tmp_path: Path) -> None:
        """``author:ts bpm`` expands only ``bpm`` -- the filter value is not text."""
        service = _service(tmp_path, vocabulary=_enabled(tmp_path))

        expansion = await self._expansion_for(
            service, "author:ts bpm", query_parser=parse_keyword_query
        )

        assert [group.original for group in expansion.groups] == ["bpm"]

    @pytest.mark.asyncio
    async def test_pattern_body_does_not_expand(self, tmp_path: Path) -> None:
        """A ``/regex/`` body contributes nothing to the matching text."""
        service = _service(tmp_path, vocabulary=_enabled(tmp_path))

        expansion = await self._expansion_for(
            service, "ts /SR01C___BPM[0-9]+/", query_parser=parse_keyword_query
        )

        assert [group.original for group in expansion.groups] == ["ts"]
        assert "sr01c" not in expansion.flattened_text.lower()

    @pytest.mark.asyncio
    async def test_parserless_module_expands_over_the_whole_query(self, tmp_path: Path) -> None:
        """With no parser declared, the whole query is the matching text."""
        service = _service(tmp_path, vocabulary=_enabled(tmp_path))

        parsed_expansion = await self._expansion_for(
            service, "ts /SR01C___BPM[0-9]+/", query_parser=parse_keyword_query
        )
        whole_expansion = await self._expansion_for(
            service, "ts /SR01C___BPM[0-9]+/", query_parser=None
        )

        assert len(parsed_expansion.groups) == len(whole_expansion.groups) == 1
        assert parsed_expansion.flattened_text != whole_expansion.flattened_text
        assert "sr01c" in whole_expansion.flattened_text.lower()

    @pytest.mark.asyncio
    async def test_quoted_phrase_is_part_of_the_matching_text(self, tmp_path: Path) -> None:
        """A quoted phrase still triggers a concept."""
        service = _service(tmp_path, vocabulary=_enabled(tmp_path))

        expansion = await self._expansion_for(
            service, '"beam position monitor" fault', query_parser=parse_keyword_query
        )

        assert [group.original for group in expansion.groups] == ["beam position monitor"]


# --- boolean-operator queries -----------------------------------------------


class TestBooleanOperatorQueries:
    """Boolean syntax cannot group alternatives, so expansion is skipped."""

    @pytest.mark.asyncio
    async def test_boolean_query_is_unexpanded_with_an_info_diagnostic(
        self, tmp_path: Path
    ) -> None:
        """``ts AND bpm`` runs unexpanded and says so."""
        recorder = _Recorder()
        service = _service(tmp_path, vocabulary=_enabled(tmp_path))
        descriptors = {
            "keyword": _descriptor(
                "keyword",
                recorder,
                accepts_expansion=True,
                query_parser=parse_keyword_query,
            )
        }

        with patch("osprey.registry.get_registry", return_value=_registry(descriptors)):
            result = await service.search("ts AND bpm", mode="keyword")

        assert "query_expansion" not in recorder.kwargs
        assert result.expanded_terms == ()
        skipped = [d for d in result.diagnostics if d.category == "expansion"]
        assert len(skipped) == 1
        assert skipped[0].level is DiagnosticLevel.INFO
        assert skipped[0].source == "service.keyword"
        assert skipped[0].message == "expansion skipped: boolean operators present"

    @pytest.mark.asyncio
    async def test_parserless_boolean_query_still_expands(self, tmp_path: Path) -> None:
        """The skip is keyword-shaped: it needs a parse to be decided."""
        recorder = _Recorder()
        service = _service(tmp_path, vocabulary=_enabled(tmp_path), modes=("keyword", "semantic"))
        descriptors = {"semantic": _descriptor("semantic", recorder, accepts_expansion=True)}

        with patch("osprey.registry.get_registry", return_value=_registry(descriptors)):
            await service.search("ts AND bpm", mode="semantic")

        assert isinstance(recorder.kwargs["query_expansion"], QueryExpansion)


# --- expanded_terms population ----------------------------------------------


class TestExpandedTermsPopulation:
    """``expanded_terms`` reports what ran, never what was merely offered."""

    @pytest.mark.asyncio
    async def test_module_output_expansion_is_reported(self, tmp_path: Path) -> None:
        """On success the groups come from the module's own ``ModuleOutput``."""
        applied = (ExpansionGroup(original="ts", alternatives=("troubleshoot",)),)
        recorder = _Recorder(result=ModuleOutput(entries=[ENTRY_ROW], expansion=applied))
        service = _service(tmp_path, vocabulary=_enabled(tmp_path))
        descriptors = {
            "keyword": _descriptor(
                "keyword",
                recorder,
                accepts_expansion=True,
                query_parser=parse_keyword_query,
            )
        }

        with patch("osprey.registry.get_registry", return_value=_registry(descriptors)):
            result = await service.search("ts bpm", mode="keyword")

        assert result.expanded_terms == ({"original": "ts", "alternatives": ["troubleshoot"]},)
        assert len(result.entries) == 1

    @pytest.mark.asyncio
    async def test_module_output_diagnostics_reach_the_result(self, tmp_path: Path) -> None:
        """A ``ModuleOutput``'s diagnostics are surfaced to the caller."""
        from osprey.services.ariel_search.models import SearchDiagnostic

        diagnostic = SearchDiagnostic(
            level=DiagnosticLevel.INFO,
            source="keyword",
            message="fuzzy fallback used",
            category="fallback",
        )
        recorder = _Recorder(result=ModuleOutput(entries=[ENTRY_ROW], diagnostics=(diagnostic,)))
        service = _service(tmp_path, vocabulary=_enabled(tmp_path))
        descriptors = {"keyword": _descriptor("keyword", recorder)}

        with patch("osprey.registry.get_registry", return_value=_registry(descriptors)):
            result = await service.search("ts bpm", mode="keyword")

        assert diagnostic in result.diagnostics

    @pytest.mark.asyncio
    async def test_bare_list_return_reports_no_expansion(self, tmp_path: Path) -> None:
        """A module that ignored the offered expansion reports empty groups."""
        recorder = _Recorder()
        service = _service(tmp_path, vocabulary=_enabled(tmp_path))
        descriptors = {
            "keyword": _descriptor(
                "keyword",
                recorder,
                accepts_expansion=True,
                query_parser=parse_keyword_query,
            )
        }

        with patch("osprey.registry.get_registry", return_value=_registry(descriptors)):
            result = await service.search("ts bpm", mode="keyword")

        assert isinstance(recorder.kwargs["query_expansion"], QueryExpansion)
        assert result.expanded_terms == ()


# --- failure paths ----------------------------------------------------------


class TestFailurePaths:
    """Timeouts and module errors still report what the statement contained."""

    @pytest.mark.asyncio
    async def test_module_error_result_carries_the_resolved_expansion(self, tmp_path: Path) -> None:
        """A failing module yields an error result with the resolved groups."""
        recorder = _Recorder(raises=RuntimeError("boom"))
        service = _service(tmp_path, vocabulary=_enabled(tmp_path))
        descriptors = {
            "keyword": _descriptor(
                "keyword",
                recorder,
                accepts_expansion=True,
                query_parser=parse_keyword_query,
            )
        }

        with patch("osprey.registry.get_registry", return_value=_registry(descriptors)):
            result = await service.search("ts bpm", mode="keyword")

        assert result.entries == ()
        assert [term["original"] for term in result.expanded_terms] == ["ts", "bpm"]
        assert result.diagnostics[0].level is DiagnosticLevel.ERROR

    @pytest.mark.asyncio
    async def test_timeout_propagates_and_carries_the_resolved_expansion(
        self, tmp_path: Path
    ) -> None:
        """``SearchTimeoutError`` reaches ``ainvoke``'s handler, not the generic one."""
        timeout = SearchTimeoutError("too slow", 10.0, "keyword search")
        recorder = _Recorder(raises=timeout)
        service = _service(tmp_path, vocabulary=_enabled(tmp_path))
        descriptors = {
            "keyword": _descriptor(
                "keyword",
                recorder,
                accepts_expansion=True,
                query_parser=parse_keyword_query,
            )
        }

        with patch("osprey.registry.get_registry", return_value=_registry(descriptors)):
            result = await service.search("ts bpm", mode="keyword")

        assert result.diagnostics[0].category == "timeout"
        assert result.diagnostics[0].source == "service.timeout"
        assert [term["original"] for term in result.expanded_terms] == ["ts", "bpm"]

    @pytest.mark.asyncio
    async def test_pattern_error_becomes_a_diagnostic_naming_the_pattern(
        self, tmp_path: Path
    ) -> None:
        """A rejected pattern is reported, not raised — and keeps the expansion.

        One bad token must not cost the caller everything the rest of the query
        meant: the result carries an ERROR diagnostic naming the offending
        pattern plus the expansion the rejected statement contained, which is
        what the MCP tools turn into an ``invalid_pattern`` error envelope.
        """
        recorder = _Recorder(raises=PatternError("invalid regex", pattern="SR0[1-4"))
        service = _service(tmp_path, vocabulary=_enabled(tmp_path))
        descriptors = {
            "keyword": _descriptor(
                "keyword",
                recorder,
                accepts_expansion=True,
                query_parser=parse_keyword_query,
            )
        }

        with patch("osprey.registry.get_registry", return_value=_registry(descriptors)):
            result = await service.search("t/s /SR0[1-4/", mode="keyword")

        assert result.entries == ()
        assert len(result.diagnostics) == 1
        diagnostic = result.diagnostics[0]
        assert diagnostic.level is DiagnosticLevel.ERROR
        assert diagnostic.category == "pattern"
        assert diagnostic.source == "service.keyword"
        assert "SR0[1-4" in diagnostic.message
        assert [term["original"] for term in result.expanded_terms] == ["t/s"]
        assert result.expanded_terms[0]["alternatives"] == ["troubleshoot"]

    @pytest.mark.asyncio
    async def test_broken_vocabulary_makes_search_raise(self, tmp_path: Path) -> None:
        """A stored vocabulary error kills the search path loudly."""
        recorder = _Recorder()
        service = _service(tmp_path, vocabulary=_enabled(tmp_path), yml=MALFORMED_VOCABULARY_YML)
        assert service.config.vocabulary_errors
        descriptors = {"keyword": _descriptor("keyword", recorder)}

        with (
            patch("osprey.registry.get_registry", return_value=_registry(descriptors)),
            pytest.raises(VocabularyError) as excinfo,
        ):
            await service.search("ts bpm", mode="keyword")

        assert excinfo.value.errors == service.config.vocabulary_errors
        assert recorder.calls == []

    @pytest.mark.asyncio
    async def test_disabled_broken_vocabulary_does_not_raise(self, tmp_path: Path) -> None:
        """Errors from a disabled block never reach the search path."""
        recorder = _Recorder()
        path = tmp_path / "vocabulary.yml"
        path.write_text(MALFORMED_VOCABULARY_YML)
        service = _service(
            tmp_path,
            vocabulary={"enabled": False, "path": str(path)},
            yml=None,
        )
        descriptors = {"keyword": _descriptor("keyword", recorder)}

        with patch("osprey.registry.get_registry", return_value=_registry(descriptors)):
            result = await service.search("ts bpm", mode="keyword")

        assert len(result.entries) == 1


# --- keyword-only truncation ------------------------------------------------


class TestTruncation:
    """The parser truncates; the service surfaces the warning it emitted."""

    @pytest.mark.asyncio
    async def test_form_past_the_cap_never_expands(self, tmp_path: Path) -> None:
        """A concept form beyond ``MAX_QUERY_LENGTH`` is cut before matching."""
        recorder = _Recorder()
        service = _service(tmp_path, vocabulary=_enabled(tmp_path))
        query = ("filler " * 200)[:1195] + " bpm"
        assert len(query) > 1000
        assert query.index("bpm") > 1000
        descriptors = {
            "keyword": _descriptor(
                "keyword",
                recorder,
                accepts_expansion=True,
                query_parser=parse_keyword_query,
            )
        }

        with patch("osprey.registry.get_registry", return_value=_registry(descriptors)):
            result = await service.search(query, mode="keyword")

        assert "query_expansion" not in recorder.kwargs
        assert result.expanded_terms == ()
        truncation = [d for d in result.diagnostics if d.category == "truncation"]
        assert len(truncation) == 1
        assert truncation[0].level is DiagnosticLevel.WARNING
