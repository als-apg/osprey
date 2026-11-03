"""Tests for the filter dimensions of an ARIEL configuration (C-DIM).

A dimension is a facet of the browse rail and a key of the filter object. The
hazards pinned here: a declared field becomes a dimension only when it says
``facet: true``; the built-ins and ``enhancement`` are always there, in a fixed
order; ``time`` is never a dimension; and a misdeclared adapter degrades to the
built-ins instead of failing every browse and search.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest

from osprey.services.ariel_search import filters as filters_module
from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.enhancement.status import EnhancementContext
from osprey.services.ariel_search.entry_fields import (
    RESERVED_DIMENSION_NAMES,
    EntryFieldDeclarationError,
    check_declarations,
)
from osprey.services.ariel_search.filters import (
    BUILTIN_DIMENSIONS,
    EntryFilter,
    FilterDimension,
    FilterValidationError,
    filter_dimensions,
)
from osprey.services.ariel_search.search.base import ParameterDescriptor
from tests.fixtures.ariel_entry_fields import (  # noqa: F401 - fixtures used by name
    example_config,
    example_descriptors,
    example_entry_fields_fixture,
)

BUILTIN_NAMES = ["author", "source_system", "tags"]


def _field(name: str, param_type: str = "text", **kw) -> ParameterDescriptor:
    return ParameterDescriptor(
        name=name,
        label=kw.pop("label", name.title()),
        description="",
        param_type=param_type,
        default=None,
        **kw,
    )


def _declare(monkeypatch, adapter, descriptors) -> None:
    monkeypatch.setattr(adapter, "get_entry_field_descriptors", lambda: list(descriptors))


def _with_logbook_facet() -> list[ParameterDescriptor]:
    descriptors = example_descriptors(declare_logbook=True)
    return [dataclasses.replace(d, facet=True) if d.name == "logbook" else d for d in descriptors]


def _names(dims: list[FilterDimension]) -> list[str]:
    return [d.name for d in dims]


@pytest.fixture(autouse=True)
def _reset_warned(monkeypatch):
    monkeypatch.setattr(filters_module, "_declaration_fallback_warned", False)


class TestDeclaredDimensions:
    def test_a_facet_field_becomes_a_declared_dimension(self, example_entry_fields, monkeypatch):
        """``logbook`` declared with facet: true is a dimension keyed metadata->>'logbook'."""
        _declare(monkeypatch, example_entry_fields, _with_logbook_facet())

        dims = filter_dimensions(example_config())

        logbook = next(d for d in dims if d.name == "logbook")
        assert logbook == FilterDimension(
            name="logbook",
            label="Logbook",
            kind="declared",
            column_sql="metadata->>'logbook'",
            multi=False,
        )

    def test_a_field_without_the_flag_is_not_a_dimension(self, example_entry_fields, monkeypatch):
        """The same field declared without facet is no dimension; neither are the others."""
        _declare(monkeypatch, example_entry_fields, example_descriptors(declare_logbook=True))

        assert _names(filter_dimensions(example_config())) == [*BUILTIN_NAMES, "enhancement"]

    def test_order_is_builtins_then_declared_in_declaration_order_then_enhancement(
        self, example_entry_fields, monkeypatch
    ):
        _declare(
            monkeypatch,
            example_entry_fields,
            [
                _field("shift", facet=True),
                _field("note"),
                _field("logbook", "select", options=[{"value": "a", "label": "A"}], facet=True),
            ],
        )

        assert _names(filter_dimensions(example_config())) == [
            *BUILTIN_NAMES,
            "shift",
            "logbook",
            "enhancement",
        ]


class TestAlwaysPresent:
    def test_builtins_are_present_without_any_adapter(self):
        """A config with no ingestion adapter still has the builtins and enhancement."""
        config = ARIELConfig.from_dict({"database": {"uri": "postgresql://test"}})

        dims = filter_dimensions(config)

        assert dims[:3] == list(BUILTIN_DIMENSIONS)
        assert _names(dims) == [*BUILTIN_NAMES, "enhancement"]

    def test_builtin_shapes(self):
        by_name = {d.name: d for d in BUILTIN_DIMENSIONS}
        assert by_name["author"].column_sql == "author"
        assert by_name["source_system"].column_sql == "source_system"
        assert by_name["tags"].multi is True
        assert not by_name["author"].multi and not by_name["source_system"].multi
        assert all(d.kind == "builtin" and d.values == () for d in BUILTIN_DIMENSIONS)

    def test_time_never_appears(self, example_entry_fields, monkeypatch):
        """``time`` is a window: not a builtin, and a facet named time is refused."""
        _declare(monkeypatch, example_entry_fields, [_field("time", facet=True)])

        assert "time" not in _names(filter_dimensions(example_config()))
        assert "time" not in _names(
            filter_dimensions(ARIELConfig.from_dict({"database": {"uri": "postgresql://test"}}))
        )


class TestEnhancementDimension:
    def test_values_are_the_enabled_modules_only_in_registry_order(self):
        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://test"},
                "enhancement_modules": {
                    "text_embedding": {"enabled": True},
                    "image_caption": {"enabled": True},
                    "semantic_processor": {"enabled": False},
                },
            }
        )

        enhancement = filter_dimensions(config)[-1]

        assert enhancement.name == "enhancement"
        assert enhancement.kind == "enhancement"
        assert enhancement.values == ("image_caption", "text_embedding")

    def test_no_enabled_modules_gives_empty_values(self):
        config = ARIELConfig.from_dict({"database": {"uri": "postgresql://test"}})

        enhancement = filter_dimensions(config)[-1]

        assert enhancement.name == "enhancement"
        assert enhancement.values == ()


class TestMisdeclaredAdapterFallback:
    def test_falls_back_to_builtins_and_enhancement_and_logs_once(
        self, example_entry_fields, monkeypatch, caplog
    ):
        _declare(monkeypatch, example_entry_fields, [_field("x", facet="yes")])

        with caplog.at_level(logging.WARNING, logger=filters_module.__name__):
            first = filter_dimensions(example_config())
            second = filter_dimensions(example_config())

        assert _names(first) == [*BUILTIN_NAMES, "enhancement"]
        assert _names(second) == _names(first)
        warnings = [r for r in caplog.records if r.name == filters_module.__name__]
        assert len(warnings) == 1
        assert "'x'" in warnings[0].getMessage()


class TestFacetDeclarationChecks:
    @pytest.mark.parametrize(
        "value",
        [
            pytest.param("true", id="string"),
            pytest.param(1, id="int"),
            pytest.param(None, id="none"),
        ],
    )
    def test_a_non_bool_flag_is_a_declaration_error(self, value):
        with pytest.raises(EntryFieldDeclarationError) as info:
            check_declarations([_field("logbook", facet=value)], adapter="Example")

        assert info.value.field == "logbook"
        assert info.value.adapter == "Example"
        assert "facet" in info.value.message

    @pytest.mark.parametrize("name", sorted(RESERVED_DIMENSION_NAMES))
    def test_a_facet_may_not_take_a_reserved_dimension_name(self, name):
        with pytest.raises(EntryFieldDeclarationError) as info:
            check_declarations([_field(name, facet=True)], adapter="Example")

        assert info.value.field == name

    @pytest.mark.parametrize("name", ["author", "source_system", "time", "enhancement"])
    def test_a_non_facet_field_may_keep_such_a_name(self, name):
        """The collision rule bites only facets; ``tags`` stays reserved on its own."""
        assert check_declarations([_field(name)], adapter="Example")[0].name == name

    def test_reserved_dimension_names(self):
        assert RESERVED_DIMENSION_NAMES == {
            "author",
            "source_system",
            "tags",
            "time",
            "enhancement",
        }


class TestDescriptorFacetSerialization:
    def test_facet_is_emitted_only_when_true(self):
        assert _field("logbook", facet=True).to_dict()["facet"] is True
        assert "facet" not in _field("logbook").to_dict()

    def test_facet_defaults_false(self):
        assert _field("logbook").facet is False


# ---------------------------------------------------------------------------
# C-FILTER: the filter object (EntryFilter)
# ---------------------------------------------------------------------------

#: The dimension table the EntryFilter tests parse against: the built-ins, one
#: declared facet, and ``enhancement`` with one text and one image module.
FILTER_DIMS: tuple[FilterDimension, ...] = (
    *BUILTIN_DIMENSIONS,
    FilterDimension("logbook", "Logbook", "declared", "metadata->>'logbook'", multi=False),
    FilterDimension(
        "enhancement",
        "Enhancements",
        "enhancement",
        "enhancement_status",
        multi=False,
        values=("text_embedding", "image_caption"),
    ),
)

#: The context the shared case table is evaluated under. ``image_caption`` is
#: currently written under marker ``m2``.
FILTER_CTX = EnhancementContext(
    markers={"image_caption": "m2"}, enabled=("text_embedding", "image_caption")
)

_T0 = datetime(2025, 10, 6, 12, 0, tzinfo=UTC)


def _entry(**kw) -> dict:
    """One entry as the repository returns it, with overridable keys."""
    entry = {
        "entry_id": "e1",
        "author": "J. Smith",
        "source_system": "ALS eLog",
        "timestamp": _T0,
        "metadata": {"tags": ["RF", "Vacuum"], "logbook": "Operations"},
        "enhancement_status": {},
        "has_images": False,
    }
    metadata = kw.pop("metadata", None)
    if metadata is not None:
        entry["metadata"] = metadata
    entry.update(kw)
    return entry


#: The shared case table: (raw filter, entry, expected match). Task 1.4 asserts
#: the SQL form agrees with :meth:`EntryFilter.matches` on every row.
MATCH_CASES = [
    pytest.param({}, _entry(), True, id="empty-filter-matches-all"),
    pytest.param({"author": ["J. Smith"]}, _entry(), True, id="author-exact"),
    pytest.param({"author": ["j. smith"]}, _entry(), True, id="author-ignores-case"),
    pytest.param(
        {"author": ["J. Smith"]}, _entry(author="J. Smithson"), False, id="author-no-substring"
    ),
    pytest.param({"author": ["Smith"]}, _entry(), False, id="author-no-partial"),
    pytest.param({"author": ["A. Other", "J. Smith"]}, _entry(), True, id="or-within-dimension"),
    pytest.param(
        {"author": ["J. Smith"], "source_system": ["Other"]},
        _entry(),
        False,
        id="and-across-dimensions",
    ),
    pytest.param(
        {"author": ["J. Smith"], "source_system": ["als elog"]},
        _entry(),
        True,
        id="and-across-both-true",
    ),
    pytest.param({"author": ["J. Smith"]}, _entry(author=None), False, id="author-null"),
    pytest.param({"tags": ["rf"]}, _entry(), True, id="tags-membership-ignores-case"),
    pytest.param(
        {"tags": ["Rf"]},
        _entry(metadata={"tags": ["rF", "Vacuum"]}),
        True,
        id="tags-mixed-case-on-both-sides",
    ),
    pytest.param({"tags": ["R"]}, _entry(), False, id="tags-no-partial"),
    pytest.param({"tags": ["Magnets", "vacuum"]}, _entry(), True, id="tags-or"),
    pytest.param(
        {"tags": ["RF"]}, _entry(metadata={"tags": "RF"}), False, id="tags-non-list-never-matches"
    ),
    pytest.param({"tags": ["RF"]}, _entry(metadata={}), False, id="tags-absent"),
    pytest.param({"logbook": ["operations"]}, _entry(), True, id="declared-from-metadata"),
    pytest.param(
        {"logbook": ["Operations"]},
        _entry(logbook="Operations", metadata={}),
        False,
        id="declared-not-from-top-level",
    ),
    pytest.param({"logbook": ["Ops"]}, _entry(), False, id="declared-no-partial"),
    pytest.param(
        {"enhancement": ["text_embedding:failed"]},
        _entry(enhancement_status={"text_embedding": {"status": "failed"}}),
        True,
        id="enhancement-failed",
    ),
    pytest.param(
        {"enhancement": ["text_embedding:failed"]},
        _entry(enhancement_status={"text_embedding": {"status": "complete"}}),
        False,
        id="enhancement-complete-is-not-failed",
    ),
    pytest.param(
        {"enhancement": ["text_embedding:failed"]},
        _entry(enhancement_status={}),
        False,
        id="enhancement-pending-is-not-failed",
    ),
    pytest.param(
        {"enhancement": ["image_caption:stale"]},
        _entry(
            has_images=True,
            enhancement_status={"image_caption": {"status": "complete", "marker": "m1"}},
        ),
        True,
        id="enhancement-stale-marker",
    ),
    pytest.param(
        {"enhancement": ["image_caption:stale"]},
        _entry(
            has_images=False,
            enhancement_status={"image_caption": {"status": "complete", "marker": "m1"}},
        ),
        False,
        id="enhancement-image-module-without-pictures-is-na",
    ),
    pytest.param(
        {"enhancement": ["image_caption:stale"]},
        _entry(
            has_images=False,
            enhancement_status={"image_caption": {"status": "failed", "marker": "m0"}},
        ),
        False,
        id="enhancement-no-images-old-marker-is-na-not-stale",
    ),
    pytest.param(
        {"enhancement": ["image_caption:failed"]},
        _entry(
            has_images=False,
            enhancement_status={"image_caption": {"status": "failed", "marker": "m2"}},
        ),
        False,
        id="enhancement-no-images-failed-is-na-not-failed",
    ),
    pytest.param(
        {"enhancement": ["image_caption:failed"]},
        _entry(
            has_images=True,
            enhancement_status={"image_caption": {"status": "failed", "marker": "m2"}},
        ),
        True,
        id="enhancement-image-failed-under-current-marker",
    ),
    pytest.param(
        {"enhancement": ["image_caption:failed"]},
        _entry(
            has_images=True,
            enhancement_status={"image_caption": {"status": "failed", "marker": "m1"}},
        ),
        False,
        id="enhancement-image-failed-under-old-marker-is-stale",
    ),
    pytest.param(
        {"enhancement": ["image_caption:stale", "text_embedding:failed"]},
        _entry(enhancement_status={"text_embedding": {"status": "failed"}}),
        True,
        id="enhancement-or",
    ),
    pytest.param(
        {"enhancement": ["text_embedding:failed"]},
        _entry(enhancement_status='{"text_embedding": {"status": "failed"}}'),
        True,
        id="enhancement-status-as-json-text",
    ),
    pytest.param(
        {"time": {"start": "2025-10-06T00:00:00+00:00", "end": None}},
        _entry(),
        True,
        id="time-after-start",
    ),
    pytest.param(
        {"time": {"start": "2025-10-07T00:00:00+00:00"}}, _entry(), False, id="time-before-start"
    ),
    pytest.param(
        {"time": {"end": "2025-10-06T12:00:00+00:00"}}, _entry(), True, id="time-end-inclusive"
    ),
    pytest.param(
        {"time": {"end": "2025-10-06T11:59:59+00:00"}}, _entry(), False, id="time-after-end"
    ),
    pytest.param(
        {"time": {"start": "2025-10-01T00:00:00+00:00"}},
        _entry(timestamp=None),
        False,
        id="time-bound-and-no-timestamp",
    ),
    pytest.param(
        {"time": {"start": "2025-10-01T00:00:00+00:00"}, "author": ["nobody"]},
        _entry(),
        False,
        id="time-and-dimension",
    ),
]


class TestEntryFilterMatches:
    @pytest.mark.parametrize(("raw", "entry", "expected"), MATCH_CASES)
    def test_matches(self, raw, entry, expected):
        assert EntryFilter.parse(raw, FILTER_DIMS).matches(entry, FILTER_CTX) is expected

    def test_a_disabled_module_never_matches(self):
        """An entry is filtered against the context's enabled modules only."""
        flt = EntryFilter.parse({"enhancement": ["text_embedding:failed"]}, FILTER_DIMS)
        entry = _entry(enhancement_status={"text_embedding": {"status": "failed"}})

        assert flt.matches(entry, EnhancementContext(enabled=("image_caption",))) is False


class TestEntryFilterToSql:
    """The SQL form of the filter: its shape, on a recording pool.

    Parity with :meth:`EntryFilter.matches` on real Postgres is pinned by
    ``integration/test_entry_filter_sql.py`` over :data:`MATCH_CASES`.
    """

    @staticmethod
    def _sql(raw, ctx=FILTER_CTX, alias="e"):
        return EntryFilter.parse(raw, FILTER_DIMS).to_sql(alias, ctx)

    @staticmethod
    async def _run(fake_pool, clauses, params):
        where = " AND ".join(clauses) or "TRUE"
        async with fake_pool.connection() as conn:
            await conn.execute(f"SELECT e.entry_id FROM enhanced_entries e WHERE {where}", params)
        return fake_pool.calls[-1]

    @pytest.mark.parametrize("raw", [pytest.param(p.values[0], id=p.id) for p in MATCH_CASES])
    def test_placeholders_equal_params_on_every_case(self, raw):
        clauses, params = self._sql(raw)
        sql = " AND ".join(clauses)
        assert sql.count("%s") == len(params)
        assert "%(" not in sql

    def test_empty_filter_is_no_clause(self):
        assert self._sql({}) == ([], [])

    async def test_scalar_dimension_is_lower_any_of_lowered_values(self, fake_pool):
        clauses, params = self._sql({"author": ["J. Smith", "A. Other"]})
        sql, sent = await self._run(fake_pool, clauses, params)

        assert clauses == ["lower(e.author) = ANY(%s)"]
        assert sent == [["j. smith", "a. other"]]
        assert "WHERE lower(e.author) = ANY(%s)" in sql

    def test_one_clause_per_dimension_in_table_order(self):
        clauses, params = self._sql({"source_system": ["ALS"], "author": ["X"]})

        assert clauses == ["lower(e.author) = ANY(%s)", "lower(e.source_system) = ANY(%s)"]
        assert params == [["x"], ["als"]]

    def test_declared_dimension_reads_metadata_under_alias(self):
        clauses, params = self._sql({"logbook": ["Operations"]})

        assert clauses == ["lower(e.metadata->>'logbook') = ANY(%s)"]
        assert params == [["operations"]]

    async def test_tags_are_a_guarded_case_insensitive_exists(self, fake_pool):
        clauses, params = self._sql({"tags": ["RF", "Vacuum"]})
        sql, sent = await self._run(fake_pool, clauses, params)

        (clause,) = clauses
        assert "jsonb_typeof(e.metadata->'tags') = 'array'" in clause
        assert "EXISTS (SELECT 1 FROM jsonb_array_elements_text(e.metadata->'tags') t" in clause
        assert "lower(t) = ANY(%s)" in clause
        assert clause.index("jsonb_typeof") < clause.index("EXISTS")
        assert sent == [["rf", "vacuum"]]
        assert clause in sql

    def test_every_column_is_qualified_by_the_alias(self):
        clauses, _ = self._sql(
            {
                "author": ["a"],
                "source_system": ["s"],
                "tags": ["t"],
                "logbook": ["l"],
                "time": {"start": "2025-10-01T00:00:00+00:00", "end": "2025-10-02T00:00:00+00:00"},
            },
            alias="x9",
        )
        sql = " AND ".join(clauses)

        for column in ("author", "source_system", "metadata", "timestamp"):
            assert f"x9.{column}" in sql
            assert f"({column}" not in sql and f" {column}" not in sql

    def test_time_window_is_inclusive_on_both_ends(self):
        start = datetime(2025, 10, 1, tzinfo=UTC)
        end = datetime(2025, 10, 2, tzinfo=UTC)
        clauses, params = self._sql(
            {"time": {"start": start.isoformat(), "end": end.isoformat()}, "author": ["a"]}
        )

        assert clauses[:2] == ["e.timestamp >= %s", "e.timestamp <= %s"]
        assert params[:2] == [start, end]
        assert params[0].tzinfo is not None

    def test_one_sided_time_window(self):
        clauses, params = self._sql({"time": {"end": "2025-10-02T00:00:00+00:00"}})

        assert clauses == ["e.timestamp <= %s"]
        assert len(params) == 1

    def test_enhancement_values_use_the_decision_table_sql(self):
        from osprey.services.ariel_search.enhancement.status import enhancement_state_sql

        clauses, params = self._sql(
            {"enhancement": ["text_embedding:failed", "image_caption:stale"]}
        )
        failed_sql, failed_params = enhancement_state_sql("text_embedding", "failed", "e")
        stale_sql, stale_params = enhancement_state_sql("image_caption", "stale", "e", marker="m2")

        assert clauses == [f"({failed_sql} OR {stale_sql})"]
        assert params == [*failed_params, *stale_params]

    def test_enhancement_honours_has_copy_state(self):
        ctx = dataclasses.replace(FILTER_CTX, has_copy_state=False)
        clauses, _ = self._sql({"enhancement": ["image_caption:failed"]}, ctx=ctx)

        assert "attachment_files" not in clauses[0]
        assert "FALSE" in clauses[0]

    def test_stale_without_a_marker_is_false(self):
        ctx = EnhancementContext(markers={}, enabled=FILTER_CTX.enabled)
        clauses, params = self._sql({"enhancement": ["image_caption:stale"]}, ctx=ctx)

        assert clauses == ["(FALSE)"]
        assert params == []

    def test_a_disabled_module_is_false(self):
        """The SQL agrees with ``matches``: a disabled module never matches."""
        ctx = EnhancementContext(enabled=("image_caption",))
        clauses, params = self._sql({"enhancement": ["text_embedding:failed"]}, ctx=ctx)

        assert clauses == ["(FALSE)"]
        assert params == []

    @pytest.mark.parametrize("alias", ["", "e; DROP TABLE x", "e.x", "1e"])
    def test_a_non_identifier_alias_is_refused(self, alias):
        with pytest.raises(ValueError):
            self._sql({"author": ["a"]}, alias=alias)

    def test_an_empty_filter_still_checks_the_alias(self):
        with pytest.raises(ValueError):
            self._sql({}, alias="bad alias")

    def test_values_are_bound_never_inlined(self):
        clauses, params = self._sql({"author": ["O'Brien'; --"], "tags": ["x' OR '1"]})
        sql = " AND ".join(clauses)

        assert "O'Brien" not in sql and "o'brien" not in sql
        assert "x' OR" not in sql
        assert params == [["o'brien'; --"], ["x' or '1"]]


class TestEntryFilterParse:
    @pytest.mark.parametrize("raw", [None, "", "  ", {}, "{}"])
    def test_nothing_parses_to_the_empty_filter(self, raw):
        flt = EntryFilter.parse(raw, FILTER_DIMS)

        assert flt.is_empty()
        assert flt.start is None and flt.end is None

    def test_json_text_parses_like_the_object(self):
        raw = {"author": ["J. Smith"], "time": {"start": "2025-10-06T00:00:00+00:00"}}

        assert EntryFilter.parse(json.dumps(raw), FILTER_DIMS) == EntryFilter.parse(
            raw, FILTER_DIMS
        )

    def test_an_empty_value_list_is_no_constraint(self):
        assert EntryFilter.parse({"author": []}, FILTER_DIMS).is_empty()

    def test_a_time_only_filter_is_not_empty(self):
        flt = EntryFilter.parse({"time": {"start": "2025-10-06"}}, FILTER_DIMS)

        assert not flt.is_empty()

    def test_null_time_bounds_are_no_constraint(self):
        assert EntryFilter.parse({"time": {"start": None, "end": None}}, FILTER_DIMS).is_empty()
        assert EntryFilter.parse({"time": None}, FILTER_DIMS).is_empty()

    def test_values_are_kept_per_dimension(self):
        flt = EntryFilter.parse({"author": ["A", "B"], "tags": ["RF"]}, FILTER_DIMS)

        assert flt.values("author") == ("A", "B")
        assert flt.values("tags") == ("RF",)
        assert flt.values("source_system") == ()

    def test_the_filter_is_frozen(self):
        flt = EntryFilter.parse({"author": ["A"]}, FILTER_DIMS)

        with pytest.raises(dataclasses.FrozenInstanceError):
            flt.start = None  # type: ignore[misc]

    @pytest.mark.parametrize(
        ("raw", "field", "reason_part"),
        [
            pytest.param({"nope": ["x"]}, "nope", "unknown", id="unknown-dimension"),
            pytest.param({"author": "J. Smith"}, "author", "list", id="non-list-value"),
            pytest.param({"author": [""]}, "author", "empty", id="empty-string"),
            pytest.param({"author": ["  "]}, "author", "empty", id="blank-string"),
            pytest.param({"author": [3]}, "author", "string", id="non-string-item"),
            pytest.param(
                {"enhancement": ["caption:failed"]}, "enhancement", "module", id="unknown-module"
            ),
            pytest.param(
                {"enhancement": ["text_embedding:complete"]},
                "enhancement",
                "state",
                id="unknown-state",
            ),
            pytest.param(
                {"enhancement": ["text_embedding"]}, "enhancement", "module>:<state", id="no-state"
            ),
            pytest.param({"time": {"start": "yesterday"}}, "time", "start", id="bad-time-start"),
            pytest.param({"time": {"end": "2025-13-01"}}, "time", "end", id="bad-time-end"),
            pytest.param({"time": "2025-10-06"}, "time", "object", id="time-not-an-object"),
            pytest.param({"time": {"from": "2025-10-06"}}, "time", "from", id="time-unknown-key"),
            pytest.param({"time": {"start": True}}, "time", "start", id="time-bool"),
            pytest.param(
                {"time": {"start": "2025-10-07", "end": "2025-10-06"}},
                "time",
                "after",
                id="time-start-after-end",
            ),
            pytest.param("{not json", "filter", "JSON", id="bad-json"),
            pytest.param(["author"], "filter", "object", id="not-an-object"),
            pytest.param("[1, 2]", "filter", "object", id="json-not-an-object"),
        ],
    )
    def test_refusals_name_the_field_and_the_reason(self, raw, field, reason_part):
        with pytest.raises(FilterValidationError) as info:
            EntryFilter.parse(raw, FILTER_DIMS)

        assert info.value.field == field
        assert reason_part in info.value.reason

    def test_time_is_reserved_even_without_a_dimension(self):
        """``time`` is never in the dimension table, yet always parses."""
        assert "time" not in _names(list(FILTER_DIMS))

        flt = EntryFilter.parse({"time": {"start": "2025-10-06"}}, FILTER_DIMS)

        assert flt.start is not None

    def test_a_dimension_missing_from_the_table_is_unknown(self):
        """A declared facet of another configuration is refused, not ignored."""
        with pytest.raises(FilterValidationError) as info:
            EntryFilter.parse({"logbook": ["Ops"]}, BUILTIN_DIMENSIONS)

        assert info.value.field == "logbook"

    def test_the_error_is_an_exception_with_a_message(self):
        err = FilterValidationError("author", "must be a list")

        assert isinstance(err, Exception)
        assert "author" in str(err) and "must be a list" in str(err)


class TestEntryFilterTime:
    """Bounds go through ``parse_time_bound``: date-only in the facility zone."""

    @pytest.fixture(autouse=True)
    def _berlin(self, monkeypatch):
        monkeypatch.setattr(
            "osprey_connectors.config.get_facility_timezone", lambda: ZoneInfo("Europe/Berlin")
        )

    def test_a_date_only_start_is_the_first_instant_of_the_facility_day(self):
        flt = EntryFilter.parse({"time": {"start": "2025-10-06"}}, FILTER_DIMS)

        assert flt.start == datetime(2025, 10, 6, 0, 0, tzinfo=ZoneInfo("Europe/Berlin"))

    def test_a_date_only_end_is_the_last_instant_of_the_facility_day(self):
        flt = EntryFilter.parse({"time": {"end": "2025-10-06"}}, FILTER_DIMS)

        assert flt.end == datetime.combine(
            datetime(2025, 10, 6).date(), time.max, tzinfo=ZoneInfo("Europe/Berlin")
        )

    @pytest.mark.parametrize(
        ("timestamp", "expected"),
        [
            # 23:30 Berlin on the 6th is 21:30 UTC: inside the inclusive end day.
            pytest.param(datetime(2025, 10, 6, 21, 30, tzinfo=UTC), True, id="late-evening-local"),
            # 22:30 UTC is 00:30 Berlin on the 7th: past the end day.
            pytest.param(datetime(2025, 10, 6, 22, 30, tzinfo=UTC), False, id="next-local-day"),
        ],
    )
    def test_the_end_day_is_inclusive_in_the_facility_zone(self, timestamp, expected):
        flt = EntryFilter.parse({"time": {"end": "2025-10-06"}}, FILTER_DIMS)

        assert flt.matches(_entry(timestamp=timestamp), FILTER_CTX) is expected

    def test_the_start_day_begins_at_facility_midnight(self):
        flt = EntryFilter.parse({"time": {"start": "2025-10-06"}}, FILTER_DIMS)
        # 22:30 UTC on the 5th is 00:30 Berlin on the 6th.
        inside = _entry(timestamp=datetime(2025, 10, 5, 22, 30, tzinfo=UTC))
        before = _entry(timestamp=datetime(2025, 10, 5, 21, 30, tzinfo=UTC))

        assert flt.matches(inside, FILTER_CTX) is True
        assert flt.matches(before, FILTER_CTX) is False

    def test_a_same_day_window_is_one_whole_day(self):
        flt = EntryFilter.parse({"time": {"start": "2025-10-06", "end": "2025-10-06"}}, FILTER_DIMS)

        assert flt.end - flt.start < timedelta(days=1)
        assert flt.matches(_entry(timestamp=datetime(2025, 10, 6, 10, tzinfo=UTC)), FILTER_CTX)

    def test_a_timestamp_given_as_iso_text_is_compared(self):
        flt = EntryFilter.parse({"time": {"end": "2025-10-06"}}, FILTER_DIMS)

        assert flt.matches(_entry(timestamp="2025-10-06T21:30:00+00:00"), FILTER_CTX) is True


class TestEntryFilterWithout:
    def test_without_drops_one_dimension(self):
        flt = EntryFilter.parse({"author": ["A"], "tags": ["RF"]}, FILTER_DIMS)

        rest = flt.without("author")

        assert rest.values("author") == ()
        assert rest.values("tags") == ("RF",)
        assert flt.values("author") == ("A",), "without must not mutate the original"

    def test_without_time_drops_the_window_only(self):
        flt = EntryFilter.parse({"author": ["A"], "time": {"start": "2025-10-06"}}, FILTER_DIMS)

        rest = flt.without("time")

        assert rest.start is None and rest.end is None
        assert rest.values("author") == ("A",)

    def test_without_the_only_dimension_is_empty(self):
        assert EntryFilter.parse({"tags": ["RF"]}, FILTER_DIMS).without("tags").is_empty()

    def test_without_an_unset_dimension_is_the_same_filter(self):
        flt = EntryFilter.parse({"tags": ["RF"]}, FILTER_DIMS)

        assert flt.without("author") == flt

    def test_self_exclusion_widens_the_match(self):
        """The facet of a dimension counts as if that dimension were unset."""
        flt = EntryFilter.parse({"author": ["A. Other"], "tags": ["RF"]}, FILTER_DIMS)

        assert flt.matches(_entry(), FILTER_CTX) is False
        assert flt.without("author").matches(_entry(), FILTER_CTX) is True
