"""Tests for facility entry-field declarations and their errors."""

from __future__ import annotations

import dataclasses
import datetime

import pytest

from osprey.services.ariel_search import entry_fields as entry_fields_module
from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.entry_fields import (
    EntryFieldDeclarationError,
    EntryFieldError,
    EntryFieldOptionsUnavailable,
    check_declarations,
    coerce_entry_value,
    entry_field_descriptors,
    resolve_entry_write,
    validate_entry_fields,
)
from osprey.services.ariel_search.search.base import ParameterDescriptor
from tests.fixtures.ariel_entry_fields import (  # noqa: F401 - fixtures used by name
    EXAMPLE_SOURCE_SYSTEM,
    OPTIONS_FAIL,
    OPTIONS_HANG,
    example_config,
    example_descriptors,
    example_entry_fields_fixture,
)


def _field(name: str, param_type: str = "text", **kwargs) -> ParameterDescriptor:
    return ParameterDescriptor(
        name=name,
        label=name.title(),
        description=f"The {name}",
        param_type=param_type,
        default=None,
        **kwargs,
    )


def _refused(descriptors, field: str) -> EntryFieldDeclarationError:
    with pytest.raises(EntryFieldDeclarationError) as info:
        check_declarations(descriptors, adapter=EXAMPLE_SOURCE_SYSTEM)
    exc = info.value
    assert exc.adapter == EXAMPLE_SOURCE_SYSTEM
    assert exc.field == field
    assert EXAMPLE_SOURCE_SYSTEM in str(exc)
    assert f"'{field}'" in str(exc)
    return exc


def test_declaration_example_adapter_passes():
    """The example adapter's declarations pass unchanged."""
    descriptors = example_descriptors()
    assert check_declarations(descriptors, adapter=EXAMPLE_SOURCE_SYSTEM) == descriptors


def test_declaration_logbook_and_shift_allowed():
    """logbook and shift are not reserved; they replace the built-in inputs."""
    descriptors = [*example_descriptors(declare_logbook=True), _field("shift")]
    assert check_declarations(descriptors, adapter=EXAMPLE_SOURCE_SYSTEM) == descriptors


def test_declaration_empty_passes():
    """No declarations is a sound declaration set."""
    assert check_declarations([]) == []


def test_declaration_duplicate_name_refused():
    """A name declared twice is refused."""
    exc = _refused([*example_descriptors(), _field("day", "date")], "day")
    assert "more than once" in str(exc)


@pytest.mark.parametrize(
    "name", ["tags", "sync_status", "created_via", "session_metadata", "title"]
)
def test_declaration_reserved_name_refused(name):
    """Every reserved name is refused."""
    exc = _refused([*example_descriptors(), _field(name)], name)
    assert "reserved" in str(exc)


def test_declaration_unknown_type_refused():
    """A param_type outside the known set is refused."""
    exc = _refused([_field("weird", "colour")], "weird")
    assert "colour" in str(exc)


@pytest.mark.parametrize("options", [None, []])
def test_declaration_select_without_options_refused(options):
    """A select with no options is refused."""
    book = dataclasses.replace(example_descriptors()[0], options=options)
    exc = _refused([book], "book")
    assert "option" in str(exc)


def test_declaration_dynamic_select_needs_no_options():
    """A dynamic_select is fine without static options."""
    assert check_declarations([_field("scan", "dynamic_select")])


def test_declaration_depends_on_unknown_field_refused():
    """depends_on naming an undeclared field is refused, naming the dependent field."""
    exc = _refused([_field("scan", "dynamic_select", depends_on=("nowhere",))], "scan")
    assert "nowhere" in str(exc)


def test_declaration_depends_on_non_static_field_refused():
    """depends_on naming a dynamic_select is refused."""
    descriptors = [
        _field("run", "dynamic_select"),
        _field("scan", "dynamic_select", depends_on=("run",)),
    ]
    exc = _refused(descriptors, "scan")
    assert "run" in str(exc)


def test_declaration_depends_on_later_static_field_passes():
    """depends_on may name a static field declared further down the list."""
    descriptors = [
        _field("scan", "dynamic_select", depends_on=("day",)),
        _field("day", "date"),
    ]
    assert check_declarations(descriptors) == descriptors


def test_declaration_default_adapter_name_in_message():
    """Without an adapter name the message still reads as a sentence."""
    with pytest.raises(EntryFieldDeclarationError, match="the facility adapter"):
        check_declarations([_field("title")])


def test_entry_field_error_carries_field():
    """EntryFieldError keeps the field and uses the message as its text."""
    exc = EntryFieldError("day", "Day must be a date")
    assert exc.field == "day"
    assert str(exc) == "Day must be a date"


def test_options_unavailable_names_field_generically():
    """EntryFieldOptionsUnavailable names the field in a generic message."""
    exc = EntryFieldOptionsUnavailable("scan")
    assert exc.field == "scan"
    assert "scan" in str(exc)


def test_helper_no_ingestion_config_yields_no_fields():
    """Without an ingestion block there is no adapter, so there are no fields."""
    config = ARIELConfig.from_dict({"database": {"uri": "postgresql://test"}})
    assert entry_field_descriptors(config) == []


@pytest.mark.usefixtures("example_entry_fields")
def test_helper_example_adapter_yields_its_three_fields():
    """The configured example adapter's declarations come back checked, in order."""
    descriptors = entry_field_descriptors(example_config())
    assert [d.name for d in descriptors] == ["book", "day", "scan"]
    assert descriptors == example_descriptors()


def test_helper_broken_adapter_raises(example_entry_fields, monkeypatch):
    """A misdeclaring adapter's error propagates, naming the adapter and field."""
    monkeypatch.setattr(
        example_entry_fields,
        "get_entry_field_descriptors",
        lambda: [*example_descriptors(), _field("title")],
    )
    with pytest.raises(EntryFieldDeclarationError) as info:
        entry_field_descriptors(example_config())
    assert info.value.adapter == EXAMPLE_SOURCE_SYSTEM
    assert info.value.field == "title"


_SELECT_OPTIONS = [
    {"value": "ops", "label": "Operations"},
    {"value": "physics", "label": "Physics"},
]


@pytest.mark.parametrize(
    ("param_type", "kwargs", "raw", "expected"),
    [
        ("int", {}, "3", 3),
        ("int", {}, " 3 ", 3),
        ("int", {}, 3, 3),
        ("int", {}, 3.0, 3),
        ("int", {}, "-7", -7),
        ("int", {"min_value": 1, "max_value": 5}, "5", 5),
        ("float", {}, "2.5", 2.5),
        ("float", {}, "3", 3.0),
        ("float", {}, 3, 3.0),
        ("float", {"min_value": 0.0, "max_value": 1.0}, "0.0", 0.0),
        ("bool", {}, "true", True),
        ("bool", {}, "False", False),
        ("bool", {}, "1", True),
        ("bool", {}, "0", False),
        ("bool", {}, "yes", True),
        ("bool", {}, "off", False),
        ("bool", {}, True, True),
        ("bool", {}, False, False),
        ("date", {}, "2026-10-04", "2026-10-04"),
        ("date", {}, datetime.date(2026, 1, 2), "2026-01-02"),
        ("text", {}, "beam dump", "beam dump"),
        ("select", {"options": _SELECT_OPTIONS}, "physics", "physics"),
        ("dynamic_select", {}, "scan-42", "scan-42"),
    ],
)
def test_coerce_valid_values(param_type, kwargs, raw, expected):
    """The string form or a native value comes back JSON-native."""
    value = coerce_entry_value(_field("f", param_type, **kwargs), raw)
    assert value == expected
    assert type(value) is type(expected)


@pytest.mark.parametrize(
    ("param_type", "kwargs", "raw"),
    [
        ("int", {}, "3.5"),
        ("int", {}, "three"),
        ("int", {}, 3.5),
        ("int", {}, True),
        ("int", {"min_value": 1}, "0"),
        ("int", {"max_value": 5}, "6"),
        ("float", {}, "abc"),
        ("float", {}, "nan"),
        ("float", {}, "inf"),
        ("float", {}, True),
        ("float", {"min_value": 0.0, "max_value": 1.0}, "1.5"),
        ("float", {"min_value": 0.0}, -0.1),
        ("bool", {}, "maybe"),
        ("bool", {}, 2),
        ("date", {}, "2026-13-01"),
        ("date", {}, "04/10/2026"),
        ("date", {}, "20261004"),
        ("date", {}, "not a date"),
        ("date", {}, datetime.datetime(2026, 1, 2, 3, 4)),
        ("date", {}, 20261004),
        ("text", {}, 5),
        ("text", {}, "x" * 201),
        ("select", {"options": _SELECT_OPTIONS}, "astrology"),
        ("select", {"options": _SELECT_OPTIONS}, "Operations"),
        ("dynamic_select", {}, "s" * 201),
        ("dynamic_select", {}, ["scan-42"]),
        ("int", {}, "1" * 201),
        ("int", {}, [3]),
    ],
)
def test_coerce_invalid_values_raise(param_type, kwargs, raw):
    """A value that does not fit its declaration raises EntryFieldError naming the field."""
    with pytest.raises(EntryFieldError) as info:
        coerce_entry_value(_field("f", param_type, **kwargs), raw)
    assert info.value.field == "f"
    assert info.value.message


@pytest.mark.parametrize(
    "param_type", ["text", "int", "float", "bool", "date", "select", "dynamic_select"]
)
@pytest.mark.parametrize("raw", ["", "   ", None])
def test_coerce_empty_means_absent(param_type, raw):
    """An empty string (or None) is an absent value, for every type."""
    kwargs = {"options": _SELECT_OPTIONS} if param_type == "select" else {}
    assert coerce_entry_value(_field("f", param_type, **kwargs), raw) is None


def test_coerce_text_at_cap_passes():
    """A 200-character string is accepted; the cap is inclusive."""
    assert coerce_entry_value(_field("note"), "x" * 200) == "x" * 200


def test_coerce_error_message_names_label_and_bound():
    """An out-of-range number names the field's label and the violated bound."""
    descriptor = _field("energy", "float", min_value=0.0, max_value=2.0)
    with pytest.raises(EntryFieldError) as info:
        coerce_entry_value(descriptor, "2.5")
    assert "Energy" in info.value.message
    assert "2" in info.value.message


def test_coerce_select_miss_names_choices():
    """A select miss lists the allowed values."""
    descriptor = _field("book", "select", options=_SELECT_OPTIONS)
    with pytest.raises(EntryFieldError) as info:
        coerce_entry_value(descriptor, "astrology")
    assert "ops" in info.value.message
    assert "physics" in info.value.message


def test_coerce_error_is_value_error():
    """EntryFieldError from coercion is a ValueError, so generic handlers see it."""
    with pytest.raises(ValueError):
        coerce_entry_value(_field("n", "int"), "x")


# --- validate_entry_fields -------------------------------------------------

_SCANS = {"2026-10-01": [{"value": "s1", "label": "Scan 1"}, {"value": "s2", "label": "Scan 2"}]}


async def _validate(adapter, values, *, partial=False, check_live=True, strict=False):
    return await validate_entry_fields(
        adapter,
        example_descriptors(),
        values,
        partial=partial,
        check_live=check_live,
        strict=strict,
    )


async def _invalid(adapter, values, field, **kwargs) -> EntryFieldError:
    with pytest.raises(EntryFieldError) as info:
        await _validate(adapter, values, **kwargs)
    assert info.value.field == field
    return info.value


async def test_validate_valid_values_coerced(example_entry_fields):
    """Every declared value comes back JSON-native, string forms included."""
    example_entry_fields.state.options_table = _SCANS
    result = await _validate(
        example_entry_fields, {"book": " physics ", "day": datetime.date(2026, 10, 1), "scan": "s2"}
    )
    assert result == {"book": "physics", "day": "2026-10-01", "scan": "s2"}
    assert example_entry_fields.state.options_calls == [("scan", {"day": "2026-10-01"})]


async def test_validate_missing_required_refused(example_entry_fields):
    """A full validation refuses a missing required value, naming it."""
    exc = await _invalid(example_entry_fields, {"day": "2026-10-01"}, "book")
    assert "required" in str(exc)
    assert example_entry_fields.state.options_calls == []


async def test_validate_empty_required_counts_as_missing(example_entry_fields):
    """An empty string for a required field is a missing value."""
    await _invalid(example_entry_fields, {"book": "  "}, "book")


async def test_validate_partial_accepts_missing_required(example_entry_fields):
    """partial=True checks only the values present."""
    assert await _validate(
        example_entry_fields, {"day": "2026-10-01"}, partial=True, check_live=False
    ) == {"day": "2026-10-01"}


async def test_validate_partial_still_refuses_wrong_type(example_entry_fields):
    """partial=True still refuses a present value of the wrong type."""
    await _invalid(example_entry_fields, {"day": "tomorrow"}, "day", partial=True)


async def test_validate_wrong_type_refused(example_entry_fields):
    """A value of the wrong type is refused, naming the field."""
    await _invalid(example_entry_fields, {"book": "ops", "day": "2026-13-01"}, "day")


async def test_validate_select_outside_options_refused(example_entry_fields):
    """A select value outside its options is refused."""
    exc = await _invalid(example_entry_fields, {"book": "poetry"}, "book")
    assert "ops" in str(exc)


async def test_validate_number_outside_bounds_refused():
    """A number outside min/max is refused."""
    descriptors = [_field("shots", "int", min_value=1, max_value=10)]
    with pytest.raises(EntryFieldError) as info:
        await validate_entry_fields(
            None, descriptors, {"shots": "11"}, partial=False, check_live=False, strict=True
        )
    assert info.value.field == "shots"


async def test_validate_dynamic_outside_live_options_refused(example_entry_fields):
    """A dynamic_select value outside the live options is refused after one call."""
    example_entry_fields.state.options_table = _SCANS
    exc = await _invalid(
        example_entry_fields, {"book": "ops", "day": "2026-10-01", "scan": "s9"}, "scan"
    )
    assert "s1" in str(exc) and "s2" in str(exc)
    assert len(example_entry_fields.state.options_calls) == 1


async def test_validate_live_choices_listed_at_most_fifty(example_entry_fields):
    """The refusal lists at most 50 allowed choices."""
    many = [{"value": f"c{i}", "label": f"C{i}"} for i in range(60)]
    example_entry_fields.state.options_table = {"2026-10-01": many}
    exc = await _invalid(
        example_entry_fields, {"book": "ops", "day": "2026-10-01", "scan": "nope"}, "scan"
    )
    assert "c49" in str(exc)
    assert "c50" not in str(exc)


async def test_validate_invalid_parent_skips_child(example_entry_fields):
    """An invalid static parent is reported and its dynamic child never checked."""
    example_entry_fields.state.options_table = _SCANS
    await _invalid(example_entry_fields, {"book": "ops", "day": "nonsense", "scan": "s1"}, "day")
    assert example_entry_fields.state.options_calls == []


async def test_validate_statics_before_dynamics(example_entry_fields):
    """A static error is reported even when a dynamic value is also wrong."""
    example_entry_fields.state.options_table = _SCANS
    await _invalid(example_entry_fields, {"scan": "s9", "day": "2026-10-01"}, "book")
    assert example_entry_fields.state.options_calls == []


async def test_validate_forwards_only_depends_on_values(example_entry_fields):
    """The options call carries only the dynamic field's parents, coerced."""
    example_entry_fields.state.options_table = _SCANS
    await _validate(example_entry_fields, {"book": "ops", "day": "2026-10-01", "scan": "s1"})
    assert example_entry_fields.state.options_calls == [("scan", {"day": "2026-10-01"})]


async def test_validate_absent_dynamic_costs_no_call(example_entry_fields):
    """No dynamic value means no options call."""
    assert await _validate(example_entry_fields, {"book": "ops"}) == {"book": "ops"}
    assert example_entry_fields.state.options_calls == []


async def test_validate_check_live_false_skips_membership(example_entry_fields):
    """check_live=False accepts any dynamic_select string without calling the adapter."""
    result = await _validate(
        example_entry_fields, {"book": "ops", "scan": "anything"}, check_live=False
    )
    assert result == {"book": "ops", "scan": "anything"}
    assert example_entry_fields.state.options_calls == []


async def test_validate_check_live_false_still_type_checks_dynamic(example_entry_fields):
    """check_live=False still refuses a non-string dynamic value."""
    await _invalid(example_entry_fields, {"book": "ops", "scan": 5}, "scan", check_live=False)


async def test_validate_options_failure_unavailable(example_entry_fields):
    """An adapter failure while listing options raises EntryFieldOptionsUnavailable."""
    example_entry_fields.state.options_mode = OPTIONS_FAIL
    with pytest.raises(EntryFieldOptionsUnavailable) as info:
        await _validate(example_entry_fields, {"book": "ops", "scan": "s1"})
    assert info.value.field == "scan"
    assert "example options lookup failed" not in str(info.value)
    assert len(example_entry_fields.state.options_calls) == 1


async def test_validate_options_timeout_unavailable(example_entry_fields, monkeypatch):
    """An options call that never returns times out as EntryFieldOptionsUnavailable."""
    monkeypatch.setattr(entry_fields_module, "OPTIONS_TIMEOUT_SECONDS", 0.05)
    example_entry_fields.state.options_mode = OPTIONS_HANG
    with pytest.raises(EntryFieldOptionsUnavailable) as info:
        await _validate(example_entry_fields, {"book": "ops", "scan": "s1"})
    assert info.value.field == "scan"


async def test_validate_strict_refuses_undeclared_key(example_entry_fields):
    """strict=True refuses an undeclared key by name."""
    exc = await _invalid(
        example_entry_fields, {"book": "ops", "colour": "red"}, "colour", strict=True
    )
    assert "colour" in str(exc)


async def test_validate_strict_without_adapter_refuses_any_key():
    """strict=True with no adapter and no declarations refuses the first key."""
    with pytest.raises(EntryFieldError) as info:
        await validate_entry_fields(
            None, [], {"colour": "red", "size": 1}, partial=True, check_live=True, strict=True
        )
    assert info.value.field == "colour"


async def test_validate_strict_without_adapter_empty_passes():
    """strict=True with no declarations accepts empty values."""
    assert (
        await validate_entry_fields(None, [], {}, partial=False, check_live=True, strict=True) == {}
    )


async def test_validate_lenient_ignores_undeclared_keys(example_entry_fields):
    """strict=False drops undeclared keys and returns only declared values."""
    result = await _validate(
        example_entry_fields, {"book": "ops", "colour": "red", "session_metadata": {}}
    )
    assert result == {"book": "ops"}


async def test_validate_never_fills_defaults(example_entry_fields):
    """A missing optional value stays absent; the declared default is not filled in."""
    result = await _validate(example_entry_fields, {"book": "ops"})
    assert "day" not in result and "scan" not in result


async def test_validate_does_not_mutate_input(example_entry_fields):
    """The caller's values dict is left untouched."""
    values = {"book": " ops ", "colour": "red"}
    await _validate(example_entry_fields, values)
    assert values == {"book": " ops ", "colour": "red"}


# --- resolve_entry_write ---------------------------------------------------


def _resolve(
    declared,
    *,
    descriptors=None,
    logbook=None,
    shift=None,
    tags=None,
    created_via="ariel-web",
    session_metadata=None,
):
    return resolve_entry_write(
        example_descriptors(declare_logbook=True) if descriptors is None else descriptors,
        declared,
        logbook=logbook,
        shift=shift,
        tags=["beam"] if tags is None else tags,
        created_via=created_via,
        session_metadata=session_metadata,
    )


def test_resolve_no_declarations_is_todays_request():
    """With no declarations the write is exactly today's: built-ins only, no mirroring."""
    result = _resolve(
        {"colour": "red", "sync_status": "synced"},
        descriptors=[],
        logbook="ops",
        shift="day",
    )
    assert result.logbook == "ops"
    assert result.shift == "day"
    assert result.adapter_metadata == {}
    assert result.local_metadata == {
        "logbook": "ops",
        "shift": "day",
        "tags": ["beam"],
        "created_via": "ariel-web",
    }


def test_resolve_no_declarations_keeps_session_metadata_locally():
    """Without declarations the session metadata still lands in the local copy only."""
    session = {"created_via": "ariel-mcp", "user": "op"}
    result = _resolve(
        {}, descriptors=[], logbook="ops", created_via="ariel-mcp", session_metadata=session
    )
    assert result.adapter_metadata == {}
    assert result.local_metadata == {
        "session_metadata": session,
        "logbook": "ops",
        "shift": None,
        "tags": ["beam"],
        "created_via": "ariel-mcp",
    }


def test_resolve_declared_values_reach_adapter_and_local_copy():
    """Declared values go to both outputs, with the resolved logbook/shift mirrored."""
    result = _resolve({"book": "ops", "day": "2026-10-01"}, logbook="control-room", shift="owl")
    assert result.logbook == "control-room"
    assert result.shift == "owl"
    assert result.adapter_metadata == {
        "book": "ops",
        "day": "2026-10-01",
        "logbook": "control-room",
        "shift": "owl",
    }
    assert result.local_metadata == {
        "book": "ops",
        "day": "2026-10-01",
        "logbook": "control-room",
        "shift": "owl",
        "tags": ["beam"],
        "created_via": "ariel-web",
    }


@pytest.mark.parametrize("builtin", [None, "", "   "])
def test_resolve_declared_logbook_with_empty_builtin_sets_both(builtin):
    """A declared logbook with no built-in value sets the request field and both metadata copies."""
    result = _resolve({"book": "ops", "logbook": "maintenance"}, logbook=builtin)
    assert result.logbook == "maintenance"
    assert result.adapter_metadata["logbook"] == "maintenance"
    assert result.local_metadata["logbook"] == "maintenance"


def test_resolve_builtin_logbook_alone_is_mirrored():
    """A built-in logbook alone is used and mirrored into metadata when fields are declared."""
    result = _resolve({"book": "ops"}, logbook="control-room")
    assert result.logbook == "control-room"
    assert result.adapter_metadata["logbook"] == "control-room"


def test_resolve_same_logbook_twice_is_not_a_conflict():
    """The same logbook given as built-in and declared is accepted."""
    result = _resolve({"book": "ops", "logbook": "maintenance"}, logbook=" maintenance ")
    assert result.logbook == "maintenance"


def test_resolve_logbook_conflict_raises():
    """Built-in and declared logbook both present and different is an EntryFieldError."""
    with pytest.raises(EntryFieldError) as info:
        _resolve({"book": "ops", "logbook": "maintenance"}, logbook="control-room")
    assert info.value.field == "logbook"
    assert "Logbook" in str(info.value)
    assert "maintenance" in str(info.value) and "control-room" in str(info.value)


def test_resolve_shift_conflict_raises():
    """A declared shift differing from the built-in shift is an EntryFieldError on 'shift'."""
    descriptors = [*example_descriptors(), _field("shift")]
    with pytest.raises(EntryFieldError) as info:
        _resolve({"book": "ops", "shift": "owl"}, descriptors=descriptors, shift="day")
    assert info.value.field == "shift"


def test_resolve_unresolved_logbook_not_mirrored():
    """With declarations but no logbook from either source, metadata carries no logbook key."""
    result = _resolve({"book": "ops"})
    assert result.logbook is None
    assert "logbook" not in result.adapter_metadata


def test_resolve_client_sync_status_dropped():
    """A client-sent sync_status or other undeclared key never reaches either output."""
    result = _resolve(
        {"book": "ops", "sync_status": "synced", "colour": "red"},
        session_metadata={"user": "op"},
    )
    for metadata in (result.adapter_metadata, result.local_metadata):
        assert "sync_status" not in metadata
        assert "colour" not in metadata


def test_resolve_ariel_keys_cannot_be_overridden():
    """ARIEL's own keys are rebuilt last, so a client value under their names loses."""
    result = _resolve(
        {"book": "ops", "created_via": "evil", "tags": ["x"], "session_metadata": {"a": 1}},
        tags=["beam"],
        created_via="ariel-web",
    )
    assert result.local_metadata["created_via"] == "ariel-web"
    assert result.local_metadata["tags"] == ["beam"]
    assert "session_metadata" not in result.local_metadata
    assert set(result.adapter_metadata) == {"book"}


@pytest.mark.parametrize("created_via", ["ariel-web", "ariel-mcp"])
def test_resolve_provenance_kept(created_via):
    """Both callers' provenance survives in the local copy and never reaches the adapter."""
    session = {"created_via": created_via, "host": "box"}
    result = _resolve({"book": "ops"}, created_via=created_via, session_metadata=session)
    assert result.local_metadata["created_via"] == created_via
    assert result.local_metadata["session_metadata"] == session
    assert "created_via" not in result.adapter_metadata
    assert "session_metadata" not in result.adapter_metadata


def test_resolve_does_not_mutate_inputs():
    """The caller's declared values, tags and session metadata are left untouched."""
    declared = {"book": "ops", "sync_status": "synced"}
    tags = ["beam"]
    session = {"user": "op"}
    result = _resolve(declared, tags=tags, session_metadata=session)
    result.local_metadata["tags"].append("x")
    result.local_metadata["session_metadata"]["user"] = "other"
    assert declared == {"book": "ops", "sync_status": "synced"}
    assert tags == ["beam"]
    assert session == {"user": "op"}
