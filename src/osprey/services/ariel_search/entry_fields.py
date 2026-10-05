"""Facility-declared entry fields: declaration checks and their errors.

A facility adapter may declare extra fields an operator fills in when writing
a logbook entry (see ``FacilityAdapter.get_entry_field_descriptors``). Every
declaration is a :class:`ParameterDescriptor`. Before any declaration reaches
a form or a write path it passes :func:`check_declarations`, so a broken
declaration fails loudly as one :class:`EntryFieldDeclarationError` naming the
adapter and the field, and nothing ever renders half a form.

Public surface:

* :class:`EntryFieldError` -- a submitted value for one field is invalid.
* :class:`EntryFieldDeclarationError` -- the adapter declared its fields wrongly.
* :class:`EntryFieldOptionsUnavailable` -- the adapter could not list the
  choices of a ``dynamic_select`` field.
* :func:`check_declarations` -- refuse a broken set of declarations.
* :func:`coerce_entry_value` -- turn one submitted value into its JSON-native
  form, or refuse it.
* :func:`entry_field_descriptors` -- the configured adapter's checked
  declarations; every caller reads the declarations through it.
* :func:`fetch_entry_field_options` -- ask the adapter for a ``dynamic_select``
  field's choices under :data:`OPTIONS_TIMEOUT_SECONDS`.
* :func:`resolve_entry_write` -- turn the built-in inputs, the checked declared
  values and the caller's provenance into a :class:`ResolvedEntryWrite`.
* :func:`validate_entry_fields` -- check a set of submitted values against the
  declarations and return them coerced.
"""

from __future__ import annotations

import asyncio
import copy
import datetime
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from osprey.services.ariel_search.exceptions import AdapterNotFoundError
from osprey.services.ariel_search.search.base import ParameterDescriptor

if TYPE_CHECKING:
    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.ingestion.base import FacilityAdapter

__all__ = [
    "ALL_PARAM_TYPES",
    "RESERVED_FIELD_NAMES",
    "STATIC_PARAM_TYPES",
    "EntryFieldDeclarationError",
    "EntryFieldError",
    "EntryFieldOptionsUnavailable",
    "MAX_LISTED_CHOICES",
    "MAX_STRING_LENGTH",
    "OPTIONS_TIMEOUT_SECONDS",
    "ResolvedEntryWrite",
    "check_declarations",
    "coerce_entry_value",
    "entry_field_descriptors",
    "fetch_entry_field_options",
    "resolve_entry_write",
    "validate_entry_fields",
]

#: Field types whose value is known without asking the adapter; only these may
#: be named in another field's ``depends_on``.
STATIC_PARAM_TYPES: frozenset[str] = frozenset({"text", "int", "float", "bool", "date", "select"})

#: Every field type a declaration may use.
ALL_PARAM_TYPES: frozenset[str] = STATIC_PARAM_TYPES | {"dynamic_select"}

#: Names an entry field may not take, because the entry already carries them.
#: ``logbook`` and ``shift`` are deliberately absent: declaring them replaces
#: the built-in input.
RESERVED_FIELD_NAMES: frozenset[str] = frozenset(
    {"tags", "sync_status", "created_via", "session_metadata", "title"}
)

#: Longest string accepted for any entry-field value, in characters.
MAX_STRING_LENGTH = 200

#: Longest the server waits for an adapter to list a field's choices, in seconds.
OPTIONS_TIMEOUT_SECONDS = 10.0

#: Most allowed choices a refusal of a ``dynamic_select`` value lists.
MAX_LISTED_CHOICES = 50

_DEFAULT_ADAPTER_NAME = "the facility adapter"

_TRUE_WORDS = frozenset({"true", "1", "yes", "on"})
_FALSE_WORDS = frozenset({"false", "0", "no", "off"})
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


class EntryFieldError(ValueError):
    """A submitted value for one declared entry field is invalid.

    Attributes:
        field: Name of the offending field.
        message: Human-readable explanation, also ``str(exc)``.
    """

    def __init__(self, field: str, message: str) -> None:
        """Initialize the error.

        Args:
            field: Name of the offending field.
            message: Human-readable explanation naming what is wrong.
        """
        super().__init__(message)
        self.field = field
        self.message = message


class EntryFieldDeclarationError(Exception):
    """The facility adapter declared its entry fields wrongly.

    Attributes:
        adapter: Display name of the adapter whose declaration is broken.
        field: Name of the offending field.
        message: Human-readable explanation naming adapter and field, also
            ``str(exc)``.
    """

    def __init__(self, message: str, *, adapter: str, field: str) -> None:
        """Initialize the error.

        Args:
            message: Human-readable explanation naming adapter and field.
            adapter: Display name of the adapter.
            field: Name of the offending field.
        """
        super().__init__(message)
        self.message = message
        self.adapter = adapter
        self.field = field


class EntryFieldOptionsUnavailable(Exception):
    """The adapter could not list the choices of a ``dynamic_select`` field.

    The message is deliberately generic: it names the field but never carries
    the adapter's own error text.

    Attributes:
        field: Name of the field whose options could not be listed.
        message: Human-readable explanation, also ``str(exc)``.
    """

    def __init__(self, field: str) -> None:
        """Initialize the error.

        Args:
            field: Name of the field whose options could not be listed.
        """
        message = f"The options for entry field '{field}' are unavailable right now."
        super().__init__(message)
        self.field = field
        self.message = message


def check_declarations(
    descriptors: Sequence[ParameterDescriptor],
    *,
    adapter: str = _DEFAULT_ADAPTER_NAME,
) -> list[ParameterDescriptor]:
    """Refuse a broken set of entry-field declarations.

    A declaration set is refused when a name is declared twice, a name is
    reserved, a ``param_type`` is unknown, a ``select`` has no options, or a
    ``depends_on`` entry names a field that is not declared or is not static.
    ``depends_on`` may name a field declared later in the list.

    Args:
        descriptors: The adapter's declarations, in form order.
        adapter: Display name of the adapter, used in the error message.

    Returns:
        The declarations as a new list, unchanged, when all of them are sound.

    Raises:
        EntryFieldDeclarationError: On the first broken declaration, naming
            the adapter and the field.
    """

    def refuse(field: str, problem: str) -> EntryFieldDeclarationError:
        return EntryFieldDeclarationError(
            f"{adapter} misdeclares entry field '{field}': {problem}",
            adapter=adapter,
            field=field,
        )

    by_name: dict[str, ParameterDescriptor] = {}
    for descriptor in descriptors:
        name = descriptor.name
        if name in by_name:
            raise refuse(name, "the name is declared more than once")
        if name in RESERVED_FIELD_NAMES:
            raise refuse(name, "the name is reserved")
        if descriptor.param_type not in ALL_PARAM_TYPES:
            known = ", ".join(sorted(ALL_PARAM_TYPES))
            raise refuse(name, f"unknown type '{descriptor.param_type}' (expected one of {known})")
        if descriptor.param_type == "select" and not descriptor.options:
            raise refuse(name, "a select needs at least one option")
        by_name[name] = descriptor

    for descriptor in by_name.values():
        for dependency in descriptor.depends_on:
            target = by_name.get(dependency)
            if target is None:
                raise refuse(
                    descriptor.name, f"depends_on names '{dependency}', which is not declared"
                )
            if target.param_type not in STATIC_PARAM_TYPES:
                raise refuse(
                    descriptor.name,
                    f"depends_on names '{dependency}', a {target.param_type} field; "
                    "only static fields (text, int, float, bool, date, select) may be named",
                )

    return list(by_name.values())


def entry_field_descriptors(config: ARIELConfig) -> list[ParameterDescriptor]:
    """Return the configured adapter's entry-field declarations, checked.

    ``get_adapter`` is looked up on :mod:`osprey.services.ariel_search.ingestion`
    at call time, so a patch of that name applies here too.

    Args:
        config: ARIEL configuration naming the ingestion adapter.

    Returns:
        The adapter's declarations in form order, or ``[]`` when no adapter is
        configured or the configured one is not found.

    Raises:
        EntryFieldDeclarationError: When the adapter declares its fields wrongly.
    """
    from osprey.services.ariel_search import ingestion

    try:
        adapter = ingestion.get_adapter(config)
    except AdapterNotFoundError:
        return []
    return check_declarations(
        adapter.get_entry_field_descriptors(), adapter=adapter.source_system_name
    )


async def fetch_entry_field_options(
    adapter: FacilityAdapter, name: str, values: dict[str, Any]
) -> list[dict[str, str]]:
    """Ask the adapter for the choices of one ``dynamic_select`` field.

    Args:
        adapter: The facility adapter that declares the field.
        name: The field's name.
        values: The coerced values of the fields it ``depends_on``.

    Returns:
        The adapter's choices as ``{"value": ..., "label": ...}`` dicts.

    Raises:
        EntryFieldOptionsUnavailable: When the adapter fails or does not answer
            within :data:`OPTIONS_TIMEOUT_SECONDS`.
    """
    try:
        return await asyncio.wait_for(
            adapter.get_entry_field_options(name, values), timeout=OPTIONS_TIMEOUT_SECONDS
        )
    except Exception as exc:
        raise EntryFieldOptionsUnavailable(name) from exc


async def validate_entry_fields(
    adapter: FacilityAdapter | None,
    descriptors: Sequence[ParameterDescriptor],
    values: dict[str, Any],
    *,
    partial: bool,
    check_live: bool,
    strict: bool,
) -> dict[str, Any]:
    """Check submitted entry-field values against their declarations.

    Static fields are checked before ``dynamic_select`` fields, in declaration
    order, and the first problem is raised; so an invalid parent is reported
    and the dynamic field depending on it is never checked. A missing value is
    never filled from the declared ``default``.

    Args:
        adapter: The adapter that declared ``descriptors``; ``None`` when no
            adapter is configured, in which case nothing is checked live.
        descriptors: The checked declarations (see :func:`entry_field_descriptors`).
        values: The submitted values keyed by field name; left unchanged.
        partial: Check only the values present; a missing required value passes.
        check_live: Check each present ``dynamic_select`` value against one
            :func:`fetch_entry_field_options` call, forwarding only the field's
            ``depends_on`` values.
        strict: Refuse any key that is not a declared field. Otherwise
            undeclared keys are ignored and left out of the result.

    Returns:
        The present declared values, coerced by :func:`coerce_entry_value`,
        keyed by field name in declaration order.

    Raises:
        EntryFieldError: On the first invalid value, naming its field.
        EntryFieldOptionsUnavailable: When a live check cannot list the choices.
    """
    declared = {descriptor.name: descriptor for descriptor in descriptors}
    if strict:
        for key in values:
            if key not in declared:
                raise EntryFieldError(key, f"'{key}' is not a declared entry field")

    statics = [d for d in descriptors if d.param_type != "dynamic_select"]
    dynamics = [d for d in descriptors if d.param_type == "dynamic_select"]
    coerced: dict[str, Any] = {}

    for descriptor in [*statics, *dynamics]:
        value = coerce_entry_value(descriptor, values.get(descriptor.name))
        if value is None:
            if descriptor.required and not partial:
                label = descriptor.label or descriptor.name
                raise EntryFieldError(descriptor.name, f"{label} is required")
            continue
        if descriptor.param_type == "dynamic_select" and check_live and adapter is not None:
            parents = {name: coerced[name] for name in descriptor.depends_on if name in coerced}
            choices = await fetch_entry_field_options(adapter, descriptor.name, parents)
            allowed = [str(choice.get("value")) for choice in choices]
            if value not in allowed:
                label = descriptor.label or descriptor.name
                listed = ", ".join(allowed[:MAX_LISTED_CHOICES])
                if not allowed:
                    problem = "has no choices available for the values given"
                elif len(allowed) > MAX_LISTED_CHOICES:
                    problem = f"must be one of: {listed}, ..."
                else:
                    problem = f"must be one of: {listed}"
                raise EntryFieldError(descriptor.name, f"{label} {problem}")
        coerced[descriptor.name] = value

    return {name: coerced[name] for name in declared if name in coerced}


@dataclass(frozen=True)
class ResolvedEntryWrite:
    """What one logbook-entry write uses, resolved by :func:`resolve_entry_write`.

    Attributes:
        logbook: Value for the request's ``logbook`` field.
        shift: Value for the request's ``shift`` field.
        adapter_metadata: Metadata handed to the facility adapter: the declared
            values only, plus the resolved ``logbook``/``shift`` when any field
            is declared; empty when none is.
        local_metadata: Metadata of ARIEL's own copy of the entry: the declared
            values, ``session_metadata`` when given, then ``logbook``, ``shift``,
            ``tags`` and ``created_via`` (when given), which no submitted value
            overrides.
            ``sync_status`` is never set here; the store that keeps the copy
            sets it.
    """

    logbook: str | None
    shift: str | None
    adapter_metadata: dict[str, Any] = field(default_factory=dict)
    local_metadata: dict[str, Any] = field(default_factory=dict)


def resolve_entry_write(
    descriptors: Sequence[ParameterDescriptor],
    declared: dict[str, Any],
    *,
    logbook: str | None,
    shift: str | None,
    tags: Sequence[str],
    created_via: str | None,
    session_metadata: dict[str, Any] | None = None,
) -> ResolvedEntryWrite:
    """Turn the inputs of one entry write into the request fields and metadata it uses.

    ``logbook`` and ``shift`` each come from the built-in input or from a
    declared field of the same name; either alone is used, and both present
    with different values is refused. A built-in value that is empty or only
    whitespace counts as absent. With no declarations the result is the write
    as it is without entry fields: the built-in ``logbook``/``shift`` unchanged,
    no adapter metadata, and a local copy of ``logbook``, ``shift``, ``tags``,
    ``created_via`` (plus ``session_metadata`` when given).

    Args:
        descriptors: The checked declarations (see :func:`entry_field_descriptors`).
        declared: Declared values as returned by :func:`validate_entry_fields`;
            any key that is not a declared field name is left out. Left unchanged.
        logbook: The built-in logbook input, if any.
        shift: The built-in shift input, if any.
        tags: The entry's tags, copied into the local metadata.
        created_via: Which ARIEL surface writes the entry, e.g. ``ariel-web``;
            ``None`` leaves the key out of the local metadata.
        session_metadata: Provenance of the writing session, kept in the local
            copy only.

    Returns:
        The resolved write; its dictionaries share nothing with the inputs.

    Raises:
        EntryFieldError: When a built-in and a declared ``logbook`` or ``shift``
            are both given and differ, naming that field.
    """
    by_name = {descriptor.name: descriptor for descriptor in descriptors}
    values = {name: copy.deepcopy(declared[name]) for name in by_name if name in declared}

    resolved: dict[str, str | None] = {}
    for name, builtin in (("logbook", logbook), ("shift", shift)):
        given = builtin if builtin is not None and builtin.strip() else None
        if name in values and values[name] is not None:
            chosen = values[name]
            if given is not None and given.strip() != str(chosen).strip():
                label = by_name[name].label or name
                raise EntryFieldError(
                    name,
                    f"{label} is given twice with different values: "
                    f"'{given.strip()}' and '{chosen}'",
                )
            resolved[name] = chosen
        else:
            resolved[name] = given if by_name else builtin

    adapter_metadata: dict[str, Any] = {}
    if by_name:
        adapter_metadata = dict(values)
        for name, value in resolved.items():
            if value is not None:
                adapter_metadata[name] = value
            else:
                adapter_metadata.pop(name, None)

    local_metadata: dict[str, Any] = copy.deepcopy(values)
    if session_metadata is not None:
        local_metadata["session_metadata"] = copy.deepcopy(session_metadata)
    local_metadata["logbook"] = resolved["logbook"]
    local_metadata["shift"] = resolved["shift"]
    local_metadata["tags"] = list(tags)
    if created_via is not None:
        local_metadata["created_via"] = created_via

    return ResolvedEntryWrite(
        logbook=resolved["logbook"],
        shift=resolved["shift"],
        adapter_metadata=adapter_metadata,
        local_metadata=local_metadata,
    )


def coerce_entry_value(descriptor: ParameterDescriptor, raw: Any) -> Any:
    """Turn one submitted value into the JSON-native value its declaration names.

    ``raw`` may be the native value or its string form (a form field, a query
    string, a JSON string). Surrounding whitespace in a string is ignored, and
    an empty string means the value is absent.

    Args:
        descriptor: The field's declaration.
        raw: The submitted value.

    Returns:
        ``None`` when the value is absent; otherwise an ``int``, ``float`` or
        ``bool`` for those types, a ``YYYY-MM-DD`` string for ``date``, and a
        string for ``text``, ``select`` and ``dynamic_select``.

    Raises:
        EntryFieldError: When the value does not fit the declaration: wrong
            type, outside ``min``/``max``, not one of a ``select``'s options,
            or a string longer than :data:`MAX_STRING_LENGTH` characters.
    """
    label = descriptor.label or descriptor.name

    def refuse(problem: str) -> EntryFieldError:
        return EntryFieldError(descriptor.name, f"{label} {problem}")

    if raw is None:
        return None
    if isinstance(raw, str):
        if len(raw) > MAX_STRING_LENGTH:
            raise refuse(f"must be at most {MAX_STRING_LENGTH} characters")
        raw = raw.strip()
        if not raw:
            return None

    param_type = descriptor.param_type
    if param_type == "int":
        value: Any = _coerce_int(raw, refuse)
        _check_bounds(descriptor, value, refuse)
        return value
    if param_type == "float":
        value = _coerce_float(raw, refuse)
        _check_bounds(descriptor, value, refuse)
        return value
    if param_type == "bool":
        return _coerce_bool(raw, refuse)
    if param_type == "date":
        return _coerce_date(raw, refuse)
    if param_type in ("text", "select", "dynamic_select"):
        if not isinstance(raw, str):
            raise refuse("must be text")
        if param_type == "select":
            allowed = [str(option.get("value")) for option in descriptor.options or []]
            if raw not in allowed:
                raise refuse(f"must be one of: {', '.join(allowed)}")
        return raw
    raise refuse(f"has an unknown type '{param_type}'")


_Refuse = Callable[[str], EntryFieldError]


def _coerce_int(raw: Any, refuse: _Refuse) -> int:
    if isinstance(raw, bool):
        raise refuse("must be a whole number")
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        if math.isfinite(raw) and raw.is_integer():
            return int(raw)
        raise refuse("must be a whole number")
    if isinstance(raw, str):
        try:
            return int(raw)
        except ValueError:
            pass
    raise refuse("must be a whole number")


def _coerce_float(raw: Any, refuse: _Refuse) -> float:
    if isinstance(raw, bool):
        raise refuse("must be a number")
    value: float | None = None
    if isinstance(raw, int | float):
        value = float(raw)
    elif isinstance(raw, str):
        try:
            value = float(raw)
        except ValueError:
            value = None
    if value is None or not math.isfinite(value):
        raise refuse("must be a number")
    return value


def _coerce_bool(raw: Any, refuse: _Refuse) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, int) and raw in (0, 1):
        return bool(raw)
    if isinstance(raw, str):
        word = raw.lower()
        if word in _TRUE_WORDS:
            return True
        if word in _FALSE_WORDS:
            return False
    raise refuse("must be true or false")


def _coerce_date(raw: Any, refuse: _Refuse) -> str:
    if isinstance(raw, datetime.datetime):
        raise refuse("must be a date (YYYY-MM-DD), not a date and time")
    if isinstance(raw, datetime.date):
        return raw.isoformat()
    if isinstance(raw, str) and _ISO_DATE.fullmatch(raw):
        try:
            return datetime.date.fromisoformat(raw).isoformat()
        except ValueError:
            pass
    raise refuse("must be a date (YYYY-MM-DD)")


def _check_bounds(descriptor: ParameterDescriptor, value: float, refuse: _Refuse) -> None:
    if descriptor.min_value is not None and value < descriptor.min_value:
        raise refuse(f"must be at least {descriptor.min_value:g}")
    if descriptor.max_value is not None and value > descriptor.max_value:
        raise refuse(f"must be at most {descriptor.max_value:g}")
