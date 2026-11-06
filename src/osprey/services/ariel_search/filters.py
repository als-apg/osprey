"""The filter dimensions an ARIEL entry list can be narrowed by.

A filter dimension is one facet of the browse rail and one key of the filter
object: the built-ins ``author``, ``source_system`` and ``tags``, every entry
field the facility adapter declares with ``facet: true``, and ``enhancement``,
whose values are the enabled enhancement modules. ``time`` is a window, not a
facet, and is never a dimension.

Public surface:

* :class:`FilterDimension` -- one dimension: its name, label, kind, the SQL
  expression it is matched against, and whether that column holds a list.
* :func:`filter_dimensions` -- the dimensions of one configuration, in rail
  order.
* :class:`EntryFilter` -- one parsed filter object: values per dimension (OR
  within a dimension, AND across them, exact ignoring case) and a ``time``
  window, matched against an entry by :meth:`EntryFilter.matches` and turned
  into WHERE clauses by :meth:`EntryFilter.to_sql`; the two agree on every
  stored entry.
* :class:`FilterValidationError` -- a filter object refused by
  :meth:`EntryFilter.parse`, naming the offending field and the reason.
* :func:`encode_cursor` / :func:`decode_cursor` -- the opaque keyset cursor of
  an entry list ordered ``timestamp DESC, entry_id DESC``.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from osprey.services.ariel_search.enhancement.status import (
    SQL_STATES,
    EnhancementContext,
    _check_alias,
    enabled_modules,
)
from osprey.services.ariel_search.entry_fields import (
    EntryFieldDeclarationError,
    entry_field_descriptors,
)

if TYPE_CHECKING:
    from osprey.services.ariel_search.config import ARIELConfig

__all__ = [
    "BUILTIN_DIMENSIONS",
    "EntryFilter",
    "FilterDimension",
    "FilterValidationError",
    "TIME_KEY",
    "decode_cursor",
    "encode_cursor",
    "filter_dimensions",
]

logger = logging.getLogger(__name__)

DimensionKind = Literal["builtin", "declared", "enhancement"]


@dataclass(frozen=True)
class FilterDimension:
    """One dimension an entry list can be filtered and faceted by.

    Attributes:
        name: The filter-object key (``author``, a declared field name, ...).
        label: Human-readable label for the rail.
        kind: ``builtin``, ``declared`` (a facility entry field) or
            ``enhancement``.
        column_sql: SQL expression over ``enhanced_entries`` the dimension's
            values are matched against.
        multi: Whether the column holds a list of values (``tags``).
        values: The allowed values; set only on ``enhancement``, where they are
            the enabled modules.
    """

    name: str
    label: str
    kind: DimensionKind
    column_sql: str
    multi: bool
    values: tuple[str, ...] = ()

    def entry_value(self, entry: Mapping[str, Any]) -> Any:
        """The entry's raw value of this dimension, or None when absent.

        A scalar built-in reads the entry's top-level column; ``tags`` and a
        declared field read their ``metadata`` key. Not meaningful for
        ``enhancement``, whose state is computed (:func:`enhancement_view`).
        """
        if self.kind == "builtin" and not self.multi:
            return entry.get(self.name)
        metadata = entry.get("metadata")
        return metadata.get(self.name) if isinstance(metadata, Mapping) else None


#: The dimensions every configuration has, in rail order.
BUILTIN_DIMENSIONS: tuple[FilterDimension, ...] = (
    FilterDimension("author", "Author", "builtin", "author", multi=False),
    FilterDimension("source_system", "Source System", "builtin", "source_system", multi=False),
    FilterDimension("tags", "Tags", "builtin", "metadata->'tags'", multi=True),
)

# One-time guard so a misdeclared adapter does not log on every request.
_declaration_fallback_warned = False


def _sql_literal(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def filter_dimensions(config: ARIELConfig) -> list[FilterDimension]:
    """Return the filter dimensions of this configuration, in rail order.

    The order is the built-ins (``author``, ``source_system``, ``tags``), then
    every declared entry field with ``facet: true`` in declaration order, keyed
    ``metadata->>'<name>'``, then ``enhancement``, whose ``values`` are the
    enabled enhancement modules in registry order. ``time`` is never listed.

    When the adapter misdeclares its entry fields, the declared dimensions are
    left out and a warning is logged once, so browse, search and capabilities
    keep working.

    Args:
        config: ARIEL configuration.

    Returns:
        The dimensions, as a new list.
    """
    global _declaration_fallback_warned

    dimensions = list(BUILTIN_DIMENSIONS)
    try:
        descriptors = entry_field_descriptors(config)
    except EntryFieldDeclarationError as exc:
        descriptors = []
        if not _declaration_fallback_warned:
            _declaration_fallback_warned = True
            logger.warning(
                "Entry field declarations refused (%s); filtering offers the "
                "built-in dimensions only.",
                exc.message,
            )
    for descriptor in descriptors:
        if descriptor.facet is not True:
            continue
        dimensions.append(
            FilterDimension(
                name=descriptor.name,
                label=descriptor.label or descriptor.name,
                kind="declared",
                column_sql=f"metadata->>{_sql_literal(descriptor.name)}",
                multi=False,
            )
        )
    dimensions.append(
        FilterDimension(
            name="enhancement",
            label="Enhancements",
            kind="enhancement",
            column_sql="enhancement_status",
            multi=False,
            values=tuple(enabled_modules(config)),
        )
    )
    return dimensions


# ---------------------------------------------------------------------------
# The filter object
# ---------------------------------------------------------------------------

#: The reserved key of the time window; never a dimension.
TIME_KEY = "time"

_TIME_BOUNDS = ("start", "end")


class FilterValidationError(ValueError):
    """A filter object :meth:`EntryFilter.parse` refuses.

    Routes answer it with HTTP 422 ``{"detail": {"field", "reason"}}``.

    Attributes:
        field: The offending dimension, ``time``, or ``filter`` for the object
            as a whole.
        reason: Why it was refused, for a human.
    """

    def __init__(self, field: str, reason: str) -> None:
        super().__init__(f"{field}: {reason}")
        self.field = field
        self.reason = reason


#: The field a refused cursor is reported under.
CURSOR_FIELD = "cursor"


def encode_cursor(ts: datetime, entry_id: str) -> str:
    """The opaque cursor positioned just after the entry ``(ts, entry_id)``.

    Base64url, unpadded, of ``"<iso_ts>|<entry_id>"`` (C-PAGE). A list read with
    it continues with the entries strictly before that one in
    ``timestamp DESC, entry_id DESC`` order.

    Args:
        ts: The timestamp of the last entry of the page.
        entry_id: The id of the last entry of the page.

    Returns:
        The cursor string, safe in a URL query.
    """
    raw = f"{ts.isoformat()}|{entry_id}".encode()
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_cursor(cursor: str) -> tuple[datetime, str]:
    """The ``(timestamp, entry_id)`` an :func:`encode_cursor` cursor carries.

    Args:
        cursor: A cursor as :func:`encode_cursor` produced it.

    Returns:
        ``(ts, entry_id)``; ``ts`` keeps the offset it was encoded with.

    Raises:
        FilterValidationError: Field ``cursor``, when ``cursor`` is not
            base64url text of ``"<iso_ts>|<entry_id>"`` with a timezone-aware
            timestamp and a non-empty entry id.
    """
    if not isinstance(cursor, str) or not cursor:
        raise FilterValidationError(CURSOR_FIELD, "must be a non-empty string")
    try:
        text = base64.urlsafe_b64decode(cursor.encode("ascii") + b"=" * (-len(cursor) % 4)).decode(
            "utf-8"
        )
    except (UnicodeError, binascii.Error, ValueError) as exc:
        raise FilterValidationError(CURSOR_FIELD, "not a valid cursor") from exc
    stamp, sep, entry_id = text.partition("|")
    if not sep or not entry_id:
        raise FilterValidationError(CURSOR_FIELD, "not a valid cursor")
    try:
        ts = datetime.fromisoformat(stamp)
    except ValueError as exc:
        raise FilterValidationError(CURSOR_FIELD, "not a valid cursor") from exc
    if ts.tzinfo is None:
        raise FilterValidationError(CURSOR_FIELD, "not a valid cursor")
    return ts, entry_id


def _norm(value: Any) -> str | None:
    """A scalar column value as the lower-cased text SQL compares, else None."""
    if isinstance(value, str):
        return value.lower()
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value).lower()
    return None


def _entry_timestamp(value: Any) -> datetime | None:
    from osprey_connectors.config import localize_facility

    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return None
    if isinstance(value, datetime):
        return localize_facility(value)
    return None


def _parse_time(raw: Any) -> tuple[datetime | None, datetime | None]:
    from osprey.services.ariel_search.models import parse_time_bound

    if raw is None:
        return (None, None)
    if not isinstance(raw, Mapping):
        raise FilterValidationError(TIME_KEY, "must be an object with start and end")
    unknown = sorted(str(k) for k in raw if k not in _TIME_BOUNDS)
    if unknown:
        raise FilterValidationError(TIME_KEY, f"unknown key {unknown[0]!r}; expected start and end")
    bounds: list[datetime | None] = []
    for key in _TIME_BOUNDS:
        value = raw.get(key)
        if value is None or (isinstance(value, str) and not value.strip()):
            bounds.append(None)
            continue
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise FilterValidationError(TIME_KEY, f"{key} is not a date, a time or epoch seconds")
        try:
            bounds.append(parse_time_bound(value, end=key == "end"))
        except ValueError as exc:
            raise FilterValidationError(TIME_KEY, f"{key}: {exc}") from None
    start, end = bounds
    if start is not None and end is not None and start > end:
        raise FilterValidationError(TIME_KEY, "start is after end")
    return (start, end)


def _parse_values(dim: FilterDimension, raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, list):
        raise FilterValidationError(dim.name, "must be a list of values")
    values: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            raise FilterValidationError(dim.name, "every value must be a string")
        if not item.strip():
            raise FilterValidationError(dim.name, "a value may not be empty")
        if dim.kind == "enhancement":
            _check_enhancement_value(dim, item)
        if item not in values:
            values.append(item)
    return tuple(values)


def _check_enhancement_value(dim: FilterDimension, value: str) -> None:
    module, sep, state = value.partition(":")
    if not sep:
        raise FilterValidationError(dim.name, f"{value!r} is not <module>:<state>")
    if module not in dim.values:
        raise FilterValidationError(dim.name, f"unknown module {module!r}")
    if state not in SQL_STATES:
        raise FilterValidationError(
            dim.name, f"unknown state {state!r}; expected one of {', '.join(SQL_STATES)}"
        )


@dataclass(frozen=True)
class EntryFilter:
    """One parsed filter object (C-FILTER).

    Build it with :meth:`parse`. Values match exactly, ignoring case; a
    dimension's values are ORed and the dimensions, and the ``time`` window,
    are ANDed.

    Attributes:
        dims: ``(dimension name, values)`` pairs, in table order, for the
            dimensions that constrain; a dimension with no values is absent.
        start: The window's first instant, inclusive, or None.
        end: The window's last instant, inclusive, or None.
        table: The dimension definitions the filter was parsed against, keyed
            by name; :meth:`matches` and the SQL form read each dimension's
            kind, column and multiplicity from here.
    """

    dims: tuple[tuple[str, tuple[str, ...]], ...] = ()
    start: datetime | None = None
    end: datetime | None = None
    table: Mapping[str, FilterDimension] = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def parse(
        cls, raw: Mapping[str, Any] | str | None, dims: Sequence[FilterDimension]
    ) -> EntryFilter:
        """Parse the wire filter object against a dimension table.

        Args:
            raw: The object, its JSON text, or None (blank text and None are
                the empty filter).
            dims: The configuration's dimensions (:func:`filter_dimensions`);
                ``time`` is reserved and handled outside them.

        Returns:
            The filter.

        Raises:
            FilterValidationError: For text that is not JSON, a value that is
                not an object, an unknown dimension, a non-list value, an empty
                or non-string value, an unknown enhancement module or state, or
                a bad time window.
        """
        table = {d.name: d for d in dims}
        if raw is None:
            return cls(table=table)
        if isinstance(raw, str):
            if not raw.strip():
                return cls(table=table)
            try:
                raw = json.loads(raw)
            except ValueError:
                raise FilterValidationError("filter", "is not valid JSON") from None
        if not isinstance(raw, Mapping):
            raise FilterValidationError("filter", "must be a JSON object")

        for key in raw:
            if key != TIME_KEY and key not in table:
                raise FilterValidationError(str(key), "unknown filter dimension")
        start, end = _parse_time(raw.get(TIME_KEY))
        parsed = []
        for dim in dims:
            if dim.name not in raw:
                continue
            values = _parse_values(dim, raw[dim.name])
            if values:
                parsed.append((dim.name, values))
        return cls(dims=tuple(parsed), start=start, end=end, table=table)

    def is_empty(self) -> bool:
        """Whether the filter constrains nothing."""
        return not self.dims and self.start is None and self.end is None

    def values(self, dim: str) -> tuple[str, ...]:
        """The values of one dimension, ``()`` when it does not constrain."""
        return dict(self.dims).get(dim, ())

    def without(self, dim: str) -> EntryFilter:
        """This filter with one dimension, or ``time``, left out.

        A facet counts its own dimension under ``without(dim)`` (self-exclusion);
        the histogram counts under ``without("time")``.
        """
        if dim == TIME_KEY:
            return EntryFilter(dims=self.dims, table=self.table)
        return EntryFilter(
            dims=tuple((name, vals) for name, vals in self.dims if name != dim),
            start=self.start,
            end=self.end,
            table=self.table,
        )

    def matches(self, entry: Mapping[str, Any], ctx: EnhancementContext) -> bool:
        """Whether ``entry`` passes the filter.

        Built-ins are read from the entry's top-level keys, except ``tags``,
        which is the ``metadata.tags`` list; declared fields from
        ``metadata[name]``; ``enhancement`` from :func:`enhancement_view` of the
        entry's ``enhancement_status`` under ``ctx``.

        Args:
            entry: An entry as the repository returns it.
            ctx: The current markers and enabled modules.
        """
        if self.start is not None or self.end is not None:
            stamp = _entry_timestamp(entry.get("timestamp"))
            if stamp is None:
                return False
            if self.start is not None and stamp < self.start:
                return False
            if self.end is not None and stamp > self.end:
                return False
        return all(self._matches_dim(name, vals, entry, ctx) for name, vals in self.dims)

    def _matches_dim(
        self,
        name: str,
        values: tuple[str, ...],
        entry: Mapping[str, Any],
        ctx: EnhancementContext,
    ) -> bool:
        dim = self.table[name]
        if dim.kind == "enhancement":
            view = ctx.view(entry)
            for value in values:
                module, _, state = value.partition(":")
                if view.get(module) == state:
                    return True
            return False
        wanted = {v.lower() for v in values}
        column = dim.entry_value(entry)
        if dim.multi:
            return isinstance(column, list) and any(_norm(item) in wanted for item in column)
        return _norm(column) in wanted

    def to_sql(self, alias: str, ctx: EnhancementContext) -> tuple[list[str], list[Any]]:
        """The filter as WHERE clauses over ``enhanced_entries <alias>``.

        One clause per constraining dimension, plus one per time bound; the
        caller ANDs them (``TRUE`` when there are none). The clauses select
        exactly the stored entries :meth:`matches` accepts under ``ctx``:

        * a scalar dimension is ``lower(<alias>.<column>) = ANY(%s)`` over the
          lower-cased values;
        * ``tags`` is an ``EXISTS`` over the elements of ``metadata->'tags'``,
          compared lower-cased, guarded by ``jsonb_typeof(...) = 'array'``;
        * each ``enhancement`` value is
          :func:`~osprey.services.ariel_search.enhancement.status.enhancement_state_sql`
          (``FALSE`` for a module ``ctx`` does not enable), ORed;
        * ``time`` is ``<alias>.timestamp >= %s`` / ``<= %s``, both inclusive.

        Args:
            alias: The alias of ``enhanced_entries`` in the enclosing query.
            ctx: The current markers, enabled modules and schema facts.

        Returns:
            ``(clauses, params)``: fragments with positional ``%s`` placeholders
            and their parameters, in order. Every value is bound, never inlined.

        Raises:
            ValueError: If ``alias`` is not a plain SQL identifier (or is ``f``
                while an image module is filtered on).
        """
        _check_alias(alias)
        clauses: list[str] = []
        params: list[Any] = []
        if self.start is not None:
            clauses.append(f"{alias}.timestamp >= %s")
            params.append(self.start)
        if self.end is not None:
            clauses.append(f"{alias}.timestamp <= %s")
            params.append(self.end)
        for name, values in self.dims:
            dim = self.table[name]
            if dim.kind == "enhancement":
                clause, dim_params = self._enhancement_sql(values, alias, ctx)
            else:
                column = f"{alias}.{dim.column_sql}"
                lowered = [v.lower() for v in values]
                if dim.multi:
                    clause = (
                        f"(jsonb_typeof({column}) = 'array' AND EXISTS ("
                        f"SELECT 1 FROM jsonb_array_elements_text({column}) t"
                        " WHERE lower(t) = ANY(%s)))"
                    )
                else:
                    clause = f"lower({column}) = ANY(%s)"
                dim_params = [lowered]
            clauses.append(clause)
            params += dim_params
        return clauses, params

    @staticmethod
    def _enhancement_sql(
        values: tuple[str, ...], alias: str, ctx: EnhancementContext
    ) -> tuple[str, list[Any]]:
        parts: list[str] = []
        params: list[Any] = []
        for value in values:
            module, _, state = value.partition(":")
            if module not in ctx.enabled:
                parts.append("FALSE")
                continue
            sql, part_params = ctx.state_sql(module, state, alias)
            parts.append(sql)
            params += part_params
        return "(" + " OR ".join(parts) + ")", params
