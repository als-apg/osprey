"""Role-typed channel fields: how a plan author *declares* what each parameter does.

A plan's ``PARAMS`` model already names the channels a run touches, but a
bare ``list[str]`` says nothing about what will be done to them. Everything
downstream -- the load gate, the enqueue pre-check, dry-run mocks, the default
figure, the pre-flight summary -- needs to know which channels the plan *moves*,
which it *triggers* and which it merely *reads*, and until now each of those
consumers guessed from the field's name. This module replaces the guessing with
a declaration.

An author annotates the field with one of the role types::

    class PARAMS(BaseModel):
        correctors: MovableChannels = Field(..., min_length=1, title="Correctors")
        monitors: ReadableChannels = Field(..., min_length=1)

Each type is a plain ``list[str]`` / ``str`` carrying one extra JSON-schema key,
``x-channel-role``, whose value is ``"movable"``, ``"triggered"`` or
``"readable"``. The singles (`MovableChannel` / `TriggeredChannel` /
`ReadableChannel`) exist for nested models, where one field holds one channel
name rather than a list. A parameter that may be left unset takes the optional
singles (`OptionalMovableChannel` / `OptionalTriggeredChannel` /
`OptionalReadableChannel`) -- their own types rather than a ``| None`` at the
field site, because pydantic only lifts an ``Annotated`` declaration into the
field's own metadata when it is the outermost annotation: spelled
``MovableChannel | None``, the role silently reads back as no declaration at
all, which is a plan whose channel no consumer collects (pinned by this
module's tests).

Two properties consumers rely on:

- **The role is additive.** It constrains nothing and validates nothing; it only
  records intent. An author still spells out ``min_length``, ``title``,
  ``description`` and any ``x-widget`` presentation hint on the field itself, and
  those survive alongside the role -- pydantic merges the annotation's
  ``json_schema_extra`` with the field's own rather than replacing it (pinned by
  this module's tests, because two shipped plans depend on it).
- **The role is introspectable from both sides.** It lands in
  ``model_fields[name].json_schema_extra`` for code walking the model, and under
  ``x-channel-role`` in ``model_json_schema()`` for anything reading the
  serialized schema -- nested inside the ``$defs`` entry when the annotated field
  lives on a nested model. Consumers that resolve ``$ref``s therefore see it too.

Reading the declaration back is `channel_roles` (what a model declares) and
`collect_channels` (what one set of parameters supplies for a role) -- the
single authority every consumer uses instead of inspecting field names.
`resolve_column` is the matching lookup on the other end, finding a channel's
column in a run's data.

Vocabulary: this module speaks only capability language -- a channel is
*movable* (the plan drives it to a value), *triggered* (the plan fires it,
waits for its act to end and records it) or *readable* (the plan records it).
That is the vocabulary of every author-facing and operator-facing surface;
`scan_metadata` is the one function that translates it into the run's standard
metadata keys, and no consumer spells those keys itself.

Kept pydantic-only (no bluesky/ophyd/tiled imports) so every consumer on the
bridge's import-clean side -- the loader, the queue routes, the figure route --
can import it freely.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Annotated, Any, Final, Literal, get_args, get_origin

from pydantic import BaseModel, Field, RootModel, TypeAdapter, ValidationError
from pydantic.fields import FieldInfo
from pydantic_core import to_jsonable_python

CHANNEL_ROLE_KEY: Final = "x-channel-role"
"""JSON-schema key under which a field's declared channel role is serialized."""

ChannelRole = Literal["movable", "triggered", "readable"]
"""What a plan does with a channel: drives it to a value, fires it and waits
for its act, or records it."""

MOVABLE_ROLE: Final[ChannelRole] = "movable"
"""Role of a channel the plan drives to a value."""

TRIGGERED_ROLE: Final[ChannelRole] = "triggered"
"""Role of a channel the plan fires, waits for and records; it never sets it."""

READABLE_ROLE: Final[ChannelRole] = "readable"
"""Role of a channel the plan records without driving."""


def _role_extra(role: ChannelRole) -> dict[str, str]:
    """One field's role as a fresh ``json_schema_extra`` mapping.

    Fresh per call: pydantic merges a field's own ``json_schema_extra`` into the
    annotation's dict, so a shared dict would let one plan's presentation hint
    leak into every other plan using the same role type.
    """
    return {CHANNEL_ROLE_KEY: role}


MovableChannels = Annotated[list[str], Field(json_schema_extra=_role_extra(MOVABLE_ROLE))]
"""A list of channel names the plan drives to values."""

TriggeredChannels = Annotated[list[str], Field(json_schema_extra=_role_extra(TRIGGERED_ROLE))]
"""A list of channel names the plan fires, waits for and records."""

ReadableChannels = Annotated[list[str], Field(json_schema_extra=_role_extra(READABLE_ROLE))]
"""A list of channel names the plan records at every point."""

MovableChannel = Annotated[str, Field(json_schema_extra=_role_extra(MOVABLE_ROLE))]
"""One channel name the plan drives to values -- the single-channel form of
`MovableChannels`, for a nested model holding one channel per entry."""

TriggeredChannel = Annotated[str, Field(json_schema_extra=_role_extra(TRIGGERED_ROLE))]
"""One channel name the plan fires, waits for and records -- the single-channel
form of `TriggeredChannels`, for a nested model holding one channel per entry."""

ReadableChannel = Annotated[str, Field(json_schema_extra=_role_extra(READABLE_ROLE))]
"""One channel name the plan records -- the single-channel form of
`ReadableChannels`, for a nested model holding one channel per entry."""

OptionalMovableChannel = Annotated[str | None, Field(json_schema_extra=_role_extra(MOVABLE_ROLE))]
"""`MovableChannel` for a parameter that may be left unset.

Its own type because ``MovableChannel | None`` would not work: pydantic lifts
an ``Annotated`` declaration into the field's metadata only when it is the
outermost annotation, so inside a union the role silently vanishes. Here the
``Annotated`` wraps the union itself, which keeps the role at the field level.
An unset (``None``) value simply contributes no channel -- `collect_channels`
skips it, so nothing downstream needs to special-case absence."""

OptionalTriggeredChannel = Annotated[
    str | None, Field(json_schema_extra=_role_extra(TRIGGERED_ROLE))
]
"""`TriggeredChannel` for a parameter that may be left unset -- see
`OptionalMovableChannel` for why the union lives inside the annotation."""

OptionalReadableChannel = Annotated[str | None, Field(json_schema_extra=_role_extra(READABLE_ROLE))]
"""`ReadableChannel` for a parameter that may be left unset -- see
`OptionalMovableChannel` for why the union lives inside the annotation."""


# ---------------------------------------------------------------------------
# Reading the declaration back
# ---------------------------------------------------------------------------

_CONTAINER_ORIGINS: Final = (list, tuple, set, frozenset, dict)
"""Annotation origins whose arguments hold *many* values rather than one.

Crossing one of these is what puts ``[]`` in a field path: ``axes: list[Axis]``
reaches ``Axis.setpoint`` as ``axes[].setpoint``. ``Union``/``Optional`` and
``Annotated`` are deliberately absent -- they re-spell one value, they do not
multiply it.
"""


def _field_role(field: FieldInfo) -> ChannelRole | None:
    """The role *field* declares, or ``None`` if it declares none.

    Anything other than one of the three known roles reads as no declaration: a
    field carrying a misspelled role is a field whose channels no consumer
    collects, which is the direction this has to fail.
    """
    extra = field.json_schema_extra
    if not isinstance(extra, dict):
        # A callable ``json_schema_extra`` mutates the schema at generation
        # time and cannot be read statically; no role type uses one.
        return None
    role = extra.get(CHANNEL_ROLE_KEY)
    for known in (MOVABLE_ROLE, TRIGGERED_ROLE, READABLE_ROLE):
        if role == known:
            return known
    return None


def _nested_models(annotation: Any, *, through_container: bool) -> Iterator[tuple[type[Any], bool]]:
    """Every ``BaseModel`` reachable from *annotation*, and whether a container
    was crossed to reach it.

    Recurses through generic arguments, so ``list[Axis]``, ``Axis | None`` and
    ``dict[str, list[Axis]]`` all reach ``Axis``.
    """
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        yield annotation, through_container
        return
    origin = get_origin(annotation)
    if origin is None:
        return
    crossed = through_container or origin in _CONTAINER_ORIGINS
    for arg in get_args(annotation):
        yield from _nested_models(arg, through_container=crossed)


def _walk_roles(
    model_cls: type[BaseModel], prefix: str, seen: frozenset[type[BaseModel]]
) -> Iterator[tuple[str, ChannelRole]]:
    """`channel_roles`' recursion: depth-first, in field-declaration order."""
    for name, field in model_cls.model_fields.items():
        path = f"{prefix}{name}"
        role = _field_role(field)
        if role is not None:
            yield path, role
        for nested, through_container in _nested_models(field.annotation, through_container=False):
            if nested in seen:
                # A model that contains itself (directly or through a cycle)
                # would otherwise recurse forever; its fields were already
                # yielded once, higher up this same path.
                continue
            sub_prefix = f"{path}[]." if through_container else f"{path}."
            yield from _walk_roles(nested, sub_prefix, seen | {nested})


def channel_roles(model_cls: type[BaseModel]) -> list[tuple[str, ChannelRole]]:
    """Every role-declaring field in *model_cls*, as ``(path, role)`` pairs.

    The single authority on what a plan declared. Consumers ask this rather
    than reading field names, which is the whole point of the contract::

        >>> channel_roles(GridScanParams)          # doctest: +SKIP
        [('monitors', 'readable'), ('axes[].setpoint', 'movable')]

    Paths are depth-first in declaration order and readable enough to name in
    an operator-facing message: a nested model reached through a list contributes
    ``field[].subfield``, one reached directly contributes ``field.subfield``.
    The path is for *reporting*; code matching values against a params dict wants
    `collect_channels`, whose rule is the leaf field name alone.

    Only the role types declare a role, so a plan that declares none comes
    back empty -- that is what the load gate reads to quarantine a plan claiming
    to write without naming anything it moves.
    """
    return list(_walk_roles(model_cls, "", frozenset({model_cls})))


def _channel_values(value: Any, keys: frozenset[str], *, key: str | None) -> Iterator[str]:
    """Every string in *value* whose NEAREST enclosing field name is in *keys*.

    The enclosing name is rebound at every mapping level and carried unchanged
    across list/tuple/set levels, so a string counts only when the field
    *immediately* above it declared a role. That "nearest" is the safety
    property: for params like ``{"axes": [{"setpoint": "COR1", "mode": "fast"}]}``
    where only ``setpoint`` is role-declared, a sticky "anywhere under a
    role-declared field" rule would also collect ``"fast"`` and refuse a
    perfectly good enqueue over a mode string -- a false refusal no agent can
    fix by changing the channel name. Rebinding at each level makes that a miss
    instead, which is the direction a convenience check must fail.
    """
    if isinstance(value, str):
        if key is not None and key in keys:
            yield value
        return
    if isinstance(value, BaseModel):
        yield from _channel_values(value.model_dump(), keys, key=key)
        return
    if isinstance(value, Mapping):
        for sub_key, sub_value in value.items():
            yield from _channel_values(sub_value, keys, key=str(sub_key))
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            yield from _channel_values(item, keys, key=key)


def collect_channels(model_cls: type[BaseModel], params: Any, role: ChannelRole) -> list[str]:
    """The channel names *params* supplies for *role*, in the order they appear.

    *params* is one plan's parameters -- the enqueued dict, or a validated model
    instance -- and *model_cls* the schema they belong to. Matching is by exact
    field name: `channel_roles` says which field names carry *role*, and a string
    counts when the field immediately enclosing it is one of them. Nested shapes
    work without any consumer knowing a plan's layout, so both a top-level
    ``correctors: MovableChannels`` and a nested ``axes[].setpoint`` come back
    from the same call.

    Duplicates are dropped (first occurrence wins) so a channel named twice is
    reported once. A params dict that simply doesn't carry a declared field
    contributes nothing rather than raising: this is what consumers use to
    describe an enqueue, and a description must not be the thing that refuses it.
    """
    keys = frozenset(
        path.rpartition(".")[2] for path, declared in channel_roles(model_cls) if declared == role
    )
    if not keys:
        return []
    ordered: dict[str, None] = {}
    for name in _channel_values(params, keys, key=None):
        ordered.setdefault(name, None)
    return list(ordered)


def declared_channels(model_cls: type[BaseModel], params: Any) -> list[str]:
    """Every channel *params* supplies under ANY role -- movable, then triggered,
    then readable, deduped.

    `collect_channels` for consumers that need the plan's whole claim rather
    than one role of it. The worker sides use it as a *bound*: the device
    mapping a plan is handed holds exactly these names, so a plan cannot reach
    a device it never declared. Both the catalog worker (`qserver_startup`) and
    the session-plan wrapper (`session_upload`) draw that bound, and drawing it
    from one function is what keeps them from drifting apart -- a role added to
    `ChannelRole` widens both at once, where two open-coded unions would widen
    whichever one someone remembered.

    A channel declared under several roles appears once, at its first position.
    A model declaring no roles contributes nothing, exactly as `collect_channels`
    does, so a plan file with no ``PARAMS`` bounds itself to no devices at all.
    """
    ordered: dict[str, None] = {}
    for role in (MOVABLE_ROLE, TRIGGERED_ROLE, READABLE_ROLE):
        for name in collect_channels(model_cls, params, role):
            ordered.setdefault(name, None)
    return list(ordered)


# ---------------------------------------------------------------------------
# The one translation into run metadata
# ---------------------------------------------------------------------------


def scan_metadata(
    movable: Sequence[str],
    readable: Sequence[str],
    points: int,
    *,
    triggered: Sequence[str] = (),
) -> dict[str, Any]:
    """Standard run metadata for a plan that moves *movable*, triggers *triggered*
    and reads *readable*.

    A plan passes the result as the ``md`` of its own run decorator, and the run
    it opens then carries the same standard description a stock scan carries.
    That is what lets analysis and data-access tooling written against the wider
    ecosystem read an OSPREY run at all: those consumers look for the standard
    keys, so a run that omits them is one they can only treat as an opaque table.

    **This function is the only place in the codebase that spells those keys.**
    Everywhere else -- authoring, error messages, docs, this module's own
    parameter names -- speaks the capability vocabulary a plan author uses:
    *movable* channels are the ones the plan drives to values, *triggered*
    channels the ones it fires and records, *readable* channels the ones it
    records. A triggered channel is recorded, so it is described beside the
    readable ones. Consumers that need a role read it from the
    declaration via `channel_roles`/`collect_channels`, never from these keys.

    Deliberately not emitted:

    - an interval count, which stock plans derive as ``points - 1``: true only
      when the points are one continuous traversal, and false for a plan that
      concatenates several sweeps into one run.
    - a dimensionality hint, for the same reason -- a serial per-channel sweep
      has no truthful value to give, and a plan wrapping a stock plan inherits
      the stock hint anyway.

    Both are omissions, not stubs: a consumer finding no hint falls back to
    something honest, where one finding a fabricated hint draws the wrong picture.

    Args:
        movable: Channel names the plan drives, in the order it drives them.
        readable: Channel names the plan records at every point.
        points: Total points the run will emit, counting every sweep.
        triggered: Channel names the plan fires, waits for and records.

    Raises:
        ValueError: if the plan neither moves nor triggers a channel, records
            none, or *points* is below one. Such a run is an authoring mistake
            worth failing on rather than a run worth opening.
    """
    if not movable and not triggered:
        raise ValueError(
            "scan_metadata needs at least one movable or triggered channel "
            "(the plan drives and triggers none)"
        )
    if not readable and not triggered:
        raise ValueError(
            "scan_metadata needs at least one readable or triggered channel (the plan records none)"
        )
    if points < 1:
        raise ValueError(f"scan_metadata needs at least one point, got {points}")
    return {
        "motors": list(movable),
        "detectors": [*triggered, *readable],
        "num_points": int(points),
    }


# ---------------------------------------------------------------------------
# Finding a channel's column in a run's data
# ---------------------------------------------------------------------------


def resolve_column(channel_name: str, columns: Iterable[str]) -> str | None:
    """The data column holding *channel_name*'s readings, or ``None``.

    One channel reaches a run's data under either of two spellings, because two
    device implementations answer to the same channel name. A device that
    declares child signals is read under ``f"{channel}-{signal}"`` -- the mock
    devices (`devices/mock.py`) do this, so a mock settable lands under
    ``"sp1-readback"``. A device that reads through the OSPREY connector
    (`devices/connector.py`) emits one entry named for the device itself, so
    the same channel lands under ``"sp1"``, plus its demand under
    ``f"sp1{SETPOINT_KEY_SUFFIX}"`` when it has a distinct readback (see
    `resolve_regressor_column`). Neither module is imported here; this is a
    rule about the data keys they produce.

    So: exact match first, then the single ``f"{channel_name}-"`` prefix rule,
    taking the first match in column order. ``None`` means the run's data has no
    column for this channel -- a channel that was never read, or a window read
    before it appeared -- and every caller treats that as "no reading", not as
    an error.

    *columns* may be any iterable of column names, including a data row itself
    (iterating a mapping yields its keys), so a caller holding one row can pass
    it directly.
    """
    prefix = f"{channel_name}-"
    fallback: str | None = None
    for column in columns:
        if column == channel_name:
            return column
        if fallback is None and column.startswith(prefix):
            fallback = column
    return fallback


SETPOINT_KEY_SUFFIX: Final = "_setpoint"
"""Suffix of the data key a connector settable with a *distinct* readback
reports its demand under: ``<name>_setpoint`` beside ``<name>``.
The same convention as ophyd's positioners (``<name>_setpoint`` beside the
``<name>`` readback), so a plan that settle-checks a slow device -- an
insertion-device gap, a ramping magnet -- reads where the device is and where
it was told to go off one device, without a second device aliasing the
setpoint channel."""

Regressor = Literal["setpoint", "readback"]
"""Which of a settable's values a response fit regresses on: the commanded
demand (``setpoint``) or the measured value (``readback``)."""


def resolve_regressor_column(
    channel_name: str, columns: Iterable[str], regressor: Regressor
) -> str | None:
    """The data column a response fit reads *channel_name*'s value from, or ``None``.

    ``readback`` resolves the channel's own column through `resolve_column`.
    ``setpoint`` takes the exact ``f"{channel_name}{SETPOINT_KEY_SUFFIX}"``
    column when the run carries one, and otherwise falls back to
    `resolve_column`: a device that reports no separate demand (an aliased
    connector device, a mock settable) reports one column, and that column is
    its demand.

    *columns* may be any iterable of column names, including a data row itself.
    """
    columns = list(columns)
    if regressor == "setpoint":
        demand = f"{channel_name}{SETPOINT_KEY_SUFFIX}"
        if demand in columns:
            return demand
    return resolve_column(channel_name, columns)


def validate_field_value(model: type[BaseModel], key: str, value: Any) -> Any:
    """Coerce and validate *value* against *model*'s field *key*; return it JSON-safe.

    The one judge of "is this value allowed for this field": a value an
    operator or agent types into a draft and a default the plan itself declares
    pass the same check, so a plan's own bounds hold whoever supplied the value.

    The adapter is ``TypeAdapter(Annotated[field.annotation, field])`` -- the
    field's own ``FieldInfo`` as the ``Annotated`` metadata, never unpacked with
    a leading ``*``: the starred form raises ``TypeError`` for a field with no
    ``Field()``-level constraints (an empty ``metadata`` list, and ``Annotated``
    requires at least one argument), while the ``FieldInfo`` itself works
    whether or not the field carries constraints. The coerced value is dumped
    with ``mode="json"``, so a nested-model field comes back as plain dicts,
    never live model instances.

    Raises `KeyError` for a field *model* does not declare; a value the field
    refuses raises pydantic's `ValidationError` unchanged.
    """
    field = model.model_fields.get(key)
    if field is None:
        raise KeyError(key)
    adapter: TypeAdapter[Any] = TypeAdapter(Annotated[field.annotation, field])
    return adapter.dump_python(adapter.validate_python(value), mode="json")


def _stated(value: Any) -> Any:
    """*value* as JSON-safe data, every model field read by its name.

    Unlike a model dump, a field declared with ``exclude=True`` is stated: it
    is still a parameter of the run, and a set that left it out would make the
    next validation fill it again.
    """
    if isinstance(value, RootModel):
        return _stated(value.root)
    if isinstance(value, BaseModel):
        return {name: _stated(getattr(value, name)) for name in type(value).model_fields}
    if isinstance(value, Mapping):
        return {to_jsonable_python(key): _stated(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_stated(item) for item in value]
    return to_jsonable_python(value)


def _validated(model: type[BaseModel], args: Mapping[str, Any]) -> dict[str, Any]:
    """*args* validated by *model* and stated field by field.

    ``ValidationError`` passes through; any other exception a default factory
    or validator raises becomes a ``ValueError`` naming it, so a model that
    cannot judge a set refuses it like one it judged invalid.
    """
    try:
        stated: dict[str, Any] = _stated(model.model_validate(args))
    except ValidationError:
        raise
    except Exception as exc:
        raise ValueError(f"the plan's model raised {type(exc).__name__}: {exc}") from exc
    return stated


def resolve_plan_args(model: type[BaseModel], args: Mapping[str, Any]) -> dict[str, Any]:
    """Validate *args* against *model* as a whole and return the complete set it states.

    The one place a plan's defaults are filled for a run: every field the
    caller left out is stated with its default, a default factory is called
    once, and the model's cross-field rules are judged on the set as a whole.

    Invariants of the returned set:

    - It validates to itself: validating it again against *model* fills no
      default, calls no factory and states the same set. A set for which that
      does not hold (a field read by an alias, a validator that rewrites its
      input) is refused here,
      because whoever validates it next would read something other than what
      was checked.
    - Its values are JSON-safe.
    - Its keys are *model*'s field names, in declaration order, including a
      field the model excludes from its dump.

    Raises:
        ValidationError: *args* is not a valid set for *model*; pydantic's own
            error, unchanged.
        ValueError: a default factory or validator raised something other than
            a validation error, or the validated set does not validate back to
            itself; the message names the error or the first field that differs.
    """
    resolved = _validated(model, args)
    try:
        again = _validated(model, resolved)
    except ValidationError as exc:
        error = exc.errors()[0]
        field = ".".join(str(part) for part in error["loc"]) or "<model>"
        raise ValueError(
            f"the validated arguments do not validate again: {field}: {error['msg']}"
        ) from exc
    if again != resolved:
        absent = object()
        field = next(
            key
            for key in (*resolved, *again)
            if resolved.get(key, absent) != again.get(key, absent)
        )
        raise ValueError(f"the validated arguments do not validate to themselves: {field}")
    return resolved
