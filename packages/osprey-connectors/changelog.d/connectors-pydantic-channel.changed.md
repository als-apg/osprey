`ChannelMetadata`, `ChannelValue`, `ChannelWriteResult` and `ChannelLimitsConfig`
are pydantic models now, mutable as before, and `LimitsPosture` is a frozen one.
A connector builds them by keyword only: `ChannelMetadata("mA", 3)` and
`LimitsPosture(True, "exclusive", None)` raise `TypeError`, so write
`ChannelMetadata(units="mA", precision=3)`. Every field holds a value of its own
type: a `units` that is not a `str`, a `precision` or `alarm_severity` that is
not an `int`, a `timestamp` that is not a `datetime`, or a `channel_address`
that is not a `str` now raises `pydantic.ValidationError` (a `ValueError`) where
the dataclass stored it. The limit pairs, `writable` and `raw_metadata` still
take whatever a transport reported and normalise it, as before. A field
assignment is validated like the construction: `ChannelMetadata` normalises a
limit pair on assignment too, and a `ChannelWriteResult` refuses an assignment
that breaks its outcome rules, which a dataclass let through. Assigning to a
`LimitsPosture` raises `ValidationError` instead of `FrozenInstanceError`.
`dataclasses.fields`, `asdict` and `replace` no longer apply: use
`Model.model_fields`, `model.model_dump()` and `osprey_connectors.model.evolve(record, **changes)`. A limit read
from the limits database keeps the `int` or `float` it was written as.
