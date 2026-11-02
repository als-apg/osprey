`subscribe(..., on_connection=...)` works on every connector, including one a
facility wrote before the keyword existed: when a subclass's `subscribe` does
not take `on_connection`, the base class passes it a wrapped callback and
reports `True` before the channel's first value (never `False`, since such a
connector never says a channel dropped). A `subscribe` that takes the keyword
receives it unchanged. The wrapper is a deprecated compatibility path: defining
such a subclass emits a `DeprecationWarning` naming it, and the wrapper will be
removed in osprey-connectors 2027.3.0.
