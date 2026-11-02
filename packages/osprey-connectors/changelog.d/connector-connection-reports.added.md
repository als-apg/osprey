A connector subscription can now say whether its channel is connected:
`subscribe()` takes a keyword-only `on_connection` callback, which EPICS CA,
PVA and the in-process simulator call with `True` before the first value and
`False` on a drop (DOOCS and Tango accept it and never call it).
`ChannelMetadata` now carries the control, warning and alarm limit pairs and
whether this client may write the channel; a bound that is not a finite
number, or a pair whose low is not below its high, reads as `None`.
