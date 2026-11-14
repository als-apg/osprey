A DOOCS or TANGO connector, or an EPICS read through aioca, used before
`connect()` raises `ConnectorUnavailableError` instead of `AttributeError`; the
MYA and MongoDB archivers raise `RuntimeError` saying they are not connected. A
service row field of the wrong type reads as unset instead of passing through:
a time, severity, status, message, user or client that is not a string, and an
alarm `enabled` or `latching` that is not a boolean. An alarm replay file whose
message key is neither a string nor null fails to load, naming the line. A
MongoDB waveform document whose value is not a list is refused as no stored
trace, and a document date that is not a datetime counts as undated. A
connector host's `validate_channel` reply that is not a boolean is refused.
