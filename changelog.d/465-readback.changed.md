A `callback`-level channel write now carries the post-write readback in its
result: the EPICS and Mock connectors read the channel once after the callback
confirms and report `readback_value` and the alarm state, while the callback
alone decides `verified`. The DOOCS connector no longer reports a string
readback as `readback_value=0.0`; non-numeric readbacks verify by equality and
carry no value. The safety rule that told the agent a callback result has no
readback is retired.
