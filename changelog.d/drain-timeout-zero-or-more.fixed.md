A `control_system.target_switch.drain_timeout_s` that is an integer beyond
float range, NaN, infinity, `true` or negative now falls back to the 5 s
default with a warning, instead of raising `OverflowError`, waiting forever
or tearing in-flight work down at once. `0` still tears the old target down
without waiting.
