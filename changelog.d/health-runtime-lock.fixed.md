Two health checks that both needed the control-system connector at the same
moment could each build one, leaving the loser of the race connected until the
process exited. `HealthRuntime` now serializes construction so a run holds
exactly one connector.
