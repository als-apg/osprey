The simulator's apply and session-artifact readers type archiver events as
read-only mappings of `object` and pass them to the connectors' series helpers
without copying; their behaviour is unchanged.
