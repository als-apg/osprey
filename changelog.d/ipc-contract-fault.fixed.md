A connector bug inside the connector-host child is reported as an internal
error; it no longer restarts the connector or reads as an unreachable control
system. A reply the connector-host child sends that cannot be rebuilt fails only
that call, the same way, and the connection stays up.
