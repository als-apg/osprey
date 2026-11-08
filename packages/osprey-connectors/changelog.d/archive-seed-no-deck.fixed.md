The archive seed no longer fails a simulator view that has a model with no
deck. `osprey_connectors.simulation.status.is_error()` is the one rule for
which model statuses are errors: `ok` and `no deck` are not.
