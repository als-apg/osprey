`osprey-connectors` has its own test suite under `packages/osprey-connectors/tests`,
and a `connectors-standalone` CI lane runs it with only that distribution
installed, from a locked sync, the first step toward releasing it on its own
cycle. The suite is self-contained: the framework-free test guards and helpers
both suites share (the environment, timezone, logging and repository guards, the
xdist scheduler, the CI diagnostics, the exit report, the port reservations and
the control-context writers) moved out of `tests/` into
`packages/osprey-connectors/tests/osprey_connectors_tests/_harness`, which the
framework's tests import, and a guard test refuses any import of `tests` or
`osprey` from the connectors suite. Connector tests that need the framework stay
in `tests/connectors`. The CI diagnostics no longer record each test twice in
the controller's event log when a run collects both suites.
