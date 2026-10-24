`osprey-connectors` has its own test suite under `packages/osprey-connectors/tests`,
which a new `connectors-standalone` CI lane runs with only that distribution
installed. The framework-free guards and helpers both suites share moved into
its `_harness`, and the framework's connector tests that need `osprey` stay in
`tests/connectors`.
