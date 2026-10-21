`osprey-connectors` has its own test suite under `packages/osprey-connectors/tests`,
and a `connectors-standalone` CI lane runs it with only that distribution
installed, the first step toward releasing it on its own cycle. The framework-free
test guards the two suites share moved out of `tests/conftest.py` into
`tests/_connectors_harness.py`; connector tests that need the framework stay in
`tests/connectors`.
