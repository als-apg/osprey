The test suite's own vocabulary is made consistent: the benchmark board scores a
`capability lane` and a `harness lane` (no longer "Tier 1" and "Tier 0", which
collided with the CI tiers), tests that hold a wall-clock budget carry one
`wall_clock` marker and run alone in a serial step on every push, the `e2e`
marker is gone (the directory is the selector), and the harness's own unit tests
move out of `tests/e2e/`. A new Architecture page, *Testing*, draws the whole
concept.
