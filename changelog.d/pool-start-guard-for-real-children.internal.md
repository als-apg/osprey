The notebook runtime reads its connector-host start bound when it builds its pool, so tests that spawn real children can widen the hang guard without changing the 30 s product default.
