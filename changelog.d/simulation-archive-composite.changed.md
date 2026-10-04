The archiver seeder module is now `osprey_connectors.simulation.archive`; `osprey.simulation.archiver_seed` imports from it.
`archive.build(view, active_set)` builds a simulator view's history at one active scenario set's start state: archiver events, drift, keyed noise and clamp per timestamp, with `bool` and `enum` channels as option indices.
