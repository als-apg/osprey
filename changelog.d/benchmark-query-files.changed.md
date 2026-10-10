The shipped benchmark query files are `in_context_queries.json` and
`tree_queries.json` under `data/benchmarks/cross_paradigm/queries/`. The build
copies the one matching `channel_finder_mode` to `data/benchmarks/queries.json`:
`in_context` takes `in_context_queries.json`, every other mode
`tree_queries.json`.
