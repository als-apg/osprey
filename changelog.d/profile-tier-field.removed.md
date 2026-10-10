The profile field `tier` is gone. A profile that spells it stops with
`facility: profile-invalid: path tier`, and the line names the `in_context` tag
on a channel as what selects the in_context subset. `osprey set` no longer
accepts `tier=`, and an emitted `profile.yml` no longer carries a commented
`tier:` block. The build picks the benchmark query set from
`channel_finder_mode`: `in_context` takes `in_context_queries.json`, every
other mode `tree_queries.json`, copied to `data/benchmarks/queries.json`. A render carries
no `data/channel_databases/tiers/`, `data/benchmarks/cross_paradigm/` or
`data/raw/`, and no `data/channel_databases/<paradigm>.json` flattened from a
tier; each channel-finder index is the view the build writes under
`data/channel_finder/`. The build's profile line reads
`profile <name> (bundle <bundle>)`.
