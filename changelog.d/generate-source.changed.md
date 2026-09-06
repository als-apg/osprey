`osprey channel-finder generate` now requires you to say where the channels
come from: `--source PATH` for your own hierarchical database, or `--demo` for
the packaged one. It also refuses to overwrite database files that already
exist unless you pass `--force`. A bare `generate` used to write the packaged
demo machine's ~2900 channels straight into `data/channel_databases/` — the
files the pipelines read — replacing a facility's own database. Generated
`in_context.json` files now record the source they were expanded from in
`_metadata`.
