`osprey ariel`'s `--adapter`, `--module` and `--mode` options now accept every
name the registry carries, so an ingestion adapter, enhancement module or search
module a deployment registered itself can be named on the command line. `osprey
ariel ingest` no longer defaults `--adapter` to `generic_json`: with no flag it
uses the `ariel.ingestion.adapter` from `config.yml`, the same rule `watch`
follows. A project whose config names an adapter and previously relied on the
`generic_json` default now ingests with the configured adapter; with neither, the
run stops and says `ariel.ingestion.adapter` is required.
