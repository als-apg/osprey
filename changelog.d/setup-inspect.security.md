The `setup_inspect` agent tool now reports `config.yml` with its `${VAR}`
placeholders intact instead of the values they resolved to, so an API key or a
database password no longer lands in the transcript. Any literal value under a
key named `*KEY*`, `*TOKEN*`, `*SECRET*` or `*PASSWORD*` — in the config or in
`.mcp.json` — is masked on top of that. The unexpanded document is also the
better diagnostic: it names the variable each key reads.
