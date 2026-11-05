The getting-started tutorials now match what the presets ship. The
control-assistant tutorial names its real default channel-finder mode,
`graph`, and describes that pipeline's tools instead of the hierarchical ones;
its first read uses a channel the example facility serves
(`SR:DIAG:DCCT:01:CURRENT:RB`) rather than `SR:BEAM:CURRENT`, which it does
not. The hello-world tutorial runs its DuckDB query with
`build/.venv/bin/python`, where DuckDB is installed; shows the `.env` line
`osprey init` prints; counts 14 commented blocks and 11 hooks; and shows
the shipped `limits.yaml` whole. The guided-setup page loses its stray
`# skip-ci` lines and no longer tells you to `cd` into a directory you are
already in. Tests now check the tutorials' addresses, profile excerpts, counts
and init summary against the presets.
