**Breaking change:** a facility tree is read from new paths, and a tree
holding a retired path is refused, not read. `osprey_facility.source_tree`
now places `classes` and `signals` under `vocabulary/`; `models`, `seeds`,
`scenarios`, `measurement` and `decks` under `simulation/`; `fixes`,
`documents` and `policy` under `provenance/`; and the benchmark queries under
`benchmarks/channel_finder/`. Each slot names its former path as retired, and
`source_tree.retired_moves()` lists them; the old paths are never read. The
new `osprey_facility.retired` module finds what a tree still holds at them
(`find_retired`), builds the shell lines that move it (`move_script`), and
`refuse_retired` raises `RetiredLayoutRefused`, a `FacilityError` whose message
is those lines. Code that spelled the old paths, or the old `deck:` prefix
`decks/`, must take them from `source_tree` or be updated.
