**Breaking change:** the old `osprey.*` import paths for modules that moved to
`osprey-connectors` are gone. Import `osprey.connectors.*`, `osprey.errors`,
and `osprey.utils.{config,dotenv,identity,log_filter,logger,relative_time,workspace}`
from `osprey_connectors` instead; the import map in
`packages/osprey-connectors/README.md` lists each replacement.
The `resolve_workspace_root`, `BaseAdapter`, `LegacyChannelDatabase` and named
`ensure_*_server` aliases are also removed; use `resolve_agent_data_root`,
`FacilityAdapter`, `FlatChannelDatabase` and `ensure_web_server("<key>")`.
Re-run `osprey build` after upgrading: a project rendered before this release
has a limits hook that imports the removed `osprey.connectors` path, so it no
longer refuses an out-of-limits write before the tool call (the connector still
refuses it at write time).
