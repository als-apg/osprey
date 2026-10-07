**Breaking change:** the old `osprey.*` import paths for modules that moved to
`osprey-connectors` are gone. Import `osprey.connectors.*`, `osprey.errors`,
`osprey.utils.{config,dotenv,identity,log_filter,logger,relative_time,workspace}`
and the `osprey.simulation` engine modules from `osprey_connectors` instead; the
import map in `packages/osprey-connectors/README.md` lists each replacement.
The `resolve_workspace_root`, `BaseAdapter`, `LegacyChannelDatabase` and named
`ensure_*_server` aliases are also removed; use `resolve_agent_data_root`,
`FacilityAdapter`, `FlatChannelDatabase` and `ensure_web_server("<key>")`.
