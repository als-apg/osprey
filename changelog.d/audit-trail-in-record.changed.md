Safety decisions are now `audit` entries in the agent record instead of lines
in per-surface files under `var/audit/<identity>/`. Each entry names the
surface that decided (`hook_approval`, `scaffold_gallery`, `preflight`,
`auth_sidecar` and the rest) in its `surface` field. The record keeps them for
`agent_record.retention_days`, and the archive keeps each closed day in
`var/archive/<day>/agent_record/entries.parquet`. `osprey reset` keeps the
record and the archive; `osprey reset --purge-audit` deletes the record and its
exports.
