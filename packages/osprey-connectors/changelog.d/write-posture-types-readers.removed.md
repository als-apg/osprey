**Breaking change:** `osprey_connectors.types` no longer exports the
`writes_enabled` readers: `WRITES_ENABLED_KEY`, `TYPE_WRITES_ENABLED_LEAF`,
`type_writes_enabled`, `target_writes_enabled`, `target_writes_enabled_key`,
`session_posture`, `any_target_writes_enabled`,
`any_armed_target_checks_limits`, `writes_enabled_key`,
`writes_enabled_answering_key` and `writes_enabled_remedy`. Use `WRITES_KEY`
with `tier_writes(section)` or `tier_allows(section)`, which no longer read
the retired keys.
