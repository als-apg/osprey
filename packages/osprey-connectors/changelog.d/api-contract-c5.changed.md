**Breaking change:** `osprey_connectors.types` is removed; its names move, by
what they answer, to `osprey_connectors.connector_types` (type names, the
built-in rows, serving and transport, the retired names, and the target words
`ControlTarget`, `TARGET_*`, `CONTROL_TARGETS`, `MachineKind`,
`is_control_target`), `osprey_connectors.targets` (`resolve_target`,
`baseline_target`, `configured_targets`, `target_configured`,
`switch_capable`, `machine_kind`), `osprey_connectors.write_tier`
(`WRITES_KEY`, `TierWord`, `TIER_WORDS`, `tier_writes`, `tier_allows`),
`osprey_connectors.limits_posture` (`LimitsMode`, `LimitsPosture` and the
limits resolvers) and `osprey_connectors.archiver_selection`
(`resolve_archiver_type` and the archiver-settings readers).
`published_machine_kind` moves to `osprey_connectors.control_context`. Import
each name from its new module; no alias is kept.
`BUILTIN_CONNECTORS` and `BuiltinConnector` move from `osprey_connectors.factory`
to `osprey_connectors.connector_types`. A row now states its `transport`,
`machine`, `baseline_target`, `invents_history`, `offered_by_init`,
`offered_by_set` and `default_port`, and the type lists (`CLI_CONTROL_SYSTEM_TYPES`,
`SET_CONTROL_SYSTEM_TYPES`, `SIMULATED_TYPES`, `STANDIN_TYPES`,
`INVENTED_HISTORY_TYPES`, `DEFAULT_VA_PORT`) are read off the rows, with the same
values and order. `ControlSystemTypeName` names the five built-in control-system
types. `BuiltinConnector.load()` is removed; nothing called it.
`simulator`, `standin` and `archive_targets` move into the
`osprey_connectors.deployment` subpackage (`osprey_connectors.deployment.simulator`,
...). `osprey_connectors.honesty` leaves the package for the framework, as
`osprey.deployment.honesty`.
The one-real-machine warning logs on `osprey_connectors.targets`, not
`osprey_connectors.types`.
