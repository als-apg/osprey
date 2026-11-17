The simulation package lists fewer names. These are now private, so stop
importing them: `series.apply_events`, `event_positions`, `epoch_seconds_array`,
`driver_key_bytes` and `wander`; `topoff.period_mean`, `tau_period` and
`tooth_depth`; `envelope.declares_motion`; `texture.channel_variable`;
`channel_faults.ChannelFault`, `FROZEN` and `DISCONNECTED` (the words are
`"frozen"` and `"disconnected"`, the keys of `CHANNEL_FAULTS`); and
`view.COMMAND_MARK`, `ENGINE_ROLES` and `MARKS` (use `records.EngineRole` and
`records.Mark`). The package root no longer re-exports `model_status`,
`read_model_status`, `GATED_TYPES` or `LABELLED_TYPES`: import them from
`osprey_connectors.simulation.status` and `osprey_connectors.simulation.values`.

The view's file and schema constants are gone (`ADDRESSES_FILE`,
`ADDRESSES_SCHEMA` and the other eight, and `SCHEMAS`): each document names its
own, as `AddressesDocument.FILE` and `AddressesDocument.SCHEMA`, and
`VIEW_DOCUMENTS` lists the five. `SimulatorView.path_for_project()` is gone:
use `view_path(project_dir)`. `SimulatorView.bindings()` is `wiring()` and
`binding(address)` is `wiring_of(address)`. The record types of a scenario's
`archiver` and `faults` entries and of a seed's `drift` and `topoff_lifetime`
are public as `ArchiverEntry`, `ModelFaultRecord`, `WanderRecord` and
`TopoffLifetimeRecord`.
