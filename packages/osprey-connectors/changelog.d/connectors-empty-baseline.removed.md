Every name a public module lists now has a user outside that module or is a
declared extension point. The names nothing used are gone or private:

- `osprey_connectors.archiver`: `channel_address()` is gone; nothing decodes a
  field name back. `to_utc()`, `require_datetime()`, `resolve_processing()`,
  `long_frame()`, `decimate_raw()`, `aggregate_series()`, `reject_non_numeric()`
  and `Processing` are no longer offered as a helper kit for connector authors;
  they stay private to the in-tree archivers.
- `osprey_connectors.archive_targets`: `ADDRESS_SETTINGS`, `APPLIANCE_TARGETS`,
  `ARCHIVER_SECTION_KEY`, `DEFAULT_ADDRESS_SETTING`, `NO_TARGETS`,
  `STORE_ARCHIVERS` and `STORE_TARGETS` are private; `archive_targets()` and
  `archive_withheld()` answer the questions.
- `osprey_connectors.control_context`: `live_report_payloads()` and
  `live_statement_reports()` are gone; a view that shows the machine reads each
  of `report_paths()` with `read_report()` and keeps the reports whose
  `display_only` is true. `state_dir()` is gone: use
  `osprey_connectors.posture_store.state_dir()`. `parse_posture()` is gone: use
  `osprey_connectors.posture_store.parse_posture_value()`. `parse_report()`,
  `posture_mirror()`, `report_path_under()`, `is_foreign_process()`,
  `REACHABLE_STATES`, `SCHEMA_VERSION` and `WRITES_STATEMENT_KEYS` are private.
- `osprey_connectors.posture_store`: `invalidate_cache()` is gone: use
  `osprey_connectors.control_context.invalidate_cache()`, which holds the cache.
  `LOGIN_TIERS_KEY` is gone; nothing read it. `agent_data_root()`,
  `AGENT_DATA_ROOT_ENV_VAR`, `LAUNCH_POSTURE_ALL_TARGETS` and `VALID_POSTURES`
  are private; `stamped_agent_data_root()` and `state_dir()` stay.
- `osprey_connectors.standin`: `archive_belongs_to_standin()`,
  `DEPLOYED_SERVICES_KEY` and `ARCHIVER_RECORDER_SERVICE` are gone;
  `osprey_connectors.archive_targets` decides whose history a store holds.
- `osprey_connectors.session_hold`: `SCHEMA_VERSION` and `EXCERPT_LIMIT` are
  private.
- `osprey_connectors.settle`: `POLL_INTERVAL_S` is private.
- `osprey_connectors.recovery`: `Origin` is private.
- `osprey_connectors.write_target`: `FALLBACK_WRITE_TOOLS` is private.
- `osprey_connectors.types`: `CLI_ARCHIVER_TYPES`, `TRANSPORT_DOOCS` and
  `TRANSPORT_TANGO` are private; `factory.BUILTIN_CONNECTORS` lists the
  built-in archivers.
- `osprey_connectors.ipc`: `DEFAULT_PVA_PORT`, `MODE_ADDR_LIST`, `ROLE_PVA`,
  `SETTLE_TIMEOUT_S` and `verify_child_report()` are no longer in the
  namespace.
- `osprey_connectors.simulation`: the package root no longer re-exports the
  active-scenario state helpers, `STATUS_NO_DECK` or `coerce`; import them from
  `simulation.state`, `simulation.status` and `simulation.values`.
- `osprey_connectors.simulation.composite`: `ENGINE_GROUP`, `INSTANCES` and
  `UDF` are private, and `STATUS_NO_DECK` is no longer re-listed (it is
  `simulation.status.STATUS_NO_DECK`).
- `osprey_connectors.simulation.pulse`: every shape constant,
  `SHAPE_CONSTANT_NAMES` and `shape_version()` are private; `trace()` and
  `SHAPE_VERSION` stay, and `SHAPE_VERSION` is unchanged, so no stored archive
  re-seeds.
- `osprey_connectors.simulation.topoff`: `shape()` is gone (`tau_period()` gives
  its `tau` and period). `DECAY_FORM`, `LIFETIME_KEY`, `REFILL_PHASE`,
  `SECONDS_PER_HOUR`, `TOPOFF_KEY`, `SHAPE_CONSTANT_NAMES` and
  `shape_version()` are private; `SHAPE_VERSION` is unchanged.
- `osprey_connectors.simulation.channel_faults`: `DISCONNECTED` and `FROZEN`
  are private; `CHANNEL_FAULTS` lists every fault.
- `osprey_connectors.simulation.values`: `VALUE_TYPES` and `WAVEFORM` are
  private.
- `osprey_connectors.simulation.view`: `NO_VIEW_MESSAGE` is private.
