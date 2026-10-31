# Changelog

All notable changes to `osprey-connectors` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this package adheres to [Calendar Versioning](https://calver.org/).

The package has its own calendar stream, released from its own
`osprey-connectors-vYYYY.M.P` tags by
`.github/workflows/release-connectors.yml`, independently of the framework's
`v*` tags. Releases are stable by default. Each release stays backward-compatible with every supported
framework release. Changes from before this stream began are recorded in the
repository's root `CHANGELOG.md`, under the framework releases that shipped
them.

Unreleased changes are written as fragments in `changelog.d/` beside this file
and folded into the `[Unreleased]` section below when a release is cut:

    python scripts/changelog_fragments.py apply --root packages/osprey-connectors

## [Unreleased]

## [2026.10.0] - 2026-10-29

The first release on the package's own calendar stream, cut independently of the
framework. Its public API is what the framework and external tools may rely on
from here on: names only the framework used have moved into the framework, and
the read-only run has one owner in `posture_env`. Framework 2026.9.0b1–b3 are
superseded and not supported with this release.

### Added

- `osprey_connectors.posture_env` is the one home of the read-only run:
  `POSTURE_ENV_VAR`, `SANDBOX_MODE` and `is_readonly_run()`. The module imports
  nothing from the package, so the posture store no longer imports the
  control-system connectors to ask it, and import-linter now holds every
  top-level module of the package in layers.

### Changed

- **Breaking change:** the EPICS connector and the connector-host pool choose
  the write gateway from the write verdict: a connector whose writes are allowed routes through
  `gateways.write_access`, otherwise it stays on `read_only`. A write allowed
  after the gateway was chosen is refused as `switching` until the connector
  reconnects. A machine with only a write gateway reads through it while its
  writes are off, refused in software; a read-only run on such a machine
  refuses to connect.

### Removed

- **Breaking change:** names only the framework used leave the public API
  before the first stable release. From `workspace`: the layer directories (`TIERS_DIRNAME`,
  `APPS_DIRNAME`, `LAYER_DIR_KINDS`, `LAYER_DIRNAMES`, `RETIRED_LAYER_DIRS`,
  `layer_dir_kind`), `PROFILE_FILENAME`, `STATE_DIR_NAME`, `IMAGE_DIR_NAME`,
  `container_image_context`, `RUNTIME_DATA_DIR_NAME`, `AUDIT_DIR_RELPATH`,
  `ARCHIVE_DIR_RELPATH`, `STATE_ZONE_DIRS`, `CHAT_UPLOADS_SUBDIR`,
  `SIMULATOR_STANDIN_LOG_SUBDIR`, `deployment_env_chain` and
  `repo_root_for_agent_data`. From `dotenv`: every `.env` writer and the
  compose-safety checks (`append_profile_env`, `write_env_merged`,
  `env_file_lock`, `env_lock_path`, `atomic_write`, `format_env_line`,
  `dotenv_line_var`, `compose_unsafe_vars`, `compose_unsafe_vars_in_chain`,
  `EnvConflict`, `ProfileEnvAppendResult` and the file banners); the chain
  reader stays. From `config`: `resolve_execution_method`,
  `EXECUTION_METHOD_SUBPROCESS`, `DEFAULT_EXECUTION_TIMEOUT_SECONDS`,
  `to_facility_iso`, `RUNTIME_WRITE_PATH_KEYS` and
  `find_runtime_write_paths_under_data`. The config load's runtime-write
  advisory now checks only `simulation.state_dir`: the other key it named,
  `services.channel_finder.pipelines.hierarchical.feedback.store_path`, is a
  spelling no reader uses, so it never matched a real config.
- `osprey_connectors.control_system` no longer exports `is_readonly_run`; import
  it from `osprey_connectors.posture_env`.
- **Breaking change:** `osprey_connectors.posture_store` no longer exports the
  old narrowing and store readers: `effective_writes`, `store_permits`,
  `store_verdict`, `store_verdict_detail`, `record_verdict_detail`,
  `StoreVerdict`, `StoreVerdictDetail`, `recorded_posture`, `target_posture`,
  `launch_permits`, `launch_narrowed_target` and `parse_launch_posture`. Ask
  `write_verdict(section, target)` instead — its `allowed` and `reason` replace
  the bool and the verdict member, and a run whose launch pin refuses answers
  `reason == "launch_pinned"`.
- **Breaking change:** names nothing outside their own module used are no longer
  public API. `config.get_framework_service_config` and `log_filter.LoggerFilter`
  are removed. These are now private: `config.RUNTIME_STATE_DIR`,
  `control_context.OWNER_KINDS`, `live_report_payloads`, `record_path_under`,
  `report_path` and `report_path_under`, `logger.QUIET_THIRD_PARTY_LOGGERS`,
  `log_filter.suppress_logger_level`,
  `process.REAP_WINDOW_S`, `standin.ARCHIVER_RECORDER_BLOCK_KEY` and
  `archive_recorder_configured`, `types.ARCHIVER_SETTINGS_LEAF`,
  `LIMITS_BLOCK_LEAVES` and `ONE_REAL_MACHINE`, `yaml_loader.safe_dumper`,
  `simulation.archive.DEFAULT_CHUNK_SIZE`, `DEFAULT_WORKERS`,
  `SEED_SCHEMA_VERSION`, `prepare_collection`, `seed_grid`,
  `synthesize_documents` and `write_manifest`,
  `simulation.composite.LOG_RECORD_MAX_BYTES`, `STATUS_MAX_BYTES` and
  `cap_status`, and `simulation.envelope.MOTION_SIGMAS` and `noise_sigma`.
  `factory` no longer exports the test-only registry helpers
  (`isolated_connector_registries`, `snapshot_connector_registries`,
  `restore_connector_registries`, `ConnectorRegistrySnapshot`).
  `control_system` no longer exports `METADATA_FIELDS`, `archiver` no longer
  exports `HOST_OVERRIDE_ENV` and `PORT_OVERRIDE_ENV`, and `ipc` no longer
  exports the wire codec and exception registry (`encode_request`,
  `FrameReader`, `ResultFrame`, `EXCEPTION_SPECS`, ...) or `CHILD`,
  `CHILD_MODULE`, `DEFAULT_CA_PORT`, `NEVER_LIVE_TYPES` and `same_endpoint`;
  `ipc` still exports `FrameDecodeError` and `FrameEncodeError`.
- **Breaking change:** `osprey_connectors.ipc` no longer exports
  `config_writes_enabled`. Ask `posture_store.write_verdict(section, target)`
  instead.
- **Breaking change:** `osprey_connectors.types` no longer exports the
  `writes_enabled` readers: `WRITES_ENABLED_KEY`, `TYPE_WRITES_ENABLED_LEAF`,
  `type_writes_enabled`, `target_writes_enabled`, `target_writes_enabled_key`,
  `session_posture`, `any_target_writes_enabled`,
  `any_armed_target_checks_limits`, `writes_enabled_key`,
  `writes_enabled_answering_key` and `writes_enabled_remedy`. Use `WRITES_KEY`
  with `tier_writes(section)` or `tier_allows(section)`, which no longer read
  the retired keys.

### Fixed

- Limits checking in `optional` mode with no `database_path` no longer needs a limits database: every channel is written as unlisted. A configured path must still exist.
