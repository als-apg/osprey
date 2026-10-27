**Breaking change:** names nothing outside their own module used are no longer
public API. `config.get_framework_service_config` and `log_filter.LoggerFilter`
are removed. These are now private: `config.RUNTIME_STATE_DIR`,
`control_context.OWNER_KINDS`, `live_report_payloads`, `record_path_under`,
`report_path` and `report_path_under`, `logger.QUIET_THIRD_PARTY_LOGGERS`,
`log_filter.suppress_logger_level`, `posture_store.parse_launch_posture`,
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
