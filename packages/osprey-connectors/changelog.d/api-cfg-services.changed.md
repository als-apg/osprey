**Breaking change:** `service_targets` takes the sections it reads:
`service_targets(cfg.pv_directory, cfg.archiver, simulator_port=..., built_simulator=...)`
(`virtual_accelerator_port` and `has_simulator` of the config). `resolve_archiver_type`,
`resolve_archiver_settings` and `archiver_settings_key` take an `ArchiverSection`
(`current_config().archiver`, or `ArchiverSection.read(mapping)`), and
`archiver_settings_errors` is gone: read `ArchiverSection.problems`. The loaded
config carries `archiver`, `pv_directory`, `logging`, `simulation` and
`agent_data`, each read once when it is loaded, never refused there.
`ConnectorSettings` is now `osprey_connectors.config_sections`' and is the base of
every settings model: the control-system connectors' is `ControlSystemSettings`
(`from_block(block, key="control_system.connector.<type>")`), and each service
backend declares a `settings_model` (`EpicsArchiverSettings`, `MongoStoreSettings`,
`MyaArchiverSettings`, `DoocsArchiverSettings`, `MockArchiverSettings`,
`ChannelFinderSettings`, `PhoebusAlarmLogSettings`, `CaPutLogSettings`,
`PhoebusKafkaSettings`, `ReplayAlarmStateSettings`) that the backend reads
`spec.settings` into as `backend.settings`. `BackendSpec.settings` is a
`ConnectorSettings` holding the block as written: read its keys with
`spec.settings.model_dump()`. An out-of-tree backend reads its keys from
`self.settings.model_extra`, or declares its own model. `ReadOnlyTransport`
takes the block's read connection (`HttpServiceSettings.connection`) instead of
the mapping and `where`.
`connection.Login` is replaced by `TokenLogin` and `PasswordLogin` (`ServiceLogin`),
whose `secret_key` and `secret_env` replace `env_key` and `env_names`;
`Credential` is `Credential(login, secret)`.
The archives refuse a block they cannot be built from with a `ConfigurationError`
naming the key, in the words they had (a missing EPICS `url`, a Mongo block missing
`name`, `collection` or its login, a mock `sample_rate_hz` that is not above 0).
New at build: a Mongo `port` that names no port and a DOOCS `avg_window` that is
not a number are refused (both failed every read before); a numeric-string Mongo
`port` is read as its number. A Kafka alarm feed whose `bootstrap_servers` or
`topic` is written as something else answers `misconfigured` rather than
`unreachable`.
