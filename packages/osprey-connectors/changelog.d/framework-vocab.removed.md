Names only the framework used leave the public API before the first stable
release. From `workspace`: the layer directories (`TIERS_DIRNAME`,
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
