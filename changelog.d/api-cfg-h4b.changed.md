Framework readers that degrade to an empty config read it through
`osprey.utils.config_paths.loaded_or_empty()`, which answers the process config
(`current_config()`: a config an entry point installed, else the located file) and
no longer installs the file it read as the process default; inside a
`deployment_config(...)` block such a reader now sees that deployment's config
rather than the working directory's. `config_paths.process_repo_root()`,
`process_agent_data_root()` and `session_agent_data_root()` answer the repo and
agent-data roots for readers that hold no config, and `repo_root_of(config)` for a
reader that holds only a mapping (a `project_root` naming no directory here gives
way to the repo the process stands in, also when no config file exists). The web
terminal's `operator_session.resolve_agent_data_root(app)` is
`shared_data_root(app)`. The registry's export directory is
`agent_data_root / file_paths.registry_exports_dir`, anchored at the config's repo
root when `project_root` is unset (it was the working directory). A theme,
model or provider read for a named config file loads that file without applying
the working directory's `.env` first.
