`get_config_value(key, default)` is gone: write `current_config().get(key, default)`,
or read a typed section (`current_config().control_system`). A reader that took
`config_path` loads that file with `load_config(path)`. `get_agent_dir` is gone:
write `current_config().agent_data_root / <subdir>`.

The loaded config names its paths: `LoadedConfig.repo_path(value)` (a configured
path, `~` expanded, a relative one anchored at `repo_root`),
`LoadedConfig.agent_data_root` (the shared root, `agent_data.base_dir` resolved),
`LoadedConfig.session_data_root(session_id)` (the session is an argument; nothing
reads `OSPREY_SESSION_ID`) and `LoadedConfig.simulation_state_dir`.
`workspace.locate_repo_root(cwd=None)` answers the repo a process with no loaded
config stands in. Gone from `osprey_connectors.workspace`: `load_osprey_config`,
`resolve_config_path` (write `config.locate_config()`), `resolve_project_root`
(`cfg.repo_root`, or `locate_repo_root()` without a config),
`resolve_shared_data_root` (`cfg.agent_data_root`), `resolve_agent_data_root`
(`cfg.session_data_root(session_id)`), `resolve_path` (`cfg.repo_path(value)`; it
no longer raises `TypeError` when `project_root` is unset, it anchors at the
config's repo root), `dotted_config_str`, and `reset_config_cache` (now
`osprey_connectors.testing.clear_config_state()`).
