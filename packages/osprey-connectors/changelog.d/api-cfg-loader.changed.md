One loader: `osprey_connectors.config.load_config(path=None, *, environ=None)`
returns a `LoadedConfig` (`path`, `repo_root`, `unresolved`, the typed
`control_system` and `system` sections;
`get(key, default)`, `section(key)`, `flag(key, *, default=...)`, `raw_tree()`,
`unexpanded_tree()`, all handing out copies). It reads the file and nothing
else: no `.env` is applied and nothing is cached or installed. `ConfigBuilder`,
`get_config_builder`, `get_configurable`, `get_full_configuration`,
`load_project_config`, `default_config_path` and `config_anchored_at` are gone,
and so is the derived view with its `model_configs`, `provider_configs`,
`service_configs` and `facility_timezone` aliases. Write
`load_config(path).raw_tree()` for `load_project_config(path)` or
`ConfigBuilder(path).raw_config`, `.unexpanded_tree()` for
`get_unexpanded_config()`, `section("models")` for the `model_configs` alias,
`install_config(load_config(path))` for `set_as_default=True`, and
`active_config()` (its `.path`) for `default_config_path()`.

The process default is `current_config()`: the config of the innermost
`using_config(...)` block, else the one `install_config` installed, else the file
`locate_config()` finds (`OSPREY_CONFIG`, then `build/config.yml`, then
`config.yml`), reloaded when it changes. `current_config().get(key, default)` reads one value.
**`CONFIG_FILE` is no longer read by the package**: a process steered by
`CONFIG_FILE` alone now finds the working directory's config; set
`OSPREY_CONFIG`. `testing.clear_config_state()` clears every cache and warn-once latch.

A load that fails raises `ConfigLoadError` (`path`): `ConfigNotFoundError`
(`searched`, still a `FileNotFoundError`), `ConfigParseError` (`line`, `key`;
a `ValueError`) for a file that is not YAML or not a mapping, and the base class
itself for a path that is a directory (no longer an `IsADirectoryError`).
`wrap_errors` is gone: the error names its path. `config_flag` raises
`ConfigurationError` (still a `ValueError`) naming the key.

`get_facility_timezone()` is `facility_timezone(config=None)`, and
`localize_facility(dt, tz)` takes the zone. The zone is read when the config is
loaded (`LoadedConfig.system.timezone`, a `config_sections.FacilityZone` with
`name`, `source` and `problem`); a zone that is not one is still UTC, reported once per load.

`apply_env_chain`, `load_project_dotenv` and `dotenv_shell_overrides` moved to
`osprey_connectors.dotenv`; the shell-override record is taken from the same
python-dotenv parse the chain is written with.
