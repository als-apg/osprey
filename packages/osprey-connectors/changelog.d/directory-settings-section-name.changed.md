`service_targets` reads the directory section under `channel_directory`, the
control-system-neutral name of the former `pv_directory` section. A config
that has no `channel_directory` key is still read under `pv_directory`, which
is what a framework release from before the rename hands over. Each
`BackendSpec` carries the key it was read under in a new `section` field
(default `channel_directory`, left out of equality), and `settings_key` is the
config key of its settings: a settings error of a directory, alarm or put-log
backend names the key that is in the config, `channel_directory.settings.url`
or `pv_directory.settings.url`.
