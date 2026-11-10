`osprey.utils.config_writer` no longer has `update_yaml_file`,
`get_control_system_type` or `set_control_system_type`, which nothing in OSPREY
called. Switch the control system with `osprey set connector=…` and
`osprey build`, and edit config fields with `config_update_fields`.
