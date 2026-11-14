The build marks every command channel `command: true` in the facility file: a
setpoint whose signal ends in `command`, or a channel a model's engine marks
`command`. `command_write` fires only marked channels, and the build stops when
an engine describes a `*_command` setpoint without the mark.
