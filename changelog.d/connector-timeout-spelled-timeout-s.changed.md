**Breaking change:** the EPICS, TANGO and virtual-accelerator connectors read their call bound from
`control_system.connector.<type>.timeout_s`, as DOOCS does; a config whose connector block still
carries `timeout` is refused when it is read, naming its new spelling. Every control-system connector
refuses a `timeout_s` that is not a positive number of seconds when it connects.
