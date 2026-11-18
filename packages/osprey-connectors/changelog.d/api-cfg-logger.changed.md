`osprey_connectors.logger.get_logger(name, /)` takes the logger's name as its
one positional argument: `get_logger(name="x", color="blue")` becomes
`get_logger("x")`, and the `component_name`, `level`, `state` and `color`
parameters are gone. `ComponentLogger` keeps the stdlib methods (`debug`,
`info`, `warning`, `error`, `critical`, `exception`) and `key_info`; call `info`
where you called `success`, `timing`, `approval`, `status` or `resume`.
`ComponentLogger(base_logger, component_name)` no longer takes `color` or
`state`, and the retired event keywords (`error_type`, `recoverable`,
`stack_trace`, …) raise `TypeError` instead of being dropped.
