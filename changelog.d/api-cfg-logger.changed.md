Code that logs through `osprey_connectors.logger.get_logger` names its logger
positionally (`get_logger("registry.loader")`, no `name=` or `color=`) and
calls `info` where it called `success`, `timing`, `approval`, `status` or
`resume`; those methods are gone. Log output is unchanged.
`logging.rich_tracebacks: 'false'` (and the other two `logging.*` switches
written as quoted words) turns the switch off, and a switch written as a null
keeps its default.
