Code that logs through `osprey_connectors.logger.get_logger` names its logger
positionally (`get_logger("registry.loader")`, no `name=` or `color=`) and
calls `info` where it called `success`, `timing`, `approval`, `status` or
`resume`; those methods are gone. Log output is unchanged.
