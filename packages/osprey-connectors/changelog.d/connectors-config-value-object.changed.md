`osprey_connectors.config.get_config_value` is annotated to return `object`
instead of `Any`. What it returns is unchanged, but code that type-checks
against it must narrow the result (for example with `isinstance`) before using
it as a specific type.
