A trigger in `triggers.yml` whose entry, `action`, `on_error` or
`source_config` is not a mapping is refused when the file loads, with an
error naming the trigger and the field. Before, the dispatcher failed to
start with an `AttributeError`, or `osprey build` stopped with a traceback. A
blank `source_config:` means an empty one.
