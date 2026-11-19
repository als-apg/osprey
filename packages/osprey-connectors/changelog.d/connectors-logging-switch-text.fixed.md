The `logging.rich_tracebacks`, `logging.show_traceback_locals` and
`logging.show_full_paths` settings now read a quoted `"false"` or a
`${VAR:-false}` value as off. Before, such a value read as on, so
`show_traceback_locals: "false"` showed local variables in tracebacks. A value
that is neither on nor off now falls back to the defaults (locals hidden, short
paths) with a warning naming the setting.
