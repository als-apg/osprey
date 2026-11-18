`osprey web` applies the `logging.*` settings of the deployment it serves, also
when it is started from another directory with `--repo`. `rich_tracebacks:
false` was ignored there, because logging was set up before the deployment was
known.
