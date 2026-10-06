The channel-finder benchmark refuses a project `config.yml` that does not parse
or is not a mapping, naming the file. `--backend auto` no longer falls back to
the default backend on such a file, and the coverage judge and the ReAct
backend no longer fail on it with an unrelated `AttributeError`.
