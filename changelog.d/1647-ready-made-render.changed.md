A tier or app the build does not render — an entry in
`modules.web_terminals.renders` with no `build_profile`, whose `project_path`
points at a project supplied ready-made — now proves its tier by that render's
own `config.yml` at build time, the same file `osprey up` reads at start. Such
a render may be the `default_tier` or sit on a shared card like any
delta-rendered one; one whose `project_path` holds no `config.yml` is still
refused there, naming the path the build looked at.
