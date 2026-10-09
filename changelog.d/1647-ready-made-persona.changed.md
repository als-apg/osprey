A persona the build does not render — a catalog entry with no `build_profile`,
whose `project_path` points at a project supplied ready-made — now proves its
tier by that render's own `config.yml` at build time, the same file `osprey up`
reads at start. Such a persona may be the `default_persona` or sit on a shared
card like any delta-rendered one; one whose `project_path` holds no `config.yml`
is still refused there, naming the path the build looked at.
