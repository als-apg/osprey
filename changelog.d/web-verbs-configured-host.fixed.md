`osprey ariel web` and `osprey channel-finder web` bind the host set in
`ariel.web.host` / `channel_finder.web.host` when `--host` is not given, as
`osprey artifacts web` does. `osprey channel-finder --project DIR web` takes
its host and port from that project's config, not the working directory's.
