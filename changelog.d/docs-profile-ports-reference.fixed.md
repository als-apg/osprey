The build-profile reference now lists every top-level key `profile.yml`
accepts, including the seventeen it had left out (`extends`, `machine`,
`port_base`, `deploy`, `dispatch`, `optimizer`, the four chat bridges,
`watchdogs` and others), and has a key table for an `mcp_servers` entry that documents
`hooks` and `writes`. The ports page's examples moved the port base with a
nested `config: deployment:` block, which replaces the whole `deployment`
subtree; they now use the dotted key or the top-level `port_base:` shorthand,
and its `dispatch:` example gains the `triggers` key the loader requires. The
environment-variable page adds `OSPREY_PROFILE_VARIANT` and the three
`OSPREY_ARCHIVE_*` variables. New tests check each table against the loader's
key sets and load each ports example through the profile loader.
