The channel-finder MCP servers find `config.yml` the way every other OSPREY
process does: `OSPREY_CONFIG`, then `build/config.yml`, then `config.yml` in the
working directory. A server started by hand from a repo root now reads the render
and resolves its database paths against it.
