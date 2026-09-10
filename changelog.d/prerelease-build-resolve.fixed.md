`osprey build` from a pre-release install no longer fails while preparing the
project environment. The build pins `osprey-framework` to the running version,
and uv admitted that beta but not the `osprey-connectors` beta it depends on;
a pre-release pin now makes the whole resolve admit pre-releases, and the
recorded `pyproject.toml` carries `[tool.uv] prerelease = "allow"` so a later
`uv sync` in the built directory resolves the same way.
