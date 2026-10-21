`[tool.osprey-connectors.compat]` in `pyproject.toml` declares the framework
releases the connectors built at HEAD must keep working under; the
`connectors-compat` CI lane and its nightly read it. Nothing in the wheel
changes.
