`osprey facility validate` no longer refuses a measurement kind whose package
the deployment lists as a local path or an `apps/<name>/`: it reads the kinds
each listed package declares from its `pyproject.toml`, building and importing
nothing.
