"""Data bundles for the shipped presets.

Each subdirectory is a packaged data bundle, named by the preset-side
``app_template:`` key in ``profiles/presets/<preset>.yml`` (a *profile* naming
it is refused — the key never survives resolution). ``osprey init``
materializes the bundle into the deployment repository it creates, and the
profile it writes there points at the copied tree with its own ``data:`` path;
that copy is what every later build reads. What a bundle holds is the content
a deployment needs beside its profile, never configuration:

- ``data/`` : the facility data tree ``osprey init`` materializes into the
  deployment repository (channel databases, channel limits, facility
  knowledge, logbook seeds, simulation scenarios, ...). Copied verbatim; see
  ``tests/templates/test_data_trees_are_not_templates.py``.
- ``mcp_servers/`` (optional) : example MCP server packages seeded into the
  repository's ``mcp_servers/`` directory and launched by the profile's
  ``mcp_servers:`` entries.
- ``web-terminal-context/`` (optional) : the bundle's ``base.md`` for the web
  terminal, overriding the framework fallback under
  ``templates/claude_code/web-terminal-context/``.

No bundle renders ``config.yml`` or ``README.md``: every project file renders
from ``templates/project/``, and what a deployment configures is spelled in
full by its profile's ``config:`` block, which the preset writes down.
"""

__all__ = []
