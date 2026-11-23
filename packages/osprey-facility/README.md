# osprey-facility

The OSPREY facility description: the `facility.json` document a deployment is
built from, the open JSON Schema it is published against, and the layers that
build it from a facility's sources. It carries none of the rest of the OSPREY
framework (agents, views, web UI, etc.).

## Install

```
pip install osprey-facility
```

The MML layer, which reads a MATLAB Middle Layer export, needs the `mml` extra:

```
pip install "osprey-facility[mml]"
```

## Public API

The modules below are the package's public API; every other module is
private. Each names the one consumer it serves: a *fork/tool* that builds or
reads `facility.json` without the framework, a *kind plug-in*, an *app* (which
may import `osprey_facility.kinds` only), or a *framework job* (views, CLI,
engines, output writing), named by the framework modules that import it.

Each public module lists its public names in `__all__`. `osprey_facility.kinds`
also serves apps; `osprey_facility.reading` also serves the framework's checked
readers. `osprey_facility.schema.published_schema()` returns the shipped JSON
Schema: the current one, or `published_schema(4)` for a format version's floor.

| Module | Consumer |
|---|---|
| `osprey_facility` | framework job: `osprey.apps.sdk`, `osprey.channel_roster`, `osprey.cli`, `osprey.facility.settings`, `osprey.facility.views`, `osprey.mcp_server.python_executor`, `osprey.simulation.model_request`, `osprey.utils.facility` |
| `osprey_facility.build` | framework job: `osprey.cli.build_cmd`, `osprey.cli.facility_cmd`, `osprey.facility.views` |
| `osprey_facility.errors` | framework job: `osprey.cli`, `osprey.facility.settings`, `osprey.facility.views`, `osprey.integrations.canopy`, `osprey.mcp_server.python_executor`, `osprey.registry`, `osprey.runtime`, `osprey.services.ariel_search`, `osprey.simulation`, `osprey.utils.facility` |
| `osprey_facility.frontmatter` | framework job: `osprey.services.facility_knowledge` |
| `osprey_facility.kinds` | kind plug-in |
| `osprey_facility.layers` | framework job: `osprey.cli.facility_cmd`, `osprey.facility.bundled` |
| `osprey_facility.layers.list` | framework job: `osprey.cli.facility_cmd` |
| `osprey_facility.layers.mml.family` | framework job: `osprey.services.channel_finder` |
| `osprey_facility.layers.mml.importer` | framework job: `osprey.cli.facility_cmd` |
| `osprey_facility.layers.mml.mapping` | framework job: `osprey.cli.facility_cmd`, `osprey.facility.bundled` |
| `osprey_facility.layers.mml.response_check` | framework job: `osprey.cli.facility_cmd` |
| `osprey_facility.layers.mml.seed` | framework job: `osprey.cli.facility_cmd` |
| `osprey_facility.layers.relations` | framework job: `osprey.cli.facility_cmd` |
| `osprey_facility.measurement_groups` | framework job: `osprey.facility.views` |
| `osprey_facility.pairing` | framework job: `osprey.facility.views` |
| `osprey_facility.reading` | fork/tool |
| `osprey_facility.records` | framework job: `osprey.apps.sdk`, `osprey.cli`, `osprey.facility.settings`, `osprey.facility.views`, `osprey.interfaces`, `osprey.mcp_server`, `osprey.services.ariel_search` |
| `osprey_facility.references` | framework job: `osprey.interfaces.ariel`, `osprey.interfaces.okf_panel`, `osprey.mcp_server.ariel`, `osprey.services.ariel_search` |
| `osprey_facility.relations` | framework job: `osprey.cli.facility_cmd`, `osprey.facility.views` |
| `osprey_facility.render` | framework job: `osprey.apps.sdk`, `osprey.cli.build_cmd`, `osprey.cli.facility_cmd`, `osprey.facility.views` |
| `osprey_facility.roots` | framework job: `osprey.cli`, `osprey.facility.bundled` |
| `osprey_facility.scenarios` | framework job: `osprey.facility.views`, `osprey.simulation.apply` |
| `osprey_facility.schema` | fork/tool |
| `osprey_facility.served` | framework job: `osprey.cli`, `osprey.facility.settings`, `osprey.facility.views` |
| `osprey_facility.sessions` | framework job: `osprey.agent_runner`, `osprey.interfaces.web_terminal`, `osprey.registry`, `osprey.simulation` |
| `osprey_facility.simulated` | framework job: `osprey.facility.views` |
| `osprey_facility.sources` | framework job: `osprey.facility.views` |
| `osprey_facility.validate` | framework job: `osprey.cli`, `osprey.facility.views`, `osprey.mcp_server.channel_finder_graph`, `osprey.registry` |
