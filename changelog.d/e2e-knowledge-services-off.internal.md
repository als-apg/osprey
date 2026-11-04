Six Docker E2E lanes that prove the virtual accelerator, the bluesky bridge or a
sidecar of it (VA substrate, ORM/grid/bump/preflight roundtrips, bluesky web,
Tiled, sandbox escape, notebook panel) no longer deploy the agent's knowledge
services: qmd, the graph store, ARIEL's Postgres and the ClickHouse record
store. Nothing in them read those services, and together they were about 9 GB of
images, enough to run a 72 GB runner out of disk. The lanes no longer build the
qmd image either; qmd is proven with its models in its own lane.
