A first deploy no longer skips seeding the facility knowledge graph. The wait
for the graph store to accept a connection was shorter than the store's own
healthcheck grace period, so on a first start the deploy gave up while the
container was still coming up — and the deployment came up with an empty graph
and nothing to say why.
