`osprey_connectors.ipc.declared_gateways(block)` classifies how a connector
block spells its gateways table (`GATEWAYS_NONE_DECLARED` for an explicit
`gateways: []`, `GATEWAYS_MISSING`, `GATEWAYS_ROWS`), and the derivation carries
it as `gateways_declared`; `declares_no_endpoints` now reads that verdict.
