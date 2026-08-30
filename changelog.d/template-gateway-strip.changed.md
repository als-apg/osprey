The Control Assistant template no longer ships ALS production gateway
addresses in its `epics` block. A stock build carries no live gateway
config at all — the gateways, `probe_channel` and
`live_gateway_acknowledged` ship as commented placeholders, and authoring
them with your facility's values is the go-live edit. Until then the live
target reads `not configured` in the roster, which is the truthful state of
a fresh deployment.
