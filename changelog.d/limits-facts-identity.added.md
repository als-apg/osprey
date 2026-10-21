**Breaking change:** `control_system.limits_checking.allow_unlisted_channels` is
replaced by `control_system.limits_checking.mode: exclusive | optional` (`false`
is `exclusive`, `true` is `optional`), and a config that still states the old
key fails the build. The `data/channel_limits.json` that `osprey build` renders
holds only the records in `data/facility/limits.yaml`; a channel with no record
follows the mode. The limits, the agent facts and the facility's display name
all come from the build's facility sources, and the profile's
`dispatch.facility_name` and the config's `facility.name`, `facility.timezone`
and `facility.ontology` are gone. Re-baselined on purpose: the
`control-assistant` render's limits database carries the three records its
`limits.yaml` holds (`SR:MAG:HCM:01:CURRENT:SP`,
`SR:RF:CAVITY:01:FREQUENCY:SP`, `SR:VAC:ION-PUMP:01:VOLTAGE:SP`) in place of
one entry per served address, and the `hello-world` render's carries its three
records with no `defaults` block.
