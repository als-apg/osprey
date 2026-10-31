The `orm` and `orbit_bump_sweep` plans run again under bluesky 1.16, where
they failed before taking a point with `Plan failed: 'setpoint'`. bluesky 1.16
changed `bps.locate` to answer by device name; the plans now read each
working point with `bps.loc`. OSPREY now requires `bluesky>=1.16.0`, and the
`writing-bluesky-plans` skill teaches `bps.loc`.
