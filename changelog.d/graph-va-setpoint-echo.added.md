A graph-mode virtual accelerator now serves setpoint-echo channels. The
channel roster read from the knowledge-graph corpus carries, for every
settable channel, the readback the corpus itself states for it: a device
(`narad_p:hasBinding`) whose write binding is named `<stem>Setpoint` and
whose read binding is named `<stem>Monitor` (`Setpoint`/`Monitor` for a
magnet's current, `GapSetpoint`/`GapMonitor` for an insertion device's gap).
`osprey build` turns each stated pair into a setpoint-echo pair in
`data/simulation/channel_manifest.json`, keyed on the setpoint's own address
with the other identity keys left empty, so a write to the setpoint is
echoed onto its readback by the IOC without a hierarchy path or a lattice
model. Every channel the corpus pairs nothing with is still served pathless
and static-noisy, and an ambiguous pair (a readback claimed by two setpoints,
a chain of setpoints) is dropped to static-noisy on both sides rather than
served as half an echo. The build's manifest fact says how many setpoints
the corpus paired; a corpus that pairs nothing still gets the "serves 0
setpoints" sentence.
