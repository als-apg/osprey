Writes to a virtual accelerator whose setpoint addresses do not end in `:SP`
now reach the lattice. Which channels are setpoints and which are their
readbacks is read from each channel's `subfield` in the manifest, as the
manifest already declared it, instead of from the address text — so drive
limits, value ranges and physics routing apply to a facility's own namespace
however it spells an address.

A build whose hierarchical channel database is levelled some other way now
says so and serves that project's channels, rather than refusing with
"could not be built … Repair the file" against a valid file, or coming up
as a silently static machine.
