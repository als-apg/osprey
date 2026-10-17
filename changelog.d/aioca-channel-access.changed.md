The EPICS connector speaks Channel Access through aioca instead of pyepics,
and pyepics is no longer a dependency: aioca is the one Channel Access client,
on one long-lived event loop per process, so a process that also uses aioca
(ophyd-async devices, for example) shares one CA context instead of two. Every
read now asks the IOC, so `fresh_reads` no longer changes anything. A write to
a channel that never connects reports that nothing was sent. A `connect()`
that points Channel Access at a different gateway gives the process a fresh CA
context, as pyepics did, closing the old one's channels; the connector's
monitors are opened again on the new context and keep delivering. `epics_ca`
dispatch triggers watch through the connector, so they stay armed across such
a repoint too. The archiver recorder samples through the connector as well,
with the `EPICS_CA_*` its container is started with, so its channels live on
the process's one CA loop; one unreachable channel still costs only itself,
and every sample is still bounded by the cadence. Text written to a numeric
channel is read by that channel's own field type before it is sent, so `"1e3"`
is refused for an integer field instead of landing as 1; text written to a
char waveform is stored as a string; and a put the IOC answers without taking
(a disabled record) is reported as failed.
