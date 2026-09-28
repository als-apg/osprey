The EPICS connector now uses pvapy (`pvaccess`) for both Channel Access and
PVAccess, and `pyepics` and `p4p` are no longer dependencies of
`osprey-connectors`. pvapy's wheels carry their own EPICS libraries, including
for linux/aarch64, so a bare-metal arm64 install no longer compiles the EPICS
client stack. Every Channel Access read now goes to the IOC, so there is no
monitor cache to go stale and no `fresh_reads` option to bypass one; write
confirmation still waits for the IOC's put-callback. Python 3.14 is not
supported until pvapy publishes wheels for it: OSPREY now requires Python 3.11,
3.12 or 3.13.
