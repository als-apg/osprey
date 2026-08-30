Folded into the write-outcome entry: a confirmed write carries the post-write
readback — the EPICS and Mock connectors read the channel once after the write
lands and report the observed value and alarm state, and the DOOCS connector no
longer reports a string readback as `0.0`.
