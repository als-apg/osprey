A TANGO or DOOCS read that cannot reach its device now reports a connection
error naming the channel, as an EPICS read already did. The control-system
tools answer it the way they answer an unreachable EPICS channel — the
operator is told the control system could not be reached, and the connector
is rebuilt on the next call — instead of reporting it as an unexpected
internal error.
