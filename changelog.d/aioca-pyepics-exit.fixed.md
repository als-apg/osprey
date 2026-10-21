A process that used aioca before the EPICS connector connected through a
gateway no longer segfaults at exit, and aioca calls on that thread no longer
time out after `connect()`. The connector speaks Channel Access through aioca
itself now, so the process has one CA context, and the connector's CA loop is
stopped before aioca tears that context down at exit.
