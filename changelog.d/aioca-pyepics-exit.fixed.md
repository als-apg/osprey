A process that used aioca before the EPICS connector connected through a
gateway no longer segfaults at exit, and aioca calls on that thread no longer
time out after `connect()`. The connector now loads pyepics' Channel Access
library lazily, on its own context, instead of adopting (and then detaching
the thread from) the context aioca created.
