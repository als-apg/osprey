The DOOCS, TANGO and simulator connectors now refuse a write when limits
validation fails for any reason, not only when the value breaks a configured
limit. An error that stopped the check from being made — a missing database, a
validator that could not run — used to be logged and the value sent anyway;
it is now reported as a refused write and nothing reaches the control system.
The EPICS connector already behaved this way.
