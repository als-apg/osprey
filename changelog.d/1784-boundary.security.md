The web terminal, nginx, login and dispatch-worker containers run with
`no-new-privileges`, so no process in them can gain a privilege through a
setuid binary. The deployment guide names the facility's EPICS access security
as the write boundary that does not depend on OSPREY's own code.
