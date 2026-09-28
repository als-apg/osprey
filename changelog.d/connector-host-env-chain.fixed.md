A connector-host child no longer loads the project `.env` from its working
directory when it reads its config file, so an `EPICS_CA_*` or `EPICS_PVA_*` line
there cannot undo the scrub the child was started with.
