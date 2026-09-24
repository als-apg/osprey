Bin edges from the mock, MongoDB, DOOCS and MYA archivers now start at the
query window, so a window opening off the old midnight-anchored lattice shifts
its bins (1-hour bins from 00:17 are labelled 00:17, 01:17 rather than 00:00,
01:00); EPICS keeps the Archiver Appliance's own epoch-anchored grid.
