A profile whose `project/` mirror ships a file under `data/simulator/` stops
the build with one `facility: profile-invalid:` line naming `data/facility/` as
the tree the facility is authored in. Agent-side writes under `data/simulator/`
are refused.
