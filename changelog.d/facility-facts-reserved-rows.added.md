A profile whose `project/` mirror ships `data/facility_facts.json` or
`data/facility_facts.md` stops `osprey build` and `osprey facility validate`
with one `facility: profile-invalid:` line naming `data/facility/` as the tree
the facility is authored in. Agent-side writes to those two paths are refused.
