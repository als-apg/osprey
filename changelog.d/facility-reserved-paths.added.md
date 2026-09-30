A profile whose `project/` mirror carries `facility.json` or anything under
`data/facility/` stops `osprey build` and `osprey facility validate` with one
`facility: profile-invalid:` line naming `data/facility/` as the tree the
facility is authored in. `osprey facility validate` writes nothing outside its
scratch render.
