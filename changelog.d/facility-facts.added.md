`osprey build` writes each render's `data/facility_facts.json` and
`data/facility_facts.md`: the facility's identity, place levels, device classes
with their aliases and families, models and channel count. The agent context
takes the facility name and these facts from that file, and a render without
one is read as a facility with no sources, named after the project.
