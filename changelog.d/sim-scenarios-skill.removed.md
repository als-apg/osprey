The `sim-scenarios` skill is no longer shipped into deployments. Simulation
scenarios are the fixtures a deployment's agent is exercised against, not
something it lists, switches or writes: applying one purges and reseeds the
logbook and rewrites the stored archive. A person switches scenarios with
`osprey sim apply`, and the simulation bundle reference holds the authoring
rules the skill used to carry. No shipped preset listed the skill; a profile
that added it to `skills:` drops the line.
