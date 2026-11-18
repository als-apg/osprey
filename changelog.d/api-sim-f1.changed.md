`osprey sim apply`, `osprey sim list`, `osprey sim status`, the deploy's archive seed and
its scenario re-apply, and the scenario-session artifacts read the active set
through the connectors' one composition (`osprey_connectors.simulation.state.compose`).
What each serves, prints and stores is unchanged. The facility build's warning
about a scenario moving a toleranced readback beyond its tolerance now judges
the scenario together with `nominal`, as the simulator serves it.
