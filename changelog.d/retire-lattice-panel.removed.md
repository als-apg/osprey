**Breaking change:** the LATTICE panel is retired: the lattice dashboard
server, its `lattice_*` workspace tools and its port family (offset 400,
`OSPREY_LATTICE_DASHBOARD_PORT`) are gone. A profile that still lists `lattice`
in `web_panels:`, names it as `default_panel`, or states `web.panels.lattice` or
a `lattice_dashboard` section builds without it and logs one warning per line.
Optics questions go to the pyat-specialist agent.
