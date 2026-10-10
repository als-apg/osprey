The lattice dashboard is part of the control-assistant preset: the LATTICE tab
launches from `lattice` in `web_panels:` with no `lattice_dashboard:` section,
reads the build's simulator view, and switches between the build's models from
its header. It opens on the first served model, and a single-pass model draws
optics only. The `lattice_init` tool and the `POST /api/state/init` route are
gone; the selected model's deck is the lattice. The rendered config of the
control-assistant preset and of its readonly, readwrite and admin personas now
carries `web.panels.lattice`; the knowledge and logbook personas leave it out.
