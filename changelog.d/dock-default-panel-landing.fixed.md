The web terminal opens on the panel a profile pins with `default_panel:` even
when that panel is one the terminal starts and health-polls, such as
JupyterLab. Before, another panel took the slot while the default's first
health probe was still in flight.
