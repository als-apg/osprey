`mml_export.m` 2.1.0 writes a sixth file, `<stem>.model.json`: what the Middle
Layer's model answers for the exported lattice (tune, chromaticity, dispersion
and their responses, in physics and hardware units). Each section reloads the
saved lattice first, and a section that fails is recorded as refused rather than
stopping the export. The exporter ships with the mml layer as well as in the
control-assistant data template, and the template README describes re-running
the export on a MATLAB host.
