A completed `chromaticity_monitor` measurement no longer fails while it is
saved: `pyaml_measure` writes each measurement as JSON itself, every array as a
list, where pyAML's own save could not write the monitor's arrays.
