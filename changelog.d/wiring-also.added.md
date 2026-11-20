A wiring record may list further channel addresses of its variable under
`also`. Each reads what the record's `address` reads at the same instant and
writes what it writes, waveforms included. The build stops when two wiring
records read the same axis of one monitor; before, the second took the first
one's readout and the first one's fault variables stopped acting.
