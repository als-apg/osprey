`osprey build` validates every simulator view file into its document model
before it writes it, and writes it through the connectors' serializer, so a
seed, channel or wiring record the simulator would refuse stops the build
(`value-invalid`, naming the file and the key) instead of the simulator. The
files it writes are byte for byte as before, except that an `HH:MM` calendar
time in `scenarios.json` is written `HH:MM:SS`. `osprey sim` and the virtual
accelerator read every view file when they open a view, and refuse one of
another schema there.
