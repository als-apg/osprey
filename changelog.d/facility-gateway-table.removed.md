**Breaking change:** `osprey set epics_gateway=<facility>` is gone, along with
the built-in table of named facilities' EPICS gateway hostnames it expanded. A
gateway address is site infrastructure, so write your own:
`osprey set config.control_system.connector.epics.gateways.read_only.address=gw.example.org`.
The command now refuses the old spelling with that line rather than writing a
key the next build rejects.
