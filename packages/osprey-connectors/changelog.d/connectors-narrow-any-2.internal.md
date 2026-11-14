A second narrowing pass replaces most of the remaining `Any` in the package
with the types the code already knows: the simulator view's records, the
archiver clients and Archiver Appliance reply, the EPICS put outcomes and
gateway config, the non-EPICS connectors' send steps and journals, the config
reads and factory classes, the logger keywords, and the connector-host frames
and pool calls.
