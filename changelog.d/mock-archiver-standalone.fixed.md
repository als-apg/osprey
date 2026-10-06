The mock archiver connects when only `osprey-connectors` is installed. Its
connect no longer fails with `No module named 'osprey'`: the simulation
file it derives from the control-system config is now resolved inside the
connectors package.
