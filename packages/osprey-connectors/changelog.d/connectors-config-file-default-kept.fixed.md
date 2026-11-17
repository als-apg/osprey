A process pointed at its config with `CONFIG_FILE` keeps reading that file
after it looks up an agent-data directory. Before, the lookup could switch the
rest of the process to the file `OSPREY_CONFIG` or the working directory
named. A connector-host child now reads the config file its parent handed it
for every lookup.
