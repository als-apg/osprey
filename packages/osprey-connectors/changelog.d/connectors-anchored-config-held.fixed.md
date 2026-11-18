A command pointed at a deployment with `--repo` keeps reading that deployment's
config, facility timezone included, after it looks up an agent-data directory.
Before, the lookup could switch the rest of the command to the config that
`OSPREY_CONFIG` or the working directory named.
