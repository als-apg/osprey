The `control-assistant` preset is now `demo`: `osprey init osprey-demo --preset
demo`. It is the full OSPREY demo on the simulated example facility, and still a
starting point to trim down (`hello-world` is the one to build up from). The old
name keeps resolving, so existing repos, `extends: control-assistant` and
`osprey scaffold pull control-assistant:…` work unchanged.
