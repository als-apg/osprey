The simulator view names its health signals in server words:
`osprey_connectors.simulation.view.SERVER_HEALTH_SIGNALS`,
`PERIODIC_SERVER_HEALTH` and `SimulatorView.server_health_by_model()`, read
from the `server_health` key of `addresses.json`. The released names
`IOC_HEALTH_SIGNALS`, `PERIODIC_IOC_HEALTH` and
`SimulatorView.ioc_health_by_model()` stay as aliases of them, and a view whose
`addresses.json` has no `server_health` is read under `ioc_health`, which is
what a framework release from before the rename writes. The contract
docstrings of the directory, alarm and put-log kinds say channel and server
where they said PV and IOC.
