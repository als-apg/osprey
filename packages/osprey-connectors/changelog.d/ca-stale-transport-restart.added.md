`ControlSystemConnector.wait_stale()` raises `ConnectorUnavailableError` — the
lost connector, which its supervisor replaces — once the connector's transport
has gone stale: the IOC serves a channel again but the client library keeps it
down. EPICS CA with a name server configured watches the channels that
connected and went down: a channel counts once another channel connected
after it went down, or once it has been down for 30 s with nothing
connecting. Only channels that connected in this process count, so in a
replacement a channel that never comes back cannot drive a restart loop.
Every other connector waits for ever.

Known limit: a process in which no channel has ever connected never reports
stale.
