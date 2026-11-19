A scenario's `stopped_servers` stops simulated servers by name: `sim-<model>`
for a served physics model, `sim-texture` for every channel no physics model
owns (`osprey_connectors.simulation.servers`). While the scenario is active no
channel of a stopped server answers: the in-process connector raises
`ChannelUnreachableError` for a read, fails a write with nothing sent and tells
subscriptions the channel disconnected, and `Composite.stopped()` lists the
addresses. A channel answers again, with its current value, once the scenario
is no longer active.
