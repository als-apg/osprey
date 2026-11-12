Dispatch workers can connect to the event dispatcher over an authenticated
ZeroMQ bus instead of listening on ports of their own: set
`dispatch.transport: bus`. Every provisioned worker then takes runs (the least
loaded first), a trigger that fires before any worker is up waits for one, and
`osprey up` mints each container its own CURVE key. The chat bridges read runs
through the dispatcher's new `/worker` routes. HTTP stays the default.
