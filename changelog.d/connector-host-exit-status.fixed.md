Putting a connector-host child down sends no signal to a child that has
already exited or already been reaped. A pid that asyncio's child watcher
thread had reaped could otherwise still be signalled, and by then it may
belong to another process.
