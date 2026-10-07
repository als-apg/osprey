A connector-host child that exits on its own is reported with the exit code it
exited with. Putting such a child down could reap it behind asyncio's child
watcher on Linux, and the pool then named its exit code as 255; the supervisor
now signals only a child that is still running and leaves an exited one for the
watcher to reap.
