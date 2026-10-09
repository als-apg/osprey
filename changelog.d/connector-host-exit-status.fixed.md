A connector-host child that exits on its own is reported with the exit code it
exited with, where a start error could say "exit code 255" for a child that
exited 3. Putting a child down no longer reaps it behind asyncio's child
watcher: the supervisor signals only a child that is still running and leaves
an exited one for the watcher to reap.
