A connector-host child that exits on its own before answering its init frame
is now reported with the exit code it exited with. Terminating a child that
had already exited but was not yet reaped stole its exit status from the event
loop, so the start error could say "exit code 255" for a child that exited 3.
