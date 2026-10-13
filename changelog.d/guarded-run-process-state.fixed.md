A pyAML tool run from a helper thread or an asyncio task inside a guarded run
now runs under that run: its writes are journaled into the run's one journal
and restored with it, where it was refused as busy against its own process. A
guarded run belongs to the process, so a context copied inside a run is
refused once the run has ended, and a child forked inside a run no longer
clears its parent's journal when it leaves the run.
