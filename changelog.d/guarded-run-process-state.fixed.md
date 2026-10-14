A pyAML tool run from a helper thread or an asyncio task inside a read-write run
now runs under that run's lock, where it was refused busy against its own
process. The lock belongs to the process, and a forked child holds no lock.
