The dispatch worker's retention sweep imports `asyncio` at module scope instead
of inside the loop function, so the name it awaits through can be replaced by
tests without touching the process-wide one.
