The virtual accelerator serves the writes already queued when a write's cycle
starts in that one publishing pass, so an apply of many settings sent
concurrently completes in about one pass instead of one pass per setting. A
write that arrives alone is still served at once.
