`osprey sim apply` left the ARIEL text and image embedding tables missing until a
manual `osprey ariel migrate`: the purge that clears the previous narrative drops
them, and a running ingest watcher never recreates them. The seed step now migrates
again after the purge, so vector and picture search work on the reseeded logbook.
