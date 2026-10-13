A Ctrl-C now stops a pyAML tool running in a helper thread of the guarded run
through its callback, as it stops one on the main thread. The Ctrl-C marks the
open run, and every `run_tool` of the run reads that mark, so the whole run is
restored and reported once.
