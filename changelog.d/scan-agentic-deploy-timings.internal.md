The Scan Stack Agentic lane reports how long its stack deploy took on a green run
too: the `osprey up` time, the wait for the bridge and the worker environment,
and `osprey up`'s own per-phase output, in the job summary. Before, that output
was printed only when the deploy failed.
