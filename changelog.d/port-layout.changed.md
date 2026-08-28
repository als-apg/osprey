**Breaking change:** the framework's default host ports follow one layout.
Every per-index family now starts at a round hundred so the index reads off the
port: web terminals `9100+i`, artifact gallery `9200+i`, ARIEL `9300+i`,
lattice `9400+i`, channel finder `9500+i`, knowledge bundle `9600+i`, system
health `9700+i` (previously `9091+i`, `9291+i`, … `9791+i`). The dispatcher
moves from `8020` to `9900` and worker *w* from `9190+(w-1)` to `9900+w`, out of
the terminal family's hundred; the qmd sidecar moves from `8180` to `9800`, out
of bluesky lane 2's. nginx (`9080`), the auth sidecar (`9070`), the bluesky
lanes (`8090`/`8190`/`8095`), Channel Access (`5064`) and every vendor default
are unchanged. A profile that spells any of these keys keeps its numbers; one
that relied on the defaults gets the new ones at its next `osprey build`, and
anything that dials a default by hand — tunnels, bookmarks, webhook callers,
`DISPATCHER_URL`/`WORKER_URL` — must follow.
