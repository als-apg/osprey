**Breaking change:** every host port a deployment publishes now derives from
one key — `deployment.port_base`, default `10000` — plus a fixed offset, so
a deployment occupies the block `port_base` to `port_base + 999`. Every
default moves, and single-user and multi-user mode now serve each panel on
the same port, as user index 0, so the two can no longer run side by side on
one host at one base. Channel Access is the exception: virtual accelerator
instance 1 keeps `5064`, the port EPICS clients expect.

Ports that move:

- nginx landing page `9080` → `10000`
- web terminal `8087` single-user, `9091+i` multi-user → `10100+i`
- event dispatcher `8020` → `10010`
- dispatch worker *w* `9190+(w-1)` → `10010+w` (host-network mode only)
- artifact gallery `8086` → `10200+i`
- ARIEL `8085` → `10300+i`
- lattice dashboard `8097` → `10400+i`
- channel finder `8092` → `10500+i`
- PostgreSQL, which required `services.postgresql.port_host` → `10800`

Services added since the last release take their layout ports directly: auth
sidecar `10001`, OpenObserve `10050`, qmd sidecar `10060`, tiled `10070`,
bluesky web `10071`, bluesky bridge lanes `10080`/`10081`, virtual
accelerator instance *n* ≥ 2 `10090+(n-2)`, knowledge (OKF) panel `10600+i`,
system health `10700+i`, MongoDB `10801`, GraphDB bolt and HTTP
`10802`/`10803`. `10900`–`10999` is reserved for facility services.

To move a deployment onto another block, set the base and rebuild:

```bash
osprey set config.deployment.port_base=20000
osprey build
```

Two deployments then coexist on one host by taking two bases; a second one
that also runs a virtual accelerator has to set
`services.virtual_accelerator.port` by hand, since Channel Access does not
move with the base. A config that spells a port explicitly still wins over
the layout, and `modules.web_terminals.web_base_port` is no longer required:
it defaults to `port_base + 100`. `dispatch.worker_port_stride` is a new
profile key (default `1`) that spaces host-mode workers inside the band. The
MongoDB archiver connector no longer guesses `27017`:
`archiver.mongodb_archiver.port` is required and a missing one is refused at
connect, with `osprey build` writing it for a project that deploys its own
store.

A deployment built before this release keeps its old ports and reports a
staleness advisory until its next `osprey build`; anything that dials a
default by hand — tunnels, bookmarks, webhook callers, `DISPATCHER_URL` /
`WORKER_URL` — has to follow the move.
