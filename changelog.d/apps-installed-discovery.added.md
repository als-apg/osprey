Applications can come from an installed package as well as from `apps/<name>/`:
a distribution declares one under the `osprey.apps` entry-point group, and the
profile's `apps:` block lists it as any other. A directory under `apps/` may not
take an installed application's name.
