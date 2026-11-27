**Breaking change:** the deployment root no longer has a `data/` directory,
and the `data:` profile key is retired. Sources move:

- `data/facility/` to `facility/`
- `data/landing/` to `web/landing/`
- `data/canopy/` (`library/`, `collections/`, `theme/`) to `web/canopy/`
- `data/<lane>_curve/` certificate directories from before the runtime move to
  `var/runtime/<lane>_curve/`

The render still writes these trees under `build/data/`, at the paths it wrote
them before. Content at the top of `data/` outside these trees no longer
reaches the render; content the render must see goes under `facility/`. The
`data/raw/` convention is retired: `data/raw/` is handled like any other
leftover.

Every command that reads the deployment root, `osprey set` included, now stops
before reading anything when a retired path, a `data:` line or anything else
left in `data/` is present. It prints one block that moves the files, rewrites
the profile values that name a moved path (keeping a `./` prefix), deletes the
`data:` line, and lists what is left for you; when nothing listed needs you
first, the block's last line runs `osprey init --revalidate <repo>`. To
migrate, run `osprey init --revalidate`, or paste the printed lines. When
nothing is left for you, one run of either takes a tree from any earlier
layout to the current one, and a second run changes nothing. A leftover in
`data/`, a clash, or a `data:` line that was not deleted keeps the repo
refused until you have dealt with it.

A directory that already exists at a new path is merged into, file by file. A
file present at both its old and its new path is a clash: it is named and left
to you, the printed lines still move everything else, and
`osprey init --revalidate` moves nothing until you have resolved it.

A `data:` value that names one other directory inside the repository is
followed: the facility and web content move out of that directory, and the
line is deleted once the directory holds nothing else.

Left for you, never moved or guessed:

- anything else still in `data/`, listed by name; only the `data/README.md`
  osprey shipped, unmodified, is deleted
- absolute and `{project_root}/` profile values that name a moved path, listed
  to check by hand
- any other `data:` value (a path outside the repository, different values in
  different profile documents, a directory that holds or lies inside a current
  entry): nothing is moved from it and the line is not deleted; move what it
  held by hand, then delete the line

Certificates are moved as they are, without minting new keys, by
`osprey init --revalidate` or the printed lines. `osprey up` moves none: it
refuses a repo that still holds the old directories.

The image's copy of the repository moves with it, so tooling that spells
source paths as `/app/<project>/data/...` inside the image now spells them
`/app/<project>/facility/...` or `/app/<project>/web/...`; paths under
`build/data/` do not change. `osprey.apps.sdk.testing.synthetic_deployment`
places the facility tree at `facility/`.
