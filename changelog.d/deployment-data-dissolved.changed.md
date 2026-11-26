**Breaking change:** the deployment root no longer has a `data/` directory,
and the `data:` profile key is retired. Sources move:

- `data/facility/` to `facility/`
- `data/landing/` to `web/landing/`
- `data/canopy/` (`library/`, `collections/`, `theme/`) to `web/canopy/`
- `data/<lane>_curve/` certificate directories from before the runtime move to
  `var/runtime/<lane>_curve/`

The render still writes the same trees under `build/data/`, so the agent, the
facility views and built images see what they saw before. Content at the top
of `data/` outside these trees no longer reaches the render; content the render
must see goes under `facility/`. The `data/raw/` convention is retired:
`data/raw/` is handled like any other leftover.

Every command that reads the deployment root, `osprey set` included, now stops
before reading anything when a retired path, a `data:` line or anything else
left in `data/` is present. It prints one block that moves the files, rewrites
the profile values that name a moved path (keeping a `./` prefix), deletes the
`data:` line, and lists what is left for you; when nothing listed needs you
first, the block's last line runs `osprey init --revalidate <repo>`. To
migrate, run `osprey init --revalidate`, or paste the printed lines. Either
way, one run takes a tree from any earlier layout to the current one, and a
second run changes nothing. A root `facility/` or `web/` already in the way is
refused, naming the clash, and nothing moves.

Left for you, never moved or guessed:

- anything else still in `data/`, listed by name; only the `data/README.md`
  osprey shipped, unmodified, is deleted
- absolute and `{project_root}/` profile values that name a moved path, listed
  to check by hand
- a `data:` value that names a directory outside the repository: nothing is
  moved from it and the line is not deleted; move what it held by hand, then
  delete the line

Certificates are moved as they are, without minting new keys, and no command
other than `osprey init --revalidate` moves them: `osprey up` refuses a repo
that still holds the old directories.

The image's copy of the repository moves with it, so tooling that spells
source paths as `/app/<project>/data/...` inside the image now spells them
`/app/<project>/facility/...` or `/app/<project>/web/...`; paths under
`build/data/` do not change. `osprey.apps.sdk.testing.synthetic_deployment`
places the facility tree at `facility/`.
