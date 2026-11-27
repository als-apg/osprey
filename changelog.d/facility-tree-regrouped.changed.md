**Breaking change:** the facility source tree under `data/facility/` is
regrouped, and a tree still laid out the old way is refused. Sources move:

- `classes.yaml`, `signals.yaml` to `vocabulary/`
- `models.yaml`, `seeds.yaml`, `decks/`, `measurement/`, `scenarios/` to
  `simulation/`
- `fixes.yaml`, `documents.yaml`, `policy.yaml` to `provenance/`
- `benchmarks/cross_paradigm/queries/` to `benchmarks/channel_finder/`

`identity.yaml`, `limits.yaml`, `records/`, `knowledge/`, `ariel/` and
`imported/` stay. Authored `deck:` values change from `decks/<model>.json` to
`simulation/decks/<model>.json`. Every command that reads or writes the tree
(`osprey build`, `osprey validate`, `osprey facility validate`, `show` and
`import`, `osprey scaffold pull`) now stops before reading anything when a
retired path is present and prints the `mkdir -p`, `git mv` (or `mv`) and
`deck:` edit lines that move the tree; when nothing listed needs you first, the
block's last line runs `osprey init --revalidate <repo>`. A file present at
both an old and a new path is reported and left to you. To migrate, run
`osprey init --revalidate`, or paste the printed lines; the tree then builds to
the same facts as before. `osprey init` writes the new layout.

App repositories that build trees for `osprey.apps.sdk.testing.write_tree`
update the keys by hand: `models.yaml` becomes `simulation/models.yaml`, and
likewise for the other moved files and directories; `model()` now defaults to
the new models file.

A machine repository mounted from another repository (`machine:`) is moved in
that repository, not by `--revalidate` in the deployment, which lists it under
what is left for you. Upgrade OSPREY in every deployment that mounts it first;
each deployment's builds then refuse the machine and print the commands that
move it. Next, move the machine repository (upgrade OSPREY there, run
`osprey init --revalidate` or the printed commands, commit), then commit the
new submodule commit in each deployment. An older OSPREY reads a moved machine
without the moved files and says nothing, so the upgrade comes first.
