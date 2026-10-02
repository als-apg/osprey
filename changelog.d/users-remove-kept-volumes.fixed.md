`osprey users remove` no longer says a user's volumes were removed when it
cannot update `profile.yml`. The message now names each volume the runtime
kept, with its reason and the `osprey users prune` command that removes it,
says the volumes were kept when no `--archive` or `--purge` was given, and
claims no removal on a re-run that finishes an earlier one.
