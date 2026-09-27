`osprey users remove` and `osprey users prune` with `--archive` or `--purge`
no longer report a volume as removed when the container runtime refused to
remove it. They finish every other step, name each volume that was kept with
the runtime's reason, and exit non-zero.
