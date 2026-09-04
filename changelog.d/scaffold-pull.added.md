The new `osprey scaffold pull PRESET[:PATH]` command copies one piece of a
shipped preset's app template, such as control-assistant's
`data/facility_knowledge` skeleton, into the current deployment repo to edit
and commit. It refuses to overwrite an existing file unless `--force` is
given, and `--list` shows every path the preset offers before you pull one.
