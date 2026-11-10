`osprey init --revalidate` now applies only what a preset changed since the
repo last took it, and leaves your own edits alone. `osprey init` keeps a plain
copy of what the presets produced in `.osprey/preset-base/`; commit it with the
repo. A key, panel, step or provider entry the preset added is added where you
lack it. One the preset removed or changed is removed or updated only where you
kept its old value. Anything you deleted is not added back, and a change you
drop from the diff before committing is not offered again. What it does not
apply but you may want, such as a changed application in `apps/`, is printed
once. `osprey validate` and `osprey build` say in one line when there is
something to apply, and `osprey up` names `osprey init --revalidate` before the
rebuild. A repo made before this starts tracking on its first `--revalidate`.
This replaces reading `# DEVIATION:` markers to keep a deleted provider entry
out.
