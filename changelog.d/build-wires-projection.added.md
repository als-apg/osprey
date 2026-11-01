`osprey build` checks a deployment's applications before writing anything: manifest and
per-login rules, pinned versions, projected ports and vendor files each refuse by name. It then
projects each application the profile's `apps:` block lists into every render, writes each
service's `services/<key>/app-config.yml`, reports an `apps/<name>/` the block does not list (nothing
of it is read, built or installed; a package that is no application is a `dependencies` local
path), and the deploy summary gains an `apps.<name>` block.
