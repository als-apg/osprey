`osprey build` now writes a registry-mode profile's `deploy.registry.url` into the
rendered `registry.url` when the profile's `config:` block names none, so the web
terminals' registry is stated once. A `registry.url` in `config:` still wins.
