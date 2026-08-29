An emitted `profile.yml` now explains its two hardest keys in place. Above
`web_panels:` it says which panels can simply be uncommented and shows what a
panel of your own looks like — a list entry plus its `web.panels.<id>.url`
address under `config:`. Above `env:` it shows the block that replaces
`env: {}`, naming `required`, `pinned`, `defaults` and `file`. Both are
comments, so nothing about what the profile builds changes.
