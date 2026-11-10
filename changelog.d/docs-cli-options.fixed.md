The CLI reference now documents every option the commands accept. Added:
`osprey init --reset` (which destroys the previous deployment's containers,
data volumes and images without asking), `osprey build --dev`,
`osprey up`/`osprey restart --reuse-stores`, `osprey web --skip-preflight`,
the `osprey channel-finder preview` display options, `osprey ariel enhance
--refresh-stale` and `--entry`, the retired `osprey scaffold web-terminals
--config`, and the short forms of several options. The overview now lists `osprey query` and
`osprey sim`, the web terminal's default port is given relative to
`port_base` instead of as a fixed number, and `init --force` names
the files it really replaces and keeps. A new docs test fails when a visible
option, or a top-level command in the overview, goes undocumented.
