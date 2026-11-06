The CLI reference now documents every option the commands accept. Added:
`osprey init --reset` (which destroys the previous deployment's containers,
data volumes and images without asking), `osprey build --dev`,
`osprey up`/`osprey restart --reuse-stores`, `osprey web --skip-preflight`,
the `osprey channel-finder preview` display options, `osprey ariel enhance
--refresh-stale`, the retired `osprey scaffold web-terminals --config`, and the
short forms of several options. The overview now lists `osprey query` and
`osprey sim`, the Artifact Gallery and web terminal default ports are given
relative to `port_base` instead of as fixed numbers, and `init --force` names
the files it really replaces and keeps. A new docs test fails when a visible
option, or a top-level command in the overview, goes undocumented.
