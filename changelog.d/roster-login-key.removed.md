**Breaking change:** `login: false` on a `modules.web_terminals.users` entry
is gone. It served one card without a login while the rest of the roster sat
behind the wall; share a card with the whole roster instead by setting
`access: any` on it, which any roster login opens with their own credential.
A profile still carrying `login: false` deploys that entry behind the login
wall like every other: replace the key with `access: any` where the card was
meant to be shared, and rebuild. `osprey users login-url` now applies to
`auth.method: token` deployments only.
