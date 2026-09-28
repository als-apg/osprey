A `type: users` section of the web-terminal landing page takes `names: hidden`,
which replaces its name cards with one "Log in to your terminal" button, so the
page no longer lists who works there. Service trays and link sections are
unchanged, and `names: shown` stays the default. `osprey build` refuses
`names: hidden` unless `auth.method` is `password` or `oidc`, since under
`token` or `none` the name card is the only way in.
