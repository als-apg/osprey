A deployment's web terminals are now logins, one file each in
`tiers/<name>.yml`, which users and roles pick with `tier:` and
`default_tier:`. The bundled ones are the access levels `readonly`,
`readwrite` and `admin`, and the narrow `logbook`, `knowledge` and
`channel-finder` logins (`lists: own`), which share the deployment's services
and data, are listed under `modules.web_terminals.shared:` and open from a
shared card on the landing page. `osprey init` writes the folder out in full,
and `osprey scaffold tiers` adds it to an existing repo. `dispatch.tier` names
the login the event dispatcher's workers run. `osprey profile card` lists the
`graph` and `event_dispatcher` servers when a render ships them.
