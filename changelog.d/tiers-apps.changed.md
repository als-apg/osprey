A deployment's web terminals are now tiers and apps. A tier is an access level
(`readonly`, `readwrite`, `admin`) in `tiers/<name>.yml`, which users and roles
pick with `tier:` and `default_tier:`. An app is a second agent that shares the
deployment's services and data (logbook, knowledge, channel finder) in
`apps/<name>.yml`, listed under `modules.web_terminals.shared:` and opened from
a shared card on the landing page. `osprey init` writes both folders out in
full, and `osprey scaffold tiers` and `osprey scaffold apps` add them to an
existing repo. `dispatch.tier` names the tier the event dispatcher's workers
run. `osprey profile card` lists the `graph` and `event_dispatcher` servers
when a render ships them.
