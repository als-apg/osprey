Every login lives in `tiers/`: a narrow one says `lists: own`, and one named by
`shared:` is sealed read-only. `osprey init --revalidate` moves an
`apps/<name>.yml` login to `tiers/` for one release; `osprey scaffold apps`
refuses, naming `osprey scaffold tiers` and `osprey app new`; landing
`type: apps` reads as `type: shared` ("Shared logins") for one release; and the
build fingerprint covers each `apps/<name>/` application directory.
