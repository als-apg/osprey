A run built for the `asksage` provider now reports a missing credential by
name instead of failing inside the build: the provider table names
`ASKSAGE_API_KEY` as the variable the key arrives in, and `.env.example`
lists it.
