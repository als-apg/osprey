An application backend served by the testing kit (`osprey.apps.sdk.testing`)
no longer refuses every write as `owner_unknown`. `serve(login)` now gives a
container that drives the target the login-to-tier map a build stages for it,
each admitted owner at the login's write ceiling, and `serve(login,
tiers={owner: "never"})` states another tier for a test of a refusal.
