The multi-user login guide names `osprey users remove` where it named a
`decommission` command that does not exist, and the seeded-password refusal
now tells a shared card to delete its stored hash from `.env.auth` and run
`osprey up` instead of naming that command. The guide now also documents the
login service's answer to nginx, how a site sign-in that is not OIDC is
brokered to it, what the roster is, and that logging out at the identity
provider does not end an OSPREY session.
