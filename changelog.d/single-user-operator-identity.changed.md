On a single-user deployment, the operator's control-context directory, write
tier and panel login now all use `deployment.host_identity`, the name
`osprey web` signs in as. Before, they used the account running `osprey build`,
so a profile-stated host identity or an `OSPREY_AUDIT_IDENTITY` set at build
time left a request the chip could not approve. Multi-user deployments are
unchanged. A build with no account name gets no directory, tier or panel login.
