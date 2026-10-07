The MongoDB archiver trusts a site certificate authority named by
`tls.ca_bundle`. Without it, a `tls=true` connection string checks the store's
certificate against the container's trust store.
