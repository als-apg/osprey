`osprey up` no longer refuses to start a service it is moving off the host
network: the host-port preflight read that service's own still-running
host-network container as "an unknown host process" holding the port the new
render publishes. A host-network `app_runtime` (or any service whose key has an
underscore) is likewise recognised again on a plain redeploy, where its
`<project>-app-runtime` container name used to go unmatched.
