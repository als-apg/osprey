Every image OSPREY builds now accepts the site CA, proxy bypass list and package
index a deployment configures under `images:`, and the compose build hands them
to each service image it builds. A build behind a TLS-intercepting proxy, or on
an internal package index, no longer stops at the seven managed service images.
