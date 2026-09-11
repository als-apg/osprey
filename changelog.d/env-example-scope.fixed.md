`.env.example` and the generated README no longer claim to list every variable
the deployment reads. The file carries the variables the deployment itself
supplies — provider keys, whatever the profile declares, and the tokens `osprey
up` mints — and points at the Environment Variables reference, which now
gathers the host-level knobs (`CONTAINER_RUNTIME`, `OSPREY_OFFLINE`,
`OSPREY_CA_BUNDLE`, `REGISTRY_PATH`, `OSPREY_TERMINAL_BIND_HOST` and the two
image axes) in one place.
