**Breaking change:** The `app_template:` profile key and the packaged app
templates behind it (`apps/*/config.yml.j2`) are gone — each preset carries
the config keys that template used to supply, and `osprey profile expand`
writes them into a profile that still names one. Two spellings that depended
on that layer go with it: `config: api.providers.*`, which is refused in
favour of the `providers.yml` catalog beside the profile, and removing the
graph store by writing `config: services.graphdb: {}` or a bare
`services.graphdb:`, which is refused because nothing injects a store to
subtract — delete the block's keys instead.
