The project image and each web-terminal persona image are now built with the
site's network settings. A top-level `offline` key and an `images.site_ca` /
`images.pip_no_proxy` / `images.pip_index_url` / `images.pip_extra_index_url`
block are passed to those builds as build arguments, so a
deployment behind a TLS-intercepting proxy, on an internal package index, or on
an air-gapped host is configured once in `config.yml` instead of only in a
hand-run `docker build`. Each is overridden for one shell by an environment
variable of the same name. A deployment that sets none of them builds exactly
what it built before.
