The web-terminal login service's image now carries the site CA. `osprey up`
installs the certificate named by `images.site_ca` into it exactly as it does
for the project image, so a deployment behind a proxy that re-signs TLS can
reach its identity provider. Before this the stack started, its health check
was green, and every login failed at the discovery fetch. The same build also
takes the site's package-index and proxy-bypass settings, so a login image on
an internal mirror no longer reaches out to PyPI.
