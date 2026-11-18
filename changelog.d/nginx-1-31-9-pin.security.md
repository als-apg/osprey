The nginx front door of a multi-user deployment is now pinned to
`nginx:1.31.9-alpine`, the release that fixes the unauthenticated HTTP/2
memory-corruption flaws CVE-2026-123007 and CVE-2026-123633 ("nginx2root").
A deployment is exposed only with `tls.enabled: true`, which is where HTTP/2
is on. The previous floating `nginx:1.31.9-alpine` tag left a host that had it
cached on the vulnerable 1.31.8; redeploy after upgrading to pull the fixed
image, and re-copy it first if `images.base_registry` names a mirror.
