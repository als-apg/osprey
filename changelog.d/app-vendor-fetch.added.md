An application's `vendor:` entries are fetched into `apps/<name>/<into>`, each
checked against its SHA256; a target outside the application, or a file name
that does not carry the version it fetches, is refused, and an offline or failed
fetch reuses a verified copy or fails naming the application and package.
`osprey vendor fetch --target DIR` fetches one manifest target, which may name
`"base": "package"`, and an application panel's `/vendor/` mount is cached as
immutable like an interface's static vendor tree; proxied panel traffic and cap
paths keep their own headers, and a route's `no-store` is never overridden.
