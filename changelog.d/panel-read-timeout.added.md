Web terminal panels take `web.panels.<id>.read_timeout_s`: a number of
seconds that bounds that panel's proxied requests, or `null` for unbounded. A
panel that omits it keeps the shared 30 s default.
