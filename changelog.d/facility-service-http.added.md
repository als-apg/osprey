A service your deployment brings with it can now say that it answers HTTP, with
`services.<name>.config.http: true`. The deploy summary then prints its address
as a link you can open instead of a bare `host:port`. Default off, so a service
speaking a binary protocol is never shown as a link that cannot open.
