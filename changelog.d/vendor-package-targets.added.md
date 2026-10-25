A vendor manifest target can name `"base": "package"` to land a file relative
to the installed `osprey` package, and `osprey vendor fetch --target DIR`
fetches only that target. A vendor asset is cached as immutable only on a
200 response, and the design-system validator skips `vendor/` in its hex
colour scan.
