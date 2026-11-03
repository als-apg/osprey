A vendor asset is cached as immutable only on a 200 response; an error or
redirect on a vendor path is no longer cached. The design-system validator
skips a panel's `vendor/` directory in its hex colour scan.
