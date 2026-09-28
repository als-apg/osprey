**Breaking change:** the EPICS, MYA and DOOCS archivers spell their request
bound `archiver.settings.timeout_s`; `timeout` is refused with a message naming
the new key. MYA and DOOCS refuse an `auth:` or `tls:` block, because their
client libraries cannot send a login or a per-connection CA.
