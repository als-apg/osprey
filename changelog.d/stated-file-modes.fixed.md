Concepts drafted into the shared facility-knowledge bundle and the session
ledger are now written group-readable (`0640`), so another web terminal or a
dispatch worker running under its own uid can read them through the
directory's group. Every other file OSPREY rewrites whole now states its mode
beside the writer, with the reader it is chosen for, instead of inheriting
`0600` from `tempfile.mkstemp`.
