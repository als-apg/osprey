`osprey build` records `deployment.host_identity` lowercased, with other
characters replaced by `-`, when the login is not already such a name. Entries
written before keep the login's spelling; on a case-sensitive filesystem the
old `var/spool/<Login>` directory stays beside the new one. The
`osprey write-authorization` and `osprey health` commands now use the recorded
identity too. A name stated in `profile.yml` is never changed.
