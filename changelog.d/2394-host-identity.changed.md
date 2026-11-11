`osprey build` records `deployment.host_identity` in lowercase, with other
characters replaced by `-`, when the login is not already such a name. Run
`osprey build` once. Entries already written under the old spelling keep it;
on a case-sensitive filesystem the old `var/spool/<Login>` directory stays
beside the new one. `osprey write-authorization` and `osprey health` now use
the recorded identity. A name stated in `profile.yml` is never changed.
