`osprey up` refuses to start a password-login deployment whose browser address is not this
machine while any login still accepts a password published in `profile.yml`, such as the
preset's demo logins. Run `osprey users passwd <user>` for each user it names.
`osprey scaffold web-terminals lint` reports the same finding as
`web_terminals.auth_seeded_password`.
