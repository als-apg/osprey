A web-terminal login whose stored password hash the login service cannot read
is now recorded as `credential_unevaluable` instead of `bad_credential`, and the
service names the user at startup. The browser still sees the ordinary refusal;
`osprey users passwd <user>` replaces the hash.
