A connector block whose gateway has an empty or non-string `address`, a `port`
that is not an integer, or a `use_name_server` that spells neither true nor
false, and a `pva_channels` that is not a glob or a list of them, is refused
by `osprey build` and when the controls server builds the connector, naming the
key; before, it built and the connection or the target check failed later.
A target's eligibility refuses only the gateway row of the role it selects.
A service connection's `auth.username` with space around it is refused.
