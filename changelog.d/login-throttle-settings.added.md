Multi-user web terminals: `modules.web_terminals.auth.throttle` sets how failed
password logins are slowed — `initial_delay_s`, `multiplier`, `max_delay_s` and
`forget_after_s`, defaulting to the previous fixed 1 s doubling to 30 s,
forgotten after 300 s quiet. Logins are slowed, never locked out; a key left out
or written with no value takes its default, and `osprey build` refuses a value
the throttle cannot use.
