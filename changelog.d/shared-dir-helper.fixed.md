The session ledger and nested facility-knowledge concepts no longer create their
directory at the process umask: under umask `077` that left a `0640` file in a
`0700` directory another uid could not enter, and when the ledger got to an
identity's state directory first the control-context record found it existing
and set no mode either. Every writer of a group-shared location now creates it
through one helper that sets each level it creates to `2770`.
