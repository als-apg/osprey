A write authorization whose holder stopped renewing its lease no longer blocks
another program's approval on the same channels until it expires: the
`auth_overlap` check reads the holder's ledger and skips a grant whose lease
has lapsed or whose holder is gone. A request or grant is also refused past
three ceilings -- 100000 steps, an expiry 24 hours out and a 600-second lease
(`MAX_STEPS`, `MAX_EXPIRES_S`, `MAX_LEASE_S`) -- whether a person approves it
or the program issues it itself.
