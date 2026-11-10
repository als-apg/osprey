A write authorization whose holder stopped renewing its lease no longer blocks
another program's approval on the same channels until it expires: the
`auth_overlap` check reads the holder's ledgers and skips a grant only when
every one it finds shows its lease time passed more than a minute ago. A
request or grant is also refused past three ceilings -- 100000 steps, an
expiry 24 hours out and a 600-second lease -- whether a person approves it or
the program issues it itself.
