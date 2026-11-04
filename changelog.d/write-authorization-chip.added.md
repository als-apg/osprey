The web terminal shows and approves programs' bounded write authorizations.
`GET /api/terminal/write-authorizations` lists the requests waiting on the
signed-in login (read from the holders' request directory,
`OSPREY_WRITE_REQUESTS_DIR`) and the login's authorizations, each with its
steps used of its total; `GET /api/terminal/write-authorizations/requests/{id}`
is the approve card (channels, ranges, the limits bound of each range, current
values read together within about 3 s (a slow machine's pool closes in the background), steps, expiry, lease, holder, label, completion,
and that a write the limits refuse still costs its step);
`POST /api/terminal/write-authorizations` `{request_id}` approves, refusing a
target the render does not configure, a login whose ceiling is `never` or
`unknown`, a read-only run, a range outside the limits database, a
non-numeric channel and a channel another program of this login holds
(`auth_overlap`); `POST /api/terminal/write-authorizations/{id}/revoke` stops
one. An approval is the confirmation for every write of the run, also for a
login whose ceiling is `asked`. The popover gains a Programs section, which
also lists a program's self-issued grant at `allowed` (`self_issued`, not
`revocable`: moving the switch off `allowed` ends it), shows expiries as the
time left, says *reading values…* while a card loads and *Approving…* while an
approval is sent, names in the Limits column a channel the limits database
does not list or marks not writable, names every channel a refusal is about,
and says *Already stopped.* for a revoke of one that had ended. `GET /api/terminal/posture` rows
gain `program_hold` and `programs` (`used` included); a grant whose lease
lapsed or whose steps are spent is not one. The chip keeps the switch word for
such a machine and adds `· program` (for example `writes off · program`) with
a ring dot, `any_armed` counts it, and the row's off control revokes it.
