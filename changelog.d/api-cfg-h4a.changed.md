A connector block's malformed gateway row for the role a target does not select
no longer refuses the target's eligibility; only the selected role's row is
refused, naming that role. `logging.rich_tracebacks: 'false'` (and the other two
`logging.*` switches written as quoted words) now turns the switch off, and a
switch written as a null keeps its default. A MongoDB archive whose `port` is
written as a numeric string now connects; one that names no port is refused when
the archive is built.
