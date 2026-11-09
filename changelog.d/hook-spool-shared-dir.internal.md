The hooks' audit spool creates its directory through the sibling
`osprey_safety_file.ensure_shared_dir`, the one statement of a group-shared
location, instead of a restated mode and its own `mkdir`/`chmod`; the
`hooks/hook-log` build artifact now stages `hooks/safety-file` beside it.
Two shifts follow: a spool directory that already exists is no longer
chmodded, and the parent levels the spool creates on the way to it now get
`2770` as well.
