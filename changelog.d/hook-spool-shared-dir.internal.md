The hooks' audit spool creates its directory through the sibling
`osprey_safety_file.ensure_shared_dir`, the one statement of a group-shared
location, instead of a restated mode and its own `mkdir`/`chmod`; the
`hooks/hook-log` build artifact now stages `hooks/safety-file` beside it.
