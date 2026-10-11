The approval prompt of a read-write `execute` or `execute_file` call reads the
control target's guarded-run directory. A journal a killed run left is listed
on the prompt: who ran it, its pid and start time, the target and generation
the journal records, then every displaced setpoint with the value it is
restored to, each string JSON-quoted. A journal holding more than 500
setpoints is refused with its path. The approved call carries
`approved_journal_sha256` (the sha256 of the listed journal, or `none` when
nothing is pending) and `approved_target`, set over any value the agent
supplied. A call is denied when a guarded run holds the target's lock (`a
guarded run is in progress on <target>`), when the journal cannot be read
(`pending guarded-run journal unreadable: <path>`), and, in a dispatch run,
when a journal is pending. A read-only call keeps its prompt and carries
neither field.
