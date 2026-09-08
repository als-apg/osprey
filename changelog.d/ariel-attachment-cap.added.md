The largest file a logbook entry may attach is now a config key,
`ariel.attachments.max_file_mb` (default 10, the previous fixed bound). The
refusal names the limit in force. Attachments live in the same Postgres as the
logbook, so raising it spends database storage.
