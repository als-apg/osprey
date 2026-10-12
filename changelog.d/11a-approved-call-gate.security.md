The python executor decides whether a read-write `execute` or `execute_file`
call was approved before it starts anything. When the tool's approval policy
asks, the approval hook's record of the call must exist and its
`approved_journal_sha256` and `approved_target` must equal the call's; otherwise
nothing runs and the result carries `failure_kind: approval_missing`. The
refusal reaches the agent as an `execution_error` whose `details.kind` is
`approval_missing`. The approval hook writes those fields into its record only
after the prompt is sent.
The sandbox binds the fields it is handed and reads no approval policy. After a
read-write run the executor files what the sandbox reported (detail
`source=sandbox`) and what it read in the target's journal itself (detail
`source=observed_on_disk`): `guarded_run_journal_left` names the setpoints the
journal still holds, and `guarded_run_journal_absent` records an abnormal exit
that left no journal entry. The restore reports reach the executor in the run's
`execution_metadata.json` as `restore_reports`; the report nonce and
`OSPREY_GUARDED_RUN_REPORT_NONCE` are gone. A read-write run whose script was
started files its record, also when its result cannot be built, and a malformed
saved-artifact manifest no longer fails the run: entries that name no file are
skipped. The executor also refuses a call approved for another target than the
run's. This check runs beside the script on one host; moving the journal, the
restore after a dead run and the report into the connector host is the planned
end state.
