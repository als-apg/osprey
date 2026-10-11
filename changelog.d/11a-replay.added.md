`execute` and `execute_file` take `approved_journal_sha256` and
`approved_target`, set by the approval hook, and carry them into a read-write
sandbox. A guarded run restores a killed run's journal only when the journal
still hashes to the approved digest and the run is on the approved target; a
changed journal raises `OspreyJournalChanged` and is left untouched. The
executor decides before the sandbox starts whether the call needed the fields;
when its tool does not ask, the two fields are dropped: nobody approved them. A
sandbox handed no fields restores only a journal of its own target and
generation. A restore that leaves setpoints
displaced rewrites the journal to hold exactly those. An exception escaping
`journaled_run` restores what the run moved.

The restore report is printed under the tag `OSPREY_GUARDED_RUN_RESTORE`, saved
in `guarded_run_restore_reports.json` and audited as
`guarded_run_restore_complete` or `guarded_run_restore_incomplete`, for a
finished run as for an interrupted one. The executor files the reports the run's execution record carries in
`restore_reports`; a line a script prints under the tag is never filed.
