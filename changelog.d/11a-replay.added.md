`execute` and `execute_file` take `approved_journal_sha256` and
`approved_target`, set by the approval hook, and carry them into a read-write
sandbox. A guarded run restores a killed run's journal only when the journal
still hashes to the approved digest and the run is on the approved target; a
changed journal raises `OspreyJournalChanged` and is left untouched. A call
without the approved fields refuses when its tool asks for approval, and
otherwise restores a journal of its own target and generation. Whether the tool
asks is read once, before user code starts, and when it does not ask the two
fields are dropped: nobody approved them. A restore that leaves setpoints
displaced rewrites the journal to hold exactly those. An exception escaping
`journaled_run` restores what the run moved.

The build records whether a render runs the approval hook as
`approval.hook_wired` in `config.yml`, from the profile's `hooks:` list; a
`config:` line for it is refused.

The restore report is tagged `OSPREY_GUARDED_RUN_RESTORE`, saved as
`guarded_run_restore_reports.json` and audited as `guarded_run_restore_complete`
or `guarded_run_restore_incomplete`, for a finished run as for an interrupted
one. The executor files a report line only when it carries the nonce drawn for
that sandbox, so a line a script prints under the tag is not a report.
