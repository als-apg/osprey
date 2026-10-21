The python executor decides whether a read-write `execute`, `execute_file` or
`pyaml_measure` call was approved before it starts anything. When the tool's
approval policy asks, the approval hook's record of the call must exist and its
`approved_target` (and, for `pyaml_measure`, its `approved_view_sha256`) must
equal the call's; a call approved for another target than the run's is refused
too. Otherwise nothing runs and the result carries
`failure_kind: approval_missing`, which reaches the agent as an
`execution_error` whose `details.kind` is `approval_missing`. The approval hook
writes those fields into its record only after the prompt is sent, and nothing
of the record reaches the sandbox. A malformed saved-artifact manifest no
longer fails the run: entries that name no file are skipped. This check runs
beside the script on one host.
