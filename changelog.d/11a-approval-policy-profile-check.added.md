`osprey build` and `osprey facility validate` stop with `profile-invalid` when a
guarded tool (`execute`, `execute_file`, `pyaml_measure`) has approval policy
`skip` while `approval.tools.channel_write` does not; the line names the key
that gives the tool `skip` and the key that gives `channel_write` its policy.
